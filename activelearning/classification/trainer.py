import copy
import tqdm
import torch
import numpy as np
import torch.nn as nn
from training.classification.trainer import ClassificationTrainer
from training.classification.classificationtracker import ClassifcationTracker
from data.datasets.classification.common.aquisition import get_dataset
from config import BaseConfig


NUMMCD_SAMPLES = 70


def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()


class ActiveLearningClassificationTrainer(ClassificationTrainer):
    def __init__(self, cfg: BaseConfig):
        super(ActiveLearningClassificationTrainer, self).__init__(cfg=cfg)

        self._train_pool = np.arange(self._loaders.data_config.train_len)
        self.n_pool = self._loaders.data_config.train_len
        self._unlabeled_loader = None
        self._unlabeled_statistics = None

    def update_loader(self, idxs: np.ndarray, unused_idxs: np.array):
        self._loaders = get_dataset(self._cfg, idxs=idxs)
        if self._cfg.active_learning.strategy == 'badge':
            self._unlabeled_loader = get_dataset(self._cfg, idxs=unused_idxs, test_bs=True)
        else:
            self._unlabeled_loader = get_dataset(self._cfg, idxs=unused_idxs, test_bs=False)
        self._unlabeled_statistics = ClassifcationTracker(self._unlabeled_loader.data_config.train_len)
        self.train_statistics = ClassifcationTracker(self._loaders.data_config.train_len)

    def get_data_statistics(self, epoch: int, data: str = 'unlabeled'):
        # sets model into eval mode -> important for dropout batchnorm. etc.
        self._model.eval()
        # initializes cool bar for visualization
        if data == 'unlabeled':
            num_samples = self._unlabeled_loader.data_config.train_len
            tbar = tqdm.tqdm(self._unlabeled_loader.train_loader)
            num_img_tr = len(self._unlabeled_loader.train_loader)
            tracker = self._unlabeled_statistics
        elif data == 'labeled':
            num_samples = self._loaders.data_config.train_len
            tbar = tqdm.tqdm(self._loaders.train_loader)
            num_img_tr = len(self._loaders.train_loader)
            tracker = self.train_statistics
        else:
            raise Exception('Prediction type not implemented yet!')

        # init statistics parameters
        train_loss = 0.0
        correct_samples = 0
        total = 0

        # iterate over all samples in each batch i
        predictions = torch.zeros(self.n_pool, dtype=torch.long, device=self._device)
        probs = torch.full((self.n_pool, self._unlabeled_loader.data_config.num_classes), 2.5, dtype=torch.float)
        indices = torch.zeros(self.n_pool)
        accs = torch.zeros(self.n_pool, dtype=int)
        switches = np.zeros(self.n_pool)
        for i, sample in enumerate(tbar):
            image, target, idxs, local_idx = sample['data'], sample['label'], sample['global_idx'], sample['idx']
            # assign each image and target to GPU
            if self._cfg.run_configs.cuda:
                image, target = image.to(self._device), target.to(self._device)
                probs, indices, idxs = probs.to(self._device), indices.to(self._device), idxs.to(self._device)
                accs = accs.to(self._device)
                local_idx = local_idx.to(self._device)

            # update model
            self._optimizer.zero_grad()

            # convert image to suitable dims
            image = image.float()

            with torch.no_grad():

                # computes output of our model
                output = self._model(image)

                _, pred = torch.max(output.data, 1)
                total += target.size(0)
                predictions[idxs] = pred
                probs_output = self._softmax(output)

                # insert to probs array
                probs[idxs.long()] = probs_output
                indices[idxs.long()] = 1

                # collect forgetting events
                acc = pred.eq(target.data)
                accs[idxs.long()] = acc.long()

                # check if prediction has changed
                predicted = pred.cpu().numpy()
                switches[idxs.long().cpu().numpy()] = tracker.get_stats(local_idx.cpu().numpy(), tracking_type='SE')
                tracker.update(acc.cpu().numpy(), predicted, local_idx.cpu().numpy())

                # Perform model update
                # calculate loss
                loss = self._loss(output, target.long())

            # extract loss value as float and add to train_loss
            train_loss += loss.item()

            # Do fun bar stuff
            tbar.set_description('Test loss: %.3f' % (train_loss / (i + 1)))
            self._writer.add_scalar('test/total_loss_iter', loss.item(), i + num_img_tr * epoch)

            correct_samples += acc.cpu().sum()

        # Update optimizer step
        if self._cfg.classification.optimization.scheduler != 'none':
            self._scheduler.step()

        # calculate accuracy
        acc = 100.0 * correct_samples.item() / total
        self._writer.add_scalar('test/total_loss_epoch', train_loss, epoch)
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self._cfg.classification.batch_size + image.data.shape[0]))
        print('Loss: %.3f' % train_loss)
        print('Test Accuracy: %.3f' % acc)

        probs = probs[indices == 1].cpu().numpy()
        predictions = predictions[indices == 1]
        accs = accs[indices == 1]
        inds = (indices == 1).nonzero().cpu().numpy()
        switches = switches[indices.cpu().numpy() == 1]

        output_dict = {
            'predictions': predictions.cpu().numpy(),
            'scalar accuracy': acc,
            'indices': inds,
            'probabilities': probs,
            'switches': switches,
            'fevents': tracker.get_all_stats('FE'),
            'sample accuracy': accs.cpu().numpy()
        }

        return output_dict

    def get_embeddings(self, loader_type: str = 'unlabeled'):
        # set model to evaluation mode
        self._model.eval()

        # get embed dim
        if self._cfg.run_configs.cuda:
            embedDim = self._model.module.get_penultimate_dim()
        else:
            embedDim = self._model.get_penultimate_dim()

        # give me more baaaaaaaaaaaaaaaaaaaaaaaaaar!!!
        if loader_type == 'labeled':
            tbar = tqdm.tqdm(self._loaders.train_loader, desc='\r')
        elif loader_type == 'unlabeled':
            tbar = tqdm.tqdm(self._unlabeled_loader.train_loader, desc='\r')
        else:
            raise Exception('You can only load labeled and unlabeled pools!')

        # init softmax layer
        softmax = torch.nn.Softmax(dim=1)

        # init embedding tesnors and indices for tracking
        embeddings = torch.zeros((self.n_pool, embedDim * self._unlabeled_loader.data_config.num_classes),
                                 dtype=torch.float)
        nongrad_embeddings = torch.zeros((self.n_pool, embedDim),
                                         dtype=torch.float)
        indices = torch.zeros(self.n_pool)

        if self._cfg.run_configs.cuda:
            embeddings, indices = embeddings.to(self._device), indices.to(self._device)
            nongrad_embeddings = nongrad_embeddings.to(self._device)

        with torch.no_grad():
            # iterate over all sample batches
            for i, sample in enumerate(tbar):
                image, target, idxs, local_idx = sample['data'], sample['label'], sample['global_idx'], sample['idx']
                # assign each image and target to GPU
                if self._cfg.run_configs.cuda:
                    image, target = image.to(self._device), target.to(self._device)

                # convert image to suitable dims
                image = image.float()

                # computes output of our model
                output = self._model(image)

                # get penultimate embedding
                if self._cfg.run_configs.cuda:
                    penultimate = self._model.module.penultimate_layer
                else:
                    penultimate = self._model.penultimate_layer

                nongrad_embeddings[idxs] = penultimate

                # get softmax probs
                probs_output = softmax(output)

                _, pred = torch.max(output.data, 1)

                # insert to embediing array
                for j in range(target.shape[0]):
                    for c in range(self._unlabeled_loader.data_config.num_classes):
                        if c == pred[j].item():
                            embeddings[idxs[j], embedDim * c: embedDim * (c + 1)] = copy.deepcopy(penultimate[j]) * \
                                                                                    (1 - probs_output[j, c].item())
                        else:
                            embeddings[idxs[j], embedDim * c: embedDim * (c + 1)] = copy.deepcopy(penultimate[j]) * \
                                                                                    (-1 * probs_output[j, c].item())
                indices[idxs.long()] = 1

        # sort idxs
        output_structure = {}
        output_structure['embeddings'] = embeddings[indices == 1].cpu().numpy()
        ind_list = (indices == 1).nonzero().cpu().numpy()
        output_structure['indices'] = ind_list
        output_structure['nongrad_embeddings'] = nongrad_embeddings[indices == 1].cpu().numpy()

        return output_structure

    def get_batchbald_probs(self, model: nn.Module, optimizer):
        # sets model into eval mode -> important for dropout batchnorm. etc.
        model.eval()
        enable_dropout(model)
        # initializes cool bar for visualization
        num_samples = self._unlabeled_loader.data_config.train_len
        tbar = tqdm.tqdm(self._unlabeled_loader.train_loader)
        num_img_tr = len(self._unlabeled_loader.train_loader)

        # logsoftmax = torch.nn.LogSoftmax(dim=1)

        # init statistics parameters
        train_loss = 0.0
        correct_samples = 0
        total = 0

        # iterate over all samples in each batch i
        probs = torch.full((self.n_pool, NUMMCD_SAMPLES, self._unlabeled_loader.data_config.num_classes),
                           2.5, dtype=torch.float)
        indices = torch.zeros(self.n_pool)
        for i, sample in enumerate(tbar):
            image, target, idxs, local_idx = sample['data'], sample['label'], sample['global_idx'], sample['idx']
            # assign each image and target to GPU
            if self._cfg.run_configs.cuda:
                image, target = image.to(self._device), target.to(self._device)
                probs, indices, idxs = probs.to(self._device), indices.to(self._device), idxs.to(self._device)
                local_idx = local_idx.to(self._device)

            total += target.size(0)

            # update model
            optimizer.zero_grad()

            # convert image to suitable dims
            image = image.float()

            with torch.no_grad():

                # computes output of our model
                for j in range(NUMMCD_SAMPLES):
                    output = model(image)
                    probs_output = self._softmax(output)

                    # insert to probs array
                    probs[idxs.long(), j] = probs_output
                    indices[idxs.long()] = 1

                # derive bnn prediction
                mean = torch.squeeze(torch.mean(probs[idxs.long()], dim=1))
                _, pred = torch.max(mean.data, 1)
                acc = pred.eq(target.data)

                # calculate loss
                loss = self._loss(output, target.long())

            # extract loss value as float and add to train_loss
            train_loss += loss.item()

            # Do fun bar stuff
            tbar.set_description('Test loss: %.3f' % (train_loss / (i + 1)))

            correct_samples += acc.cpu().sum()

        # calculate accuracy
        acc = 100.0 * correct_samples.item() / total
        print('Loss: %.3f' % train_loss)
        print('Test Accuracy: %.3f' % acc)

        probs = probs[indices == 1]
        inds = (indices == 1).nonzero()

        output_dict = {
            'scalar accuracy': acc,
            'indices': inds,
            'probabilities': probs,
        }

        return output_dict

    def batchbald_train(self, model, optimizer):
        '''Trains the model in the given epoch. It uses the training loader to get the dataset and trains the model
        for one epoch'''
        # sets model into training mode -> important for dropout batchnorm. etc.
        model.train()
        # initializes cool bar for visualization
        tbar = tqdm.tqdm(self._loaders.train_loader)
        num_img_tr = len(self._loaders.train_loader)

        # init statistics parameters
        train_loss = 0.0
        correct_samples = 0
        total = 0

        # iterate over all samples in each batch i
        for i, sample in enumerate(tbar):
            image, target, idxs = sample['data'], sample['label'], sample['idx']
            # assign each image and target to GPU
            if self._cfg.run_configs.cuda:
                image, target = image.to(self._device), target.to(self._device)

            # update model
            optimizer.zero_grad()

            # convert image to suitable dims
            image = image.float()

            # computes output of our model
            output = model(image)

            _, pred = torch.max(output.data, 1)
            total += target.size(0)

            # collect forgetting events
            acc = pred.eq(target.data)

            # Perform model update
            # calculate loss
            loss = self._loss(output, target.long())
            # perform backpropagation
            loss.backward()

            # update params with gradients
            optimizer.step()

            # extract loss value as float and add to train_loss
            train_loss += loss.item()

            # Do fun bar stuff
            tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))

            correct_samples += acc.cpu().sum()

        # calculate accuracy
        acc = 100.0 * correct_samples.item() / total
        print('Loss: %.3f' % train_loss)
        print('Training Accuracy: %.3f' % acc)
        return model, optimizer

    def save_statistics(self, folder: str):
        self.train_statistics.save_statistics(folder, ld_type='training')
