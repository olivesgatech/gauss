import os
import copy
import tqdm
import torch
import torch.nn as nn
import numpy as np
from models.architectures import build_architecture
from training.common.trainutils import determine_multilr_milestones
from data.datasets.classification.common.aquisition import get_dataset
from training.classification.classificationtracker import ClassifcationTracker
from training.common.saver import Saver
from training.common.summaries import TensorboardSummary
from config import BaseConfig


class ClassificationTrainer:
    def __init__(self, cfg: BaseConfig):
        self._cfg = cfg
        self._epochs = cfg.classification.epochs
        self._device = cfg.run_configs.gpu_id

        # Define Saver
        self._saver = Saver(cfg)

        # Define Tensorboard Summary
        self._summary = TensorboardSummary(self._saver.experiment_dir)
        self._writer = self._summary.create_summary()

        self._loaders = get_dataset(cfg)

        self._model = build_architecture(self._cfg.classification.model, self._loaders.data_config, cfg)

        self._softmax = nn.Softmax(dim=1)

        if self._cfg.classification.loss == 'ce':
            self._loss = nn.CrossEntropyLoss()
        else:
            raise Exception('Loss not implemented yet!')

        if self._cfg.classification.optimization.optimizer == 'adam':
            self._optimizer = torch.optim.Adam(self._model.parameters(),
                                               lr=self._cfg.classification.optimization.lr)
        elif self._cfg.classification.optimization.optimizer == 'sgd':
            self._optimizer = torch.optim.SGD(self._model.parameters(),
                                              lr=self._cfg.classification.optimization.lr,
                                              momentum=0.9,
                                              nesterov=True,
                                              weight_decay=5e-4)
        else:
            raise Exception('Optimizer not implemented yet!')

        if self._cfg.classification.optimization.scheduler == 'multiLR':
            milestones = determine_multilr_milestones(self._epochs, self._cfg.classification.optimization.multiLR_steps)
            self._scheduler = torch.optim.lr_scheduler.MultiStepLR(self._optimizer,
                                                                   milestones=milestones,
                                                                   gamma=self._cfg.classification.optimization.gamma)
        elif self._cfg.classification.optimization.scheduler == 'none':
            pass
        else:
            raise Exception('Scheduler not implemented yet!')

        # Using cuda
        if self._cfg.run_configs.cuda:
            # use multiple GPUs if available
            self._model = torch.nn.DataParallel(self._model,
                                                device_ids=[self._cfg.run_configs.gpu_id])
        else:
            self._device = torch.device('cpu')

        # LD stats
        self.train_statistics = ClassifcationTracker(self._loaders.data_config.train_len)
        self.test_statistics = ClassifcationTracker(self._loaders.data_config.test_len)

        if self._cfg.run_configs.resume != 'none':
            resume_file = self._cfg.run_configs.resume
            # we have a checkpoint
            if not os.path.isfile(resume_file):
                raise RuntimeError("=> no checkpoint found at '{}'".format(resume_file))
            # load checkpoint
            checkpoint = torch.load(resume_file)
            # minor difference if working with cuda
            if self._cfg.run_configs.cuda:
                self._model.load_state_dict(checkpoint['state_dict'])
            else:
                self._model.load_state_dict(checkpoint['state_dict'])
            self._optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(resume_file, checkpoint['epoch']))

    def get_device(self):
        return torch.device(f'cuda:{self._device}') if self._cfg.run_configs.cuda else torch.device('cpu')

    def training(self, epoch, save_checkpoint=False, track_summaries=False):
        '''Trains the model in the given epoch. It uses the training loader to get the dataset and trains the model
        for one epoch'''
        # sets model into training mode -> important for dropout batchnorm. etc.
        self._model.train()
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
            self._optimizer.zero_grad()

            # convert image to suitable dims
            image = image.float()

            # computes output of our model
            output = self._model(image)

            _, pred = torch.max(output.data, 1)
            total += target.size(0)

            # collect forgetting events
            acc = pred.eq(target.data)

            # check if prediction has changed
            predicted = pred.cpu().numpy()
            self.train_statistics.update(acc.cpu().numpy(), predicted, idxs.cpu().numpy())

            # Perform model update
            # calculate loss
            loss = self._loss(output, target.long())
            # perform backpropagation
            loss.backward()

            # update params with gradients
            self._optimizer.step()

            # extract loss value as float and add to train_loss
            train_loss += loss.item()
            tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))
            if track_summaries:
                self._writer.add_scalar('train/total_loss_iter', loss.item(), i + num_img_tr * epoch)

            correct_samples += acc.cpu().sum()

        # Update optimizer step
        if self._cfg.classification.optimization.scheduler != 'none':
            self._scheduler.step(epoch)

        # calculate accuracy
        acc = 100.0 * correct_samples.item() / total
        if track_summaries:
            self._writer.add_scalar('train/total_loss_epoch', train_loss, epoch)
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self._cfg.classification.batch_size + image.data.shape[0]))
        print('Loss: %.3f' % train_loss)
        print('Training Accuracy: %.3f' % acc)

        # save checkpoint
        if save_checkpoint:
            self._saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self._model.state_dict(),
                'optimizer': self._optimizer.state_dict(),
            })
        return acc

    def testing(self, epoch, alternative_loader_struct = None, track_embeddings: bool = False, name: str = None):
        """
        tests the model on a given holdout set. Provide an alterantive loader structure if you do not want to test on
        the test set.
        :param epoch:
        :param alternative_loader_struct:
        :return:
        """
        # sets model into training mode -> important for dropout batchnorm. etc.
        self._model.eval()
        # initializes cool bar for visualization
        if alternative_loader_struct is None:
            num_samples = self._loaders.data_config.test_len
            num_classes = self._loaders.data_config.num_classes
            tbar = tqdm.tqdm(self._loaders.test_loader)
            num_img_tr = len(self._loaders.test_loader)
            tracker = self._test_statistics
        else:
            alternative_loader = alternative_loader_struct[0]
            tracker = alternative_loader_struct[1]
            num_classes = alternative_loader.data_config.num_classes
            if name == 'train':
                num_samples = alternative_loader.data_config.train_len
                tbar = tqdm.tqdm(alternative_loader.train_loader)
                num_img_tr = len(alternative_loader.train_loader)
            else:
                num_samples = alternative_loader.data_config.test_len
                tbar = tqdm.tqdm(alternative_loader.test_loader)
                num_img_tr = len(alternative_loader.test_loader)

        # init statistics parameters
        train_loss = 0.0
        correct_samples = 0
        total = 0

        # get embed dim
        if self._cfg.run_configs.cuda:
            embedDim = self._model.module.get_penultimate_dim()
        else:
            embedDim = self._model.get_penultimate_dim()

        # iterate over all samples in each batch i
        predictions = torch.zeros(num_samples, dtype=torch.long, device=self._device)
        probabilites = torch.zeros((num_samples, num_classes), dtype=torch.float, device=self._device)
        embeddings = torch.zeros((num_samples, embedDim * num_classes), dtype=torch.float)
        nongrad_embeddings = torch.zeros((num_samples, embedDim), dtype=torch.float)

        if self._cfg.run_configs.cuda:
            embeddings, nongrad_embeddings = embeddings.to(self._device), nongrad_embeddings.to(self._device)

        for i, sample in enumerate(tbar):
            image, target, idxs = sample['data'], sample['label'], sample['idx']
            # assign each image and target to GPU
            if self._cfg.run_configs.cuda:
                image, target = image.to(self._device), target.to(self._device)

            # update model
            self._optimizer.zero_grad()

            # convert image to suitable dims
            image = image.float()

            with torch.no_grad():

                # computes output of our model
                output = self._model(image)

                probs_output = self._softmax(output)
                probabilites[idxs] = probs_output

                _, pred = torch.max(output.data, 1)
                total += target.size(0)
                predictions[idxs] = pred

                if track_embeddings:
                    # get penultimate embedding
                    if self._cfg.run_configs.cuda:
                        penultimate = self._model.module.penultimate_layer
                    else:
                        penultimate = self._model.penultimate_layer

                    nongrad_embeddings[idxs] = penultimate

                    # insert to embediing array
                    for j in range(target.shape[0]):
                        for c in range(num_classes):
                            if c == pred[j].item():
                                embeddings[idxs[j], embedDim * c: embedDim * (c + 1)] = copy.deepcopy(penultimate[j]) * \
                                                                                        (1 - probs_output[j, c].item())
                            else:
                                embeddings[idxs[j], embedDim * c: embedDim * (c + 1)] = copy.deepcopy(penultimate[j]) * \
                                                                                        (-1 * probs_output[j, c].item())

                # collect forgetting events
                acc = pred.eq(target.data)

                # check if prediction has changed
                predicted = pred.cpu().numpy()
                tracker.update(acc.cpu().numpy(), predicted, idxs.cpu().numpy())

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
        if track_embeddings:
            out = {'predictions': predictions.cpu().numpy(),
                   'probs': probabilites.cpu().numpy(),
                   'grad_embeddings': embeddings.cpu().numpy(),
                   'nongrad_embeddings': nongrad_embeddings.cpu().numpy()}
        else:
            out = {'predictions': predictions.cpu().numpy(),
                   'probs': probabilites.cpu().numpy()}

        return out, acc

    def get_loaders(self, mode: str):
        if mode == 'test' or mode == 'train':
            return self._loaders
        else:
            raise Exception('Test mode not implemented yet')
