import os
import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from data.datasets.classification.common.dataobjects import ClassificationStructure, LoaderObject
from data.datasets.classification.CIFAR100.dataset import LoaderCIFAR100
from data.datasets.classification.common.custom_transforms import Cutout
from config import BaseConfig


def get_cifar100(cfg: BaseConfig, idxs: np.ndarray = None, test_bs: bool = False, num_classes: int = None):
    path = cfg.data.data_loc
    print(os.path.expanduser(path) + '/CIFAR100')
    raw_tr = datasets.CIFAR100(path + '/CIFAR100', train=True, download=cfg.data.download)
    raw_te = datasets.CIFAR100(path + '/CIFAR100', train=False, download=cfg.data.download)
    train_targets = np.array(raw_tr.targets)
    trainset = raw_tr.data
    test_targets = np.array(raw_te.targets)
    testset = raw_te.data

    if num_classes is not None:
        if num_classes == 0:
            raise ValueError('Dataset must have at least one class!')
        print(f'Size of training {trainset.shape[0]} and test set {testset.shape[0]} with all classes')
        trainset = trainset[train_targets < num_classes]
        train_targets = train_targets[train_targets < num_classes]

        testset = testset[test_targets < num_classes]
        test_targets = test_targets[test_targets < num_classes]
        print(f'Size of training {trainset.shape[0]} and test set {testset.shape[0]} with {num_classes} classes')
        if cfg.data.unbalance:
            print(f'Unbalancing dataset....')
            for i in range(num_classes):
                if i < num_classes // 2:
                    im_idxs = np.argwhere(train_targets == i)
                    remove_idxs = im_idxs[:400]
                    remove_idxs = np.squeeze(remove_idxs)

                    trainset = np.delete(trainset, remove_idxs, axis=0)
                    train_targets = np.delete(train_targets, remove_idxs)
                else:
                    im_idxs = np.argwhere(test_targets == i)
                    remove_idxs = im_idxs[:90]
                    remove_idxs = np.squeeze(remove_idxs)

                    testset = np.delete(testset, remove_idxs, axis=0)
                    test_targets = np.delete(test_targets, remove_idxs)

                print(f'Class {i}: {len(trainset[train_targets == i])}  training '
                      f'{len(testset[test_targets == i])} test')

            print(f'Total samples training {len(trainset)} test {len(testset)}')


    # if idxs is not None:
    #    raw_tr.data = raw_tr.data[idxs]
    #    raw_tr.targets = np.array(raw_tr.targets)[idxs]

    # init data configs
    data_config = ClassificationStructure()
    data_config.train_set = trainset
    data_config.train_labels = torch.from_numpy(train_targets)
    data_config.test_set = testset
    data_config.test_labels = torch.from_numpy(test_targets)
    if idxs is not None:
        data_config.train_len = len(idxs)
    else:
        data_config.train_len = len(data_config.train_labels)
    data_config.test_len = len(data_config.test_labels)
    data_config.num_classes = num_classes if num_classes is not None else 100
    data_config.img_size = 32

    data_config.is_configured = True

    # add transforms
    train_transform = transforms.Compose([])

    if cfg.data.augmentations.random_hflip:
        train_transform.transforms.append(transforms.RandomHorizontalFlip())
    if cfg.data.augmentations.random_crop:
        train_transform.transforms.append(transforms.RandomCrop(32, padding=4))

    # mandatory transforms
    mean = [x / 255.0 for x in [125.3, 123.0, 113.9]]
    std = [x / 255.0 for x in [63.0, 62.1, 66.7]]
    train_transform.transforms.append(transforms.ToTensor())
    train_transform.transforms.append(transforms.Normalize(mean=mean, std=std))

    # cutout requires tensor inputs
    if cfg.data.augmentations.cutout:
        train_transform.transforms.append(Cutout(n_holes=1, length=16))

    # test transforms
    test_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean=mean, std=std)])

    # create loaders
    bs = cfg.classification.batch_size
    testbs = cfg.classification.test_batch_size
    train_loader = DataLoader(LoaderCIFAR100(data_config=data_config,
                                             split='train',
                                             transform=train_transform, current_idxs=idxs),
                              batch_size=bs,
                              shuffle=True
                              )
    test_loader = DataLoader(LoaderCIFAR100(data_config=data_config,
                                            split='test',
                                            transform=test_transform),
                             batch_size=testbs,
                             shuffle=False)
    loaders = LoaderObject(train_loader=train_loader,
                           test_loader=test_loader,
                           data_configs=data_config)

    return loaders
