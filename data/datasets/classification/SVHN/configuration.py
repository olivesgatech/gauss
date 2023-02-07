import os
import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from data.datasets.classification.common.dataobjects import ClassificationStructure, LoaderObject
from data.datasets.classification.SVHN.dataset import LoaderSVHN
from config import BaseConfig


def get_svhn(cfg: BaseConfig, idxs: np.ndarray = None):
    path = cfg.data.data_loc
    print(os.path.expanduser(path) + '/SVHN')
    raw_tr = datasets.SVHN(os.path.expanduser(path) + '/SVHN', split='train', download=cfg.data.download)
    raw_te = datasets.SVHN(os.path.expanduser(path) + '/SVHN', split='test', download=cfg.data.download)

    # if idxs is not None:
    #     raw_tr.data = raw_tr.train_data[idxs]
    #     raw_tr.targets = raw_tr.train_labels[idxs]
    #     raise Exception('Active learning not implemented for loader yet!')

    # init data configs
    data_config = ClassificationStructure()
    data_config.train_set = raw_tr.data
    data_config.train_labels = torch.from_numpy(raw_tr.labels)
    data_config.test_set = raw_te.data
    data_config.test_labels = torch.from_numpy(raw_te.labels)
    data_config.train_len = len(data_config.train_labels)
    data_config.test_len = len(data_config.test_labels)
    data_config.num_classes = 10
    data_config.img_size = 32

    data_config.is_configured = True

    # add transforms
    train_transform = transforms.Compose([])

    if cfg.data.augmentations.random_hflip:
        train_transform.transforms.append(transforms.RandomHorizontalFlip())
    if cfg.data.augmentations.random_crop:
        train_transform.transforms.append(transforms.RandomCrop(32, padding=4))

    # mandatory transforms
    mean = [0.4377, 0.4438, 0.4728]
    std = [0.1980, 0.2010, 0.1970]
    train_transform.transforms.append(transforms.ToTensor())
    train_transform.transforms.append(transforms.Normalize(mean=mean, std=std))

    # test transforms
    test_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean=mean, std=std)])

    # create loaders
    train_loader = DataLoader(LoaderSVHN(data_config=data_config,
                                         split='train',
                                         transform=train_transform,
                                         current_idxs=idxs),
                              batch_size=cfg.classification.batch_size,
                              shuffle=True
                              )
    test_loader = DataLoader(LoaderSVHN(data_config=data_config,
                                         split='test',
                                         transform=test_transform),
                              batch_size=cfg.classification.batch_size,
                              shuffle=False)
    loaders = LoaderObject(train_loader=train_loader,
                           test_loader=test_loader,
                           data_configs=data_config)

    return loaders