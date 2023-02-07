import os
import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from data.datasets.classification.common.dataobjects import ClassificationStructure, LoaderObject
from data.datasets.classification.CINIC10.dataset import LoaderCINIC10
from data.datasets.classification.common.custom_transforms import Cutout
from config import BaseConfig


def get_cinic10(cfg: BaseConfig, idxs: np.ndarray = None, test_bs: bool = False):
    path = cfg.data.data_loc
    print(os.path.expanduser(path) + '/CINIC10')
    raw_tr = datasets.ImageFolder(path + '/CINIC10/train')
    raw_te = datasets.ImageFolder(path + '/CINIC10/test')

    #if idxs is not None:
    #    raw_tr.data = raw_tr.data[idxs]
    #    raw_tr.targets = np.array(raw_tr.targets)[idxs]

    # init data configs
    data_config = ClassificationStructure()
    data_config.train_set = raw_tr
    data_config.test_set = raw_te
    data_config.train_len = len(raw_tr)
    data_config.test_len = len(raw_te)
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
    mean = [0.47889522, 0.47227842, 0.43047404]
    std = [0.24205776, 0.23828046, 0.25874835]
    train_transform.transforms.append(transforms.ToTensor())
    train_transform.transforms.append(transforms.Normalize(mean=mean, std=std))

    # cutout requires tesnor inputs
    if cfg.data.augmentations.cutout:
        train_transform.transforms.append(Cutout(n_holes=1, length=16))

    # test transforms
    test_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean=mean, std=std)])

    # create loaders
    #if test_bs:
    #    bs = 1
    #else:
    bs = cfg.classification.batch_size
    train_loader = DataLoader(LoaderCINIC10(data_config=data_config,
                                            split='train',
                                            transform=train_transform, current_idxs=idxs),
                              batch_size=bs,
                              shuffle=True
                              )
    test_loader = DataLoader(LoaderCINIC10(data_config=data_config,
                                           split='test',
                                           transform=test_transform),
                             batch_size=bs,
                             shuffle=False)
    loaders = LoaderObject(train_loader=train_loader,
                           test_loader=test_loader,
                           data_configs=data_config)

    return loaders
