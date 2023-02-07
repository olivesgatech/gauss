import os
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from data.datasets.classification.common.dataobjects import ClassificationStructure, LoaderObject
from data.datasets.classification.CURETSR.dataset import LoaderCURETSR
from data.datasets.classification.common.custom_transforms import Cutout
from data.datasets.classification.CURETSR.utils import *
from config import BaseConfig


def get_curetsr(cfg: BaseConfig, idxs: np.ndarray = None, test_bs: bool = False):
    path = cfg.data.data_loc
    print(os.path.expanduser(path) + '/CURE-TSR')
    train_dir = os.path.expanduser(path) + '/CURE-TSR/Real_Train/ChallengeFree'
    test_dir = os.path.expanduser(path) + '/CURE-TSR/Real_Test/ChallengeFree'
    tr_data = make_dataset(train_dir)
    te_data = make_dataset(test_dir)
    tr_files = [obj[0] for obj in tr_data]
    tr_labels = [obj[1] for obj in tr_data]
    te_files = [obj[0] for obj in te_data]
    te_labels = [obj[1] for obj in te_data]

    # init data configs
    data_config = ClassificationStructure()
    data_config.train_set = tr_files
    data_config.train_labels = np.array(tr_labels)
    data_config.test_set = te_files
    data_config.test_labels = np.array(te_labels)
    data_config.train_len = len(tr_files)
    data_config.test_len = len(te_files)
    data_config.num_classes = 14
    if cfg.classification.model[:5] == 'dense':
        data_config.img_size = 32
    else:
        data_config.img_size = 28


    data_config.is_configured = True

    # add transforms
    train_transform = transforms.Compose([])
    train_transform.transforms.append(transforms.Resize([data_config.img_size, data_config.img_size]))
    if cfg.data.augmentations.random_hflip:
        train_transform.transforms.append(transforms.RandomHorizontalFlip())
    if cfg.data.augmentations.random_crop:
        train_transform.transforms.append(transforms.RandomCrop(data_config.img_size, padding=3))

    # mandatory transforms
    mean = [x / 255.0 for x in [125.3, 123.0, 113.9]]
    std = [x / 255.0 for x in [63.0, 62.1, 66.7]]
    train_transform.transforms.append(transforms.ToTensor())
    train_transform.transforms.append(transforms.Normalize(mean=mean, std=std))

    # cutout requires tesnor inputs
    if cfg.data.augmentations.cutout:
        train_transform.transforms.append(Cutout(n_holes=1, length=16))

    # test transforms
    test_transform = transforms.Compose([transforms.Resize([data_config.img_size, data_config.img_size]),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=mean, std=std)])

    # create loaders
    #if test_bs:
    #    bs = 1
    #else:
    bs = cfg.classification.batch_size
    train_loader = DataLoader(LoaderCURETSR(data_config=data_config,
                                            split='train',
                                            transform=train_transform, current_idxs=idxs),
                              batch_size=bs,
                              shuffle=True
                              )
    test_loader = DataLoader(LoaderCURETSR(data_config=data_config,
                                           split='test',
                                           transform=test_transform),
                             batch_size=bs,
                             shuffle=False)
    loaders = LoaderObject(train_loader=train_loader,
                           test_loader=test_loader,
                           data_configs=data_config)

    return loaders