import os
import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from data.datasets.classification.common.dataobjects import ClassificationStructure, LoaderObject
from data.datasets.classification.CIFAR10.dataset import LoaderCIFAR10
from data.datasets.classification.common.custom_transforms import Cutout
from config import BaseConfig

CORRUPTIONS = {
    'brightness': True,
    'contrast': True,
    'defocus_blur': True,
    'elastic_transform': True,
    'fog': True,
    'frost': True,
    'gaussian_blur': True,
    'gaussian_noise': True,
    'glass_blur': True,
    'impulse_noise': True,
    'jpeg_compression': True,
    # 'labels': True,
    'motion_blur': True,
    'pixelate': True,
    'saturate': True,
    # 'shot_noise': True,
    'snow': True,
    'spatter': True,
    # 'speckle_noise': True,
    'zoom_blur': True
}

LEVELS = [2, 5]


def get_cifar10c(cfg: BaseConfig):
    path = cfg.data.data_loc
    print(os.path.expanduser(path) + '/CIFAR-10-C/')
    output_loader_list = []

    for corruption in CORRUPTIONS:
        for level in LEVELS:
            x_corrupted_te = np.load(os.path.expanduser(path) + '/CIFAR-10-C/' + corruption + '.npy')
            y_corrupted_te = np.load(os.path.expanduser(path) + '/CIFAR-10-C/labels.npy')
            x_corrupted_te = x_corrupted_te[(level - 1) * 10000:level * 10000]
            y_corrupted_te = torch.from_numpy(y_corrupted_te[(level - 1) * 10000:level * 10000])

            # init data configs
            data_config = ClassificationStructure()
            data_config.test_set = x_corrupted_te
            data_config.test_labels = torch.from_numpy(np.array(y_corrupted_te))
            data_config.test_len = 10000
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
            mean = [x / 255.0 for x in [125.3, 123.0, 113.9]]
            std = [x / 255.0 for x in [63.0, 62.1, 66.7]]
            train_transform.transforms.append(transforms.ToTensor())
            train_transform.transforms.append(transforms.Normalize(mean=mean, std=std))

            # cutout requires tesnor inputs
            if cfg.data.augmentations.cutout:
                train_transform.transforms.append(Cutout(n_holes=1, length=16))

            # test transforms
            test_transform = transforms.Compose([transforms.ToTensor(),
                                                 transforms.Normalize(mean=mean, std=std)])

            bs = cfg.classification.batch_size
            test_loader = DataLoader(LoaderCIFAR10(data_config=data_config,
                                                   split='test',
                                                   transform=test_transform),
                                     batch_size=bs,
                                     shuffle=False)
            loaders = LoaderObject(test_loader=test_loader,
                                   train_loader=None,
                                   data_configs=data_config,
                                   name=corruption + str(level))

            output_loader_list.append(loaders)

    return output_loader_list
