import os
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from data.datasets.classification.common.dataobjects import ClassificationStructure, LoaderObject
from data.datasets.classification.MNIST.dataset import LoaderMNIST


def get_mnist(cfg, idxs: np.ndarray = None):
    path = cfg.data.data_loc
    print(os.path.expanduser(path) + '/MNIST')
    raw_tr = datasets.MNIST(os.path.expanduser(path) + '/MNIST', train=True, download=cfg.data.download)
    raw_te = datasets.MNIST(os.path.expanduser(path) + '/MNIST', train=False, download=cfg.data.download)

    # if idxs is not None:
    #    raw_tr.data = raw_tr.train_data[idxs]
    #    raw_tr.targets = raw_tr.train_labels[idxs]
    #    raise Exception('Active learning not implemented for loader yet!')

    # init data configs
    data_config = ClassificationStructure()
    data_config.train_set = raw_tr.train_data
    data_config.train_labels = raw_tr.train_labels
    data_config.test_set = raw_te.test_data
    data_config.test_labels = raw_te.test_labels
    data_config.train_len = len(raw_tr.train_data)
    data_config.test_len = len(raw_te.test_data)
    data_config.num_classes = 10
    data_config.img_size = 28

    data_config.is_configured = True

    # add transforms
    train_transform = transforms.Compose([])

    if cfg.data.augmentations.random_hflip:
        train_transform.transforms.append(transforms.RandomHorizontalFlip())
    if cfg.data.augmentations.random_crop:
        #train_transform.transforms.append(transforms.RandomCrop(28, padding=6))
        train_transform.transforms.append(transforms.RandomCrop(28, padding=3))

    # mandatory transforms
    train_transform.transforms.append(transforms.Grayscale(num_output_channels=3))
    mean = [x / 255.0 for x in [125.3, 123.0, 113.9]]
    std = [x / 255.0 for x in [63.0, 62.1, 66.7]]
    train_transform.transforms.append(transforms.ToTensor())
    train_transform.transforms.append(transforms.Normalize(mean=mean, std=std))

    # test transforms
    #test_transform = transforms.Compose([transforms.RandomCrop(28, padding=4),
    test_transform = transforms.Compose([transforms.Grayscale(num_output_channels=3),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=mean, std=std)])

    # create loaders
    train_loader = DataLoader(LoaderMNIST(data_config=data_config,
                                          split='train',
                                          transform=train_transform,
                                          current_idxs=idxs),
                              batch_size=cfg.classification.batch_size,
                              shuffle=True)
    test_loader = DataLoader(LoaderMNIST(data_config=data_config,
                                         split='test',
                                         transform=test_transform),
                              batch_size=cfg.classification.batch_size,
                              shuffle=False)
    loaders = LoaderObject(train_loader=train_loader,
                           test_loader=test_loader,
                           data_configs=data_config)

    return loaders