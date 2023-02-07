import numpy as np
from data.datasets.classification.MNIST.configuration import get_mnist
from data.datasets.classification.SVHN.configuration import get_svhn
from data.datasets.classification.CIFAR10.configuration import get_cifar10
from data.datasets.classification.CIFAR100.configuration import get_cifar100
from data.datasets.classification.STL10.configuration import get_stl10
from data.datasets.classification.CURETSR.configuration import get_curetsr
from data.datasets.classification.CIFAR10C.configuration import get_cifar10c
from data.datasets.classification.CINIC10.configuration import get_cinic10
from config import BaseConfig


def get_dataset(cfg: BaseConfig, override: str = None, idxs: np.ndarray = None, test_bs: bool = False):
    if override is not None:
        dataset = override
    else:
        dataset = cfg.data.dataset

    if dataset == 'MNIST':
        return get_mnist(cfg, idxs)
    elif dataset == 'SVHN':
        return get_svhn(cfg, idxs)
    elif dataset == 'CIFAR10':
        return get_cifar10(cfg, idxs, test_bs)
    elif dataset == 'CIFAR100':
        return get_cifar100(cfg, idxs, test_bs, num_classes=cfg.data.num_classes)
    elif dataset == 'CURETSR':
        return get_curetsr(cfg, idxs, test_bs)
    elif dataset == 'STL10':
        return get_stl10(cfg, idxs)
    elif dataset == 'CINIC10':
        return get_cinic10(cfg, idxs)
    elif dataset == 'CIFAR10C':
        if not override:
            raise Exception('CIFAR10C can not be trained on as it is a corruption dataset!')
        return get_cifar10c(cfg)
    else:
        raise Exception('Dataset not implemented yet!')