import os
import numpy as np
from config import BaseConfig
from data.datasets.classification.common.aquisition import get_dataset
from training.classification.classificationtracker import ClassifcationTracker


class USPECTracker:
    def __init__(self, set_shape: tuple, num_seeds: int):
        self._predictions = np.squeeze(np.zeros((num_seeds,) + set_shape))
        self._seen = {}

    def update(self, preds: np.ndarray, seed: int):
        if seed in self._seen.keys():
            raise Exception('Seed has already been run!')

        self._seen[seed] = True
        self._predictions[seed, ...] = preds
        return

    def save_statistics(self, directory: str, ld_type: str):
        path = directory + '/uspec_statistics/' + ld_type
        if not os.path.exists(path):
            os.makedirs(path)
        np.save(path + '/predictions.npy', self._predictions)


def get_uspec_inputs(cfg: BaseConfig):
    uspec_inputs = []

    if cfg.run_configs.train:
        name = 'train'
        loader = get_dataset(cfg=cfg)
        shape = (loader.data_config.train_len,)
        uspec_tracker = USPECTracker(shape, num_seeds=cfg.uspec_configs.num_seeds)
        fe_tracker = ClassifcationTracker(loader.data_config.train_len)
        loader_struct = (loader, fe_tracker)
        uspec_inputs.append((uspec_tracker, name, loader_struct))
    if cfg.run_configs.test:
        name = 'test'
        loader = get_dataset(cfg=cfg)
        shape = (loader.data_config.test_len,)
        uspec_tracker = USPECTracker(shape, num_seeds=cfg.uspec_configs.num_seeds)
        fe_tracker = ClassifcationTracker(loader.data_config.test_len)
        loader_struct = (loader, fe_tracker)
        uspec_inputs.append((uspec_tracker, name, loader_struct))
    if cfg.run_configs.ood.mnist:
        name = 'MNIST'
        loader = get_dataset(cfg=cfg, override=name)
        shape = (loader.data_config.test_len,)
        uspec_tracker = USPECTracker(shape, num_seeds=cfg.uspec_configs.num_seeds)
        fe_tracker = ClassifcationTracker(loader.data_config.test_len)
        loader_struct = (loader, fe_tracker)
        uspec_inputs.append((uspec_tracker, name, loader_struct))
    if cfg.run_configs.ood.svhn:
        name = 'SVHN'
        loader = get_dataset(cfg=cfg, override=name)
        shape = (loader.data_config.test_len,)
        uspec_tracker = USPECTracker(shape, num_seeds=cfg.uspec_configs.num_seeds)
        fe_tracker = ClassifcationTracker(loader.data_config.test_len)
        loader_struct = (loader, fe_tracker)
        uspec_inputs.append((uspec_tracker, name, loader_struct))
    if cfg.run_configs.ood.stl10:
        name = 'STL10'
        loader = get_dataset(cfg=cfg, override=name)
        shape = (loader.data_config.test_len,)
        uspec_tracker = USPECTracker(shape, num_seeds=cfg.uspec_configs.num_seeds)
        fe_tracker = ClassifcationTracker(loader.data_config.test_len)
        loader_struct = (loader, fe_tracker)
        uspec_inputs.append((uspec_tracker, name, loader_struct))
    if cfg.run_configs.ood.cifar10:
        name = 'CIFAR10'
        loader = get_dataset(cfg=cfg, override=name)
        shape = (loader.data_config.test_len,)
        uspec_tracker = USPECTracker(shape, num_seeds=cfg.uspec_configs.num_seeds)
        fe_tracker = ClassifcationTracker(loader.data_config.test_len)
        loader_struct = (loader, fe_tracker)
        uspec_inputs.append((uspec_tracker, name, loader_struct))
    if cfg.run_configs.ood.cifar100:
        name = 'CIFAR100'
        loader = get_dataset(cfg=cfg, override=name)
        shape = (loader.data_config.test_len,)
        uspec_tracker = USPECTracker(shape, num_seeds=cfg.uspec_configs.num_seeds)
        fe_tracker = ClassifcationTracker(loader.data_config.test_len)
        loader_struct = (loader, fe_tracker)
        uspec_inputs.append((uspec_tracker, name, loader_struct))
    if cfg.run_configs.ood.cinic10:
        name = 'CINIC10'
        loader = get_dataset(cfg=cfg, override=name)
        shape = (loader.data_config.test_len,)
        uspec_tracker = USPECTracker(shape, num_seeds=cfg.uspec_configs.num_seeds)
        fe_tracker = ClassifcationTracker(loader.data_config.test_len)
        loader_struct = (loader, fe_tracker)
        uspec_inputs.append((uspec_tracker, name, loader_struct))
    if cfg.run_configs.ood.cifar10C:
        name = 'CIFAR10C'
        loaders = get_dataset(cfg=cfg, override=name)
        for loader in loaders:
            name = loader.name
            shape = (loader.data_config.test_len,)
            uspec_tracker = USPECTracker(shape, num_seeds=cfg.uspec_configs.num_seeds)
            fe_tracker = ClassifcationTracker(loader.data_config.test_len)
            loader_struct = (loader, fe_tracker)
            uspec_inputs.append((uspec_tracker, name, loader_struct))
    if len(uspec_inputs) == 0:
        raise Exception('At least one set must be specified!')

    return uspec_inputs