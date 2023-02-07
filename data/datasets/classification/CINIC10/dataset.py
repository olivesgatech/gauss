import PIL.Image as Image
import torch
import numpy as np
from torch.utils.data import Dataset
from data.datasets.classification.common.dataobjects import DatasetStructure


class LoaderCINIC10(Dataset):
    def __init__(self, data_config: DatasetStructure, split: str, current_idxs: list = None, transform=None):
        # make sure split is in correct format
        if split != 'train' and split != 'test':
            raise Exception('Dataset handling is not correct!!! Split name should be either \'train\' or \'test\'!!!')

        # initialize data
        if split == 'train':
            self._data = data_config.train_set
        else:
            self._data = data_config.test_set

        self._current_idxs = current_idxs
        if current_idxs is not None:
            self._data = torch.utils.data.Subset(self._data, current_idxs)

        self.transform = transform

    def __getitem__(self, index):
        x, y = self._data[index]
        if self.transform is not None:
            # im = Image.fromarray(x)
            x = self.transform(x)

        if self._current_idxs is not None:
            global_idx = self._current_idxs[index]
        else:
            global_idx = index

        sample = {'data': x, 'label': y, 'idx': index, 'global_idx': global_idx}
        return sample

    def __len__(self):
        return len(self._data)
