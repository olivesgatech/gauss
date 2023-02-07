import PIL.Image as Image
from torch.utils.data import Dataset
from data.datasets.classification.common.dataobjects import DatasetStructure


class LoaderMNIST(Dataset):
    def __init__(self, data_config: DatasetStructure, split: str, current_idxs: list = None, transform=None):
        # make sure split is in correct format
        if split != 'train' and split != 'test':
            raise Exception('Dataset handling is not correct!!! Split name should be either \'train\' or \'test\'!!!')


        # initialize data
        if split == 'train':
            self.X = data_config.train_set
            self.Y = data_config.train_labels
        else:
            self.X = data_config.test_set
            self.Y = data_config.test_labels

        self.transform = transform
        self._current_idxs = current_idxs
        if current_idxs is not None:
            self.X = self.X[current_idxs]
            self.Y = self.Y[current_idxs]

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        if self.transform is not None:
            x = Image.fromarray(x.numpy(), mode='L')
            x = self.transform(x)

        if self._current_idxs is not None:
            global_idx = self._current_idxs[index]
        else:
            global_idx = index

        sample = {'data': x, 'label': y, 'idx': index, 'global_idx': global_idx}
        return sample

    def __len__(self):
        return len(self.X)