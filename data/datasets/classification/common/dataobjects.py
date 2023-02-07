from torch.utils.data import DataLoader
from data.datasets.shared.utils import DatasetStructure


class ClassificationStructure(DatasetStructure):
    def __init__(self):
        super(ClassificationStructure, self).__init__()


class LoaderObject:
    def __init__(self, train_loader: DataLoader, test_loader: DataLoader, data_configs: DatasetStructure,
                 name: str = None):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.data_config = data_configs
        self.name = name