from models.classification.mlp import MLP
from models.classification.resnet import ResNet18, ResNet34, ResNetn
from models.classification.torchmodels import *
from models.segmentation.deeplabv3.deeplab import DeepLab
from data.datasets.shared.utils import DatasetStructure
from config import BaseConfig


def build_architecture(architecture: str, data_cfg: DatasetStructure, cfg: BaseConfig, bnn: bool = False):
    if architecture == 'MLP':
        if bnn:
            raise NotImplementedError('BNN not implemented yet!')
        dim = (data_cfg.img_size ** 2)*3
        return MLP(num_classes=data_cfg.num_classes, dim=dim)
    elif architecture == 'resnet-18':
        if cfg.classification.pretrained:
            out = ResNetn(num_classes=data_cfg.num_classes, type=18, pretrained=True, is_dropout=bnn)
        else:
            out = ResNet18(num_classes=data_cfg.num_classes, is_dropout=bnn)
        return out
    elif architecture == 'resnet-34':
        if cfg.classification.pretrained:
            out = ResNetn(num_classes=data_cfg.num_classes, type=34, pretrained=True, is_dropout=bnn)
        else:
            out = ResNet34(num_classes=data_cfg.num_classes, is_dropout=bnn)
        return out
    elif architecture == 'densenet-121':
        out = DenseNet(type=121, num_classes=data_cfg.num_classes, pretrained=cfg.classification.pretrained,
                       is_dropout=bnn)
        return out
    elif architecture == 'vgg-11':
        out = VGG(type=11, num_classes=data_cfg.num_classes, pretrained=cfg.classification.pretrained,
                  is_dropout=bnn)
        return out
    elif architecture == 'vgg-16':
        out = VGG(type=16, num_classes=data_cfg.num_classes, pretrained=cfg.classification.pretrained,
                  is_dropout=bnn)
        return out
    elif architecture == 'efficientnet_b0':
        out = EfficientNet(num_classes=data_cfg.num_classes, pretrained=cfg.classification.pretrained,
                           is_dropout=bnn)
        return out
    else:
        raise Exception('Architecture not implemented yet')


def build_segmentation(architecture: str, data_cfg: DatasetStructure):
    if architecture == 'deeplab-v3':
        return DeepLab(num_classes=data_cfg.num_classes)
    else:
        raise Exception('Architecture not implemented yet')
