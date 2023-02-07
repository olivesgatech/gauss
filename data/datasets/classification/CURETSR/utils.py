import os
from PIL import Image
import torch


def make_dataset (traindir):
    img = []
    for fname in sorted(os.listdir(traindir)):
        target = int(fname[3:5]) - 1
        path = os.path.join(traindir, fname)
        item = (path, target)
        img.append(item)
    return img


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def _is_tensor_image(img):
    return torch.is_tensor(img) and img.ndimension() == 3


def standardization(tensor):
    if not _is_tensor_image(tensor):
        raise TypeError('Tensor is not a torch image')

    for t in tensor:
        t.sub_(t.mean()).div_(t.std())

    return tensor


def l2normalize(tensor):
    if not _is_tensor_image(tensor):
        raise TypeError('Tensor is not a torch image')

    tensor = tensor.mul(255)
    norm_tensor = tensor / torch.norm(tensor)
    return norm_tensor