import os
import shutil
import torch
from collections import OrderedDict
import glob
import matplotlib.pyplot as plt
import numpy as np
import torchvision.utils as vutils


def plot_batch(sample_batch, device):
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(np.transpose(vutils.make_grid(sample_batch.to(device)[: 64], padding=2, normalize=True).cpu(),
                            (1, 2, 0)))


class Saver(object):

    def __init__(self, cfg):
        self._cfg = cfg
        self.directory = os.path.join(self._cfg.run_configs.output_folder_name, self._cfg.data.dataset)
        self.runs = sorted(glob.glob(os.path.join(self.directory, 'experiment_*')))
        run_id = int(self.runs[-1].split('_')[-1]) + 1 if self.runs else 0

        self.experiment_dir = os.path.join(self.directory, 'experiment_{}'.format(str(run_id)))
        if not os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)

    def save_checkpoint(self, state, filename='checkpoint.pth.tar'):
        """Saves checkpoint to disk"""
        filename = os.path.join(self.experiment_dir, filename)
        torch.save(state, filename)
        shutil.copyfile(filename, os.path.join(self.directory, 'model_best.pth.tar'))

