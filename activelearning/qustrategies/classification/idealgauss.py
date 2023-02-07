import os
import numpy as np
from activelearning.qustrategies.classification.sampler import Sampler
from activelearning.qustrategies.classification.gmmss import gausssampling
from config import BaseConfig


class IdealGauSSSampling(Sampler):
    '''Class for ideal example forgetting algorithm. Assumes that the forgetting events are known.
     Inherits from sampler.'''
    def __init__(self, n_pool: int, start_idxs: np.ndarray, cfg: BaseConfig):
        super(IdealGauSSSampling, self).__init__(n_pool, start_idxs, cfg)
        # set index array and init pointer
        fevents = np.load(os.path.expanduser(self._cfg.active_learning.stats.eventsampling_file))
        self._unlabeled = np.delete(fevents, start_idxs, axis=0)
        self._indices = np.arange(n_pool)
        self._indices = np.delete(self._indices, start_idxs)
        print('Loaded sampling file: ' + os.path.expanduser(self._cfg.active_learning.stats.eventsampling_file))

    def query(self, n, trainer):
        '''Reads the top samples from the example forgetting list'''
        # read current samples from list
        switching_inds = gausssampling(switching_list=self._unlabeled, cfg=self._cfg, n=n)

        inds = self._indices[switching_inds]

        self._unlabeled = np.delete(self._unlabeled, switching_inds, axis=0)
        self._indices = np.delete(self._indices, switching_inds, axis=0)

        return inds
