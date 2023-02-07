import os
import numpy as np
from activelearning.qustrategies.classification.sampler import Sampler
from config import BaseConfig


class SoftmaxSampling(Sampler):
    '''Class for ideal example forgetting algorithm. Assumes that the forgetting events are known.
     Inherits from sampler.'''
    def __init__(self, n_pool: int, start_idxs: np.ndarray, cfg: BaseConfig):
        super(SoftmaxSampling, self).__init__(n_pool, start_idxs, cfg)
        # set index array and init pointer
        probs = np.load(os.path.expanduser(self._cfg.active_learning.stats.probsampling_file))
        if self._cfg.active_learning.stats.secondary_samping_type == 'entropy':
            logs = np.log2(probs)
            mult = logs * probs
            entropy = np.sum(mult, axis=1)
            prob_inds = np.argsort(entropy)
        elif self._cfg.active_learning.stats.secondary_samping_type == 'margin':
            # get smallest margins
            sorted_probs = np.sort(probs, axis=1)
            margins = sorted_probs[:, -1] - sorted_probs[:, -2]
            prob_inds = np.argsort(margins)
        elif self._cfg.active_learning.stats.secondary_samping_type == 'lconf':
            # get max probs
            probabilities = np.max(probs, axis=1)
            prob_inds = np.argsort(probabilities)
        else:
            raise Exception('Type not implemented yet')

        self.ef_samples = prob_inds
        self.ef_samples = self.ef_samples[~np.isin(self.ef_samples, tuple(start_idxs))]
        print('Loaded sampling file: ' + os.path.expanduser(self._cfg.active_learning.stats.probsampling_file))
        self.query_type = self._cfg.active_learning.stats.stat_sampling_type

        if self.query_type == 'SV':
            # read from bottom to top
            self.reading_pos = 0
        elif self.query_type == 'nSV':
            # read from top to bottom
            self.reading_pos = self.ef_samples.shape[0]
        else:
            raise NotImplementedError

    def query(self, n, trainer):
        '''Reads the top samples from the example forgetting list'''
        # read current samples from list
        if self.query_type == 'SV':
            bottom = self.reading_pos
            top = self.reading_pos + n

            # reset the reading position
            self.reading_pos = top
        elif self.query_type == 'nSV':
            bottom = self.reading_pos - n
            top = self.reading_pos

            # reset the reading position
            self.reading_pos = bottom
        else:
            raise NotImplementedError

        inds = self.ef_samples[bottom:top]

        return inds
