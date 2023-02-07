import numpy as np
from activelearning.qustrategies.classification.sampler import Sampler
from config import BaseConfig


class SwitchSampler(Sampler):
    def __init__(self, n_pool, start_idxs, cfg: BaseConfig):
        super(SwitchSampler, self).__init__(n_pool, start_idxs, cfg)

    def action(self, trainer):
        print('Tracking switches')
        _ = trainer.get_data_statistics(0)

    def query(self, n: int, trainer):
        # get probabilities and their indices
        print('Sampling Switches')
        unl_dict = trainer.get_data_statistics(0)
        indices, switches = unl_dict['indices'], unl_dict['switches']

        # get max entropy
        if self._cfg.active_learning.stats.stat_sampling_type == 'SV':
            target_inds = np.argsort(switches)[-n:]
        elif self._cfg.active_learning.stats.stat_sampling_type == 'nSV':
            target_inds = np.argsort(switches)[:n]
        else:
            raise Exception('stat sampling type not implemented yet!')

        # derive final indices
        inds = indices[target_inds]

        return inds
