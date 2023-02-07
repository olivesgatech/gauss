import numpy as np
from activelearning.qustrategies.classification.sampler import Sampler
from activelearning.qustrategies.classification.coreset import CoresetSampler
from activelearning.qustrategies.classification.lconfsampling import LeastConfidenceSampler
from config import BaseConfig


class ALBLSampler(Sampler):
    '''Class for sampling the highest entropy. Inherits from sampler.'''
    def __init__(self, n_pool, start_idxs, cfg: BaseConfig):
        super(ALBLSampler, self).__init__(n_pool, start_idxs, cfg)
        self._n_pool = n_pool
        self._strategy_list = [CoresetSampler(n_pool, start_idxs, cfg), LeastConfidenceSampler(n_pool, start_idxs, cfg)]
        self._nstrategies = len(self._strategy_list)
        self._delta = 0.1
        self._w = np.ones((self._nstrategies,))
        self._pmin = 1.0 / (self._nstrategies * 10.0)
        self._start = True
        self._aw = np.zeros(n_pool)
        self._aw[self.idx_current] = 1.0
        self._s_idx = None

    def query(self, n: int, trainer):
        if not self._start:
            idxs_labeled = np.arange(self._n_pool)[self.idx_current]
            l_stats = trainer.get_data_statistics(0, 'labeled')
            fn = l_stats['sample accuracy'].astype(float)
            reward = (fn / self._aw[idxs_labeled]).mean()

            self._w[self._s_idx] *= np.exp(self._pmin / 2.0 * (
                        reward + 1.0 / self.last_p * np.sqrt(np.log(self._nstrategies / self._delta) / self._nstrategies)))

        self._start = False
        W = self._w.sum()
        p = (1.0 - self._nstrategies * self._pmin) * self._w / W + self._pmin

        for i, stgy in enumerate(self._strategy_list):
            print('  {} {}'.format(p[i], type(stgy).__name__))

        self._s_idx = np.random.choice(np.arange(self._nstrategies), p=p)
        print('  select {}'.format(type(self._strategy_list[self._s_idx]).__name__))
        q_idxs = self._strategy_list[self._s_idx].query(n, trainer)
        self._aw[q_idxs] = p[self._s_idx]
        self.last_p = p[self._s_idx]

        return q_idxs

    def update(self, new_idx):
        super(ALBLSampler, self).update(new_idx)
        # self.idx_current = np.append(self.idx_current, new_idx)
        # self.total_pool[new_idx] = 1

        for i, stgy in enumerate(self._strategy_list):
            stgy.update(new_idx)
