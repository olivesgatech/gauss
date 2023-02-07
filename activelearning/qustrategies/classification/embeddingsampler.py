import os
import numpy as np
from activelearning.qustrategies.classification.sampler import Sampler
from activelearning.qustrategies.classification.badge import init_centers
from activelearning.qustrategies.classification.coreset import furthest_first
from config import BaseConfig


class EmbeddingSampling(Sampler):
    '''Class for ideal example forgetting algorithm. Assumes that the forgetting events are known.
     Inherits from sampler.'''
    def __init__(self, n_pool: int, start_idxs: np.ndarray, cfg: BaseConfig):
        super(EmbeddingSampling, self).__init__(n_pool, start_idxs, cfg)
        # set index array and init pointer
        embeddings = np.load(os.path.expanduser(self._cfg.active_learning.stats.embedsampling_file))
        self._labeled_emebeddings = embeddings[start_idxs, :]
        self._embeddings = np.delete(embeddings, start_idxs, axis=0)
        self._indices = np.arange(n_pool)
        self._indices = np.delete(self._indices, start_idxs)

        print('Loaded sampling file: ' + os.path.expanduser(self._cfg.active_learning.stats.embedsampling_file))

    def query(self, n, trainer):
        if self._cfg.active_learning.stats.secondary_samping_type == 'badge':
            print(self._embeddings.shape)
            embed_inds = init_centers(self._embeddings, n)
        elif self._cfg.active_learning.stats.secondary_samping_type == 'coreset':
            embed_inds = furthest_first(self._embeddings, self._labeled_emebeddings, n)
        else:
            raise Exception('Type not implemented yet')

        inds = self._indices[embed_inds]

        # update embeddings and indices
        self._labeled_emebeddings = np.concatenate((self._labeled_emebeddings, self._embeddings[embed_inds, :]), axis=0)
        self._embeddings = np.delete(self._embeddings, embed_inds, axis=0)
        self._indices = np.delete(self._indices, embed_inds)
        return inds
