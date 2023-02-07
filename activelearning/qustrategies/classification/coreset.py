import numpy as np
import tqdm
from sklearn.metrics import pairwise_distances
from activelearning.qustrategies.classification.sampler import Sampler
from config import BaseConfig


# implementation inspired from https://github.com/JordanAsh/badge/blob/master/query_strategies/core_set.py
def furthest_first(X, X_set, n):
    m = np.shape(X)[0]
    if np.shape(X_set)[0] == 0:
        min_dist = np.tile(float("inf"), m)
    else:
        dist_ctr = pairwise_distances(X, X_set)
        min_dist = np.amin(dist_ctr, axis=1)

    idxs = []
    iterator = [i for i in range(n)]
    tbar = tqdm.tqdm(iterator)
    for i, _ in enumerate(tbar):
        idx = min_dist.argmax()
        idxs.append(idx)
        dist_new_ctr = pairwise_distances(X, X[[idx], :])
        for j in range(m):
            # tbar.set_description('M: %d' % j)
            min_dist[j] = min(min_dist[j], dist_new_ctr[j, 0])

    return idxs


class CoresetSampler(Sampler):
    def __init__(self, n_pool, start_idxs, cfg: BaseConfig):
        super(CoresetSampler, self).__init__(n_pool, start_idxs, cfg)

    def query(self, n: int, trainer):
        unlabeled_embeddings = trainer.get_embeddings(loader_type='unlabeled')
        labeled_embeddings = trainer.get_embeddings(loader_type='labeled')
        unlabeled_indices = unlabeled_embeddings['indices']

        # do coreset algorithm
        chosen = furthest_first(unlabeled_embeddings['nongrad_embeddings'], labeled_embeddings['nongrad_embeddings'], n)

        # derive final indices
        inds = unlabeled_indices[chosen]

        return inds
