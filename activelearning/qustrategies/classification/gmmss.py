import numpy as np
from sklearn.mixture import GaussianMixture
from activelearning.qustrategies.classification.sampler import Sampler
from config import BaseConfig


def gausssampling(switching_list: np.ndarray, cfg: BaseConfig, n: int):
    # init gmm and predict samples
    gmm = GaussianMixture(n_components=2).fit(switching_list.reshape(-1, 1))
    pred = gmm.predict(switching_list.reshape(-1, 1))
    probs = gmm.predict_proba(switching_list.reshape(-1, 1))
    # read current samples from list
    if cfg.active_learning.stats.stat_sampling_type == 'SV':
        if gmm.means_[0][0] < gmm.means_[1][0]:
            relevant_gaussian = 1
        else:
            relevant_gaussian = 0
        backup = np.argsort(switching_list)[len(switching_list) - n:]
    elif cfg.active_learning.stats.stat_sampling_type == 'nSV':
        if gmm.means_[0][0] > gmm.means_[1][0]:
            relevant_gaussian = 1
        else:
            relevant_gaussian = 0
        backup = np.argsort(switching_list)[:n]
    else:
        raise NotImplementedError

    # get relevant indices
    rel_inds = (pred == relevant_gaussian).nonzero()[0]
    length = rel_inds.shape[0]

    # get corresponding probs
    probs = probs[:, relevant_gaussian]
    probs = probs[rel_inds]
    norm_probs = probs / np.sum(probs)

    # check if sufficient samples are available
    if length > n:
        # sample all elements from relevant gaussian with probability distribution
        print('Sampling from RV: ' + cfg.active_learning.stats.stat_sampling_type)
        switching_inds = np.random.choice(rel_inds, size=n, replace=False, p=norm_probs)
    else:
        # not enough elements from prediction include other ones
        print('Not enough samples of forgettable RV or gmm not specified -> using highest/lowest values. Type: '
              + cfg.active_learning.stats.stat_sampling_type)
        switching_inds = backup

    return switching_inds


class GauSSSampler(Sampler):
    def __init__(self, n_pool, start_idxs, cfg: BaseConfig):
        super(GauSSSampler, self).__init__(n_pool, start_idxs, cfg)

    def action(self, trainer):
        print('Tracking switches')
        _ = trainer.get_data_statistics(0)

    def query(self, n: int, trainer):
        # get probabilities and their indices
        print('Sampling Switches')
        unl_dict = trainer.get_data_statistics(0)
        indices, switching_list = unl_dict['indices'], unl_dict['switches']

        switching_inds = gausssampling(switching_list=switching_list, cfg=self._cfg, n=n)

        # derive final indices
        inds = indices[switching_inds]

        return inds
