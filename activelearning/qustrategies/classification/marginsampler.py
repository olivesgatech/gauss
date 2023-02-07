import numpy as np
from activelearning.qustrategies.classification.sampler import Sampler
from config import BaseConfig


class MarginSampler(Sampler):
    def __init__(self, n_pool, start_idxs, cfg: BaseConfig):
        super(MarginSampler, self).__init__(n_pool, start_idxs, cfg)

    def query(self, n: int, trainer):
        # get probabilities and their indices
        print('Sampling Margin Probs')
        unl_dict = trainer.get_data_statistics(0)
        probabilities, indices = unl_dict['probabilities'], unl_dict['indices']

        # get smallest margins
        sorted_probs = np.sort(probabilities, axis=1)
        margins = sorted_probs[:, -1] - sorted_probs[:, -2]
        prob_inds = np.argsort(margins)[:n]

        # derive final indices
        inds = indices[prob_inds]

        return inds
