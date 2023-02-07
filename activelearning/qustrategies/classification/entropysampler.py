import numpy as np
from activelearning.qustrategies.classification.sampler import Sampler
from config import BaseConfig


class EntropySampler(Sampler):
    '''Class for sampling the highest entropy. Inherits from sampler.'''
    def __init__(self, n_pool, start_idxs, cfg: BaseConfig):
        '''Constructor implemented in sampler'''
        super(EntropySampler, self).__init__(n_pool, start_idxs, cfg)

    def query(self, n: int, trainer):
        '''Returns samples with highest entropy in the output distribution.
        Parameters:
            :param probs: datastructure containing the sigmoid probabilities and the index list
            :type probs: dict
            :param n: number of samples to be queried
            :type n: int'''
        # get probabilities and their indices
        print('Sampling Entropy Probs')
        unl_dict = trainer.get_data_statistics(0)
        probabilities, indices = unl_dict['probabilities'], unl_dict['indices']

        # get max entropy
        logs = np.log2(probabilities)
        mult = logs*probabilities
        entropy = np.sum(mult, axis=1)
        prob_inds = np.argsort(entropy)[:n]

        # derive final indices
        inds = indices[prob_inds]

        return inds
