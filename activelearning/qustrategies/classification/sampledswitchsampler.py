import numpy as np
import pdb
from scipy import stats
from sklearn.metrics import pairwise_distances
from activelearning.qustrategies.classification.sampler import Sampler
from config import BaseConfig


# one-dimensional kmeans ++ initialization
def init_centers(uncertainties: np.ndarray, n: int, order: str = 'max', distance_type: str = 'l2', debug: bool = True):
    if order == 'max':
        ind = np.argmax(uncertainties.flatten())
    else:
        ind = np.argmin(uncertainties.flatten())
    mu = [uncertainties[ind]]
    indsAll = [ind]
    centInds = [0.] * len(uncertainties)
    cent = 0
    print('Starting K-Means++')
    if debug:
        print('#Samps\tTotal Distance')
    # uncertainties = uncertainties[..., np.newaxis]
    while len(mu) < n:
        # the pairwise distance will be zero for uncertainties already included in mu
        if len(mu) == 1:
            D2 = pairwise_distances(uncertainties, mu, metric=distance_type).ravel().astype(float)
        else:
            newD = pairwise_distances(uncertainties, [mu[-1]], metric=distance_type).ravel().astype(float)
            for i in range(uncertainties.shape[0]):
                if D2[i] > newD[i]:
                    centInds[i] = cent
                    D2[i] = newD[i]
        if debug:
            print(str(len(mu)) + '\t' + str(sum(D2)), flush=True)

        # multiply minimum distances with uncertainty weights
        D2 = uncertainties.flatten() * D2

        if np.sum(D2) == 0.0:
            print('Distances are all zero! Giving the min or max uncertainty values!')
            leftover_inds = np.arange(uncertainties.shape[0])
            leftover_inds = np.delete(leftover_inds, indsAll)
            uncertainties = uncertainties[leftover_inds].ravel()
            newn = n - len(indsAll)
            if order == 'max':
                rel_inds = np.argsort(uncertainties)[-newn:]
            else:
                rel_inds = np.argsort(uncertainties)[:newn]
            indsAll.extend(leftover_inds[rel_inds])
            break
        else:
            D2 = D2.ravel().astype(float)
            Ddist = (D2 ** 2) / sum(D2 ** 2)
            customDist = stats.rv_discrete(name='custm', values=(np.arange(len(D2)), Ddist))
            ind = customDist.rvs(size=1)[0]
            mu.append(uncertainties[ind])
            indsAll.append(ind)
            cent += 1
    return indsAll


class SampledSwitchSampler(Sampler):
    def __init__(self, n_pool, start_idxs, cfg: BaseConfig):
        super(SampledSwitchSampler, self).__init__(n_pool, start_idxs, cfg)

    def action(self, trainer):
        print('Tracking switches')
        _ = trainer.get_data_statistics(0)

    def query(self, n: int, trainer):
        # get probabilities and their indices
        print('Sampling Switches')
        unl_dict = trainer.get_data_statistics(0)
        indices, switches = unl_dict['indices'], unl_dict['switches']

        switches = switches[..., np.newaxis]

        if self._cfg.active_learning.stats.stat_sampling_type == 'SV':
            target_inds = init_centers(switches, n)
        elif self._cfg.active_learning.stats.stat_sampling_type == 'nSV':
            target_inds = init_centers(switches, n)
        else:
            raise Exception('stat sampling type not implemented yet!')

        # derive final indices
        inds = indices[target_inds]

        return inds
