import numpy as np
import pdb
from scipy import stats
from sklearn.metrics import pairwise_distances
from activelearning.qustrategies.classification.sampler import Sampler
from config import BaseConfig


# kmeans ++ initialization from https://github.com/JordanAsh/badge/blob/master/query_strategies/badge_sampling.py
def init_centers(X, K, firstmax=True, debug=False):
    if firstmax:
        ind = np.argmax([np.linalg.norm(s, 2) for s in X])
    else:
        ind = np.argmin([np.linalg.norm(s, 2) for s in X])
    mu = [X[ind]]
    indsAll = [ind]
    centInds = [0.] * len(X)
    cent = 0
    print('Starting K-Means++')
    if debug:
        print('#Samps\tTotal Distance')
    while len(mu) < K:
        if len(mu) == 1:
            D2 = pairwise_distances(X, mu).ravel().astype(float)
        else:
            newD = pairwise_distances(X, [mu[-1]]).ravel().astype(float)
            for i in range(len(X)):
                if D2[i] > newD[i]:
                    centInds[i] = cent
                    D2[i] = newD[i]
        if debug:
            print(str(len(mu)) + '\t' + str(sum(D2)), flush=True)
        if sum(D2) == 0.0:
            pdb.set_trace()
        D2 = D2.ravel().astype(float)
        Ddist = (D2 ** 2) / sum(D2 ** 2)
        customDist = stats.rv_discrete(name='custm', values=(np.arange(len(D2)), Ddist))
        ind = customDist.rvs(size=1)[0]
        mu.append(X[ind])
        indsAll.append(ind)
        cent += 1
    gram = np.matmul(X[indsAll], X[indsAll].T)
    val, _ = np.linalg.eig(gram)
    val = np.abs(val)
    vgt = val[val > 1e-2]
    return indsAll


class BadgeSampler(Sampler):
    def __init__(self, n_pool, start_idxs, cfg: BaseConfig):
        super(BadgeSampler, self).__init__(n_pool, start_idxs, cfg)

    def query(self, n: int, trainer):
        embeddings = trainer.get_embeddings(loader_type='unlabeled')
        # get probabilities and their indices
        indices = embeddings['indices']
        grad_embedding = embeddings['embeddings']

        # get smallest margins
        embed_inds = init_centers(grad_embedding, n)

        # derive final indices
        inds = indices[embed_inds]

        return inds
