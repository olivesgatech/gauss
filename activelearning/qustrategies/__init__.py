import numpy as np
from activelearning.qustrategies.classification.randomsampler import RandomSampling
from activelearning.qustrategies.classification.entropysampler import EntropySampler
from activelearning.qustrategies.classification.marginsampler import MarginSampler
from activelearning.qustrategies.classification.lconfsampling import LeastConfidenceSampler
from activelearning.qustrategies.classification.switchsampler import SwitchSampler
from activelearning.qustrategies.classification.coreset import CoresetSampler
from activelearning.qustrategies.classification.badge import BadgeSampler
from activelearning.qustrategies.classification.gmmss import GauSSSampler
from activelearning.qustrategies.classification.eventsampler import EventSampling
from activelearning.qustrategies.classification.softmaxsampler import SoftmaxSampling
from activelearning.qustrategies.classification.embeddingsampler import EmbeddingSampling
from activelearning.qustrategies.classification.idealgauss import IdealGauSSSampling
from activelearning.qustrategies.classification.sampledswitchsampler import SampledSwitchSampler
from activelearning.qustrategies.classification.forgettingsampler import ForgettingSampler
from activelearning.qustrategies.classification.albl import ALBLSampler
from activelearning.qustrategies.classification.batchbald import BatchBaldSampler
from config import BaseConfig


def get_sampler(cfg: BaseConfig, n_pool: int, start_idxs: np.ndarray):
    if cfg.active_learning.strategy == 'random':
        print('Using Random Sampler')
        sampler = RandomSampling(n_pool=n_pool, start_idxs=start_idxs, cfg=cfg)
    elif cfg.active_learning.strategy == 'entropy':
        print('Using Entropy Sampler')
        sampler = EntropySampler(n_pool=n_pool, start_idxs=start_idxs, cfg=cfg)
    elif cfg.active_learning.strategy == 'margin':
        print('Using Margin Sampler')
        sampler = MarginSampler(n_pool=n_pool, start_idxs=start_idxs, cfg=cfg)
    elif cfg.active_learning.strategy == 'lconf':
        print('Using Least Confidence Sampler')
        sampler = LeastConfidenceSampler(n_pool=n_pool, start_idxs=start_idxs, cfg=cfg)
    elif cfg.active_learning.strategy == 'switch':
        print('Using Switch Sampler')
        sampler = SwitchSampler(n_pool=n_pool, start_idxs=start_idxs, cfg=cfg)
    elif cfg.active_learning.strategy == 'forgetting':
        print('Using Forgetting Event Sampler')
        sampler = ForgettingSampler(n_pool=n_pool, start_idxs=start_idxs, cfg=cfg)
    elif cfg.active_learning.strategy == 'badge':
        print('Using BADGE Sampler')
        sampler = BadgeSampler(n_pool=n_pool, start_idxs=start_idxs, cfg=cfg)
    elif cfg.active_learning.strategy == 'coreset':
        print('Using Coreset Sampler')
        sampler = CoresetSampler(n_pool=n_pool, start_idxs=start_idxs, cfg=cfg)
    elif cfg.active_learning.strategy == 'gauss':
        print('Using GauSS Sampler')
        sampler = GauSSSampler(n_pool=n_pool, start_idxs=start_idxs, cfg=cfg)
    elif cfg.active_learning.strategy == 'eventsampling':
        print('Using Optimal Event Sampler')
        sampler = EventSampling(n_pool=n_pool, start_idxs=start_idxs, cfg=cfg)
    elif cfg.active_learning.strategy == 'softmaxsampling':
        print('Using Optimal Softmax Sampler: ' + cfg.active_learning.stats.secondary_samping_type)
        sampler = SoftmaxSampling(n_pool=n_pool, start_idxs=start_idxs, cfg=cfg)
    elif cfg.active_learning.strategy == 'embeddingsampling':
        print('Using Optimal Embedding Sampler: ' + cfg.active_learning.stats.secondary_samping_type)
        sampler = EmbeddingSampling(n_pool=n_pool, start_idxs=start_idxs, cfg=cfg)
    elif cfg.active_learning.strategy == 'idealgauss':
        print('Using Optimal GauSS Sampler')
        sampler = IdealGauSSSampling(n_pool=n_pool, start_idxs=start_idxs, cfg=cfg)
    elif cfg.active_learning.strategy == 'sampledswitch':
        print('Using Sampled Switch Sampler')
        sampler = SampledSwitchSampler(n_pool=n_pool, start_idxs=start_idxs, cfg=cfg)
    elif cfg.active_learning.strategy == 'albl':
        print('Using ALBL Sampler')
        sampler = ALBLSampler(n_pool=n_pool, start_idxs=start_idxs, cfg=cfg)
    elif cfg.active_learning.strategy == 'batchbald':
        print('Using Batchbald Sampler')
        sampler = BatchBaldSampler(n_pool=n_pool, start_idxs=start_idxs, cfg=cfg)
    else:
        raise Exception('Strategy not implemented yet')

    return sampler
