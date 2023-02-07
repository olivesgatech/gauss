import math
from dataclasses import dataclass
from typing import List

import torch
from toma import toma
from tqdm.auto import tqdm
import torch.nn as nn
from models.architectures import build_architecture
from training.common.trainutils import determine_multilr_milestones
from data.datasets.classification.common.aquisition import get_dataset

from activelearning.qustrategies.classification.sampler import Sampler
from activelearning.qustrategies.utilities.batchbald import DynamicJointEntropy, SampledJointEntropy
from config import BaseConfig


# Implementation from https://github.com/BlackHC/batchbald_redux/blob/master/01_batchbald.ipynb
def compute_conditional_entropy(log_probs_N_K_C: torch.Tensor) -> torch.Tensor:
    N, K, C = log_probs_N_K_C.shape

    entropies_N = torch.empty(N, dtype=torch.double)

    pbar = tqdm(total=N, desc="Conditional Entropy", leave=False)

    @toma.execute.chunked(log_probs_N_K_C, 1024)
    def compute(log_probs_n_K_C, start: int, end: int):
        nats_n_K_C = log_probs_n_K_C * torch.exp(log_probs_n_K_C)

        entropies_N[start:end].copy_(-torch.sum(nats_n_K_C, dim=(1, 2)) / K)
        pbar.update(end - start)

    pbar.close()

    return entropies_N


def compute_entropy(log_probs_N_K_C: torch.Tensor) -> torch.Tensor:
    N, K, C = log_probs_N_K_C.shape

    entropies_N = torch.empty(N, dtype=torch.double)

    pbar = tqdm(total=N, desc="Entropy", leave=False)

    @toma.execute.chunked(log_probs_N_K_C, 1024)
    def compute(log_probs_n_K_C, start: int, end: int):
        mean_log_probs_n_C = torch.logsumexp(log_probs_n_K_C, dim=1) - math.log(K)
        nats_n_C = mean_log_probs_n_C * torch.exp(mean_log_probs_n_C)

        entropies_N[start:end].copy_(-torch.sum(nats_n_C, dim=1))
        pbar.update(end - start)

    pbar.close()

    return entropies_N

# Cell


@dataclass
class CandidateBatch:
    scores: List[float]
    indices: List[int]


def get_batchbald_batch(log_probs_N_K_C: torch.Tensor, batch_size: int,
                        num_samples: int, dtype=None, device=None) -> CandidateBatch:
    N, K, C = log_probs_N_K_C.shape

    batch_size = min(batch_size, N)

    candidate_indices = []
    candidate_scores = []

    if batch_size == 0:
        return CandidateBatch(candidate_scores, candidate_indices)

    conditional_entropies_N = compute_conditional_entropy(log_probs_N_K_C)

    batch_joint_entropy = DynamicJointEntropy(
        num_samples, batch_size - 1, K, C, dtype=dtype, device=device
    )

    # We always keep these on the CPU.
    scores_N = torch.empty(N, dtype=torch.double, pin_memory=torch.cuda.is_available())

    for i in tqdm(range(batch_size), desc="BatchBALD", leave=False):
        if i > 0:
            latest_index = candidate_indices[-1]
            batch_joint_entropy.add_variables(log_probs_N_K_C[latest_index : latest_index + 1])

        shared_conditinal_entropies = conditional_entropies_N[candidate_indices].sum()

        batch_joint_entropy.compute_batch(log_probs_N_K_C, output_entropies_B=scores_N)

        scores_N -= conditional_entropies_N + shared_conditinal_entropies
        scores_N[candidate_indices] = -float("inf")

        candidate_score, candidate_index = scores_N.max(dim=0)

        candidate_indices.append(candidate_index.item())
        candidate_scores.append(candidate_score.item())

    return CandidateBatch(candidate_scores, candidate_indices)


class BatchBaldSampler(Sampler):
    """
    Batch bald sampler.
    """
    def __init__(self, n_pool, start_idxs, cfg: BaseConfig):
        super(BatchBaldSampler, self).__init__(n_pool, start_idxs, cfg)
        self._cfg = cfg
        loaders = get_dataset(cfg)
        self._model = build_architecture(self._cfg.classification.model, loaders.data_config, cfg, bnn=True)

        if self._cfg.classification.optimization.optimizer == 'adam':
            self._optimizer = torch.optim.Adam(self._model.parameters(),
                                               lr=self._cfg.classification.optimization.lr)
        elif self._cfg.classification.optimization.optimizer == 'sgd':
            self._optimizer = torch.optim.SGD(self._model.parameters(),
                                              lr=self._cfg.classification.optimization.lr,
                                              momentum=0.9,
                                              nesterov=True,
                                              weight_decay=5e-4)
        else:
            raise Exception('Optimizer not implemented yet!')

        if self._cfg.run_configs.cuda:
            # use multiple GPUs if available
            self._model = torch.nn.DataParallel(self._model,
                                                device_ids=[self._cfg.run_configs.gpu_id])

    def action(self, trainer):
        print('Tracking switches')
        self._model, self._optimizer = trainer.batchbald_train(self._model, self._optimizer)

    def query(self, n: int, trainer):
        # get probabilities and their indices
        print('Sampling Batchbald Probs')
        unl_dict = trainer.get_batchbald_probs(self._model, self._optimizer)
        probabilities, indices = unl_dict['probabilities'], unl_dict['indices']

        # batchbald expects log softmax outputs
        log_probs = torch.log(probabilities)
        cand_batch = get_batchbald_batch(log_probs_N_K_C=log_probs, batch_size=n, num_samples=probabilities.shape[0],
                                         device=trainer.get_device())

        # convert to numpy
        indices = indices.cpu().numpy()

        # derive final indices
        inds = indices[cand_batch.indices]

        return inds
