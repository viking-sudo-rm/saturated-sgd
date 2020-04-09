from typing import Callable, Dict, Iterable, Optional

from overrides import overrides
import torch
from torch.nn import Parameter, Module
import numpy as np

from allennlp.training.metrics.metric import Metric

from src.utils.saturate import saturate

EPS = np.finfo(np.float).eps


def cos(vec1, vec2):
    norm1 = torch.clamp(vec1.norm(dim=-1), min=EPS)
    norm2 = torch.clamp(vec2.norm(dim=-1), min=EPS)
    return torch.sum(vec1 * vec2, dim=-1) / (norm1 * norm2)


@Metric.register("cosine_similarity")
class CosineSimilarity(Metric):

    """Measure cosine similarity, which will be applied between soft and hard logits."""

    def __init__(self, key: str) -> None:
        self.key = key
        self.sum = 0.0
        self.num = 0.0

    def __call__(
        self,
        output_dict: Dict,
        hard_output_dict: Dict,
        mask=None,
    ):
        logits = output_dict[self.key]
        hard_logits = hard_output_dict[self.key]

        if mask is not None:
            logits = logits * mask.unsqueeze(-1)
            hard_logits = hard_logits * mask.unsqueeze(-1)

        # Assume 0 is the batch dimension.
        sims = cos(logits.flatten(start_dim=1), hard_logits.flatten(start_dim=1))
        self.sum += torch.sum(sims)
        self.num += sims.size(0)

    @overrides
    def get_metric(self, reset: bool = False):
        if self.num == 0:
            mean_sim = 0.
        else:
            mean_sim = float(self.sum) / float(self.num)
        if reset:
            self.reset()
        return mean_sim

    @overrides
    def reset(self):
        self.sum = 0.0
        self.num = 0.0
