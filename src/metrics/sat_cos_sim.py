from typing import Callable, Iterable, Optional

from overrides import overrides
import torch
from torch.nn import Parameter
import numpy as np

from allennlp.training.metrics.metric import Metric

from src.utils.saturate import saturate

EPS = np.finfo(np.float).eps


def cos(vec1, vec2):
    norm1 = torch.clamp(vec1.norm(dim=-1), min=EPS)
    norm2 = torch.clamp(vec2.norm(dim=-1), min=EPS)
    return torch.sum(vec1 * vec2, dim=-1) / (norm1 * norm2)


@Metric.register("saturation_cos_sim")
class SaturationCosSim(Metric):

    def __init__(self, infinity: float = 1e3) -> None:
        self.infinity = infinity
        self.sum = 0.
        self.num = 0.

    def __call__(
        self, logits: torch.FloatTensor, model, logits_callback, mask=None,
    ):
        # TODO: Unify the API for these saturation metrics.
        with saturate(model, self.infinity):
            hard_logits = logits_callback()

        if mask is not None:
            logits = logits * mask.unsqueeze(-1)
            hard_logits = hard_logits * mask.unsqueeze(-1)
        
        sims = cos(logits.flatten(start_dim=1), hard_logits.flatten(start_dim=1))
        self.sum += torch.sum(sims)
        self.num += sims.size(0)

    @overrides
    def get_metric(self, reset: bool = False):
        mean_sim = float(self.sum) / float(self.num)
        if reset:
            self.reset()
        return mean_sim

    @overrides
    def reset(self):
        self.sum = 0.
        self.num = 0.
