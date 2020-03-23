from typing import Iterable

from overrides import overrides
import torch
import torch.nn.functional as F
from math import sqrt

from allennlp.training.metrics.metric import Metric


@Metric.register("param_norm")
class ParamNorm(Metric):

    """Track the norm of the parameters."""

    def __init__(self) -> None:
        self.l1_mean_norm = 0.
        self.l2_mean_norm = 0.
        self.l1_norm = 0.
        self.l2_norm = 0.
        self.max_abs = 0.
        self.min_abs = 0.
        self.mean_abs = 0.
        self.std_abs = 0.

    def __call__(
        self,
        params: Iterable[torch.FloatTensor],
    ):
        params = list(params)

        # Mean norms.
        # TODO: Can compute these more efficiently with running average if memory becomes issue.
        l1_sum = 0.
        l2_sum = 0.
        num = 0
        for param in params:
            if len(param.size()) != 2:
                continue
            num += param.size(0)
            l1_sum += torch.sum(param.norm(dim=1, p=1) / param.size(1))
            l2_sum += torch.sum(param.norm(dim=1, p=2) / sqrt(param.size(1)))
        self.l1_mean_norm = l1_sum.item() / num
        self.l2_mean_norm = l2_sum.item() / num

        # Global norms.
        params = torch.cat([torch.flatten(param) for param in params])
        self.l1_norm = torch.norm(params, p=1).item()
        self.l2_norm = torch.norm(params, p=2).item()

        # Parameter norms.
        abs_params = torch.abs(params)
        self.max_abs = torch.max(abs_params).item()
        self.min_abs = torch.min(abs_params).item()
        self.mean_abs = torch.mean(abs_params).item()
        self.std_abs = torch.std(abs_params).item()

    @overrides
    def get_metric(self, reset: bool = False):
        results = {
            "norm": {
                "l1": self.l1_norm,
                "l2": self.l2_norm,
            },
            "mean_norm": {
                "l1": self.l1_mean_norm,
                "l2": self.l2_mean_norm,
            },
            "abs": {
                "max": self.max_abs,
                "min": self.min_abs,
                "mean": self.mean_abs,
                "std": self.std_abs,
            },
        }
        if reset:
            self.reset()
        return results

    @overrides
    def reset(self):
        self.l1_mean_norm = 0.
        self.l2_mean_norm = 0.
        self.l1_norm = 0.
        self.l2_norm = 0.
        self.max_abs = 0.
        self.min_abs = 0.
        self.mean_abs = 0.
        self.std_abs = 0.
