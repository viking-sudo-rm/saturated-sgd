from typing import List

from overrides import overrides
import torch
import torch.nn.functional as F

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
        params: List[torch.FloatTensor],
    ):
        # Mean norms.
        params = [torch.flatten(param) for param in params]
        # TODO: Can compute these more efficiently with running average if memory becomes issue.
        self.l1_mean_norm = torch.mean(torch.stack([param.norm(p=1) for param in params])).item()
        self.l2_mean_norm = torch.mean(torch.stack([param.norm(p=2) for param in params])).item()

        # Global norms.
        params = torch.cat(params)
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
