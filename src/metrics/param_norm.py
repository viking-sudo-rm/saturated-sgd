from typing import List

from overrides import overrides
import torch
import torch.nn.functional as F

from allennlp.training.metrics.metric import Metric


@Metric.register("param_norm")
class ParamNorm(Metric):

    """Track the norm of the parameters."""

    def __init__(self) -> None:
        self.norm = 0.
        self.max_abs = 0.
        self.min_abs = 0.
        self.mean_abs = 0.
        self.std_abs = 0.

    def __call__(
        self,
        params: List[torch.FloatTensor],
    ):
        params = [torch.flatten(param) for param in params]
        params = torch.cat(params)
        self.norm = torch.norm(params, p=2).item()

        abs_params = torch.abs(params)
        self.max_abs = torch.max(abs_params).item()
        self.min_abs = torch.min(abs_params).item()
        self.mean_abs = torch.mean(abs_params).item()
        self.std_abs = torch.std(abs_params).item()

    @overrides
    def get_metric(self, reset: bool = False):
        results = {
            "norm": self.norm,
            "max_abs": self.max_abs,
            "min_abs": self.min_abs,
            "mean_abs": self.mean_abs,
            "std_abs": self.std_abs,
        }
        if reset:
            self.reset()
        return results

    @overrides
    def reset(self):
        self.norm = 0.
        self.max_abs = 0.
        self.min_abs = 0.
        self.mean_abs = 0.
        self.std_abs = 0.
