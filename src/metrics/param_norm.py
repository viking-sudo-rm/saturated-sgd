from typing import List

from overrides import overrides
import torch
import torch.nn.functional as F

from allennlp.training.metrics.metric import Metric


@Metric.register("param_norm")
class ParamNorm(Metric):

    """Track the norm of the parameters."""

    def __init__(self) -> None:
        self._absolute_error = 0.0
        self._total_count = 0.0

    def __call__(
        self,
        params: List[torch.FloatTensor],
    ):

        params = [torch.flatten(param) for param in params]
        params = torch.cat(params)
        norm = torch.norm(params, p=2)

        self._total_count += 1
        self._absolute_error += norm.item()

    def get_metric(self, reset: bool = False):
        mean_absolute_error = float(self._absolute_error) / float(self._total_count)
        if reset:
            self.reset()
        return mean_absolute_error

    @overrides
    def reset(self):
        self._absolute_error = 0.0
        self._total_count = 0.0
