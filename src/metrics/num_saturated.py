from typing import Optional, List

from overrides import overrides
import torch
import torch.nn.functional as F

from allennlp.training.metrics.metric import Metric


@Metric.register("num_saturated")
class NumSaturated(Metric):

    """A metric to count the percentage of activations that are saturated.
    
    We approximate this by rounding the activations to the nearest integer and measuring the
    distance from the actual value to the rounded value.
    """

    def __init__(self, delta: float = 0.1, values: List[float] = [-1.0, 1.0]) -> None:
        self.delta = delta
        self.values = values  # The values that we consider asymptotes for saturation.
        self._absolute_error = 0.0
        self._total_count = 0.0

    def __call__(
        self, activations: torch.Tensor, mask: Optional[torch.Tensor] = None,
    ):
        is_saturated = torch.zeros_like(activations, dtype=torch.bool)
        for value in self.values:
            is_saturated |= torch.abs(activations - value) < self.delta

        if mask is not None:
            mask = mask.unsqueeze(-1)
            n_hidden = activations.size(-1)
            self._total_count += torch.sum(mask) * n_hidden
            self._absolute_error += torch.sum((is_saturated * mask).float())
        else:
            self._total_count += is_saturated.numel()
            self._absolute_error += torch.sum(is_saturated.float())

    def get_metric(self, reset: bool = False):
        mean_absolute_error = float(self._absolute_error) / float(self._total_count)
        if reset:
            self.reset()
        return mean_absolute_error

    @overrides
    def reset(self):
        self._absolute_error = 0.0
        self._total_count = 0.0
