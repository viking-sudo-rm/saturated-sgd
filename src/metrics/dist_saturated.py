from typing import Optional

from overrides import overrides
import torch
import torch.nn.functional as F

from allennlp.training.metrics.metric import Metric


@Metric.register("sat_dist")
class SaturatedDist(Metric):

    """A metric to count the distance to the saturated value of the activation."""

    def __init__(self, min_value: float = -1.0, max_value: float = 1.0) -> None:
        self.min_value = min_value
        self.max_value = max_value
        self._absolute_error = 0.0
        self._total_count = 0.0

    def __call__(
        self, activations: torch.Tensor, mask: Optional[torch.Tensor] = None,
    ):
        mid_value = (self.max_value + self.min_value) / 2
        distances = torch.where(
            activations > mid_value,
            self.max_value - activations,
            activations - self.min_value,
        )

        if mask is not None:
            mask = mask.unsqueeze(-1)
            n_hidden = activations.size(-1)
            self._total_count += torch.sum(mask) * n_hidden
            self._absolute_error += torch.sum((distances * mask).float())
        else:
            self._total_count += distances.numel()
            self._absolute_error += torch.sum(distances.float())

    def get_metric(self, reset: bool = False):
        mean_absolute_error = float(self._absolute_error) / float(self._total_count)
        if reset:
            self.reset()
        return mean_absolute_error

    @overrides
    def reset(self):
        self._absolute_error = 0.0
        self._total_count = 0.0
