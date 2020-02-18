from typing import Optional, List

from overrides import overrides
import torch

from allennlp.training.metrics.metric import Metric


def atanh(x):
    return 0.5 * (x.log1p() - (-x).log1p())


@Metric.register("preactivation_abs")
class PreactivationAbs(Metric):

    """Track the absolute values of the preactivations.
    
    Assumes that the activations passed in use tanh activation function.
    """

    def __init__(self, infinity: float = 10.) -> None:
        self.infinity = infinity
        self._absolute_error = 0.0
        self._total_count = 0.0

    def __call__(
        self, activations: torch.Tensor, mask: Optional[torch.Tensor] = None,
    ):
        preactivations = atanh(activations)
        preactivations = torch.abs(preactivations)

        # Empirically, the inverse tanh seems to be numerically unstable beyond around 10.
        infinities = self.infinity * torch.ones_like(preactivations)
        preactivations = torch.where(torch.isinf(preactivations), infinities, preactivations)

        if mask is not None:
            mask = mask.unsqueeze(-1)
            n_hidden = activations.size(-1)
            self._total_count += torch.sum(mask) * n_hidden
            self._absolute_error += torch.sum((preactivations * mask).float())
        else:
            self._total_count += preactivations.numel()
            self._absolute_error += torch.sum(preactivations.float())

    def get_metric(self, reset: bool = False):
        mean_absolute_error = float(self._absolute_error) / float(self._total_count)
        if reset:
            self.reset()
        return mean_absolute_error

    @overrides
    def reset(self):
        self._absolute_error = 0.0
        self._total_count = 0.0
