from typing import Callable, Iterable

from overrides import overrides
import torch
from torch.nn import Parameter

from allennlp.models import Model
from allennlp.training.metrics.metric import Metric


@Metric.register("saturation_error")
class SaturationError(Metric):

    """Compute the error between the saturated and unsaturated networks."""

    def __init__(self, infinity: float = 1e3) -> None:
        self.infinity = infinity
        self.sat_error = float("inf")

    def __call__(
        self,
        loss: torch.FloatTensor,
        parameters: Iterable[Parameter],
        loss_callback: Callable,
    ):
        parameters = list(parameters)
        old_params = [param.clone().detach() for param in parameters]
        for param in parameters:
            param.data.requires_grad = False
            param.data.mul_(self.infinity)
        
        with torch.no_grad():
            hard_loss = loss_callback()

            # Set to old value to avoid numerical instability in division.
            for param, old_param in zip(parameters, old_params):
                param.set_(old_param)
                param.data.requires_grad = True

        self.sat_error = (loss.detach() - hard_loss).abs().item()

    @overrides
    def get_metric(self, reset: bool = False):
        error = self.sat_error
        if reset:
            self.reset()
        return error

    @overrides
    def reset(self):
        self.sat_error = float("inf")
