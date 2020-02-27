from typing import Callable, Iterable, Optional

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
        self.error_sum = 0.
        self.error_num = 0

    def __call__(
        self,
        logits: torch.FloatTensor,
        parameters: Iterable[Parameter],
        logits_callback: Callable,
        mask: Optional[torch.Tensor] = None,
    ):
        parameters = list(parameters)
        old_params = [param.clone().detach() for param in parameters]
        
        with torch.no_grad():
            for param in parameters:
                param.data.mul_(self.infinity)

            hard_logits = logits_callback()

            # Set to old value to avoid numerical instability in division.
            for param, old_param in zip(parameters, old_params):
                param.set_(old_param)

        rank = logits.detach().argsort(-1)
        hard_rank = hard_logits.argsort(-1)

        if mask is None:
            self.error_sum = torch.sum(rank != hard_rank)
            self.error_num = rank.numel()
        else:
            self.error_sum = torch.sum(mask.unsqueeze(-1) * (rank != hard_rank))
            self.error_num = torch.sum(mask)

    @overrides
    def get_metric(self, reset: bool = False):
        error = float(self.error_sum) / float(self.error_num)
        if reset:
            self.reset()
        return error

    @overrides
    def reset(self):
        self.error_sum = 0.
        self.error_num = 0
