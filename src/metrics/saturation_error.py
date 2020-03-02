from typing import Callable, Iterable, Optional

from overrides import overrides
import torch
from torch.nn import Parameter

from allennlp.training.metrics.metric import Metric


@Metric.register("saturation_error")
class SaturationError(Metric):

    """Measure the error in terms of argmax ranking between saturated and unsaturated networks."""

    def __init__(self, infinity: float = 1e3) -> None:
        self.infinity = infinity
        self.sorted_sum = 0.
        self.sorted_num = 0
        self.max_sum = 0.
        self.max_num = 0

    def __call__(
        self,
        logits: torch.FloatTensor,
        parameters: Iterable[Parameter],
        logits_callback: Callable,
        mask: Optional[torch.Tensor] = None,
    ):
        parameters = list(parameters)
        old_param_data = [param.data for param in parameters]
        
        with torch.no_grad():
            for param in parameters:
                param.data = param.data.mul(self.infinity)

            hard_logits = logits_callback()

            # Set to old value to avoid numerical instability in division.
            for param, data in zip(parameters, old_param_data):
                param.data = data

        rank = logits.detach().argsort(-1)
        hard_rank = hard_logits.argsort(-1)
        pred = rank.select(-1, -1)
        hard_pred = hard_rank.select(-1, -1)

        if mask is None:
            self.sorted_sum = torch.sum(rank != hard_rank)
            self.sorted_num = rank.numel()
            self.max_sum = torch.sum(pred != hard_pred)
            self.max_num = pred.numel()
        else:
            self.sorted_sum = torch.sum(mask * (rank != hard_rank))
            self.sorted_num = torch.sum(mask)
            self.max_sum = torch.sum(mask * (pred != hard_pred))
            self.max_num = torch.sum(mask)

    @overrides
    def get_metric(self, reset: bool = False):
        results = {
            "sorted": float(self.sorted_sum) / float(self.sorted_num),
            "max": float(self.max_sum) / float(self.max_num),
        }
        if reset:
            self.reset()
        return results

    @overrides
    def reset(self):
        self.sorted_sum = 0.
        self.sorted_num = 0
        self.max_sum = 0.
        self.max_num = 0