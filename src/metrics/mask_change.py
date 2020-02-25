from typing import List

from overrides import overrides
import torch
import torch.nn.functional as F

from allennlp.training.metrics.metric import Metric


@Metric.register("mask_change")
class MaskChange(Metric):

    """Track the norm of the parameters."""

    def __init__(self, percent: float = 0.5) -> None:
        self.percent = percent
        self.last_mask = None

    def __call__(
        self,
        params: List[torch.FloatTensor],
    ):
        # TODO: Implement this.
        return NotImplemented

    def get_metric(self, reset: bool = False):
        return NotImplemented

    @overrides
    def reset(self):
        self.last_mask = None
