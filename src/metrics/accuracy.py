from typing import Callable, Dict, Iterable, Optional

from overrides import overrides
import torch
import numpy as np

from allennlp.training.metrics.metric import Metric


@Metric.register("accuracy")
class Accuracy(Metric):

    """Measure accuracy (percent of times the tensors are the same) between soft and hard logits."""

    def __init__(self, key: str, ignore_mask: bool = False) -> None:
        self.key = key
        self.ignore_mask = ignore_mask

        self.sum = 0.0
        self.num = 0.0

    def __call__(
        self,
        output_dict: Dict,
        hard_output_dict: Dict,
        mask=None,
    ):
        logits = output_dict[self.key]
        hard_logits = hard_output_dict[self.key]

        if mask is not None and not self.ignore_mask:
            overlap = (logits == hard_logits) * mask.unsqueeze(-1)
            numel = torch.sum(mask) * logits.size(-1)
        else:
            overlap = logits == hard_logits
            numel = logits.numel()

        # Assume 0 is the batch dimension.
        self.sum += torch.sum(overlap)
        self.num += numel

    @overrides
    def get_metric(self, reset: bool = False):
        if self.num == 0:
            mean_sim = 0.
        else:
            mean_sim = float(self.sum) / float(self.num)
        if reset:
            self.reset()
        return mean_sim

    @overrides
    def reset(self):
        self.sum = 0.0
        self.num = 0.0
