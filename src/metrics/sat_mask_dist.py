from typing import Optional, List, Iterable

from math import sqrt
from overrides import overrides
import torch
from torch.nn import Parameter

from allennlp.training.metrics.metric import Metric

from src.utils.percentile import percentile


@Metric.register("mask_change")
class MaskChange(Metric):

    """Track the difference in the saturation mask, i.e. the set of highly saturated activations between consecutive batches."""

    def __init__(self, percent: float, normalize: bool = False) -> None:
        self.percent = percent
        self.normalize = normalize
        self.hamming_dist = None
        self.last_mask = None

    def __call__(
        self, parameters: Iterable[Parameter],
    ):
        if self.normalize:
            # Normalize by the square root of the input dimension.
            activation_norms = [param.norm(dim=1) / sqrt(param.size(1)) for param in parameters if len(param.size()) == 2]
        else:
            activation_norms = [param.norm(dim=1) for param in parameters if len(param.size()) == 2]
        activation_norms = torch.cat(activation_norms)

        threshold = percentile(activation_norms, self.percent)
        mask = activation_norms > threshold

        if self.last_mask is None:
            self.hamming_dist = 0.
        else:
            self.hamming_dist = torch.sum(self.last_mask != mask).item()
        self.last_mask = mask

    def get_metric(self, reset: bool = False):
        hamming_dist = self.hamming_dist
        if reset:
            self.reset()
        return hamming_dist

    @overrides
    def reset(self):
        self.hamming_dist = None
        self.last_mask = None
