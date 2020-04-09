"""Pruning methods implemented within the PyTorch pruning API."""

from overrides import overrides
import torch
from torch.nn.utils.prune import BasePruningMethod
from math import sqrt

LEAST = "LEAST"
MOST = "MOST"
RANDOM = "RANDOM"  # TODO: Implement a pruner for this.


class ThresholdPruning(BasePruningMethod):

    def __init__(self, threshold: float, prune_weights: bool = True, prune_mode: str = LEAST):
        self.threshold = threshold
        self.prune_weights = prune_weights
        self.prune_mode = prune_mode

    @overrides
    def compute_mask(self, param: torch.FloatTensor, default_mask: torch.Tensor) -> torch.Tensor:
        """Assume the parameter is a matrix."""
        if self.prune_weights:
            norms = param.abs()
        else:
            norms = param.norm(dim=1, p=2) / sqrt(param.size(1))
            norms = norms.unsqueeze(dim=1)

        if self.prune_mode == LEAST:
            return default_mask * (norms > self.threshold)
        elif self.prune_mode == MOST:
            return default_mask * (norms <= self.threshold)
        else:
            return NotImplemented


class RandomPruning(BasePruningMethod):

    def __init__(self, percent: float, prune_weights: bool = True):
        self.percent = percent
        self.prune_weights = prune_weights
    
    @overrides
    def compute_mask(self, param: torch.FloatTensor, default_mask: torch.Tensor) -> torch.Tensor:
        if self.prune_weights:
            probs = (1 - self.percent) * torch.ones_like(default_mask)
        else:
            size = default_mask.size(0)
            probs = (1 - self.percent) * torch.ones(size, 1, device=param.device)
        return default_mask * probs.bernoulli().bool()
