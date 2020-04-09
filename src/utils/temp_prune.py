from typing import List
import torch
from torch.nn import Module, Linear, Parameter

from src.utils.saturate import masked_saturate
from src.utils.percentile import percentile


class temp_prune:

    """Context manager that does temporary pruning by wrapping a saturation context manager."""

    def __init__(self, model: Module, percent: float):
        params = self._get_params(model)
        norms = torch.cat([param.abs().flatten() for param in params])
        threshold = percentile(norms, percent)
        # We want to saturate (i.e. zero out) weights below the threshold.
        masks = [param.abs() < threshold for param in params]
        self.saturate = masked_saturate(params, masks, infinity=0.)

    @staticmethod
    def _get_params(model: Module) -> List[Parameter]:
        # Need to change this method based on the model.
        encoder = model._seq2seq_encoder
        return [param for param in encoder.parameters() if len(param.size()) == 2]

    def __enter__(self):
        self.saturate.__enter__()
    
    def __exit__(self, type, value, traceback):
        self.saturate.__exit__(type, value, traceback)
