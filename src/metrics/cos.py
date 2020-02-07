from typing import Optional

from overrides import overrides
import torch
import torch.nn.functional as F

from allennlp.training.metrics.metric import Metric


def _escape_norm(vectors):
    norms = torch.norm(vectors, p=2, dim=-1)
    ones = torch.ones_like(norms)
    return torch.where(torch.isnan(norms), ones, norms)


@Metric.register("cos")
class CosDistance(Metric):

    """Cosine distance metric adapted from AllenNLP implementation of L1 (mean absolute error)."""

    def __init__(self) -> None:
        self._absolute_error = 0.0
        self._total_count = 0.0

    def __call__(
        self,
        predictions: torch.Tensor,
        gold_labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):

        distances = torch.sum(predictions * gold_labels, dim=-1)
        if mask is not None:
            distances *= mask
        
        distances /= _escape_norm(predictions)
        distances /= _escape_norm(gold_labels)

        self._total_count += distances.numel()
        self._absolute_error += torch.sum(distances)

    def get_metric(self, reset: bool = False):
        mean_absolute_error = float(self._absolute_error) / float(self._total_count)
        if reset:
            self.reset()
        return mean_absolute_error

    @overrides
    def reset(self):
        self._absolute_error = 0.0
        self._total_count = 0.0
