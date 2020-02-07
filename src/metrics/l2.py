from typing import Optional

from overrides import overrides
import torch
import torch.nn.functional as F

from allennlp.training.metrics.metric import Metric


@Metric.register("l2")
class L2Error(Metric):

    """L2 metric adapted from AllenNLP implementation of L1 (mean absolute error).
    
    Set setting `normalize=True`, we can get a metric that resembles (ish) cosine distance.
    """

    def __init__(self, normalize: bool = False) -> None:
        self._absolute_error = 0.0
        self._total_count = 0.0
        self._normalize = normalize

    def __call__(
        self,
        predictions: torch.Tensor,
        gold_labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        if self._normalize:
            predictions = F.normalize(predictions, p=2, dim=-1)
            gold_labels = F.normalize(gold_labels, p=2, dim=-1)

        errors = predictions - gold_labels
        distances = torch.sum(errors * errors, dim=-1)
        if mask is not None:
            distances *= mask
        distances = torch.sqrt(distances)

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
