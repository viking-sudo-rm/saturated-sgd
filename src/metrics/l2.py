from typing import Optional

from overrides import overrides
import torch
import torch.nn.functional as F

from allennlp.training.metrics.metric import Metric


@Metric.register("l2")
class L2Error(Metric):

    """L2 metric adapted from AllenNLP implementation of L1 (mean absolute error).
    
    Set setting `normalize=True`, cosine distance can also be computed.
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
        predictions, gold_labels, mask = self.unwrap_to_tensors(predictions, gold_labels, mask)

        errors = predictions - gold_labels
        if mask is not None:
            errors *= mask
        distances = torch.sum(errors * errors, dim=-1)

        if self._normalize:
            distances /= F.normalize(predictions, p=2, dim=-1)
            distances /= F.normalize(gold_labels, p=2, dim=-1)

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
