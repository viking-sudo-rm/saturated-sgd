from typing import List

from overrides import overrides
import torch

from allennlp.training.metrics.metric import Metric


@Metric.register("ortho")
class Ortho(Metric):

    """Two measures of angular drift, using cosine distance."""

    def __init__(self) -> None:
        self.value = 1.

    def __call__(
        self,
        params: List[torch.FloatTensor],
    ):
        params = [param for param in params if param.requires_grad]

        flat_params = [torch.flatten(param) for param in params]
        flat_params = torch.cat(flat_params)

        flat_grads = [torch.flatten(param.grad) for param in params]
        flat_grads = torch.cat(flat_grads)

        self.value = torch.dot(flat_params, flat_grads).item()

    def get_metric(self, reset: bool = False):
        value = self.value
        if reset:
            self.reset()
        return value

    @overrides
    def reset(self):
        self.value = 1.
