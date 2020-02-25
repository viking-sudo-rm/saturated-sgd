from typing import List

from overrides import overrides
import torch

from allennlp.training.metrics.metric import Metric


@Metric.register("step_ratio")
class StepRatio(Metric):

    """Check to what degree we are increasing magnitude with our parameter step."""

    def __init__(self, norm_scaled: bool = False) -> None:
        self.norm_scaled = norm_scaled
        # Tensors to memoize the computation.
        self.last_params = None
        self.last_norm = None
        self.last_norm_params = None
        self.metric = None

    def __call__(
        self,
        params: List[torch.FloatTensor],
    ):
        params = [torch.flatten(param) for param in params]
        params = torch.cat(params)
        norm = torch.norm(params, p=2)
        norm_params = params / norm

        if self.last_params is None:
            self.metric = 0.

        else:
            dir_norm = torch.norm(norm_params - self.last_norm_params, p=2)
            if self.norm_scaled:
                dir_norm *= self.last_norm
            mag_norm = torch.norm(params - self.last_params, p=2)
            self.metric = (dir_norm / mag_norm).item()

        self.last_params = params
        self.last_norm = norm
        self.last_norm_params = norm_params

    def get_metric(self, reset: bool = False):
        if reset:
            self.reset()
        return self.metric

    @overrides
    def reset(self):
        self.last_params = None
        self.last_norm = None
        self.last_norm_params = None
        self.metric = None
