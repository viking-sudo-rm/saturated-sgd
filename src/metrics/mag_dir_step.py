from typing import List

from overrides import overrides
import torch

from allennlp.training.metrics.metric import Metric


@Metric.register("mag_dir_step")
class MagnitudeDirectionStep(Metric):

    """Compute the parameter step size in magnitude/direction space."""

    def __init__(self) -> None:
        self.last_magnitude = None
        self.last_direction = None
        self.magnitude_metric = None
        self.direction_metric = None

    def __call__(
        self,
        params: List[torch.FloatTensor],
    ):
        params = [torch.flatten(param) for param in params]
        params = torch.cat(params)

        magnitude = torch.norm(params, p=2)
        direction = params / magnitude

        if self.last_magnitude is None:
            self.magnitude_metric = 0.
            self.direction_metric = 0.

        else:
            self.magnitude_metric = torch.abs(self.last_magnitude - magnitude).item()
            self.direction_metric = torch.norm(self.last_direction - direction).item()

        self.last_magnitude = magnitude
        self.last_direction = direction

    def get_metric(self, reset: bool = False):
        result = {
            "magnitude_step": self.magnitude_metric,
            "direction_step": self.direction_metric,
        }
        if reset:
            self.reset()
        return result

    @overrides
    def reset(self):
        self.last_magnitude = None
        self.last_direction = None
        self.magnitude_metric = None
        self.direction_metric = None
