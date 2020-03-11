from typing import Optional, List

from overrides import overrides
import torch
import torch.nn.functional as F
from math import sqrt

from allennlp.training.metrics.metric import Metric


@Metric.register("num_saturated")
class NumSaturated(Metric):

    """A metric to count the percentage of activations that are saturated.
    
    There are two ways to do this:
    1. Weight saturation: Count the percent of weights with magnitude above some threshold.
    2. Activation saturation: Count the percent of activations with norm above some threshold.

    TODO: Which is a better norm here? L1 or L2? Using L2 now for default.
    """

    def __init__(self, weight_delta: float, act_delta: float, act_norm: int = 2) -> None:
        self.weight_delta = weight_delta
        self.act_delta = act_delta
        self.act_norm = act_norm

        self.n_weight = 0
        self.n_act = 0

    def __call__(self, parameters):
        self.reset()
        parameters = list(parameters)

        # Compute activation-level metrics.
        for param in parameters:
            if len(param.size()) != 2:
                continue
            norm = param.norm(dim=1, p=self.act_norm)
            if self.act_norm == 1:
                norm = norm / param.size(1)
            else:
                norm = norm / sqrt(param.size(1))
            self.n_act += torch.sum(norm > self.act_delta)

        # Compute weight-level metrics.
        parameters = torch.cat([param.flatten() for param in parameters])
        self.n_weight = torch.sum(parameters.abs() > self.weight_delta)

    def get_metric(self, reset: bool = False):
        results = {
            "weight": self.n_weight.item(),
            "act": self.n_act.item(),
        }
        if reset:
            self.reset()
        return results

    @overrides
    def reset(self):
        self.n_act = 0
        self.n_weight = 0
