from typing import List

from overrides import overrides
import torch

from allennlp.training.metrics.metric import Metric


@Metric.register("outward_projection")
class OutwardProjection(Metric):

    """Two measures of angular drift, using cosine distance."""

    def __init__(self) -> None:
        self.cos_sim = torch.nn.CosineSimilarity(dim=0)
        self.last_params = None
        self.alpha = None
        self.beta = None

    def __call__(
        self,
        params: List[torch.FloatTensor],
    ):
        params = [torch.flatten(param) for param in params]
        params = torch.cat(params)

        if self.alpha is None:
            self.alpha = 0.
            self.beta = 0.

        else:
            step = params - self.last_params
            self.alpha = torch.abs(self.cos_sim(params, self.last_params)).item()
            self.beta = torch.abs(self.cos_sim(step, self.last_params)).item()

        self.last_params = params.clone()

    def get_metric(self, reset: bool = False):
        results = {
            "alpha": self.alpha,
            "beta": self.beta,
        }
        if reset:
            self.reset()
        return results

    @overrides
    def reset(self):
        self.last_params = None
        self.alpha = None
        self.beta = None
