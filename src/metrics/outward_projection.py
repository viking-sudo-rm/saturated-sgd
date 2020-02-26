from typing import List

from overrides import overrides
import torch

from allennlp.training.metrics.metric import Metric


@Metric.register("outward_projection")
class OutwardProjection(Metric):

    """Two measures of angular drift, using cosine distance."""

    def __init__(self, use_step: bool = False) -> None:
        # Tensors to memoize the computation.
        self.use_step = use_step
        self.cos_sim = torch.nn.CosineSimilarity(dim=0)
        self.last_params = None
        self.metric = None

    def __call__(
        self,
        params: List[torch.FloatTensor],
    ):
        params = [torch.flatten(param) for param in params]
        params = torch.cat(params)

        if self.last_params is None:
            self.metric = 0.

        if self.use_step:
            step = params - self.last_params
            self.metric = self.cos_sim(step, self.last_params)
            self.metric = torch.abs(self.metric).item()

        else:
            self.metric = self.cos_sim(params, self.last_params)
            self.metric = torch.abs(self.metric).item()

        self.last_params = params.clone()

    def get_metric(self, reset: bool = False):
        if reset:
            self.reset()
        return self.metric

    @overrides
    def reset(self):
        self.last_params = None
        self.metric = None
