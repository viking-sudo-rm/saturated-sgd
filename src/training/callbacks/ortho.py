# This is commented out because CallbackTrainer was removed.

from abc import abstractmethod
from typing import Any, Dict
from torch.nn import Module

from allennlp.training.metrics import Metric
from allennlp.training.trainer import BatchCallback


@BatchCallback.register("parameter_metrics")
class ParamMetricsCallback(BatchCallback):

    """A callback that adds the dot product gradient/param metric to training."""

    def __init__(self, metrics: Dict[str, Metric] = {}):
        self.parameter_metrics = metrics

    def __call__(
        self, trainer: "GradientDescentTrainer", *args, **kwargs
    ) -> None:
        # This is insane.
        writer = trainer._tensorboard._train_log
        parameters = list(trainer.model.parameters())
        for name, metric in self.parameter_metrics.items():
            metric(parameters)
            value = metric.get_metric(reset=False)
            writer.add_scalar(name, value)
