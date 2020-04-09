# This is commented out because CallbackTrainer was removed.

from abc import abstractmethod
from typing import Any, Dict
from torch.nn import Module

from allennlp.training.trainer import EpochCallback


class Prunable:

    """Interface that all prunable modules should implement."""

    @abstractmethod
    def prune(self) -> None:
        return NotImplemented


@EpochCallback.register("pruner")
class PrunerCallback(EpochCallback):

    """A callback function that initiates pruning at a specific epoch."""

    def __init__(self, epoch: int):
        self.epoch = epoch

    def __call__(
        self, trainer: "GradientDescentTrainer", metrics: Dict[str, Any], epoch: int
    ) -> None:
        if epoch == self.epoch:
            for module in trainer.model.modules():
                if isinstance(module, Prunable):
                    module.prune()
