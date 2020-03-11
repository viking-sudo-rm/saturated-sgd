# This is commented out because CallbackTrainer was removed.

# from torch.nn import Module
from abc import abstractmethod

# from allennlp.training.callbacks.callback import Callback, handle_event
# from allennlp.training.callbacks.events import Events
# from allennlp.training.callback_trainer import CallbackTrainer

class Prunable:

    """Interface that all prunable modules should implement."""

    @abstractmethod
    def prune(self) -> None:
        return NotImplemented


# @Callback.register("pruner")
# class PrunerCallback(Callback):

#     """A callback function that initiates pruning at a specific epoch, or the end of training."""

#     def __init__(self, epoch: int):
#         self.epoch = epoch  # Set to -1 to do pruning after training has finished.

#     @handle_event(Events.EPOCH_END)
#     def on_epoch_end(self, trainer: CallbackTrainer):
#         if trainer.epoch_number == self.epoch:
#             self._prune_all_modules(trainer.model)

#     @handle_event(Events.TRAINING_END)
#     def on_training_end(self, trainer: CallbackTrainer):
#         if trainer.epoch_number == -1:
#             self._prune_all_modules(trainer.model)

#     @classmethod
#     def _prune_all_modules(cls, module: Module):
#         if isinstance(module, Prunable):
#             module.prune()

#         for submodule in module.modules():
#             cls._prune_all_modules(submodule)
