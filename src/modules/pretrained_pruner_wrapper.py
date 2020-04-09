from typing import List, Optional, Iterable, Tuple, Union
from overrides import overrides
import torch
from torch.nn import Parameter, Module, Linear
from transformers import *
from math import sqrt

from allennlp.modules.token_embedders import TokenEmbedder
from allennlp.modules.token_embedders.pretrained_transformer_embedder import (
    PretrainedTransformerEmbedder,
)
from allennlp.modules.token_embedders.pretrained_transformer_mismatched_embedder import (
    PretrainedTransformerMismatchedEmbedder,
)

from src.training.callbacks.pruner import Prunable
from src.utils.percentile import percentile
from src.utils.pruning import ThresholdPruning, RandomPruning, LEAST, MOST, RANDOM

# SHAPES = {(768, 768), (768, 3072), (3072, 768)}


@TokenEmbedder.register("pretrained_pruner_wrapper")
class PretrainedPrunerWrapper(TokenEmbedder, Prunable):
    def __init__(
        self,
        pretrained_embedder: TokenEmbedder,
        percent: float,
        prune_weights: bool = True,  # Whether to prune weights or activations.
        prune_mode: str = LEAST,
    ):
        super().__init__()
        self.pretrained_embedder = pretrained_embedder
        self.percent = percent
        self.prune_weights = prune_weights
        self.prune_mode = prune_mode

    @overrides
    def forward(self, *args, **kwargs,) -> torch.Tensor:  # type: ignore
        return self.pretrained_embedder(*args, **kwargs)

    @overrides
    def prune(self):
        with torch.no_grad():
            prunables = self._get_prunable_modules()
            parameters = [prunable.weight for prunable in prunables]
            threshold = self._get_threshold(parameters)
            for prunable in prunables:
                if self.prune_mode == RANDOM:
                    RandomPruning.apply(
                        prunable,
                        "weight",
                        self.percent,
                        prune_weights=self.prune_weights,
                    )
                else:
                    ThresholdPruning.apply(
                        prunable,
                        "weight",
                        threshold,
                        prune_weights=self.prune_weights,
                        prune_mode=self.prune_mode,
                    )

    def _get_prunable_modules(self) -> List[Module]:
        embedder = self.pretrained_embedder
        if isinstance(embedder, PretrainedTransformerMismatchedEmbedder):
            embedder = embedder._matched_embedder
        model = embedder.transformer_model
        if isinstance(model, BertModel):
            return [mod for mod in model.encoder.modules() if isinstance(mod, Linear)]
        else:
            return NotImplemented

    def _get_threshold(self, parameters: List[Parameter]) -> float:
        if self.prune_weights:
            norms = torch.cat([param.abs().flatten() for param in parameters])
        else:
            norms = torch.cat(
                [param.norm(dim=1, p=2) / sqrt(param.size(1)) for param in parameters]
            )
        return percentile(norms, self.percent)
