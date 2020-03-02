from typing import Optional
from overrides import overrides
import torch

from allennlp.modules.token_embedders import TokenEmbedder

from src.training.callbacks.pruner import Prunable
from src.utils.percentile import percentile
from src.modules.masked_linear import MaskedLinear

# Unlike Gordon et al., this will also prune the pooling layer.
SHAPES = {(768, 768), (768, 3072), (3072, 768)}


BASIC = "BASIC"
ROWS = "ROWS"
COLS = "COLS"


@TokenEmbedder.register("bert_pruner_wrapper")
class BertPrunerWrapper(TokenEmbedder, Prunable):
    def __init__(
        self,
        bert_embedder: TokenEmbedder,
        percent: float,
        mode: str = BASIC,
    ):
        super().__init__()
        self.bert_embedder = bert_embedder
        self.percent = percent
        self.mode = mode

    @overrides
    def forward(
        self,
        token_ids: torch.LongTensor,
        mask: torch.LongTensor,
        type_ids: Optional[torch.LongTensor] = None,
        segment_concat_mask: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:  # type: ignore
        return self.bert_embedder(token_ids, mask, type_ids, segment_concat_mask)

    @overrides
    def prune(self):
        # Set to 0 and set requires_grad=False.
        # TODO FIXME finish this
        parameters = [param for param in self.bert_embedder.parameters() if tuple(param.size()) in SHAPES]

        if self.mode == BASIC:
            norms = torch.cat([param.flatten() for param in parameters]).abs()

        elif self.mode == ROWS:
            # This should correspond to saturation.
            norms = torch.cat([param.mean(dim=1) for param in parameters])
        
        elif self.mode == COLS:
            norms = torch.cat([param.mean(dim=0) for param in parameters])
        
        threshold = percentile(norms, self.percent)
