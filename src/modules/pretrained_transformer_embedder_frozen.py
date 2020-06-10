import math
from typing import Optional, Tuple

from overrides import overrides

import torch
import torch.nn.functional as F
from transformers import XLNetConfig
from transformers.modeling_auto import AutoModel

from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from allennlp.modules.token_embedders.token_embedder import TokenEmbedder
from allennlp.modules.token_embedders.pretrained_transformer_embedder import (
    PretrainedTransformerEmbedder,
)
from allennlp.nn.util import batched_index_select


@TokenEmbedder.register("pretrained_transformer_frozen")
class PretrainedTransformerEmbedderFrozen(TokenEmbedder):
    def __init__(self, embedder: PretrainedTransformerEmbedder):
        super().__init__()
        self.embedder = embedder
        for param in self.embedder.parameters():
            param.requires_grad = False

    @overrides
    def get_output_dim(self):
        return self.embedder.get_output_dim()

    @overrides
    def forward(
        self,
        token_ids: torch.LongTensor,
        mask: torch.BoolTensor,
        type_ids: Optional[torch.LongTensor] = None,
        segment_concat_mask: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:  # type: ignore
        return self.embedder(
            token_ids, mask, type_ids=type_ids, segment_concat_mask=segment_concat_mask
        )
