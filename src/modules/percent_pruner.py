from typing import Union
import torch

from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder

from src.modules.saturated_dropout import (
    saturated_dropout,
    saturated_ste_dropout,
    random_dropout,
)
from src.saturation_norms import SaturationNorm, RnnNorm
from src.utils.percentile import percentile


@Seq2SeqEncoder.register("percent_saturated_dropout")
class PercentSaturatedDropout(Seq2SeqEncoder):

    """Drop the tanh activations below a certain saturation level."""

    def __init__(
        self,
        encoder: Seq2SeqEncoder,
        percent: float,
        ste: bool = False,
        eval_only: bool = False,
        random_baseline: bool = False,
        norm: SaturationNorm = RnnNorm(),
        prune_saturated: bool = False,
    ):
        super().__init__()
        self.encoder = encoder
        self.percent = percent
        self.ste = ste
        self.eval_only = eval_only
        self.random_baseline = random_baseline
        self.norm = norm
        self.prune_saturated = prune_saturated

    def forward(self, inputs, mask=None):
        activations = self.encoder(inputs, mask)
        if self.percent == 0.0 or (self.eval_only and self.training):
            return activations

        mean_norms = self.norm.get_mean_norms(activations, mask)
        # Want to get the smallest norms.

        threshold = (
            -percentile(-mean_norms, self.percent)
            if not self.prune_saturated
            else percentile(mean_norms, self.percent)
        )

        if self.ste:
            return saturated_ste_dropout(activations, mean_norms, threshold, self.prune_saturated)
        elif self.random_baseline:
            return random_dropout(activations, self.percent)
        else:
            return saturated_dropout(activations, mean_norms, threshold, self.prune_saturated)

    def get_input_dim(self):
        return self.encoder.get_input_dim()

    def get_output_dim(self):
        return self.encoder.get_output_dim()

    def is_bidirectional(self):
        return self.encoder.is_bidirectional()
