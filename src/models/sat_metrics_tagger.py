from typing import Optional

from overrides import overrides

import torch.nn.functional as F

from allennlp.models import SimpleTagger, Model
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder
from allennlp.data.vocabulary import Vocabulary
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.training.metrics.mean_absolute_error import MeanAbsoluteError

from src.metrics.l2 import L2Error


@Model.register("sat_metrics_tagger")
class SatMetricsTagger(SimpleTagger):

    """A tagger that logs metrics between saturated and unsaturated networks."""

    def __init__(
        self,
        vocab: Vocabulary,
        text_field_embedder: TextFieldEmbedder,
        encoder: Seq2SeqEncoder,
        calculate_span_f1: bool = None,
        label_encoding: Optional[str] = None,
        label_namespace: str = "labels",
        verbose_metrics: bool = False,
        initializer: InitializerApplicator = InitializerApplicator(),
        regularizer: Optional[RegularizerApplicator] = None,
    ) -> None:
        super().__init__(
            vocab,
            text_field_embedder,
            encoder,
            calculate_span_f1,
            label_encoding,
            label_namespace,
            verbose_metrics,
            initializer,
            regularizer,
        )

        assert hasattr(encoder, "saturated")
        self.sat_metrics = {
            "l1": MeanAbsoluteError(),
            "l2": L2Error(),
            "cos": L2Error(normalize=True),
        }

    def forward(self, tokens, tags, metadata):
        self.encoder.saturated = False

        # Quick-and-dirty copied from SimpleTagger.

        embedded_text_input = self.text_field_embedder(tokens)
        batch_size, sequence_length, _ = embedded_text_input.size()
        mask = get_text_field_mask(tokens)
        encoded_text = self.encoder(embedded_text_input, mask)

        logits = self.tag_projection_layer(encoded_text)
        reshaped_log_probs = logits.view(-1, self.num_classes)
        class_probabilities = F.softmax(reshaped_log_probs, dim=-1).view(
            [batch_size, sequence_length, self.num_classes]
        )

        output_dict = {"logits": logits, "class_probabilities": class_probabilities}

        if tags is not None:
            loss = sequence_cross_entropy_with_logits(logits, tags, mask)
            for metric in self.metrics.values():
                metric(logits, tags, mask.float())
            if self._f1_metric is not None:
                self._f1_metric(logits, tags, mask.float())
            output_dict["loss"] = loss

        if metadata is not None:
            output_dict["words"] = [x["words"] for x in metadata]

        # === Here starts my part. ===

        self.encoder.saturated = True
        sat_encodings = self.encoder(embedded_text_input, mask)

        for metric in self.sat_metrics.values():
            metric(encoded_text, sat_encodings)

        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False):
        metrics = super.get_metrics(reset=reset)
        for name, metric in self.sat_metrics.items():
            metrics[name] = metric.get_metric(reset)
        return metrics
