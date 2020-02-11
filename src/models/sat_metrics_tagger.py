from typing import Optional, Dict

from overrides import overrides

import torch.nn.functional as F

from allennlp.models import SimpleTagger, Model
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder
from allennlp.data.vocabulary import Vocabulary
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.training.metrics import Metric

from src.metrics.l2 import L2Error


@Model.register("sat_metrics_tagger")
class SatMetricsTagger(SimpleTagger):

    """A tagger that logs metrics between saturated and unsaturated networks."""

    def __init__(
        self,
        vocab: Vocabulary,
        text_field_embedder: TextFieldEmbedder,
        encoder: Seq2SeqEncoder,
        parameter_metrics: Dict[str, Metric] = {},
        saturated_metrics: Dict[str, Metric] = {},
        activation_metrics: Dict[str, Metric] = {},
        **kwargs,
    ) -> None:
        super().__init__(
            vocab,
            text_field_embedder,
            encoder,
            **kwargs
        )

        self.saturated_metrics = saturated_metrics
        self.parameter_metrics = parameter_metrics
        self.activation_metrics = activation_metrics

        self.saturated = hasattr(self._contextualizer, "saturated")

    def forward(self, tokens, tags, metadata=None):

        if self.saturated:
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

        if self.saturated:
            self.encoder.saturated = True
            sat_encodings = self.encoder(embedded_text_input, mask)
            for metric in self.saturated_metrics.values():
                metric(encoded_text, sat_encodings, mask=mask)
    
        for metric_fn in self.parameter_metrics.values():
            metric_fn(self.parameters())
        
        for metric_fn in self.activation_metrics.values():
            metric_fn(encoded_text, mask=mask)

        return output_dict

    def get_metrics(self, reset: bool = False):
        metrics = super().get_metrics(reset=reset)

        if self.saturated:
            for name, metric in self.saturated_metrics.items():
                metrics[name] = metric.get_metric(reset=reset)

        for name, metric in self.parameter_metrics.items():
            metrics[name] = metric.get_metric(reset=reset)
        
        for name, metric in self.activation_metrics.items():
            metrics[name] = metric.get_metric(reset=reset)

        return metrics
