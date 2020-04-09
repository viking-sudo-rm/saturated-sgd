from typing import Dict
import torch

from allennlp.training.metrics import Metric
from allennlp.models import Model, BasicClassifier
from allennlp.nn.util import get_text_field_mask
from allennlp.data import Vocabulary
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder
from allennlp.modules.text_field_embedders import TextFieldEmbedder

from src.utils.metrics import update_metrics
from src.utils.saturate import saturate
from src.utils.temp_prune import temp_prune


@Model.register("sat_metrics_classifier")
class SatMetricsClassifier(BasicClassifier):

    """Check the saturation of the final output layer in a text classifier.
    
    Based heavily on default AllenNLP `basic_classifier`.
    """

    def __init__(
        self,
        vocab: Vocabulary,
        text_field_embedder: TextFieldEmbedder,
        seq2vec_encoder: Seq2VecEncoder,
        parameter_metrics: Dict[str, Metric] = {},
        activation_metrics: Dict[str, Metric] = {},
        prune_metrics: Dict[str, Metric] = {},
        prune_percent: float = 0.9,
        **kwargs,
    ):
        super().__init__(vocab, text_field_embedder, seq2vec_encoder, **kwargs)
        self.parameter_metrics = parameter_metrics
        self.activation_metrics = activation_metrics
        self.prune_metrics = prune_metrics
        self.prune_percent = 0.9

    def forward(  # type: ignore
        self, tokens, label, _saturated=False,
    ) -> Dict[str, torch.Tensor]:
        # Quick-and-dirty copied.
        embedded_sequence = self._text_field_embedder(tokens)
        mask = get_text_field_mask(tokens)

        if self._seq2seq_encoder:
            embedded_sequence = self._seq2seq_encoder(embedded_sequence, mask=mask)

        embedded_text = self._seq2vec_encoder(embedded_sequence, mask=mask)

        if self._dropout:
            embedded_text = self._dropout(embedded_text)

        if self._feedforward is not None:
            embedded_text = self._feedforward(embedded_text)

        logits = self._classification_layer(embedded_text)
        probs = torch.nn.functional.softmax(logits, dim=-1)

        output_dict = {
            "embedded_sequence": embedded_sequence,
            "logits": logits,
            "probs": probs,
        }

        if _saturated:
            return output_dict

        if label is not None:
            loss = self._loss(logits, label.long().view(-1))
            output_dict["loss"] = loss
            self._accuracy(logits, label)

        if self.activation_metrics:
            with saturate(self, infinity=1e3):
                sat_output_dict = self.forward(tokens, None, _saturated=True)
        
        if self.prune_metrics and not self.training:
            with temp_prune(self, percent=self.prune_percent):
                prune_output_dict = self.forward(tokens, None, _saturated=True)

        for metric_fn in self.parameter_metrics.values():
            metric_fn(self.parameters())

        # These metrics compare the network to its saturated version.
        for metric_fn in self.activation_metrics.values():
            metric_fn(output_dict, sat_output_dict, mask=mask)
        
        # These metrics compare the network to its pruned version.
        if not self.training:  # Expensive, so only compute during eval time.
            for metric_fn in self.prune_metrics.values():
                metric_fn(output_dict, prune_output_dict, mask=mask)

        return output_dict
    
    def get_metrics(self, reset: bool = False):
        metrics = super().get_metrics(reset=reset)
        for metrics_dict in [self.parameter_metrics, self.activation_metrics, self.prune_metrics]:
            for name, metric in metrics_dict.items():
                value = metric.get_metric(reset=reset)
                update_metrics(metrics, name, value)
        return metrics
