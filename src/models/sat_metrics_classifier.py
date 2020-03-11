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
from src.metrics.saturation_error import SaturationError
from src.metrics.sat_cos_sim import SaturationCosSim
from src.utils.saturate import saturate


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
        **kwargs,
    ):
        super().__init__(vocab, text_field_embedder, seq2vec_encoder, **kwargs)
        self.parameter_metrics = parameter_metrics
        self.activation_metrics = activation_metrics
        self.saturation_error = SaturationError()
        self.saturation_sim = SaturationCosSim()



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

        with saturate(self, 1e3):
            sat_output_dict = self.forward(tokens, None, _saturated=True)

        logits_callback = lambda: ["logits"]
        self.saturation_error(logits, self.parameters(), lambda: sat_output_dict["logits"])
        self.saturation_sim(embedded_sequence, self, lambda: sat_output_dict["embedded_sequence"], mask)

        for metric_fn in self.parameter_metrics.values():
            metric_fn(self.parameters())

        for metric_fn in self.activation_metrics.values():
            metric_fn(embedded_sequence, mask=mask.float())

        return output_dict
    
    def get_metrics(self, reset: bool = False):
        metrics = super().get_metrics(reset=reset)

        # TODO: Refactor these saturation metrics under a common signature; probably just pass two dictionaries of results.
        value = self.saturation_error.get_metric(reset=reset)
        update_metrics(metrics, "sat_error", value)
        value = self.saturation_sim.get_metric(reset=reset)
        update_metrics(metrics, "sat_sim", value)

        for name, metric in self.parameter_metrics.items():
            value = metric.get_metric(reset=reset)
            update_metrics(metrics, name, value)
        
        for name, metric in self.activation_metrics.items():
            value = metric.get_metric(reset=reset)
            update_metrics(metrics, name, value)

        return metrics
