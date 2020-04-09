# from typing import Dict
# import torch

# from allennlp.training.metrics import Metric
# from allennlp.models import Model, LanguageModel
# from allennlp.nn.util import get_text_field_mask
# from allennlp.data import Vocabulary
# from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder
# from allennlp.modules.text_field_embedders import TextFieldEmbedder


# @Model.register("sat_metrics_lm")
# class SatMetricsLanguageModel(LanguageModel):

#     def __init__(
#         self,
#         vocab: Vocabulary,
#         text_field_embedder: TextFieldEmbedder,
#         contextualizer: Seq2SeqEncoder,
#         parameter_metrics: Dict[str, Metric] = {},
#         saturated_metrics: Dict[str, Metric] = {},
#         activation_metrics: Dict[str, Metric] = {},
#         **kwargs,
#     ):
#         super().__init__(vocab, text_field_embedder, contextualizer, **kwargs)

#         self.saturated_metrics = saturated_metrics
#         self.parameter_metrics = parameter_metrics
#         self.activation_metrics = activation_metrics

#         self.saturated = hasattr(self._contextualizer, "saturated")
    
#     def forward(  # type: ignore
#         self, source
#     ) -> Dict[str, torch.Tensor]:
#         if self.saturated:
#             self._contextualizer.saturated = False

#         return_dict = super().forward(source)
#         unsaturated = return_dict["lm_embeddings"]
#         mask = get_text_field_mask(source)

#         # Note: Is this saturated? What about the embedding layer?
#         if self.saturated:
#             self._contextualizer.saturated = True
#             embeddings = return_dict["noncontextual_token_embeddings"]
#             saturated = self._contextualizer(embeddings, mask)

#             for metric_fn in self.saturated_metrics.values():
#                 metric_fn(unsaturated, saturated, mask=mask)
        
#         for metric_fn in self.parameter_metrics.values():
#             metric_fn(self.parameters())
        
#         for metric_fn in self.activation_metrics.values():
#             metric_fn(unsaturated, mask=mask)

#         return return_dict
    
#     def get_metrics(self, reset: bool = False):
#         metrics = super().get_metrics(reset=reset)

#         if self.saturated:
#             for name, metric in self.saturated_metrics.items():
#                 metrics[name] = metric.get_metric(reset=reset)

#         for name, metric in self.parameter_metrics.items():
#             metrics[name] = metric.get_metric(reset=reset)
        
#         for name, metric in self.activation_metrics.items():
#             metrics[name] = metric.get_metric(reset=reset)

#         return metrics
