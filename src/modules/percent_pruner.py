from typing import Union
import torch

from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder

from src.modules.saturated_dropout import saturated_dropout, saturated_ste_dropout


# Thanks to https://gist.github.com/spezold/42a451682422beb42bc43ad0c0967a30.
def percentile(t: torch.tensor, q: float) -> Union[int, float]:
    """
    Return the ``q``-th percentile of the flattened input tensor's data.
    
    CAUTION:
     * Needs PyTorch >= 1.1.0, as ``torch.kthvalue()`` is used.
     * Values are not interpolated, which corresponds to
       ``numpy.percentile(..., interpolation="nearest")``.
       
    :param t: Input tensor.
    :param q: Percentile to compute, which must be between 0 and 100 inclusive.
    :return: Resulting value (scalar).
    """
    # Note that ``kthvalue()`` works one-based, i.e. the first sorted value
    # indeed corresponds to k=1, not k=0! Use float(q) instead of q directly,
    # so that ``round()`` returns an integer, even if q is a np.float32.
    k = 1 + round(float(q) * (t.numel() - 1))
    result = t.view(-1).kthvalue(k).values.item()
    return result


@Seq2SeqEncoder.register("percent_saturated_dropout")
class PercentSaturatedDropout(Seq2SeqEncoder):

    """Drop the tanh activations below a certain saturation level."""

    def __init__(
        self,
        encoder: Seq2SeqEncoder,
        percent: float,
        ste: bool = True,
        eval_only: bool = False,
    ):
        super().__init__()
        self.encoder = encoder
        self.percent = percent
        self.ste = ste
        self.eval_only = eval_only

    def forward(self, inputs, mask=None):
        activations = self.encoder(inputs, mask)
        if self.eval_only and self.training:
            return activations
        norms = torch.abs(activations)

        if mask is not None:
            sum_norms = torch.sum(norms * mask.unsqueeze(-1), dim=[0, 1])
            num_norms = torch.sum(mask)
            mean_norms = sum_norms / num_norms
        else:
            mean_norms = torch.mean(norms, dim=[0, 1])

        threshold = percentile(mean_norms, self.percent)

        if self.ste:
            return saturated_ste_dropout(activations, mean_norms, threshold)
        else:
            return saturated_dropout(activations, mean_norms, threshold)

    def get_input_dim(self):
        return self.encoder.get_input_dim()

    def get_output_dim(self):
        return self.encoder.get_output_dim()

    def is_bidirectional(self):
        return self.encoder.is_bidirectional()
