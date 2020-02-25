from overrides import overrides
import torch
from torch.nn import _VF
from torch.nn.utils.rnn import PackedSequence
from torch.nn import Parameter

from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper, Seq2SeqEncoder

from src.training.callbacks.pruner import Prunable
from src.utils.percentile import percentile


@Seq2SeqEncoder.register("magnitude_prunable_rnn")
class MagnitudePrunableRnn(PytorchSeq2SeqWrapper):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        percent: float,
        activations: bool = False,
    ):
        module = _MagnitudePrunableRnn(input_size, hidden_size, percent, activations)
        super().__init__(module=module)


class _MagnitudePrunableRnn(torch.nn.Module, Prunable):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        percent: float,
        activations: bool = False,
    ):
        super().__init__()
        self.rnn = torch.nn.RNN(input_size, hidden_size)
        self.percent = percent
        self.activations = activations  # FIXME: Support this.

        self.xmask = Parameter(
            torch.ones_like(self.rnn._flat_weights[0]), requires_grad=False
        )
        self.hmask = Parameter(
            torch.ones_like(self.rnn._flat_weights[1]), requires_grad=False
        )

    @overrides
    def forward(self, inputs, hx=None):
        # Largely copied from PyTorch RNN wrapper at:
        #   https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/rnn.py#L79

        is_packed = isinstance(inputs, PackedSequence)
        if is_packed:
            inputs, batch_sizes, sorted_indices, unsorted_indices = inputs
            max_batch_size = batch_sizes[0]
            max_batch_size = int(max_batch_size)
        else:
            batch_sizes = None
            max_batch_size = inputs.size(0) if self.rnn.batch_first else inputs.size(1)
            sorted_indices = None
            unsorted_indices = None

        if hx is None:
            num_directions = 2 if self.rnn.bidirectional else 1
            hx = torch.zeros(
                self.rnn.num_layers * num_directions,
                max_batch_size,
                self.rnn.hidden_size,
                dtype=inputs.dtype,
                device=inputs.device,
            )

        weights = (
            self.xmask * self.rnn._flat_weights[0],
            self.hmask * self.rnn._flat_weights[1],
            self.rnn._flat_weights[2],
            self.rnn._flat_weights[3],
        )

        # FIXME:
#          greatly increasing memory usage. To compact weights again call flatten_parameters().
# /pytorch/aten/src/ATen/native/cudnn/RNN.cpp:1238: UserWarning: RNN module weights are not part of single contiguous chunk of memory. This means they need to be compacted at every call, possibly greatly increasing memory usage. To compact weights^C

        if batch_sizes is None:
            result = _VF.rnn_tanh(
                inputs,
                hx,
                weights,
                self.rnn.bias,
                self.rnn.num_layers,
                self.rnn.dropout,
                self.rnn.training,
                self.rnn.bidirectional,
                self.rnn.batch_first,
            )
        else:
            result = _VF.rnn_tanh(
                inputs,
                batch_sizes,
                hx,
                weights,
                self.rnn.bias,
                self.rnn.num_layers,
                self.rnn.dropout,
                self.rnn.training,
                self.rnn.bidirectional,
            )

        output = result[0]
        hidden = result[1]

        if is_packed:
            output = PackedSequence(
                output, batch_sizes, sorted_indices, unsorted_indices
            )
        return output, self.rnn.permute_hidden(hidden, unsorted_indices)

    @overrides
    def prune(self):
        # Note: this thing is better defined for transformers since they have layer norm.
        weights = torch.cat(
            [self.rnn._flat_weights[0], self.rnn._flat_weights[1]], dim=-1
        )

        if not self.activations:
            abs_values = torch.abs(weights)
            threshold = percentile(abs_values, self.percent)
            self.xmask *= self.rnn._flat_weights[0] >= threshold
            self.hmask *= self.rnn._flat_weights[1] >= threshold

        else:
            norms = torch.norm(weights, p=2, dim=-1)
            threshold = percentile(norms, self.percent)
            self.xmask *= (norms >= threshold).unsqueeze(dim=-1)
            self.hmask *= (norms >= threshold).unsqueeze(dim=-1)
