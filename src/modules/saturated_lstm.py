"""An LSTM encoder that can be toggled between saturated and unsaturated modes."""

import torch
from overrides import overrides

from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder


def _sigmoid(tensor):
    return (tensor > 0).float()


def _tanh(tensor):
    ones = torch.ones_like(tensor)
    return torch.where(tensor > 0, ones, -ones)


class _Gate(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        soft_fn: callable = torch.sigmoid,
        hard_fn: callable = _sigmoid,
    ):
        super().__init__()
        self.inp = torch.nn.Linear(input_dim, hidden_dim, bias=False)
        self.hid = torch.nn.Linear(hidden_dim, hidden_dim)
        self.soft_fn = soft_fn
        self.hard_fn = hard_fn

    @overrides
    def forward(self, inp, hidden, saturated):
        inputs = self.inp(inp) + self.hid(hidden)
        if saturated:
            return self.hard_fn(inputs.detach())
        return self.soft_fn(inputs)


@Seq2SeqEncoder.register("saturated_lstm")
class SaturatedLstm(Seq2SeqEncoder):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_dim = input_size
        self.output_dim = hidden_size

        self.include = _Gate(input_size, hidden_size)
        self.forget = _Gate(input_size, hidden_size)
        self.output = _Gate(input_size, hidden_size)
        self.c_twiddle = _Gate(input_size, hidden_size, torch.tanh, _tanh)

        self.saturated = False

    @overrides
    def forward(self, inputs: torch.FloatTensor, mask: torch.ByteTensor):
        batch_size, seq_len, _ = inputs.size()
        mask = mask.unsqueeze(-1)

        hid_state = torch.zeros(batch_size, self.output_dim, device=inputs.device)
        cell_state = torch.zeros(batch_size, self.output_dim, device=inputs.device)
        states = []

        for time in range(seq_len):
            inp = inputs[:, time, :]

            include = self.include(inp, hid_state, self.saturated)
            forget = self.forget(inp, hid_state, self.saturated)
            output = self.output(inp, hid_state, self.saturated)
            c_twiddle = self.c_twiddle(inp, hid_state, self.saturated)

            cell_state = mask * (include * inp + forget * c_twiddle)
            hid_state = mask * (output * torch.tanh(cell_state))
            states.append(hid_state)

        return torch.stack(states, dim=1)

    @overrides
    def get_input_dim(self):
        return self.input_dim

    @overrides
    def get_output_dim(self):
        return self.output_dim

    @overrides
    def is_bidirectional(self):
        return False
