import torch
from overrides import overrides

from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder


@Seq2SeqEncoder.register("norm_rnn")
class NormRnn(Seq2SeqEncoder):
    def __init__(self, input_dim: int, hidden_dim: int, scale: float = 1.):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.scale = scale

        self.in_proj = torch.nn.Linear(input_dim, hidden_dim, bias=False)
        self.hid_proj = torch.nn.Linear(hidden_dim, hidden_dim, bias=True)
    
    @overrides
    def forward(self, inputs: torch.Tensor, mask: torch.BoolTensor = None) -> torch.Tensor:
        batch_size, seq_len, _ = inputs.size()
        state = torch.zeros(batch_size, self.hidden_dim, device=inputs.device)
        norm = torch.cat([param.flatten() for param in self.parameters()]).norm(p=2)
        
        states = []
        for time in range(seq_len):
            inp = inputs[:, time, :]
            preact = self.in_proj(inp) + self.hid_proj(state)
            state = torch.tanh(self.scale / norm * preact)
            states.append(state)
        return torch.stack(states, dim=1) * mask.unsqueeze(dim=-1)

    @overrides
    def get_input_dim(self):
        return self.input_dim
    
    @overrides
    def get_output_dim(self):
        return self.hidden_dim
    
    @overrides
    def is_bidirectional(self):
        return False
