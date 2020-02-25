import torch
from unittest import TestCase

from src.modules.magnitude_prunable_rnn import _MagnitudePrunableRnn


INPUTS = torch.zeros(16, 32, 10)


class MagnitudePrunableRnnTest(TestCase):

    RNN = _MagnitudePrunableRnn(10, 20, .5)

    def test_rnn_prune_and_forward(self):
        self.RNN.prune()
        self.RNN.forward(INPUTS)
    
    def test_rnn_forward(self):
        self.RNN.forward(INPUTS)

    def test_rnn_prune_activations(self):
        rnn = _MagnitudePrunableRnn(10, 20, .5, activations=True)
        rnn.prune()
        rnn.forward(INPUTS)
