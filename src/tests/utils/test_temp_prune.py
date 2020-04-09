from unittest import TestCase
import torch
from torch.nn import Parameter

from src.utils.temp_prune import temp_prune


class Mock:
    pass


class TempPruneTest(TestCase):

    def setUp(self):
        self.parameters = [
            Parameter(torch.tensor([[1., 2.], [3., 4.]])),
            Parameter(torch.tensor(-10.)),
        ]

        self.model = Mock()
        self.model._seq2seq_encoder = Mock()
        self.model._seq2seq_encoder.parameters = lambda: self.parameters

    def test_temp_prune(self):
        with temp_prune(self.model, percent=.5):
            torch.testing.assert_allclose(self.parameters[0], torch.tensor([[0., 0.], [3., 4.]]))
            torch.testing.assert_allclose(self.parameters[1], torch.tensor(-10.))
        torch.testing.assert_allclose(self.parameters[0], torch.tensor([[1., 2.], [3., 4.]]))
        torch.testing.assert_allclose(self.parameters[1], torch.tensor(-10.))
