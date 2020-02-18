import torch
from unittest import TestCase

from allennlp.modules.seq2seq_encoders import PassThroughEncoder

from src.modules.percent_pruner import PercentSaturatedDropout


class PercentSaturatedDropoutTest(TestCase):

    def test_saturated_dropout_no_mask(self):
        encoder = PassThroughEncoder(input_dim=1)
        pruner = PercentSaturatedDropout(encoder, percent=.25)
        inputs = torch.arange(0, 10).unsqueeze(0).unsqueeze(0).float()
        dropped = pruner(inputs)
        # Rounds up on the boundary.
        exp_dropped = torch.tensor([[[0., 0., 2., 3., 4., 5., 6., 7., 8., 9.]]])
        torch.testing.assert_allclose(dropped, exp_dropped)

    def test_saturated_dropout_trivial_mask(self):
        encoder = PassThroughEncoder(input_dim=1)
        pruner = PercentSaturatedDropout(encoder, percent=.25)
        inputs = torch.arange(0, 10).unsqueeze(0).unsqueeze(0).float()
        mask = torch.ones(1, 1)
        dropped = pruner(inputs, mask)
        # Rounds up on the boundary.
        exp_dropped = torch.tensor([[[0., 0., 2., 3., 4., 5., 6., 7., 8., 9.]]])
        torch.testing.assert_allclose(dropped, exp_dropped)

    def test_saturated_dropout_negative(self):
        encoder = PassThroughEncoder(input_dim=1)
        pruner = PercentSaturatedDropout(encoder, percent=0.)
        inputs = torch.arange(0, 10).unsqueeze(0).unsqueeze(0).float() + 1
        dropped = pruner(inputs)
        # Rounds up on the boundary.
        exp_dropped = torch.tensor([[[1., 2., 3., 4., 5., 6., 7., 8., 9., 10.]]])
        torch.testing.assert_allclose(dropped, exp_dropped)
