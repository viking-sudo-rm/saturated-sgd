import torch
from unittest import TestCase

from allennlp.modules.seq2seq_encoders import PassThroughEncoder

from src.modules.percent_pruner import PercentSaturatedDropout


INPUTS = (torch.arange(0, 10).unsqueeze(0).unsqueeze(0).float() + 1) / 10.0


class PercentSaturatedDropoutTest(TestCase):

    """Test cases for pruning unsaturated activations.
    
    Most of these test cases utilize the RnnNorm, i.e. assume close to zero == unsaturated.
    """

    def test_saturated_dropout_no_mask(self):
        encoder = PassThroughEncoder(input_dim=1)
        pruner = PercentSaturatedDropout(encoder, percent=0.25)
        dropped = pruner(INPUTS)
        exp_dropped = torch.tensor([[[0.0, 0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]]])
        torch.testing.assert_allclose(dropped, exp_dropped)

    def test_saturated_dropout_no_mask_10(self):
        encoder = PassThroughEncoder(input_dim=1)
        pruner = PercentSaturatedDropout(encoder, percent=0.1)
        dropped = pruner(INPUTS)
        exp_dropped = torch.tensor([[[0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]]])
        torch.testing.assert_allclose(dropped, exp_dropped)

    def test_saturated_dropout_trivial_mask(self):
        encoder = PassThroughEncoder(input_dim=1)
        pruner = PercentSaturatedDropout(encoder, percent=0.25)
        mask = torch.ones(1, 1)
        dropped = pruner(INPUTS, mask)
        exp_dropped = torch.tensor([[[0.0, 0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]]])
        torch.testing.assert_allclose(dropped, exp_dropped)

    def test_saturated_dropout_zero(self):
        encoder = PassThroughEncoder(input_dim=1)
        pruner = PercentSaturatedDropout(encoder, percent=0.0)
        dropped = pruner(INPUTS)
        exp_dropped = torch.tensor([[[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]]])
        torch.testing.assert_allclose(dropped, exp_dropped)
