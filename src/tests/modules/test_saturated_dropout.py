import torch
from unittest import TestCase

from src.modules.saturated_dropout import saturated_dropout, saturated_ste_dropout


class SaturatedDropoutTest(TestCase):

    def test_saturated_dropout(self):
        activations = torch.tensor([[[.5, .1, .9], [.2, .1, .4], [.1, .1, .2]]])
        mean_norms = torch.mean(activations, dim=[0, 1])
        threshold = .2
        
        exp_dropped = torch.tensor([[[.5, 0., .9], [.2, 0., .4], [.1, 0., .2]]])
        dropped = saturated_dropout(activations, mean_norms, threshold)
        torch.testing.assert_allclose(dropped, exp_dropped)

    def test_saturated_ste_dropout(self):
        activations = torch.tensor([[[.5, .1, .9], [.2, .1, .4], [.1, .1, .2]]])
        mean_norms = torch.mean(activations, dim=[0, 1])
        threshold = .2
        
        exp_dropped = torch.tensor([[[.5, 0., .9], [.2, 0., .4], [.1, 0., .2]]])
        dropped = saturated_ste_dropout(activations, mean_norms, threshold)
        torch.testing.assert_allclose(dropped, exp_dropped)