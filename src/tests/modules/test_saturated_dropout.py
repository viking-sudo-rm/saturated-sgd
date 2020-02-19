import torch
from unittest import TestCase

from src.modules.saturated_dropout import saturated_dropout, saturated_ste_dropout


class SaturatedDropoutTest(TestCase):
    def test_saturated_dropout(self):
        activations = torch.tensor(
            [[[0.5, 0.11, 0.9], [0.2, 0.05, 0.4], [0.1, 0.1, 0.2]]]
        )
        mean_norms = torch.mean(activations, dim=[0, 1])
        threshold = 0.2

        exp_dropped = torch.tensor(
            [
                [
                    [0.0000, 0.1100, 0.0000],
                    [0.0000, 0.0500, 0.0000],
                    [0.0000, 0.1000, 0.0000],
                ]
            ]
        )
        dropped = saturated_dropout(activations, mean_norms, threshold)
        torch.testing.assert_allclose(dropped, exp_dropped)

    def test_saturated_ste_dropout(self):
        activations = torch.tensor(
            [[[0.5, 0.11, 0.9], [0.2, 0.05, 0.4], [0.1, 0.1, 0.2]]]
        )
        mean_norms = torch.mean(activations, dim=[0, 1])
        threshold = 0.2

        exp_dropped = torch.tensor(
            [
                [
                    [0.0000, 0.1100, 0.0000],
                    [0.0000, 0.0500, 0.0000],
                    [0.0000, 0.1000, 0.0000],
                ]
            ]
        )
        dropped = saturated_ste_dropout(activations, mean_norms, threshold)
        torch.testing.assert_allclose(dropped, exp_dropped)
