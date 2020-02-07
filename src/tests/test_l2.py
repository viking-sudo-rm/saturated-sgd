import torch
from unittest import TestCase
from math import sqrt

from src.metrics.l2 import L2Error


PREDS = torch.tensor(
    [[1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 0.0],]
)

GOLD = torch.tensor(
    [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.0, 0.0, 0.0],]
)


class L2Test(TestCase):
    def test_l2(self):
        criterion = L2Error()
        criterion(PREDS, GOLD)
        value = criterion.get_metric(reset=True)
        exp_value = (sqrt(1) + 2 * sqrt(2)) / 4
        torch.testing.assert_allclose(value, exp_value)

    def test_normalized_l2(self):
        criterion = L2Error(normalize=True)
        criterion(PREDS, GOLD)
        value = criterion.get_metric(reset=True)
        exp_value = 0.7948951125144958
        torch.testing.assert_allclose(value, exp_value)
