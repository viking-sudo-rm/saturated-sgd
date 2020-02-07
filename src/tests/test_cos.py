import torch
from unittest import TestCase

from src.metrics.cos import CosDistance


class CosDistanceTest(TestCase):

    def test_magnitude_invariance(self):
        preds = torch.tensor([[10., 0., 0.]])
        gold = torch.tensor([[1., 0., 0.]])
        criterion = CosDistance()
        criterion(preds, gold)
        value = criterion.get_metric(reset=True)
        exp_value = 1.
        torch.testing.assert_allclose(value, exp_value)
    
    def test_orthogonal(self):
        preds = torch.tensor([[10., 0., 0.]])
        gold = torch.tensor([[0., 4., 0.]])
        criterion = CosDistance()
        criterion(preds, gold)
        value = criterion.get_metric(reset=True)
        exp_value = 0.
        torch.testing.assert_allclose(value, exp_value)
