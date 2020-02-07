from typing import List
import torch
import os
import pathlib
from unittest import TestCase
from nltk.tree import Tree

from src.metrics.l2 import L2Error


PREDS = torch.tensor([
    [1., 1., 0.],
    [0., 1., 0.],
    [1., 1., 0.],
    [0., 0., 0.],
])

GOLD = torch.tensor([
    [0., 1., 0.],
    [0., 0., 1.],
    [0., 1., 1.],
    [0., 0., 0.],
])


class L2Test(TestCase):

    def test_l2(self):
        criterion = L2Error()
        criterion(PREDS, GOLD)
        value = criterion.get_metric(reset=True)
        exp_value = 5/4
        torch.testing.assert_allclose(value, exp_value)
    
    def test_normalized(self):
        # This test is designed to probe whether normalization is working properly.
        preds = torch.tensor([[100., 20.]])
        gold = torch.tensor([[2., -4.]])
        criterion = L2Error(normalize=True)
        criterion(preds, gold)
        value = criterion.get_metric(reset=True)
        exp_value = 1.4737651348114014
        torch.testing.assert_allclose(value, exp_value)
