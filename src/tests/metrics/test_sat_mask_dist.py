import torch
from unittest import TestCase

from src.metrics.sat_mask_dist import MaskChange


class MaskChangeTest(TestCase):
    def test_metric_identity(self):
        torch.manual_seed(2)
        params = [torch.randn(10, 10)]
        metric = MaskChange(.5)
        metric(params)
        metric(params)
        assert metric.get_metric() == 0.
