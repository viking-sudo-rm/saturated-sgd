import torch
from torch.nn import Parameter
from unittest import TestCase

from src.metrics.saturation_error import SaturationError


class SaturationErrorTest(TestCase):

    # TODO: Should we also test mocked here?

    def test_relu(self):
        torch.manual_seed(2)
        metric = SaturationError()

        # This network should not be strongly saturating.
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 10),
            torch.nn.Sigmoid(),
            torch.nn.Linear(10, 10),
        )
        cross_entropy = torch.nn.CrossEntropyLoss()
        inputs = torch.randn([1, 10])
        label = torch.tensor([4])
        
        loss = cross_entropy(model(inputs), label)
        parameters = list(model.parameters())
        loss_callback = lambda: cross_entropy(model(inputs), label)
        
        metric(loss, parameters, loss_callback)
        error = metric.get_metric(reset=True)
        exp_error = 1047.397705078125
        torch.testing.assert_allclose(error, exp_error)

        for param, new_param in zip(parameters, model.parameters()):
            torch.testing.assert_allclose(param, new_param)

    def test_sigmoid(self):
        torch.manual_seed(2)
        metric = SaturationError()

        # This network should be strongly saturating.
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 10),
            torch.nn.ReLU(),
        )
        cross_entropy = torch.nn.NLLLoss()
        inputs = torch.randn([1, 10])
        label = torch.tensor([4])
        
        loss = cross_entropy(model(inputs), label)
        parameters = list(model.parameters())
        loss_callback = lambda: cross_entropy(model(inputs), label)        
        metric(loss, parameters, loss_callback)

        error = metric.get_metric(reset=True)
        exp_error = 0.
        torch.testing.assert_allclose(error, exp_error)

        for param, new_param in zip(parameters, model.parameters()):
            torch.testing.assert_allclose(param, new_param)
