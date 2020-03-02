import torch
from torch.nn import Parameter
from unittest import TestCase
from parameterized import parameterized

from src.metrics.saturation_error import SaturationError

# Add new models below these ones so that randomness isn't disturbed.
torch.manual_seed(3)

RELU = torch.nn.Sequential(
    torch.nn.Linear(10, 10),
    torch.nn.ReLU(),
    torch.nn.Linear(10, 10),
)

# Does this network have zero cross entropy loss? What about the ranking of logits?
SIGMOID = torch.nn.Sequential(
    torch.nn.Linear(10, 10),
    torch.nn.Sigmoid(),
    torch.nn.Linear(10, 10),
)

WEIRD_RELU = torch.nn.Sequential(
    torch.nn.Linear(10, 10, bias=False),
    torch.nn.ReLU(),
    torch.nn.Linear(10, 10, bias=False),
    torch.nn.ReLU(),
)


class SaturationErrorTest(TestCase):

    @parameterized.expand([
        ["sigmoid", SIGMOID, {"max": .0, "sorted": .2}],
        ["relu", RELU, {"max": .0, "sorted": .7}],
        ["weird_relu", WEIRD_RELU, {"max": .0, "sorted": .0}],
    ])
    def test_logits(self, _name, model, exp_results):
        metric = SaturationError()
        old_params = [param.clone() for param in model.parameters()]

        inputs = torch.randn([1, 10])
        logits_callback = lambda: model(inputs)
        logits = logits_callback()
        
        metric(logits, model.parameters(), logits_callback)
        results = metric.get_metric(reset=True)
        assert results == exp_results

        # Check that parameters are reset properly
        for old_param, new_param in zip(old_params, model.parameters()):
            torch.testing.assert_allclose(old_param, new_param)
