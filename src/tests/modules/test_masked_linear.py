import torch
from unittest import TestCase

from src.modules.masked_linear import MaskedLinear, add_masks_to_module


INPUTS = (torch.arange(0, 10).unsqueeze(0).unsqueeze(0).float() + 1) / 10.0


class _MockModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(10, 10)
        self.linear2 = torch.nn.Linear(3, 3)
        self.other = torch.nn.BatchNorm1d(10)


class MaskedLinearTest(TestCase):

    """Test cases for pruning unsaturated activations.
    
    Most of these test cases utilize the RnnNorm, i.e. assume close to zero == unsaturated.
    """

    def test_masked_linear(self):
        torch.random.manual_seed(1066)
        inputs = torch.randn(10)
        linear = torch.nn.Linear(10, 10)
        masked_linear = MaskedLinear.convert(linear)

        torch.testing.assert_allclose(masked_linear(inputs), linear(inputs))
        masked_linear.mask.data.fill_(0.)
        torch.testing.assert_allclose(masked_linear(inputs), linear.bias)

    def test_add_masks_to_model_sequential(self):
        torch.random.manual_seed(37)
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 10),
            torch.nn.Softmax(dim=-1),
        )
        add_masks_to_module(model)

        modules = list(model.children())
        assert isinstance(modules[0], MaskedLinear)
        assert isinstance(modules[1], torch.nn.ReLU)
        assert isinstance(modules[2], MaskedLinear)
        assert isinstance(modules[3], torch.nn.Softmax)
    
    def test_add_masks_to_model_update_attr_refs(self):
        model = _MockModel()
        add_masks_to_module(model)
        self.assertIsInstance(model.linear1, MaskedLinear)
        self.assertIsInstance(model.linear2, MaskedLinear)
        self.assertIsInstance(model.other, torch.nn.BatchNorm1d)
