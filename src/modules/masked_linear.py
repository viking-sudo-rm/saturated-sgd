from overrides import overrides
import torch
from torch.nn import Parameter, Module, Linear
from torch.nn import functional as F


def add_masks_to_module(module: Module) -> None:
    """Add masks to all the linear layers within this module."""
    # Update object attributes.
    for attr_name in dir(module):
        submodule = getattr(module, attr_name)
        if isinstance(submodule, Linear):
            masked = MaskedLinear.convert(submodule)
            setattr(module, attr_name, masked)
        elif isinstance(submodule, Module) and submodule is not module:
            add_masks_to_module(submodule)
        
    # Also update any submodules that might not be object attributes.
    pairs = list(module.named_modules())
    for name, submodule in pairs:
        if isinstance(submodule, Linear):
            module._modules[name] = MaskedLinear.convert(submodule)


class MaskedLinear(Module):

    def __init__(self, weight: Parameter, bias: Parameter):
        super().__init__()
        self.weight = weight
        self.bias = bias
        self.mask = Parameter(torch.ones_like(weight), requires_grad=False)

    @classmethod
    def convert(cls, linear: Linear):
        weight = Parameter(linear.weight.clone())
        bias = None if linear.bias is None else Parameter(linear.bias.clone())
        return MaskedLinear(weight, bias)

    @overrides
    def forward(self, inputs):
        return F.linear(inputs, self.mask * self.weight, self.bias)

    def extra_repr(self):
        out_features, in_features = self.weight.size()
        return 'in_features={}, out_features={}, bias={}'.format(
            in_features, out_features, self.bias is not None
        )
