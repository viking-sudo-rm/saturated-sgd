from overrides import overrides
import torch
from torch.nn import Parameter, Module, Linear
from torch.nn import functional as F


def add_masks_to_model(model: Module):
    for name, module in model.named_modules():
        if isinstance(module, Linear):
            model._modules[name] = MaskedLinear.convert(module)
        elif module is not model:
            add_masks_to_model(module)


class MaskedLinear(Module):

    def __init__(self, weight: Parameter, bias: Parameter):
        super().__init__()
        self.weight = weight
        self.bias = bias
        self.mask = Parameter(torch.ones_like(weight))
    
    @classmethod
    def convert(cls, linear: Linear):
        weight = linear.weight.clone()
        bias = None if linear.bias is None else linear.bias.clone()
        return MaskedLinear(weight, bias)

    @overrides
    def forward(self, inputs):
        return F.linear(inputs, self.mask * self.weight, self.bias)

    def extra_repr(self):
        out_features, in_features = self.weight.size()
        return 'in_features={}, out_features={}, bias={}'.format(
            in_features, out_features, self.bias is not None
        )
