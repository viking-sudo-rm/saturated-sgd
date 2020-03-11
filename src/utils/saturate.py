import torch
from torch.nn import Module


class saturate:

    """Context manager in which a model will appear to be saturated."""

    def __init__(self, model: Module, infinity: float = 1000):
        self.model = model
        self.infinity = infinity
        self.old_param_data = []
        self.no_grad = torch.no_grad()
    
    def __enter__(self):
        self.no_grad.__enter__()
        for param in self.model.parameters():
            self.old_param_data.append(param.data)
            param.data = param.data.mul(self.infinity)
    
    def __exit__(self, type, value, traceback):
        for param, data in zip(self.model.parameters(), self.old_param_data):
            param.data = data
        self.no_grad.__exit__(type, value, traceback)
