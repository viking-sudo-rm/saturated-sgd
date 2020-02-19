from abc import abstractmethod, ABCMeta
import torch
from overrides import overrides

from allennlp.common import Registrable


class SaturationNorm(Registrable, metaclass=ABCMeta):

    """A norm for computing how saturated each activation is. For a set of saturation points S,
    this is defined as:

        min_{s \in S} |x - s|
    
    Rather than implementing this general approach, we write
    more efficient subclasses for each type of network.
    """

    def get_mean_norms(
        self, activations: torch.FloatTensor, mask: torch.Tensor
    ) -> torch.FloatTensor:
        norms = self.get_norms(activations)
        if mask is None:
            return torch.mean(norms, dim=[0, 1])
        
        sum_norms = torch.sum(norms * mask.unsqueeze(-1), dim=[0, 1])
        num_norms = torch.sum(mask)
        return sum_norms / num_norms

    @abstractmethod
    def get_norms(self, activations: torch.FloatTensor) -> torch.FloatTensor:
        return NotImplemented


@SaturationNorm.register("rnn")
class RnnNorm(SaturationNorm):
    @overrides
    def get_norms(self, activations):
        return 1 - torch.abs(activations)


@SaturationNorm.register("gru")
class GruNorm(SaturationNorm):
    @overrides
    def get_norms(self, activations):
        zero_dist = torch.abs(activations)
        extreme_dist = 1 - zero_dist
        return min(zero_dist, extreme_dist)
