from typing import List, Any, Dict, Tuple
import torch.optim as optim
import torch

from allennlp.training.optimizers import Optimizer, make_parameter_groups


class NormSGD(optim.SGD):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None or not p.requires_grad:
                    continue

                d_p = p.grad / p.grad.norm()
                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf

                p.add_(d_p, alpha=-group['lr'])

        return loss


@Optimizer.register("norm_sgd")
class NormSgdOptimizer(Optimizer, NormSGD):
    """Wrap the boi for the allennlp."""

    def __init__(
        self,
        model_parameters: List[Tuple[str, torch.nn.Parameter]],
        lr: float,
        parameter_groups: List[Tuple[List[str], Dict[str, Any]]] = None,
        momentum: float = 0.0,
        dampening: float = 0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
    ):
        super().__init__(
            params=make_parameter_groups(model_parameters, parameter_groups),
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )