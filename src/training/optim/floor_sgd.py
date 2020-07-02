from typing import List, Any, Dict, Tuple
import torch.optim as optim
import torch

from allennlp.training.optimizers import Optimizer, make_parameter_groups

EPS = torch.finfo(torch.float64).eps

class FloorSGD(optim.SGD):
    def __init__(self, *args, min_step: float = 1e-1, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_step = min_step
    
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

                # The conditional FloorSGD computation.
                # TODO: Apply across each row in each matrix??? Don't treat all matrices the same.
                d_p = p.grad
                norm = p.grad.norm(p=2)
                if norm < self.min_step:
                    d_p = d_p * self.min_step / (norm + EPS)

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


@Optimizer.register("floor_sgd")
class FloorSgdOptimizer(Optimizer, FloorSGD):
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
        min_step: float = 1e-1,
    ):
        super().__init__(
            params=make_parameter_groups(model_parameters, parameter_groups),
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
            min_step = min_step,
        )