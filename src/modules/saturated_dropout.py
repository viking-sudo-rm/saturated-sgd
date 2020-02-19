import torch


def saturated_ste_dropout(
    activations: torch.FloatTensor,
    mean_norms: torch.FloatTensor,
    threshold: float,
    prune_saturated: bool = False,
) -> torch.FloatTensor:
    """Compute saturated dropout at a certain threshold. Preserve gradients on the backward pass.
    
    `activations` is a [batch_size, seq_len, hid_dim] tensor giving the raw activations.
    `mean_norms` is a [hid_dim] tensor giving the degree of saturation for each hidden unit.
    `threshold` is the the norm value below which activations are dropped.
    """
    conditions = (
        (mean_norms <= threshold) if not prune_saturated else (mean_norms >= threshold)
    )
    conditions = conditions.unsqueeze(0).unsqueeze(0)
    zeros = (-activations).detach() + activations
    return torch.where(conditions, activations, zeros)


def saturated_dropout(
    activations: torch.FloatTensor,
    mean_norms: torch.FloatTensor,
    threshold: float,
    prune_saturated: bool = False,
) -> torch.FloatTensor:
    """Compute saturated dropout at a certain threshold. Preserve gradients on the backward pass.
    
    `activations` is a [batch_size, seq_len, hid_dim] tensor giving the raw activations.
    `mean_norms` is a [hid_dim] tensor giving the degree of saturation for each hidden unit.
    `threshold` is the the norm value below which activations are dropped.
    """
    conditions = (
        (mean_norms <= threshold) if not prune_saturated else (mean_norms >= threshold)
    )
    conditions = conditions.unsqueeze(0).unsqueeze(0)
    zeros = torch.zeros_like(activations)
    return torch.where(conditions, activations, zeros)


def random_dropout(activations: torch.FloatTensor, percent: float) -> torch.FloatTensor:
    """A random dropout baseline.
    
    Note that percent is different than threshold.
    """
    probs = percent * torch.ones(activations.size(-1), dtype=activations.dtype)
    conditions = torch.bernoulli(1 - probs).unsqueeze(0).unsqueeze(0).byte()
    zeros = torch.zeros_like(activations)
    return torch.where(conditions, activations, zeros)
