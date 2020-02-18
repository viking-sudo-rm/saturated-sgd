import torch


def saturated_ste_dropout(
    activations: torch.FloatTensor, mean_norms: torch.FloatTensor, threshold: float
) -> torch.FloatTensor:
    """Compute saturated dropout at a certain threshold. Preserve gradients on the backward pass.
    
    `activations` is a [batch_size, seq_len, hid_dim] tensor giving the raw activations.
    `mean_norms` is a [hid_dim] tensor giving the degree of saturation for each hidden unit.
    `threshold` is the the norm value below which activations are dropped.
    """
    conditions = (mean_norms >= threshold).unsqueeze(0).unsqueeze(0)
    zeros = (torch.zeros_like(activations) - activations).detach() + activations
    return torch.where(conditions, activations, zeros)


def saturated_dropout(
    activations: torch.FloatTensor, mean_norms: torch.FloatTensor, threshold: float
) -> torch.FloatTensor:
    """Compute saturated dropout at a certain threshold. Preserve gradients on the backward pass.
    
    `activations` is a [batch_size, seq_len, hid_dim] tensor giving the raw activations.
    `mean_norms` is a [hid_dim] tensor giving the degree of saturation for each hidden unit.
    `threshold` is the the norm value below which activations are dropped.
    """
    conditions = (mean_norms >= threshold).unsqueeze(0).unsqueeze(0)
    zeros = torch.zeros_like(activations)
    return torch.where(conditions, activations, zeros)
