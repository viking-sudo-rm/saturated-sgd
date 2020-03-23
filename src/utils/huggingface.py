"""Utils for working with scripts working with Huggingface models."""

from typing import Dict, List, Tuple

from collections import defaultdict
import torch
from torch.nn import Module, Parameter
from transformers import *
import numpy as np
from math import sqrt


def cos(vec1, vec2):
    """Return the cosine similarity between two vectors.

    TODO: Handle zero case with an epsilon."""
    return torch.sum(vec1 * vec2, dim=-1) / (vec1.norm(dim=-1) * vec2.norm(dim=-1))


def get_tokenizer_and_model(model_name: str):
    """Get Tokenizer and Model for a model name."""
    if model_name.startswith("t5"):
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        model = T5Model.from_pretrained(model_name)
    else:
        return NotImplemented
    return tokenizer, model


def get_parameters(
    model: Module, exclude_embed: bool = True, exclude_decoder: bool = True
) -> List[Parameter]:
    """Get parameters for a model, potentially excluding the embedding layer."""
    if not exclude_embed:
        return list(model.parameters())
    # Exclude the embedding layer from pruning.
    return [
        param
        for name, param in model.named_parameters()
        if (not exclude_embed or "embed" not in name.lower())
        and (not exclude_decoder or "decoder" not in name.lower())
    ]


def get_paired_mag_and_act_norms(
    parameters: List[Parameter],
) -> Tuple[List[torch.Tensor], List[float]]:
    """Return coindexed lists of (1) tensors of magnitudes for each weight in an activation and (2) the activation norm."""
    # TODO: Normalize by parentheses.
    with torch.no_grad():
        all_mag_norms = []
        all_act_norms = []
        for param in parameters:
            if len(param.size()) != 2:
                continue
            mag_norms = param.abs()
            act_norms = param.norm(dim=1, p=2) / sqrt(param.size(1))
            all_mag_norms.extend(norm for norm in mag_norms)
            all_act_norms.extend(norm.item() for norm in act_norms)
        return all_mag_norms, all_act_norms


def get_activation_norms(parameters: List[Parameter]) -> List[float]:
    """Return a list of all the activation norms."""
    with torch.no_grad():
        all_norms = []
        for param in parameters:
            if len(param.size()) != 2:
                continue
            norms = param.norm(dim=1, p=2) / sqrt(param.size(1))
            all_norms.extend(norm.item() for norm in norms)
        return all_norms


def get_weight_norms(parameters: List[Parameter]) -> np.ndarray:
    """Get an array of all weight magnitudes in the parameter list."""
    with torch.no_grad():
        norms = torch.cat([param.abs().flatten() for param in parameters])
        return norms.numpy()


def get_params_by_layer(model) -> Dict:
    """Get dictionary of parameters partitioned by layer number."""
    layers = defaultdict(list)
    for name, param in model.named_parameters():
        pieces = name.split(".")
        if pieces[0] == "encoder" and pieces[1] == "block":
            layer = int(pieces[2])
            layers[layer].append(param)
    return layers
