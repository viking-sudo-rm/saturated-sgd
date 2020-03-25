"""Utils for working with scripts working with Huggingface models."""

from typing import Callable, Dict, List, Tuple

from collections import defaultdict
import torch
from torch.nn import Module, Parameter
from transformers import *
import numpy as np
from math import sqrt

EPS = np.finfo(np.float).eps


def cos(vec1: torch.FloatTensor, vec2: torch.FloatTensor) -> torch.FloatTensor:
    """Return the cosine similarity between two vectors."""
    norm1 = torch.clamp(vec1.norm(dim=-1), min=EPS)
    norm2 = torch.clamp(vec2.norm(dim=-1), min=EPS)
    return torch.sum(vec1 * vec2, dim=-1) / (norm1 * norm2)


def get_tokenizer_and_model(model_name: str):
    """Get `Tokenizer` and `Model` for a model name."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.output_hidden_states = True
    return tokenizer, model


def wrap_contextualize(model, input_ids) -> torch.FloatTensor:
    """Wrap a forward pass to the model and pick out the hidden states of the encoder."""
    results = model(input_ids=input_ids)
    assert model.output_hidden_states
    assert len(results[0].shape) == 3
    return results[0]


def get_prunable_parameters(
    model: Module,
    only_matrices: bool = True,
    # TODO: Replace exclude with a filter function per model.
    exclude: List[str] = ["embed", "decoder.", "pooler.", "shared."],
) -> List[Parameter]:
    """Get parameters for a model, potentially excluding the embedding layer."""
    # names = [name for name, _ in model.named_parameters()]
    # import pdb; pdb.set_trace()
    return [
        param
        for name, param in model.named_parameters()
        if (
            (not only_matrices or len(param.size()) == 2)
            and not any(substr in name for substr in exclude)
        )
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
