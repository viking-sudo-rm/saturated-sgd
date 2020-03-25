"""Experiment with cosine similarity and pruning.

This enables two different experiments:
1. Prune the least saturated activations in the contextualizer, and measure cosine similarity to normal representation.
2. Saturate the least saturated activations in the contextualizer, and measure cosine similarity to normal representation.

Pruning can be seen as scaling the magnitude to zero. Saturating can be seen as scaling the weight norm to infinity.

Note: run from the base saturated-sgd directory for path to work properly.
"""
from typing import Tuple, Iterable, List
from collections import defaultdict
from transformers import *
import torch
from torch.nn import Module, Parameter
from math import sqrt
import os
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import random

from src.utils.saturate import masked_saturate
from src.utils.percentile import percentile
from src.utils.huggingface import cos, get_tokenizer_and_model, get_prunable_parameters


PATH = "images/"

# Type representing a subnetwork for pruning.
PruneTuple = Tuple[List[Parameter], List[torch.Tensor]]


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="t5-base")
    parser.add_argument("--mode", choices=PRUNE_FNS.keys(), default=PRUNE_FNS.keys(), action="append")
    parser.add_argument("--prune_embed", action="store_true")
    return parser.parse_args()


sentences = [
    "Hello to my little friend.",
    "It's a great day in Seattle, besides the virus.",
    "Working from home is great.",
    "Wow, who needs pre-annotated corpora?",
]


def prune_activations(
    parameters: List[Parameter],
    percent: float,
    least_saturated: bool = True,
) -> PruneTuple:
    """Take as argument percent of parameters to prune."""
    if not least_saturated:
        percent = 1 - percent
    with torch.no_grad():
        activation_norms = [
            param.norm(dim=1) / sqrt(param.size(1)) for param in parameters
        ]
        threshold = percentile(torch.cat(activation_norms), percent)
        subnetwork = []
        masks = []
        for param in parameters:
            assert len(param.size()) == 2
            norm = param.norm(dim=1) / sqrt(param.size(1))
            mask = (norm < threshold) if least_saturated else (norm >= threshold)
            mask = mask.unsqueeze(1)
            subnetwork.append(param)
            masks.append(mask)
        return subnetwork, masks


def prune_weights(
    parameters: List[Parameter],
    percent: float,
    least_saturated: bool = True,  # unused
) -> PruneTuple:
    """Take as argument percent of parameters to prune."""
    if not least_saturated:
        percent = 1 - percent
    with torch.no_grad():
        param_norms = torch.cat([param.flatten().abs() for param in parameters if len(param.size()) == 2])
        threshold = percentile(param_norms, percent)
        subnetwork = []
        masks = []
        for param in parameters:
            assert len(param.size()) == 2
            norm = param.abs()
            mask = (norm < threshold) if least_saturated else (norm >= threshold)
            subnetwork.append(param)
            masks.append(mask)
        return subnetwork, masks


def prune_random_activations(
    parameters: List[Parameter],
    percent: float,
    _: bool = True,
) -> PruneTuple:
    """Take as argument percent of parameters to prune."""
    with torch.no_grad():
        subnetwork = []
        masks = []
        for param in parameters:
            assert len(param.size()) == 2
            if random.random() < percent:
                mask = torch.ones_like(param).bool()
                subnetwork.append(param)
                masks.append(mask)
    return subnetwork, masks


def prune_random_weights(
    parameters: List[Parameter],
    percent: float,
    _: bool = True,
) -> PruneTuple:
    """Take as argument percent of parameters to prune."""
    with torch.no_grad():
        subnetwork = []
        masks = []
        for param in parameters:
            assert len(param.size()) == 2
            probs = percent * torch.ones_like(param)
            mask = probs.bernoulli().bool()
            subnetwork.append(param)
            masks.append(mask)
    return subnetwork, masks


# Register different kinds of pruning here to compare easily.
PRUNE_FNS = {
    "least-activations": prune_activations,
    "least-weights": prune_weights,
    "most-activations": lambda params, p: prune_activations(params, p, least_saturated=False),
    "most-weights": lambda params, p: prune_weights(params, p, least_saturated=False),
    "random-activations": prune_random_activations,
    "random-weights": prune_random_weights,
}


def get_similarity(
    tokenizer,
    model: Module,
    subnetwork: Iterable[Parameter],
    masks: Iterable[torch.Tensor],
    scale: float,
):
    sims = []
    for sentence in sentences:
        input_ids = torch.tensor(tokenizer.encode(sentence)).unsqueeze(0)
        # Index 0 is the pooled value, 1 is the tensor for the full sequence.
        soft = model(input_ids=input_ids)[1]
        with masked_saturate(subnetwork, masks, scale):
            hard = model(input_ids=input_ids)[1]
        sim = cos(soft.flatten(start_dim=1), hard.flatten(start_dim=1))
        sims.append(sim)
    return torch.mean(torch.stack(sims)).item()


def main(args):
    # Options from params.
    model_name = args.model
    inf = 0.0
    print(f"Loading tokenizer and model for {model_name}...")
    tokenizer, model = get_tokenizer_and_model(model_name)

    # Make sure that necessary directories exist.
    model_dir_path = os.path.join(PATH, model_name)
    if not os.path.isdir(model_dir_path):
        os.mkdir(model_dir_path)
    dir_path = os.path.join(PATH, model_name, "prune-cos")
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)

    # Prune at various percents and record data.
    percents = list(x / 100.0 for x in range(0, 110, 10))
    sims = defaultdict(list)
    for mode in args.mode:
        print(f"Starting {mode}...")
        prune_fn = PRUNE_FNS[mode]
        for percent in percents:
            print(f"  > Pruning at rate {percent}...")
            # This should handle both T5 and RoBERTa. For other model types, should check.
            exclude_params = ["embed", "decoder", "pooler"]
            parameters = get_prunable_parameters(model)
            subnetwork, masks = prune_fn(parameters, percent)
            sim = get_similarity(tokenizer, model, subnetwork, masks, inf)
            sims[mode].append(sim)

    # TODO: Might want to put these on same plot.
    for mode, data in sims.items():
        plt.figure()
        plt.plot(percents, data)
        plt.title(f"Pruning curve for {mode} ({model_name})")
        plt.xlabel("Prune Rate")
        plt.ylabel("Cos Sim")
        path = os.path.join(dir_path, f"{mode}.png")
        plt.savefig(path)
        print(f"Saved {path}.")


if __name__ == "__main__":
    args = parse_args()
    main(args)
