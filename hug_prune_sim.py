"""Experiment with cosine similarity and pruning.

This enables two different experiments:
1. Prune the least saturated activations in the contextualizer, and measure cosine similarity to normal representation.
2. Saturate the least saturated activations in the contextualizer, and measure cosine similarity to normal representation.

Pruning can be seen as scaling the magnitude to zero. Saturating can be seen as scaling the weight norm to infinity.

Note: run from the base saturated-sgd directory for path to work properly.
"""
from typing import Tuple, Iterable, List
from transformers import *
import torch
from torch.nn import Module, Parameter
from math import sqrt
import os
import matplotlib.pyplot as plt
from argparse import ArgumentParser

from src.utils.saturate import masked_saturate
from src.utils.percentile import percentile
from src.utils.huggingface import cos, get_tokenizer_and_model, get_parameters


PATH = "images/"

# Type representing a subnetwork for pruning.
PruneTuple = Tuple[List[Parameter], List[torch.Tensor]]


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="t5-base")
    parser.add_argument("--mode", choices=["activations", "weights"], default="activations")
    parser.add_argument("--saturate", action="store_true")
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
    with torch.no_grad():
        activation_norms = [
            param.norm(dim=1) / sqrt(param.size(1))
            for param in parameters
            if len(param.size()) == 2
        ]
        threshold = percentile(torch.cat(activation_norms), percent)

        subnetwork = []
        masks = []
        for param in parameters:
            if len(param.size()) != 2:
                continue
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
    with torch.no_grad():
        param_norms = torch.cat([param.flatten().abs() for param in parameters if len(param.size()) == 2])
        threshold = percentile(param_norms, percent)

        subnetwork = []
        masks = []
        for param in parameters:
            if len(param.size()) != 2:
                continue
            norm = param.abs()
            mask = (norm < threshold) if least_saturated else (norm >= threshold)
            subnetwork.append(param)
            masks.append(mask)

        return subnetwork, masks


# Different types of pruning. Currently support pruning by weight magnitude or by activation norm.
PRUNE_FNS = {
    "activations": prune_activations,
    "weights": prune_weights,
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
        soft = model(input_ids=input_ids)[0]
        with masked_saturate(subnetwork, masks, scale):
            hard = model(input_ids=input_ids)[0]
        sim = cos(soft.flatten(start_dim=1), hard.flatten(start_dim=1))
        sims.append(sim)
    return torch.mean(torch.stack(sims))


def main(args):
    # Options from params.
    model_name = args.model
    least_saturated = not args.saturate
    prune_fn = PRUNE_FNS[args.mode]
    inf = 0.0 if least_saturated else 1000.0
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
    percents = list(x / 100.0 for x in range(0, 100, 10))
    sims = []
    for percent in percents:
        print(f"Pruning at rate {percent}...")
        parameters = get_parameters(model, exclude_embed=not args.prune_embed)
        subnetwork, masks = prune_fn(parameters, percent, least_saturated)
        sim = get_similarity(tokenizer, model, subnetwork, masks, inf)
        sims.append(sim)

    if least_saturated:
        plt.figure()
        plt.plot(percents, sims)
        plt.title(f"Cosine similarity curve for pruning ({args.mode})")
        plt.xlabel("Activation Prune Rate")
        plt.ylabel("Cos Sim")
        path = os.path.join(dir_path, f"least-{args.mode}.png")
        plt.savefig(path)
        print(f"Saved {path}.")

    else:
        plt.figure()
        plt.plot(percents, sims)
        plt.title(f"Cosine similarity curve for saturating ({args.mode})")
        plt.xlabel("1 - Activation Saturation Rate")
        plt.ylabel("Cos Sim")
        path = os.path.join(dir_path, f"most-{args.mode}.png")
        plt.savefig(path)
        print(f"Saved {path}.")


if __name__ == "__main__":
    args = parse_args()
    main(args)
