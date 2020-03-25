# import seaborn as sns
from typing import Tuple, List
from argparse import ArgumentParser
import torch
from torch.nn import Parameter
from transformers import *
from tqdm import tqdm
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import os
import random

from src.utils.saturate import saturate
from src.metrics.param_norm import ParamNorm
from src.utils.huggingface import (
    cos,
    get_activation_norms,
    get_weight_norms,
    get_paired_mag_and_act_norms,
    get_params_by_layer,
    get_prunable_parameters,
    get_tokenizer_and_model,
    wrap_contextualize,
)


PATH = "images/"
NORM = ParamNorm()


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="t5-base")
    return parser.parse_args()


def get_norm_metric(parameters: List[Parameter]):
    NORM(parameters)
    return NORM.get_metric(reset=True)


def get_similarity(sentences, tokenizer, model, infinity: float = 10) -> float:
    sims = []
    for sentence in sentences:
        input_ids = torch.tensor(tokenizer.encode(sentence)).unsqueeze(0)
        outputs = wrap_contextualize(model, input_ids)
        with saturate(model, infinity):
            sat_outputs = wrap_contextualize(model, input_ids)

        sim = cos(outputs.flatten(start_dim=1), sat_outputs.flatten(start_dim=1))
        sims.append(sim)

    return torch.mean(torch.stack(sims, dim=0)).item()


sentences = [
    "Hello to my little friend.",
    "It's a great day in Seattle, besides the virus.",
    "Working from home is great.",
    "Wow, who needs pre-annotated corpora?",
]
infs = [1, 1.1, 1.2, 1.3, 1.4, 1.5, 2.0, 10.0, 100.0]


def main(args):
    model_name = args.model
    print(f"Loading {model_name} tokenizer and model...")
    tokenizer, model = get_tokenizer_and_model(model_name)
    model_dir = os.path.join(PATH, model_name)
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    sims = defaultdict(list)
    norms = []

    act_norm_dists = {}
    weight_norm_dists = {}
    paired_norm_dists = {}

    for inf in infs:
        sim = get_similarity(sentences, tokenizer, model, infinity=inf)
        sims[model_name].append(sim)
        print(f"({model_name} * {inf:.2f}) sim={sim:.2f}")

    # This part excludes everything besides the encoder.
    # TODO: Should probably redo old plots only with the encoder weights.

    parameters = list(model.parameters())  # Can also use `get_prunable_parameters` here.
    act_norm_dists[model_name] = get_activation_norms(parameters)
    weight_norm_dists[model_name] = get_weight_norms(parameters)

    params_by_layer = get_params_by_layer(model)
    paired_norm_dists[model_name] = {
        layer: get_paired_mag_and_act_norms(params)
        for layer, params in params_by_layer.items()
    }

    norm_metric = get_norm_metric(parameters)
    norm = norm_metric["mean_norm"]["l2"]
    norms.append(norm)
    print(f"({model_name}) mean_norm/l2={norm:.2f}")


    # print("Drawing norm vs. norm plot...")
    # for model_name, scatter_by_layer in paired_norm_dists.items():
    #     for layer, (all_mags, acts) in scatter_by_layer.items():
    #         pairs = []
    #         for mags, act in tqdm(zip(all_mags, acts), total=len(acts)):
    #             # This is pretty slow; have to copy activation norm for each weight in the activation.
    #             pairs.extend((mag.item(), act) for mag in mags)
    #         # print("Downsampling list...")
    #         # pairs = random.sample(pairs, 5000)
    #         plt.scatter(*zip(*pairs), label=f"Layer {layer}")
    # plt.title(f"Activation norm vs. weight magnitude by layer for {model_name}")
    # plt.xlabel("Weight magnitude")
    # plt.ylabel("Activation norm")
    # plt.legend()
    # path = os.path.join(PATH, f"norm-vs-norm/{prefix}.png")
    # plt.savefig(path)
    # print(f"Saved norm versus norm by layer plot to {path}.")
    # quit()

    # TODO: Restructure image saving by model type.

    print("Computing all the data...")
    name_to_data = {}
    for model_name, data in act_norm_dists.items():
        a = plt.hist(data, label=model_name, bins=200)
        bin_counts, bins = a[:2]
        cum_dist = np.cumsum(bin_counts) / np.sum(bin_counts)
        name_to_data[model_name] = (cum_dist, bins)
    plt.title("Activation norm distribution by model type")
    plt.xlabel("Activation norm")
    plt.legend()
    path = os.path.join(model_dir, f"act-norm.png")
    plt.savefig(path)
    print(f"Saved {path}.")

    plt.figure()
    for model_name, (bins, cum_dist) in name_to_data.items():
        plt.plot(np.array([0] + list(bins)), cum_dist, label=model_name)

    plt.title("Cumulative distribution of activation norm")
    plt.legend()
    plt.xlabel("Norm")
    plt.xlabel("Percentile")
    path = os.path.join(model_dir, f"act-norm-cdf.png")
    plt.savefig(path)
    print(f"Saved {path}.")

    # Plot the distribution of weight magnitudes.
    plt.figure()
    for model_name, data in weight_norm_dists.items():
        plt.hist(data, label=model_name, bins=200)
    plt.title("Weight magnitude distribution by model type")
    plt.xlabel("Weight magnitude")
    plt.legend()
    path = os.path.join(model_dir, f"weight-mag.png")
    plt.savefig(path)
    print(f"Saved {path}.")

    # Plot norm curve.
    # plt.figure()
    # plt.scatter(norms, sims, s=['a', 'b', 'c'])
    # plt.title("BERT-like saturation vs. norm")
    # plt.xlabel("Activation Norm")
    # plt.ylabel("Saturation Cos Sim")
    # # plt.legend()
    # plt.savefig(f"images/bertlike-sat-vs-norm.png")

    # Plot the cosine similarity curve.
    plt.figure()
    plt.title(f"Cosine similarity with increasing weight multiplier")
    plt.ylabel("Cos Similarity")
    plt.xlabel("Weight Multiplier")
    plt.xscale("log")
    for model_name, data in sims.items():
        plt.plot(infs, data, label=model_name)
    plt.legend()
    path = os.path.join(model_dir, f"cos-sim.png")
    plt.savefig(path)
    print(f"Saved {path}.")


if __name__ == "__main__":
    main(parse_args())
