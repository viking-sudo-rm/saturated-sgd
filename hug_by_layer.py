"""Experiments comparing saturation by layer."""

import torch
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from collections import defaultdict
import numpy as np
import os

from src.utils.saturate import masked_saturate
from src.utils.percentile import percentile
from src.utils.huggingface import get_tokenizer_and_model, get_activation_norms, get_weight_norms, get_params_by_layer


PATH = "images/"


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="t5-base")
    parser.add_argument("--mode", choices=["activations", "weights"], default="activations")
    return parser.parse_args()


def main(args):
    tokenizer, model = get_tokenizer_and_model(args.model)
    params_by_layer = get_params_by_layer(model)
    act_norms_by_layer = {layer: get_activation_norms(params) for layer, params in params_by_layer.items()}
    weight_norms_by_layer = {layer: get_weight_norms(params) for layer, params in params_by_layer.items()}

    mean_act_norms_by_layer = {layer: np.mean(norms) for layer, norms in act_norms_by_layer.items()}
    mean_weight_norms_by_layer = {layer: np.mean(norms) for layer, norms in weight_norms_by_layer.items()}

    # Make sure that necessary directories exist.
    model_dir_path = os.path.join(PATH, args.model)
    if not os.path.isdir(model_dir_path):
        os.mkdir(model_dir_path)
    dir_path = os.path.join(model_dir_path, "layerwise")
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)

    plt.figure()
    plt.plot(list(mean_act_norms_by_layer.keys()), list(mean_act_norms_by_layer.values()))
    plt.title("Mean activation norm by encoder layer")
    plt.xlabel("Layer")
    plt.ylabel("Mean act norm")
    path = os.path.join(dir_path, "activations.png")
    plt.savefig(path)
    print(f"Saved activations plot to {path}.")

    plt.figure()
    plt.plot(list(mean_weight_norms_by_layer.keys()), list(mean_weight_norms_by_layer.values()))
    plt.title("Mean weight norm by encoder layer")
    plt.xlabel("Layer")
    plt.ylabel("Mean weight norm")
    path = os.path.join(dir_path, "weights.png")
    plt.savefig(path)
    print(f"Saved weights plot to {path}.")


if __name__ == "__main__":
    args = parse_args()
    main(args)
