"""Experiments comparing saturation by layer."""

import torch
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from collections import defaultdict
import numpy as np
import os
from transformers import *

from src.utils.saturate import masked_saturate
from src.utils.percentile import percentile
from src.utils.huggingface import (
    get_tokenizer_and_model,
    get_activation_norms,
    get_weight_norms,
    get_params_by_layer,
    get_embed_params,
)


PATH = "images/"


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--embed", action="store_true")
    return parser.parse_args()


def cat_and_norm(params):
    params = torch.cat([param.flatten() for param in params])
    return params.norm(p=2)


def main(args):
    random_model = T5Model(T5Config.from_pretrained("t5-base"))
    model = T5Model.from_pretrained("t5-base")

    random_params_by_layer = get_params_by_layer(random_model)
    random_act_norms_by_layer = {
        layer: cat_and_norm(params).item()
        for layer, params in random_params_by_layer.items()
    }

    params_by_layer = get_params_by_layer(model)
    act_norms_by_layer = {
        layer: cat_and_norm(params).item()
        for layer, params in params_by_layer.items()
    }

    # Make sure that necessary directories exist.
    model_dir_path = os.path.join(PATH, "t5-base")
    if not os.path.isdir(model_dir_path):
        os.mkdir(model_dir_path)
    dir_path = os.path.join(model_dir_path, "layerwise")
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)

    # Use the layer indexing from the rest of the paper.
    layers = list(range(1, 13))

    plt.figure()
    plt.plot(layers, list(random_act_norms_by_layer.values()), label="Random init")
    plt.plot(layers, list(act_norms_by_layer.values()), label="Pretrained")
    plt.title("Parameter norm by encoder layer")
    plt.xlabel("Layer")
    plt.ylabel("Parameter norm")
    plt.legend()
    path = os.path.join(dir_path, "parameter.png")
    plt.savefig(path)
    print(f"Saved norm-by-layer plot to {path}.")


if __name__ == "__main__":
    args = parse_args()
    main(args)
