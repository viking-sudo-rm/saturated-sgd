"""Script to analyze weight trajectory during training of T5.

To download the model checkpoints from GCP, you can do:
gsutil -m cp -r gs://t5-data/pretrained_models/ data/t5-models

But also, this should work if you set INIT_CHECKPONT to a GCP path.
"""

import os
import numpy as np
from math import sqrt
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt

from t5.models.mtf_model import MtfModel
import t5.data
import gin

# MIXTURE_NAME = 'all_mix'
MIXTURE_NAME = "c4_v020_unsupervised"

CKPT_PATH = "/home/willm/data/bsl/bsl-0/checkpoint"
FILE_FORMAT = """model_checkpoint_path: "{ckpt}"
all_model_checkpoint_paths: "{ckpt}"
"""

FILTER_PRED = lambda name: name.startswith("shared/")


def get_checkpoints(model_dir):
    # return tf.train.get_checkpoint_state(model_dir).all_model_checkpoint_paths
    ckpts = []
    for file_name in os.listdir(model_dir):
        if file_name.endswith(".index"):
            ckpts.append(file_name.replace(".index", ""))
    ckpts = sorted(ckpts, key=lambda ckpt: int(ckpt.split("-")[1]))
    return list(ckpts)


def downsample(li: list, samples: int = 5):
    step = len(li) // samples
    for idx, item in enumerate(li):
        if idx % step == 0:
            yield item


def write_checkpoint_file(ckpt):
    with open(CKPT_PATH, "w") as fh:
        contents = FILE_FORMAT.format(ckpt=ckpt)
        fh.write(contents)


def _operative_config_path(model_dir):
    return os.path.join(model_dir, "operative_config.gin")


def get_param_norm(estimator, normalize: bool = False):
    # In the PyTorch code, we have many different options for norms, but they don't seem to differ much. Would it make sense to swap in one of these quantities here?
    params = estimator.get_variable_names()
    values = [estimator.get_variable_value(p) for p in params]
    flat = np.concatenate([value.flatten() for value in values])
    norm = np.linalg.norm(flat)
    if normalize:
        norm /= sqrt(len(flat))
    return norm


def get_all_norms(estimator):
    params = [
        estimator.get_variable_value(p)
        for p in estimator.get_variable_names()
        if not FILTER_PRED(p)
    ]
    batched_norms = [
        np.linalg.norm(param, axis=1) / param.shape[1]
        for param in params
        if hasattr(param, "shape") and len(param.shape) == 2
    ]
    return [item for sublist in batched_norms for item in sublist]


def main():
    # Can look at both the histogram and the norm of the weights.
    ckpt_ids = []
    norms = []
    all_all_norms = {}
    for trial in range(1):
        model = MtfModel(f"/home/willm/data/bsl/bsl-{trial}/", tpu=None)
        gin.parse_config_file(_operative_config_path(model._model_dir))
        vocabulary = t5.data.get_mixture_or_task(MIXTURE_NAME).get_vocabulary()
        ckpts = get_checkpoints(model._model_dir)
        print(f"All {len(ckpts)} ckpts:", ckpts)
        ckpts = downsample(ckpts, samples=5)
        for n, ckpt in enumerate(ckpts):
            print(f"Starting ckpt {ckpt}...")
            ckpt_id = int(ckpt.split("-")[1])
            ckpt_ids.append(ckpt_id)
            write_checkpoint_file(ckpt)

            estimator = model.estimator(vocabulary, init_checkpoint=ckpt)
            all_all_norms[ckpt] = get_all_norms(estimator)
            continue

            # get_sat_cos_sim(estimator, os.path.join(model._model_dir, ckpt))
            norm = get_param_norm(estimator, normalize=False)
            norms.append(norm)
            print(f"({n}/{len(ckpts)}) norm({ckpt_id}) = {norm:.0f}")

    # Save distributions for each checkpoint.
    plt.figure()
    for ckpt, data in all_all_norms.items():
        plt.hist(data, label=ckpt, bins=500)
    plt.legend()
    plt.xlim(0.0, 1.0)
    limits = plt.xlim()
    plt.xlabel("Normalized Activation Norm")
    plt.ylabel("Density")
    plt.title("Distribution of activation norms over time")
    plt.savefig("images/t5/dists/summary.png")
    print("Saved summary distribution plot.")

    # Save a distribution plot for each checkpoint, all on the same axis.
    for ckpt, data in all_all_norms.items():
        plt.figure()
        plt.hist(data, bins=500)
        plt.xlim(limits)
        plt.xlabel("Normalized Activation Norm")
        plt.ylabel("Density")
        plt.title("Distribution of activation norms over time")
        path = f"images/t5/dists/{ckpt}.png"
        plt.savefig(path)
        print(f"Saved {path}.")

    quit()

    # Save a plot of the norm data.
    plt.figure()
    plt.plot(ckpt_ids, norms, marker="o")
    plt.savefig("images/t5/norm.png")
    print("Saved figure.")

    # Save the norm data, which is expensive to compute.
    import pickle

    with open("/home/willm/data/bsl/t5-deriv/norms.dat", "wb") as fh:
        pickle.dump(norms, fh)


if __name__ == "__main__":
    main()
