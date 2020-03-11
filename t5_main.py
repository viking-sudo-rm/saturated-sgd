"""Script to analyze weight trajectory during training of T5.

To download the model checkpoints from GCP, you can do:
gsutil -m cp -r gs://t5-data/pretrained_models/ data/t5-models

But also, this should work if you set INIT_CHECKPONT to a GCP path.
"""

import os
import numpy as np
from math import sqrt

from t5.models.mtf_model import MtfModel
import t5.data
import gin

MIXTURE_NAME = 'all_mix'
DIR_NAME = "/home/willm/data/t5-models/pretrained_models/small"
BASE_CKPT = "/home/willm/data/t5-models/pretrained_models/small/model.ckpt-1000000"
# CKPTS = [BASE_CKPT + ".data-%05d-of-%05d" % (idx, 16) for idx in range(16)]
# CKPTS = "gargabe"  # This shows that weights are not restored properly.
# CKPT = "gs://t5-data/pretrained_models/small/model.ckpt-1000000"


def _operative_config_path(model_dir):
  return os.path.join(model_dir, "operative_config.gin")


def get_param_norm(estimator, normalize: bool = False):
    params = estimator.get_variable_names()
    values = [estimator.get_variable_value(p) for p in params]
    flat = np.concatenate([value.flatten() for value in values])
    norm = np.linalg.norm(flat)
    if normalize:
        norm /= sqrt(len(flat))
    return norm


# model = MtfModel(DIR_NAME, tpu=None)
# gin.parse_config_file(_operative_config_path(model._model_dir))
# vocabulary = t5.data.get_mixture_or_task(MIXTURE_NAME).get_vocabulary()

# norms = []
# for ckpt in CKPTS:
#     print("Computing norm for " + ckpt + "..")
#     estimator = model.estimator(vocabulary, ckpt)
#     import pdb; pdb.set_trace()
#     norm = get_param_norm(estimator)
#     norms.append(norm)
#     print(f"Norm = {norm}")


# Can look at both the histogram and the norm of the weights.
norms = []
for model_name in ["small", "base", "large", "3B", "11B"]:
    model = MtfModel("/home/willm/data/t5-models/pretrained_models/" + model_name, None)
    gin.parse_config_file(_operative_config_path(model._model_dir))
    vocabulary = t5.data.get_mixture_or_task(MIXTURE_NAME).get_vocabulary()
    norm = get_param_norm(model.estimator(vocabulary), normalize=False)
    norms.append(norm)
    print(model_name, "=", norm)

import matplotlib.pyplot as plt
plt.plot(norms)
plt.savefig("images/t5-small-norm.png")
print("Saved figure.")