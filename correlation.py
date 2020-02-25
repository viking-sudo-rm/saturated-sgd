"""Experiment to check the correlation between norm of an activation's parameters and its
saturated distance.

Currently just doing this with random projections. Probably want to redo this with a trained model.
"""

from scipy.stats import pearsonr
import torch
from torchvision.transforms import Normalize

BATCH_SIZE = 1000
INPUT_DIM = 64
HIDDEN_DIM = 100


inputs = torch.randn(BATCH_SIZE, INPUT_DIM)

layer = torch.nn.Sequential(
    torch.nn.Linear(INPUT_DIM, HIDDEN_DIM),
    torch.nn.Tanh(),
)

activations = layer(inputs)
sat_dists = torch.mean(1 - torch.abs(activations), dim=0)

weight = layer[0].weight
norms = torch.sqrt(torch.sum(weight * weight, dim=1))

sat_dists = sat_dists.detach().numpy()
norms = norms.detach().numpy()

import matplotlib.pyplot as plt
plt.scatter(norms, sat_dists)
plt.savefig("images/correlation.png")

corr, _ = pearsonr(norms, sat_dists)
print(f"R^2: {corr * corr:.5f}")
