import torch
from transformers import *
from math import sqrt
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

from src.utils.saturate import saturate


def get_norm(parameters, normalize: bool = False):
    params = torch.cat([param.flatten() for param in parameters])
    norm = torch.norm(params)
    if normalize:
        norm /= sqrt(len(params))
    return norm


def cos(vec1, vec2):
    return torch.sum(vec1 * vec2, dim=-1) / (vec1.norm(dim=-1) * vec2.norm(dim=-1))


def get_similarity(sentences, model, infinity=10):
    sims = []
    for sentence in sentences:
        input_ids = torch.tensor(tokenizer.encode(sentence)).unsqueeze(0)

        outputs = model(input_ids=input_ids)[0]
        with saturate(model, infinity):
            sat_outputs = model(input_ids=input_ids)[0]

        sim = cos(outputs.flatten(start_dim=1), sat_outputs.flatten(start_dim=1))
        sims.append(sim)

    norm = get_norm(model.parameters())
    norm_norm = get_norm(model.parameters(), normalize=True)
    sim = torch.mean(torch.stack(sims, dim=0))
    return norm, norm_norm, sim


size = "base"
models = [
    (f"t5-{size}", T5Tokenizer.from_pretrained(f"t5-{size}"), T5Model.from_pretrained(f"t5-{size}")),
    # (f"roberta-{size}", RobertaTokenizer.from_pretrained(f"roberta-{size}"), RobertaModel.from_pretrained(f"roberta-{size}")),
    (f'xlnet-{size}-cased', XLNetTokenizer.from_pretrained(f"xlnet-{size}-cased"), XLNetModel.from_pretrained(f"xlnet-{size}-cased")),
]

sentences = [
    "Hello to my little friend.",
    "It's a great day in Seattle, besides the virus.",
    "Working from home is great.",
    "Wow, who needs pre-annotated corpora?",
]
infs = [1, 10, 100, 1000]  # np.linspace(1, 2, 10)

for model_name, tokenizer, model in models:
    sims = []
    for inf in infs:
        norm, norm_norm, sim = get_similarity(sentences, model, infinity=inf)
        sims.append(sim)
        print(f"({model_name}) norm={norm:.0f}, norm_norm={norm_norm:.2f}, sim={sim:.2f}")
    plt.plot(infs, sims, label=model_name)

plt.title(f"Cosine similarity with increasing weight multiplier ({size})")
plt.ylabel("Cos Similarity")
plt.xlabel("Weight Multiplier")
plt.legend()
plt.savefig(f"images/infs/all-{size}.png")