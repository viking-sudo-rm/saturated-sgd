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
from rich import print
from torch.nn.utils.rnn import pad_sequence
import warnings

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

# See https://github.com/huggingface/transformers/issues/37.
PATH = "images/sim-by-layer"


class Avg(object):
    """ Computes and stores the average and current value """

    def __init__(self, name="generic", fmt=":f", write_val=True, write_avg=True):
        self.name = name
        self.fmt = fmt
        self.reset()

        self.write_val = write_val
        self.write_avg = write_avg

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = (self.sum / self.count)

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class SumAvg:
    """Confusingly, this is a different kind of average meter
    
    Takes sequences of different lengths, returning the average where each scalar element is equally weighted."""

    def __init__(self):
        self.sum = 0
        self.num = 0

    def update(self, tensor: torch.Tensor):
        self.sum += tensor.sum().item()
        self.num += tensor.numel()

    def get(self):
        return self.sum / self.num


class saturate_directionally:
    """Saturate by moving in the gradient direction."""

    def __init__(self, model, grad_dict, time: int):
        self.named_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad and p.grad is not None]
        self.grad_dict = grad_dict
        self.time = time
        self.old_param_data = []
        self.no_grad = torch.no_grad()

    def __enter__(self):
        self.no_grad.__enter__()
        for name, param in self.named_params:
            self.old_param_data.append(param.data)
            if param.data.shape != self.grad_dict[name].shape:
                import pdb; pdb.set_trace()
            param.data = param.data - self.grad_dict[name] * self.time

    def __exit__(self, type, value, traceback):
        for (_, param), data in zip(self.named_params, self.old_param_data):
            param.data = data
        self.no_grad.__exit__(type, value, traceback)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--model", action="append")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--mock_sents", action="store_true")
    parser.add_argument("--num_sents", type=int, default=100)
    parser.add_argument("--random_init", action="store_true")
    return parser.parse_args()


def get_sentences(args):
    if args.mock_sents:
        return [
            "Hello to my little friend.",
            "It's a great day in Seattle, besides the virus.",
            "Working from home is great.",
            "Wow, who needs pre-annotated corpora?",
        ]

    with open("/home/willm/data/brown.txt") as fh:
        text = fh.read()
        sentences = [line for line in text.split("\n\n") if not line.startswith("#")]
        return sentences[: args.num_sents]


def main(args):
    sentences = get_sentences(args)

    tokenizers = [
        AutoTokenizer.from_pretrained("bert-base-cased"),
        AutoTokenizer.from_pretrained("roberta-base"),
    ]

    model_names = ["bert-base-cased", "roberta-base"]

    if not args.random_init:
        models = [
            BertForMaskedLM.from_pretrained(
                "bert-base-cased", output_hidden_states=True
            ),
            # RobertaForMaskedLM.from_pretrained(
            #     "roberta-base", output_hidden_states=True
            # ),
        ]

    else:
        models = [
            BertForMaskedLM(
                BertConfig.from_pretrained("bert-base-cased", output_hidden_states=True)
            ),
            # RobertaForMaskedLM(
            #     RobertaConfig.from_pretrained("roberta-base", output_hidden_states=True)
            # ),
        ]
    
    if args.cuda:
        models = [model.cuda(0) for model in models]

    sims_by_model = {}

    for name, tokenizer, model in zip(model_names, tokenizers, models):
        print(f"[green]=>[/green] {type(model).__name__}...")

        sim_avgs = [Avg() for _ in range(13)]
        grad_dict = defaultdict(lambda: Avg("grads"))
        all_preds = []
        all_states = []
        all_input_ids = []

        print("[green]=>[/green] Computing average gradients")
        for sentence in tqdm(sentences):
            input_ids = torch.tensor(
                tokenizer.encode(sentence, max_length=512)
            ).unsqueeze(dim=0)
            if args.cuda:
                input_ids = input_ids.cuda(0)

            loss, scores, states = model(input_ids, masked_lm_labels=input_ids)
            preds = scores.argmax(dim=-1)

            all_input_ids.append(input_ids)
            all_states.append(states)
            all_preds.append(preds)

            model.zero_grad()
            loss.backward()

            for n, p in model.named_parameters():
                if p.grad is not None:
                    grad_dict[n].update(p.grad)

        with torch.no_grad():
            params = torch.cat(
                [p.flatten() for p in model.parameters() if p.grad is not None]
            )
            grads = torch.cat(
                [
                    grad_dict[n].avg.flatten()
                    for n, p in model.named_parameters()
                    if p.grad is not None
                ]
            )

            print("=> Computing saturation agreement and similarity along direction")
            print(f"\t (similarity to grads: {((params @ grads) / (params.norm() * grads.norm())).item():.2f})")

        norm = grads.norm(p=2)
        norm_grad_dict = {n: meter.avg / norm for n, meter in grad_dict.items()}

        print("=> Starting to saturate networks.")

        # times = [.001, .01, .1]
        times = np.linspace(0., 10., 10)
        agree_avgs = {t: SumAvg() for t in times}
        sim_avgs = {t: SumAvg() for t in times}
        for t in tqdm(times):
            for input_ids, preds, states in zip(all_input_ids, all_preds, all_states):
                with saturate_directionally(model, norm_grad_dict, t):
                    # _, _, hard_states = model(input_ids)
                    hard_loss, hard_scores, hard_states = model(input_ids, masked_lm_labels=input_ids)
                    hard_preds = hard_scores.argmax(dim=-1)

                assert isinstance(states, tuple)
                assert isinstance(hard_states, tuple)
                sims = [
                    cos(state, hard_state) for state, hard_state in zip(states, hard_states)
                ]
                sim_avgs[t].update(sims[-1])
                # for sim, avg in zip(sims, sim_avgs):
                #     avg.update(sim)

                agree = (preds == hard_preds).float()
                agree_avgs[t].update(agree)

        # for layer, avg in enumerate(sim_avgs):
        #     print(f"[red]Layer #{layer} Sim[/red]: {avg.get():.2f}")

        # sims_by_model[name] = [avg.get() for avg in sim_avgs]

        for t in times:
            print(f"[red]Agree@{t}[/red]: {agree_avgs[t].get():.2f}")
            print(f"[red]Sim@{t}[/red]: {sim_avgs[t].get():.2f}")

    import matplotlib.pyplot as plt

    for model, data in sims_by_model.items():
        plt.plot(data, label=model)
    plt.xlabel("Layer #")
    plt.ylabel("Sat Sim")
    plt.title(
        "Randomly initialized saturation similarity"
        if args.random_init
        else "Pretrained saturation similarity"
    )
    plt.legend()
    if args.random_init:
        path = os.path.join(PATH, "random-init.png")
    else:
        path = os.path.join(PATH, "pretrained.png")
    plt.savefig(path)
    print(f"[green]=>[/green] Saved fig to {path}.")


if __name__ == "__main__":
    main(parse_args())
