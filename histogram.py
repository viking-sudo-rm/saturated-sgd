"""Script to extract parses from trained models and evaluate F1 against gold-standard parses.

Before running this evaluation, you will need to build EVALB and set the path appropriately. See
https://github.com/allenai/allennlp/blob/master/allennlp/training/metrics/evalb_bracketing_scorer.py
for how to do this.
"""

import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np

from allennlp.common.util import import_module_and_submodules
from allennlp.common.params import Params
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model


def _load_model(vocab, model_params, args):
    """Load a model, either on the GPU or CPU."""
    model = Model.from_params(params=model_params, vocab=vocab)

    if args.nocuda:
        model = model.cpu()
        if args.random_init:
            return model
        with open("%s/best.th" % args.model_path, "rb") as fh:
            cpu = torch.device("cpu")
            model.load_state_dict(torch.load(fh, map_location=cpu))
        return model
    
    with open("%s/best.th" % args.model_path, "rb") as fh:
        model.load_state_dict(torch.load(fh))
    return model.cuda(0)


def main(args):
    """Driver function whose behavior is configured by a rich set of flags."""
    for package_name in args.include_package:
        import_module_and_submodules(package_name)

    vocab = Vocabulary.from_files("%s/vocabulary" % args.model_path)
    params = Params.from_file("%s/config.json" % args.model_path)
    model = _load_model(vocab, params.pop("model"), args)

    if not args.activations:
        x_name = "weight magnitude"
        parameters = [torch.flatten(param) for param in model.parameters()]
        parameters = torch.cat(parameters).abs()
        data = parameters.detach().numpy()
    
    else:
        x_name = "activation magnitude"
        sizes = {(768, 768), (768, 3072), (3072, 768)}
        parameters = [param for param in model.parameters() if tuple(param.size()) in sizes]
        activations = torch.cat([param.norm(p=2, dim=1) for param in parameters])
        data = activations.detach().numpy()

    plt.hist(data, bins=args.bins, range=[args.min, args.max])
    plt.xlabel(x_name)
    if args.log:
        plt.yscale("log")
    plt.savefig(args.save)
    print(f"Saved {args.save}.")


def parse_args():
    """Flags controlling the behavior of this script."""
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=str)
    parser.add_argument("--save", type=str, default="images/tmp.png")
    parser.add_argument("--nocuda", action="store_true")
    parser.add_argument(
        "--include-package",
        type=str,
        action="append",
        default=[],
        help="additional packages to include",
    )
    parser.add_argument("--random_init", action="store_true")
    parser.add_argument("--activations", action="store_true")
    parser.add_argument("--log", action="store_true")
    parser.add_argument("--min", type=float, default=0.0)
    parser.add_argument("--max", type=float, default=1.)
    parser.add_argument("--bins", type=int, default=200)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
