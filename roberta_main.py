import torch
from math import sqrt
from fairseq.models.roberta import RobertaModel

from src.utils.saturate import saturate


def get_norm(parameters, normalize: bool = False):
    params = torch.cat([param.flatten() for param in parameters])
    norm = torch.norm(params)
    if normalize:
        norm /= sqrt(len(params))
    return norm


def cos(vec1, vec2):
    return torch.sum(vec1 * vec2, dim=-1) / (vec1.norm(dim=-1) * vec2.norm(dim=-1))


for model_name in ["roberta.base", "roberta.large"]:
    roberta = RobertaModel.from_pretrained("/home/willm/data/roberta-models/" + model_name, checkpoint_file="model.pt")
    norm = get_norm(roberta.parameters(), normalize=True)
    print(model_name, "=", norm.item())

    tokens = roberta.encode("Hello to my little friend.")
    features = roberta.extract_features(tokens)
    with saturate(roberta):
        sat_features = roberta.extract_features(tokens)
    
    sim = cos(features.flatten(), sat_features.flatten())
    import pdb; pdb.set_trace()
