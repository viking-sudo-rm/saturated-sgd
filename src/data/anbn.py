from typing import List

from allennlp.data import Instance
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import TextField, LabelField
from allennlp.data.tokenizers import Token
from allennlp.data.token_indexers import SingleIdTokenIndexer

DELTAS = [-1, 0, 1]


@DatasetReader.register("anbn")
class AnbnDatasetReader(DatasetReader):

    def __init__(self):
        super().__init__(lazy=False)
        self._token_indexers = {"tokens": SingleIdTokenIndexer()}

    def _read(self, path: str):
        min_n, max_n = [int(x) for x in path.split(":")]
        for n in range(min_n, max_n):
            for delta in DELTAS:
                tokens, label = self._get_tokens_and_acceptance(n, delta)
                yield self.text_to_instance(tokens, label)

    def text_to_instance(self, tokens: List[str], label: bool = None):
        fields = {
            "tokens": TextField([Token(tok) for tok in tokens], self._token_indexers)
        }
        if label is not None:
            fields["label"] = LabelField(str(label))
        return Instance(fields)
    
    def _get_tokens_and_acceptance(self, length: int, delta: int):
        tokens = ["a" for _ in range(length)]
        tokens.extend("b" for _ in range(length + delta))
        return tokens, (delta == 0)
