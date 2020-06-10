from typing import Dict, List, Optional

from overrides import overrides

from allennlp.data import Instance
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import TextField, LabelField
from allennlp.data.tokenizers import Token, Tokenizer
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers.whitespace_tokenizer import WhitespaceTokenizer


@DatasetReader.register("agreement")
class AgreementReader(DatasetReader):

    def __init__(self,
                 tokenizer: Optional[Tokenizer] = None,
                 token_indexers: Optional[Dict[str, TokenIndexer]] = None,
                 lazy: bool = False):
        super().__init__(lazy=lazy)
        self.tokenizer = tokenizer or WhitespaceTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    @overrides
    def _read(self, path: str):
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                label, text = line.split("\t")
                yield self.text_to_instance(text, label)

    @overrides
    def text_to_instance(self, text: str, label: str):
        tokens = self.tokenizer.tokenize(text)
        tokens_field = TextField(tokens, self._token_indexers)
        label_field = LabelField(label)
        
        return Instance({
            "tokens": tokens_field,
            "label": label_field,
        })
