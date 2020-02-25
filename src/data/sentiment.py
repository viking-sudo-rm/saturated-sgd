from typing import Callable, Dict, List, Optional

from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.tree import Tree
from overrides import overrides
from typing import Dict

from allennlp.data import Instance
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import TextField, SequenceLabelField, LabelField
from allennlp.data.tokenizers import Token, Tokenizer
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer


def _binarize_sentiment(sentiment):
    """Return 0/1 for negative/positive, and discard neutral sentences.

    https://github.com/allenai/allennlp/blob/master/allennlp/data/dataset_readers/stanford_sentiment_tree_bank.py
    """
    sentiment = int(sentiment)
    if sentiment < 2:
        return 0
    elif sentiment == 2:
        return None
    else:
        return 1


@DatasetReader.register("basic_sentiment")
class BasicSentimentReader(DatasetReader):

    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 binary_sentiment: bool = False,
                 lazy: bool = False):
        super().__init__(lazy=lazy)
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._binary_sentiment = binary_sentiment

    @overrides
    def _read(self, file_path):
        with open(file_path) as in_file:
            for line in in_file.readlines():
                if not line:
                    continue

                tree = Tree.fromstring(line)
                sentiment = tree.label()
                if self._binary_sentiment:
                    sentiment = _binarize_sentiment(sentiment)
                    if sentiment is None:
                        continue

                yield self.text_to_instance(tree.leaves(), sentiment)

    @overrides
    def text_to_instance(self,
                         tokens: List[str],
                         sentiment: Optional[int] = None) -> Instance:
        tokens = [Token(token) for token in tokens]
        tokens_field = TextField(tokens, self._token_indexers)
        fields = {"tokens": tokens_field}

        if sentiment is not None:
            sentiment = str(sentiment)
            fields["label"] = LabelField(sentiment)

        return Instance(fields)
