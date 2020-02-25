from typing import List
import os
import pathlib
from unittest import TestCase
from nltk.tree import Tree

from src.data.sentiment import BasicSentimentReader


class PtbWsjReaderTest(TestCase):

    FIXTURES = pathlib.Path(__file__).parent / ".." / "fixtures"
    PATH = os.path.join(FIXTURES, "sentiment.txt")

    def test_get_gold_trees(self):
        reader = BasicSentimentReader()
        instances = reader.read(self.PATH)
        all_tokens = [[tok.text for tok in instance["tokens"]] for instance in instances]
        all_labels = [instance["label"].label for instance in instances]

        exp_all_tokens =  [['It', "'s", 'a', 'lovely', 'film', 'with', 'lovely', 'performances', 'by', 'Buy', 'and', 'Accorsi', '.'], ['No', 'one', 'goes', 'unindicted', 'here', ',', 'which', 'is', 'probably', 'for', 'the', 'best', '.'], ['And', 'if', 'you', "'re", 'not', 'nearly', 'moved', 'to', 'tears', 'by', 'a', 'couple', 'of', 'scenes', ',', 'you', "'ve", 'got', 'ice', 'water', 'in', 'your', 'veins', '.'], ['A', 'warm', ',', 'funny', ',', 'engaging', 'film', '.'], ['Uses', 'sharp', 'humor', 'and', 'insight', 'into', 'human', 'nature', 'to', 'examine', 'class', 'conflict', ',', 'adolescent', 'yearning', ',', 'the', 'roots', 'of', 'friendship', 'and', 'sexual', 'identity', '.'], ['Half', 'Submarine', 'flick', ',', 'Half', 'Ghost', 'Story', ',', 'All', 'in', 'one', 'criminally', 'neglected', 'film'], ['Entertains', 'by', 'providing', 'good', ',', 'lively', 'company', '.'], ['Dazzles', 'with', 'its', 'fully-written', 'characters', ',', 'its', 'determined', 'stylishness', '-LRB-', 'which', 'always', 'relates', 'to', 'characters', 'and', 'story', '-RRB-', 'and', 'Johnny', 'Dankworth', "'s", 'best', 'soundtrack', 'in', 'years', '.'], ['Visually', 'imaginative', ',', 'thematically', 'instructive', 'and', 'thoroughly', 'delightful', ',', 'it', 'takes', 'us', 'on', 'a', 'roller-coaster', 'ride', 'from', 'innocence', 'to', 'experience', 'without', 'even', 'a', 'hint', 'of', 'that', 'typical', 'kiddie-flick', 'sentimentality', '.'], ['Nothing', "'s", 'at', 'stake', ',', 'just', 'a', 'twisty', 'double-cross', 'you', 'can', 'smell', 'a', 'mile', 'away', '--', 'still', ',', 'the', 'derivative', 'Nine', 'Queens', 'is', 'lots', 'of', 'fun', '.']]
        assert all_tokens == exp_all_tokens

        exp_all_labels = ['3', '2', '3', '4', '4', '2', '3', '4', '4', '3']
        assert all_labels == exp_all_labels
