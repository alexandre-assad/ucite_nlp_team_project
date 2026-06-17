import tempfile
import unittest
from pathlib import Path

from configuration import Configuration
from data_utils import read_conllu
from evaluator import evaluate
from oracle import get_oracle_transition
from parser import parse_sentence
from perceptron import AveragedPerceptron
from transitions import apply_transition


def make_tiny_sentence():
    return [
        {"id": 0, "form": "ROOT", "lemma": "ROOT", "upos": "ROOT", "head": -1, "deprel": "ROOT"},
        {"id": 1, "form": "Je", "lemma": "je", "upos": "PRON", "head": 2, "deprel": "nsubj"},
        {"id": 2, "form": "mange", "lemma": "manger", "upos": "VERB", "head": 0, "deprel": "root"},
    ]


class BaselineSanityTests(unittest.TestCase):
    def test_conllu_reader_skips_comments_and_multiword_tokens(self):
        sample = """# sent_id = 1
# text = au revoir
1-2\tau\t_\t_\t_\t_\t_\t_\t_\t_
1\ta\tà\tADP\t_\t_\t2\tcase\t_\t_
2\trevoir\trevoir\tNOUN\t_\t_\t0\troot\t_\t_

"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sample.conllu"
            path.write_text(sample, encoding="utf-8")
            sentences = read_conllu(str(path))
        self.assertEqual(len(sentences), 1)
        self.assertEqual(sentences[0][0]["form"], "ROOT")
        self.assertEqual(len(sentences[0]), 3)
        self.assertEqual(sentences[0][1]["form"], "a")
        self.assertEqual(sentences[0][2]["head"], 0)

    def test_initial_configuration(self):
        sentence = make_tiny_sentence()
        config = Configuration(sentence)
        self.assertEqual(config.stack, [0])
        self.assertEqual(config.buffer, [1, 2])
        self.assertEqual(config.arcs, [])
        self.assertFalse(config.is_terminal())

    def test_static_oracle_sequence_on_small_example(self):
        sentence = make_tiny_sentence()
        config = Configuration(sentence)

        self.assertEqual(get_oracle_transition(config), "SHIFT")
        apply_transition(config, "SHIFT")
        self.assertEqual(get_oracle_transition(config), "LEFT-ARC:nsubj")
        apply_transition(config, "LEFT-ARC:nsubj")
        self.assertEqual(get_oracle_transition(config), "RIGHT-ARC:root")
        apply_transition(config, "RIGHT-ARC:root")
        self.assertEqual(get_oracle_transition(config), "REDUCE")

    def test_parser_termination_and_single_root(self):
        sentence = make_tiny_sentence()
        labels = ["nsubj", "root"]
        
        class DeterministicModel:
            def __init__(self):
                self.sequence = ["SHIFT", "LEFT-ARC:nsubj", "RIGHT-ARC:root", "REDUCE"]
                self.step = 0

            def predict(self, features, valid_classes=None):
                choice = self.sequence[self.step]
                self.step += 1
                if choice not in valid_classes:
                    raise AssertionError(f"{choice} not in valid set {valid_classes}")
                return choice, 1.0

        model = DeterministicModel()
        predicted, details = parse_sentence(sentence, model, labels, return_details=True)
        self.assertTrue(details["terminal_reached"])
        self.assertEqual(sum(1 for tok in predicted[1:] if tok["head"] == 0), 1)
        self.assertTrue(all(tok["head"] is not None for tok in predicted[1:]))

    def test_evaluator_gold_vs_gold_is_perfect(self):
        sentence = make_tiny_sentence()
        uas, las = evaluate([sentence], [sentence])
        self.assertEqual(uas, 100.0)
        self.assertEqual(las, 100.0)


if __name__ == "__main__":
    unittest.main()
