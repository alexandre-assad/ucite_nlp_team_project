import unittest

from configuration import Configuration
from oracle import StaticArcStandardOracle
from parser import parse_sentence
from transitions import ArcStandardSystem


def make_arc_standard_sentence():
    return [
        {"id": 0, "form": "ROOT", "lemma": "ROOT", "upos": "ROOT", "head": -1, "deprel": "ROOT"},
        {"id": 1, "form": "Je", "lemma": "je", "upos": "PRON", "head": 2, "deprel": "nsubj"},
        {"id": 2, "form": "mange", "lemma": "manger", "upos": "VERB", "head": 0, "deprel": "root"},
    ]


def make_arc_standard_sentence_with_object():
    return [
        {"id": 0, "form": "ROOT", "lemma": "ROOT", "upos": "ROOT", "head": -1, "deprel": "ROOT"},
        {"id": 1, "form": "Je", "lemma": "je", "upos": "PRON", "head": 2, "deprel": "nsubj"},
        {"id": 2, "form": "mange", "lemma": "manger", "upos": "VERB", "head": 0, "deprel": "root"},
        {"id": 3, "form": "une", "lemma": "un", "upos": "DET", "head": 4, "deprel": "det"},
        {"id": 4, "form": "pomme", "lemma": "pomme", "upos": "NOUN", "head": 2, "deprel": "obj"},
    ]


class ArcStandardTests(unittest.TestCase):
    def setUp(self):
        self.system = ArcStandardSystem()
        self.oracle = StaticArcStandardOracle()
        self.labels = ["nsubj", "root"]

    def test_shift_behavior(self):
        config = self.system.initial_config(make_arc_standard_sentence())
        self.system.apply(config, "SHIFT")
        self.assertEqual(config.stack, [0, 1])
        self.assertEqual(config.buffer, [2])

    def test_left_arc_behavior(self):
        config = self.system.initial_config(make_arc_standard_sentence())
        self.system.apply(config, "SHIFT")
        self.system.apply(config, "SHIFT")
        self.system.apply(config, "LEFT-ARC:nsubj")
        self.assertEqual(config.stack, [0, 2])
        self.assertIn((2, 1, "nsubj"), config.arcs)

    def test_right_arc_behavior(self):
        config = self.system.initial_config(make_arc_standard_sentence())
        self.system.apply(config, "SHIFT")
        self.system.apply(config, "SHIFT")
        self.system.apply(config, "LEFT-ARC:nsubj")
        self.system.apply(config, "RIGHT-ARC:root")
        self.assertEqual(config.stack, [0])
        self.assertIn((0, 2, "root"), config.arcs)

    def test_oracle_reconstructs_small_gold_tree(self):
        sentence = make_arc_standard_sentence()
        config = self.system.initial_config(sentence)
        transitions = []

        while not self.system.is_terminal(config):
            transition = self.oracle.choose(config)
            transitions.append(transition)
            self.system.apply(config, transition)

        self.assertEqual(
            transitions,
            ["SHIFT", "SHIFT", "LEFT-ARC:nsubj", "RIGHT-ARC:root"],
        )
        self.assertIn((2, 1, "nsubj"), config.arcs)
        self.assertIn((0, 2, "root"), config.arcs)

    def test_parsing_terminates(self):
        sentence = make_arc_standard_sentence()

        class DeterministicModel:
            def __init__(self):
                self.sequence = ["SHIFT", "SHIFT", "LEFT-ARC:nsubj", "RIGHT-ARC:root"]
                self.index = 0

            def predict(self, features, valid_classes=None):
                choice = self.sequence[self.index]
                self.index += 1
                return choice, 1.0

        parsed, details = parse_sentence(
            sentence,
            DeterministicModel(),
            self.labels,
            return_details=True,
            transition_system=self.system,
        )
        self.assertTrue(details["terminal_reached"])
        self.assertTrue(all(token["head"] is not None for token in parsed[1:]))

    def test_root_never_gets_a_head(self):
        sentence = make_arc_standard_sentence()

        class DeterministicModel:
            def __init__(self):
                self.sequence = ["SHIFT", "SHIFT", "LEFT-ARC:nsubj", "RIGHT-ARC:root"]
                self.index = 0

            def predict(self, features, valid_classes=None):
                choice = self.sequence[self.index]
                self.index += 1
                return choice, 1.0

        parsed = parse_sentence(
            sentence,
            DeterministicModel(),
            self.labels,
            transition_system=self.system,
        )
        self.assertEqual(parsed[0]["head"], -1)

    def test_oracle_reconstructs_je_mange_une_pomme(self):
        sentence = make_arc_standard_sentence_with_object()
        config = self.system.initial_config(sentence)
        transitions = []

        while not self.system.is_terminal(config):
            transition = self.oracle.choose(config)
            transitions.append(transition)
            self.system.apply(config, transition)

        self.assertEqual(
            transitions,
            [
                "SHIFT",
                "SHIFT",
                "LEFT-ARC:nsubj",
                "SHIFT",
                "SHIFT",
                "LEFT-ARC:det",
                "RIGHT-ARC:obj",
                "RIGHT-ARC:root",
            ],
        )
        self.assertIn((2, 1, "nsubj"), config.arcs)
        self.assertIn((4, 3, "det"), config.arcs)
        self.assertIn((2, 4, "obj"), config.arcs)
        self.assertIn((0, 2, "root"), config.arcs)


if __name__ == "__main__":
    unittest.main()
