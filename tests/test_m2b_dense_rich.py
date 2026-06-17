import unittest
from pathlib import Path

from configuration import Configuration
from dense_embeddings import RandomStaticEmbeddingProvider
from dense_rich_features import DenseRichFeatureExtractor, build_label_vocab, build_pos_vocab
from dense_rich_model import DenseRichTransitionClassifier
from transitions import ArcStandardSystem


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def rich_sentence():
    return [
        {"id": 0, "form": "ROOT", "lemma": "ROOT", "upos": "ROOT", "head": -1, "deprel": "ROOT"},
        {"id": 1, "form": "Je", "lemma": "je", "upos": "PRON", "head": 2, "deprel": "nsubj"},
        {"id": 2, "form": "mange", "lemma": "manger", "upos": "VERB", "head": 0, "deprel": "root"},
        {"id": 3, "form": "une", "lemma": "un", "upos": "DET", "head": 4, "deprel": "det"},
        {"id": 4, "form": "pomme", "lemma": "pomme", "upos": "NOUN", "head": 2, "deprel": "obj"},
        {"id": 5, "form": "verte", "lemma": "vert", "upos": "ADJ", "head": 4, "deprel": "amod"},
        {"id": 6, "form": "aujourd'hui", "lemma": "aujourd'hui", "upos": "ADV", "head": 2, "deprel": "advmod"},
    ]


class DenseRichM2bTests(unittest.TestCase):
    def setUp(self):
        self.provider = RandomStaticEmbeddingProvider(embedding_dim=50, seed=0)
        self.pos_vocab = build_pos_vocab([rich_sentence()])
        self.label_vocab = build_label_vocab([rich_sentence()])
        self.extractor = DenseRichFeatureExtractor(self.provider, self.pos_vocab, self.label_vocab)

    def test_selected_positions_are_extracted_correctly(self):
        config = Configuration(rich_sentence())
        config.stack = [0, 1, 2, 3]
        config.buffer = [4, 5, 6]
        config.arcs = [(3, 1, "nsubj"), (3, 6, "advmod"), (2, 5, "obj")]
        info = self.extractor.inspect(config)
        self.assertEqual(
            info["positions"],
            {
                "s0": 3,
                "s1": 2,
                "s2": 1,
                "b0": 4,
                "b1": 5,
                "b2": 6,
                "s0_left": 1,
                "s0_right": 6,
                "s1_left": 5,
                "s1_right": 5,
            },
        )

    def test_pos_vocab_contains_root_null_unk(self):
        self.assertIn("ROOT", self.pos_vocab)
        self.assertIn("NULL", self.pos_vocab)
        self.assertIn("UNK", self.pos_vocab)

    def test_label_vocab_contains_null_and_dependency_labels(self):
        self.assertIn("NULL", self.label_vocab)
        self.assertIn("nsubj", self.label_vocab)
        self.assertIn("root", self.label_vocab)
        self.assertIn("obj", self.label_vocab)

    def test_missing_children_use_null_token_and_null_label(self):
        config = Configuration(rich_sentence())
        info = self.extractor.inspect(config)
        self.assertEqual(info["positions"]["s0_left"], -1)
        self.assertEqual(info["positions"]["s0_right"], -1)
        self.assertEqual(info["child_labels"]["s0_left_label"], "NULL")
        self.assertEqual(info["child_labels"]["s0_right_label"], "NULL")

    def test_model_input_has_expected_dimensions(self):
        config = Configuration(rich_sentence())
        features = self.extractor.extract(config)
        model = DenseRichTransitionClassifier(
            transition_classes=["SHIFT", "LEFT-ARC:nsubj", "RIGHT-ARC:root"],
            word_embedding_dim=50,
            pos_vocab_size=len(self.pos_vocab),
            label_vocab_size=len(self.label_vocab),
            pos_embedding_dim=32,
            label_embedding_dim=32,
            hidden_dim=64,
            dropout=0.1,
            device="cpu",
        )
        batch = model.prepare_batch([features])
        flattened = model.feature_net.forward_features(batch)
        self.assertEqual(model.feature_dim, 986)
        self.assertEqual(flattened.shape[1], 986)

    def test_mlp_outputs_one_score_per_transition_class(self):
        config = Configuration(rich_sentence())
        features = self.extractor.extract(config)
        model = DenseRichTransitionClassifier(
            transition_classes=["SHIFT", "LEFT-ARC:nsubj", "RIGHT-ARC:root"],
            word_embedding_dim=50,
            pos_vocab_size=len(self.pos_vocab),
            label_vocab_size=len(self.label_vocab),
            pos_embedding_dim=32,
            label_embedding_dim=32,
            hidden_dim=64,
            dropout=0.1,
            device="cpu",
        )
        logits = model.logits(model.prepare_batch([features]))
        self.assertEqual(tuple(logits.shape), (1, 3))

if __name__ == "__main__":
    unittest.main()
