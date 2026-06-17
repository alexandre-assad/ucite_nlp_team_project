import unittest
from pathlib import Path

from configuration import Configuration
from dense_embeddings import RandomStaticEmbeddingProvider, build_embedding_provider
from dense_features import DenseFeatureExtractor
from dense_model import DenseMLPTransitionClassifier
from parser import parse_sentence
from transitions import ArcStandardSystem


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def tiny_sentence():
    return [
        {"id": 0, "form": "ROOT", "lemma": "ROOT", "upos": "ROOT", "head": -1, "deprel": "ROOT"},
        {"id": 1, "form": "Je", "lemma": "je", "upos": "PRON", "head": 2, "deprel": "nsubj"},
        {"id": 2, "form": "mange", "lemma": "manger", "upos": "VERB", "head": 0, "deprel": "root"},
    ]


class DenseM2Tests(unittest.TestCase):
    def test_dense_feature_extractor_returns_expected_size(self):
        config = Configuration(tiny_sentence())
        provider = RandomStaticEmbeddingProvider(embedding_dim=50, seed=0)
        extractor = DenseFeatureExtractor(provider)
        vector = extractor.extract(config)
        self.assertEqual(vector.shape[0], 200)

    def test_missing_positions_use_zero_vectors(self):
        provider = RandomStaticEmbeddingProvider(embedding_dim=8, seed=0)
        extractor = DenseFeatureExtractor(provider)
        sentence = tiny_sentence()
        vector = extractor.extract_from_positions(sentence, None, None, 1, None)
        self.assertTrue((vector[:8] == 0).all())
        self.assertTrue((vector[8:16] == 0).all())
        self.assertTrue((vector[24:32] == 0).all())

    def test_unknown_words_are_handled_safely(self):
        provider = RandomStaticEmbeddingProvider(embedding_dim=12, seed=0)
        token = {"form": "Inconnu", "lemma": "_"}
        vector = provider.vector_for_token(token)
        self.assertEqual(vector.shape[0], 12)

    def test_mlp_outputs_one_score_per_transition_class(self):
        model = DenseMLPTransitionClassifier(
            transition_classes=["SHIFT", "LEFT-ARC:nsubj", "RIGHT-ARC:root"],
            input_dim=40,
            hidden_dim=16,
            dropout=0.1,
            device="cpu",
        )
        choice, score = model.predict([0.0] * 40)
        self.assertIn(choice, {"SHIFT", "LEFT-ARC:nsubj", "RIGHT-ARC:root"})
        self.assertIsInstance(score, float)

    def test_greedy_decoding_terminates_on_tiny_sentence(self):
        sentence = tiny_sentence()
        labels = ["nsubj", "root"]
        system = ArcStandardSystem()
        provider = RandomStaticEmbeddingProvider(embedding_dim=8, seed=0)
        extractor = DenseFeatureExtractor(provider)

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
            labels,
            return_details=True,
            transition_system=system,
            feature_extractor=extractor,
        )
        self.assertTrue(details["terminal_reached"])
        self.assertTrue(all(token["head"] is not None for token in parsed[1:]))

    def test_missing_fasttext_file_gives_clear_error(self):
        config = {
            "source": "fasttext_bin",
            "path": "data/embeddings/cc.fr.300.bin",
            "embedding_dim": 300,
            "oov_strategy": "fasttext_subword_vector",
            "seed": 0,
        }
        with self.assertRaises(FileNotFoundError) as ctx:
            build_embedding_provider(config, PROJECT_ROOT)
        self.assertIn("data/embeddings/cc.fr.300.bin", str(ctx.exception))

if __name__ == "__main__":
    unittest.main()
