import unittest
from pathlib import Path

from configuration import Configuration
from contextual_embeddings import CamembertEmbeddingProvider
from contextual_rich_features import ContextualRichFeatureExtractor, build_label_vocab, build_pos_vocab
from contextual_rich_model import DenseRichTransitionClassifier


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def french_sentence():
    return [
        {"id": 0, "form": "ROOT", "lemma": "ROOT", "upos": "ROOT", "head": -1, "deprel": "ROOT"},
        {"id": 1, "form": "Je", "lemma": "je", "upos": "PRON", "head": 2, "deprel": "nsubj"},
        {"id": 2, "form": "mange", "lemma": "manger", "upos": "VERB", "head": 0, "deprel": "root"},
        {"id": 3, "form": "une", "lemma": "un", "upos": "DET", "head": 4, "deprel": "det"},
        {"id": 4, "form": "pomme", "lemma": "pomme", "upos": "NOUN", "head": 2, "deprel": "obj"},
    ]


class ContextualM3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.provider = CamembertEmbeddingProvider(
            model_name="camembert-base",
            device="cpu",
            pooling="average",
            local_files_only=True,
        )
        cls.pos_vocab = build_pos_vocab([french_sentence()])
        cls.label_vocab = build_label_vocab([french_sentence()])
        cls.extractor = ContextualRichFeatureExtractor(cls.provider, cls.pos_vocab, cls.label_vocab)

    def test_camembert_alignment_works(self):
        alignment = self.provider.align_tokens(french_sentence())
        self.assertIn("▁Je", alignment["tokens"])
        self.assertEqual(sorted(set(w for w in alignment["word_ids"] if w is not None)), [0, 1, 2, 3])

    def test_token_level_contextual_vectors_have_expected_dimension(self):
        matrix = self.provider.get_sentence_embeddings(french_sentence())
        self.assertEqual(matrix.shape[0], len(french_sentence()))
        self.assertEqual(matrix.shape[1], 768)

    def test_sentence_embeddings_are_cached(self):
        before = self.provider.cache_size()
        first = self.provider.get_sentence_embeddings(french_sentence())
        middle = self.provider.cache_size()
        second = self.provider.get_sentence_embeddings(french_sentence())
        after = self.provider.cache_size()
        self.assertIs(first, second)
        self.assertEqual(middle, after)
        self.assertGreaterEqual(after, before)

    def test_missing_positions_use_null_vectors(self):
        sentence = french_sentence()
        features = self.extractor.materialize_from_snapshot(
            sentence,
            {
                "token_ids": __import__("numpy").asarray([-1, -1, 1, -1, -1, -1, -1, -1, -1, -1]),
                "child_label_ids": __import__("numpy").asarray([0, 0, 0, 0]),
                "distance_bucket": 0,
                "stack_size_bucket": 1,
                "buffer_size_bucket": 1,
            },
        )
        self.assertTrue((features["word_vectors"][0] == 0).all())
        self.assertTrue((features["word_vectors"][1] == 0).all())

    def test_mlp_output_dimension_equals_transition_classes(self):
        config = Configuration(french_sentence())
        features = self.extractor.extract(config)
        model = DenseRichTransitionClassifier(
            transition_classes=["SHIFT", "LEFT-ARC:nsubj", "RIGHT-ARC:root"],
            word_embedding_dim=768,
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
