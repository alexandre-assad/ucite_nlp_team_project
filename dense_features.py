"""Dense feature extractor for the M2 parser."""

import numpy as np

from framework import FeatureExtractor


class DenseFeatureExtractor(FeatureExtractor):
    """Concatenate static embeddings for s1, s0, b0, and b1."""

    def __init__(self, embedding_provider):
        self.embedding_provider = embedding_provider
        self.embedding_dim = embedding_provider.embedding_dim
        self.output_dim = 4 * self.embedding_dim

    def extract(self, config):
        return self.extract_from_positions(
            config.sentence,
            config.get_stack_second(),
            config.get_stack_top(),
            config.get_buffer_front(),
            config.get_buffer_second(),
        )

    def extract_from_positions(self, sentence, s1_idx, s0_idx, b0_idx, b1_idx):
        vectors = [
            self._vector(sentence, s1_idx),
            self._vector(sentence, s0_idx),
            self._vector(sentence, b0_idx),
            self._vector(sentence, b1_idx),
        ]
        return np.concatenate(vectors, axis=0).astype(np.float32)

    def _vector(self, sentence, idx):
        if idx is None:
            return self.embedding_provider.zero_vector
        return self.embedding_provider.vector_for_token(sentence[idx])
