"""Contextual variant of the rich M2b feature extractor."""

import numpy as np

from dense_rich_features import (
    CHILD_LABEL_NAMES,
    NULL_LABEL,
    NULL_POS,
    NULL_TOKEN_ID,
    POSITION_NAMES,
    ROOT_POS,
    UNK_POS,
    build_label_vocab,
    build_pos_vocab,
)
from framework import FeatureExtractor


class ContextualRichFeatureExtractor(FeatureExtractor):
    """Use cached CamemBERT token vectors plus POS/label/structure features."""

    def __init__(self, embedding_provider, pos_vocab, label_vocab):
        self.embedding_provider = embedding_provider
        self.pos_vocab = pos_vocab
        self.label_vocab = label_vocab
        self.word_embedding_dim = embedding_provider.embedding_dim
        self.token_position_count = len(POSITION_NAMES)
        self.child_label_count = len(CHILD_LABEL_NAMES)

    def extract(self, config):
        snapshot = self.extract_snapshot(config)
        return self.materialize_from_snapshot(config.sentence, snapshot)

    def extract_snapshot(self, config):
        token_ids = [
            self._stack_idx(config, 0),
            self._stack_idx(config, 1),
            self._stack_idx(config, 2),
            self._buffer_idx(config, 0),
            self._buffer_idx(config, 1),
            self._buffer_idx(config, 2),
        ]

        s0_left, s0_left_label = self._child_info(config, token_ids[0], side="left")
        s0_right, s0_right_label = self._child_info(config, token_ids[0], side="right")
        s1_left, s1_left_label = self._child_info(config, token_ids[1], side="left")
        s1_right, s1_right_label = self._child_info(config, token_ids[1], side="right")

        token_ids.extend([s0_left, s0_right, s1_left, s1_right])
        child_labels = [s0_left_label, s0_right_label, s1_left_label, s1_right_label]

        return {
            "token_ids": np.asarray(token_ids, dtype=np.int64),
            "child_label_ids": np.asarray(
                [self.label_vocab[label] for label in child_labels],
                dtype=np.int64,
            ),
            "distance_bucket": self._distance_bucket(token_ids[0], token_ids[3]),
            "stack_size_bucket": min(len(config.stack), 5),
            "buffer_size_bucket": min(len(config.buffer), 5),
        }

    def materialize_from_snapshot(self, sentence, snapshot):
        token_ids = snapshot["token_ids"]
        word_vectors = np.stack(
            [self.embedding_provider.vector_for_token(sentence, token_id) for token_id in token_ids]
        ).astype(np.float32)
        pos_ids = np.asarray([self._pos_id(sentence, token_id) for token_id in token_ids], dtype=np.int64)
        null_flags = np.asarray([1.0 if token_id == NULL_TOKEN_ID else 0.0 for token_id in token_ids], dtype=np.float32)
        root_flags = np.asarray([1.0 if token_id == 0 else 0.0 for token_id in token_ids], dtype=np.float32)

        return {
            "word_vectors": word_vectors,
            "pos_ids": pos_ids,
            "label_ids": snapshot["child_label_ids"],
            "distance_bucket": snapshot["distance_bucket"],
            "stack_size_bucket": snapshot["stack_size_bucket"],
            "buffer_size_bucket": snapshot["buffer_size_bucket"],
            "null_flags": null_flags,
            "root_flags": root_flags,
        }

    def inspect(self, config):
        snapshot = self.extract_snapshot(config)
        inverse_label_vocab = {idx: label for label, idx in self.label_vocab.items()}
        return {
            "positions": {
                name: int(token_id)
                for name, token_id in zip(POSITION_NAMES, snapshot["token_ids"])
            },
            "child_labels": {
                name: inverse_label_vocab[int(label_id)]
                for name, label_id in zip(CHILD_LABEL_NAMES, snapshot["child_label_ids"])
            },
            "distance_bucket": int(snapshot["distance_bucket"]),
            "stack_size_bucket": int(snapshot["stack_size_bucket"]),
            "buffer_size_bucket": int(snapshot["buffer_size_bucket"]),
        }

    def _stack_idx(self, config, depth):
        if len(config.stack) > depth:
            return config.stack[-1 - depth]
        return NULL_TOKEN_ID

    def _buffer_idx(self, config, offset):
        if len(config.buffer) > offset:
            return config.buffer[offset]
        return NULL_TOKEN_ID

    def _child_info(self, config, parent_idx, side):
        if parent_idx in {None, NULL_TOKEN_ID}:
            return NULL_TOKEN_ID, NULL_LABEL
        dependents = config.get_dependents(parent_idx)
        if not dependents:
            return NULL_TOKEN_ID, NULL_LABEL
        child_idx = min(dependents) if side == "left" else max(dependents)
        return child_idx, self._arc_label(config, child_idx)

    def _arc_label(self, config, dep_idx):
        for _, dep, label in config.arcs:
            if dep == dep_idx:
                return label
        return NULL_LABEL

    def _distance_bucket(self, s0_idx, b0_idx):
        if s0_idx in {None, NULL_TOKEN_ID} or b0_idx in {None, NULL_TOKEN_ID}:
            return 0
        return min(abs(b0_idx - s0_idx), 5)

    def _pos_id(self, sentence, token_id):
        if token_id == NULL_TOKEN_ID:
            return self.pos_vocab[NULL_POS]
        token = sentence[token_id]
        upos = token["upos"]
        return self.pos_vocab.get(upos, self.pos_vocab[UNK_POS])
