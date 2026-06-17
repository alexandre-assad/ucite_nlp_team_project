"""Frozen CamemBERT sentence encodings for contextual parsing features."""

from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


class CamembertEmbeddingProvider:
    """Cache token-level CamemBERT embeddings for full CoNLL-U sentences."""

    def __init__(self, model_name="camembert-base", device=None, pooling="average", local_files_only=True):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.pooling = pooling
        self.local_files_only = local_files_only
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
        self.model = AutoModel.from_pretrained(model_name, local_files_only=local_files_only).to(self.device)
        self.model.eval()
        self.embedding_dim = int(self.model.config.hidden_size)
        self.zero_vector = np.zeros(self.embedding_dim, dtype=np.float32)
        self._cache = {}

    def cache_size(self):
        """Expose the current number of cached sentence encodings for tests."""
        return len(self._cache)

    def vector_for_token(self, sentence, token_id):
        """Return the contextual vector for one token in a sentence."""
        if token_id in {None, -1}:
            return self.zero_vector
        sentence_matrix = self.get_sentence_embeddings(sentence)
        return sentence_matrix[token_id]

    def get_sentence_embeddings(self, sentence):
        """Return a cached matrix of token embeddings including ROOT at index 0."""
        key = self._sentence_key(sentence)
        if key not in self._cache:
            self._cache[key] = self._encode_sentence(sentence)
        return self._cache[key]

    def _encode_sentence(self, sentence):
        words = [token["form"] for token in sentence[1:]]
        encoded = self.tokenizer(
            words,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
        )
        encoded = {name: tensor.to(self.device) for name, tensor in encoded.items()}

        with torch.no_grad():
            outputs = self.model(**encoded)

        hidden = outputs.last_hidden_state[0].detach().cpu().numpy()
        word_alignment = self.tokenizer(
            words,
            is_split_into_words=True,
            truncation=True,
        )
        alignment = word_alignment.word_ids()

        token_vectors = [self.zero_vector.copy()]
        for word_index in range(len(words)):
            subword_indices = [i for i, aligned_word in enumerate(alignment) if aligned_word == word_index]
            if not subword_indices:
                token_vectors.append(self.zero_vector.copy())
                continue
            subword_vectors = hidden[subword_indices]
            if self.pooling == "first":
                token_vectors.append(subword_vectors[0].astype(np.float32))
            else:
                token_vectors.append(subword_vectors.mean(axis=0).astype(np.float32))

        return np.stack(token_vectors).astype(np.float32)

    def align_tokens(self, sentence):
        """Return token/subword alignment info for sanity tests."""
        words = [token["form"] for token in sentence[1:]]
        encoded = self.tokenizer(
            words,
            is_split_into_words=True,
            truncation=True,
        )
        tokens = self.tokenizer.convert_ids_to_tokens(encoded["input_ids"])
        return {
            "tokens": tokens,
            "word_ids": encoded.word_ids(),
        }

    def _sentence_key(self, sentence):
        return tuple((token["id"], token["form"], token["lemma"], token["upos"]) for token in sentence)
