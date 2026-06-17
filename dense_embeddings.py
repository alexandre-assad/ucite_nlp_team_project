"""Static dense embedding providers for M2."""

import hashlib
from pathlib import Path

import numpy as np


def lookup_key(token):
    """Use lemma if available, otherwise fall back to form."""
    lemma = token.get("lemma", "_")
    if lemma and lemma != "_":
        return lemma.lower()
    return token.get("form", "_").lower()


def _stable_seed(seed, key):
    digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


class RandomStaticEmbeddingProvider:
    """Deterministic pseudo-random static embeddings keyed by token string."""

    def __init__(self, embedding_dim, seed):
        self.embedding_dim = embedding_dim
        self.seed = seed
        self._cache = {}
        self.zero_vector = np.zeros(self.embedding_dim, dtype=np.float32)
        self.oov_strategy = "deterministic_hashed_vector"

    def vector_for_token(self, token):
        key = lookup_key(token)
        return self.vector_for_key(key)

    def vector_for_key(self, key):
        if key not in self._cache:
            rng = np.random.default_rng(_stable_seed(self.seed, key))
            self._cache[key] = rng.normal(loc=0.0, scale=0.1, size=self.embedding_dim).astype(
                np.float32
            )
        return self._cache[key]


class FastTextBinEmbeddingProvider:
    """Static embeddings backed by a local FastText `.bin` model."""

    def __init__(self, model_path, expected_dim=None):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                "FastText embedding file not found. Expected it at "
                f"'{self.model_path.as_posix()}'. Place 'cc.fr.300.bin' under "
                "'data/embeddings/'."
            )

        import fasttext

        self.model = fasttext.load_model(str(self.model_path))
        self._vector_lookup = self._resolve_vector_lookup(self.model)
        self.embedding_dim = self._resolve_dimension(self.model, expected_dim)
        if expected_dim is not None and self.embedding_dim != expected_dim:
            raise ValueError(
                f"FastText dimension mismatch: config expects {expected_dim}, "
                f"but model provides {self.embedding_dim}."
            )
        self.zero_vector = np.zeros(self.embedding_dim, dtype=np.float32)
        self.oov_strategy = "fasttext_subword_vector"

    def vector_for_token(self, token):
        return self.vector_for_key(lookup_key(token))

    def vector_for_key(self, key):
        return np.asarray(self._vector_lookup(key), dtype=np.float32)

    def _resolve_vector_lookup(self, model):
        if hasattr(model, "get_word_vector"):
            return model.get_word_vector

        internal = getattr(model, "f", None)
        if internal is not None and hasattr(internal, "getWordVector"):
            return internal.getWordVector

        raise RuntimeError(
            "The installed 'fasttext' package can load the .bin file but does not expose "
            "word-vector lookup methods. Your current installation appears to support only "
            "text classification prediction. To run full M2 with "
            "'data/embeddings/cc.fr.300.bin', install a FastText Python package build that "
            "supports embedding lookup (for example one exposing 'get_word_vector'), then rerun "
            "'python run_m2_dense_experiment.py'."
        )

    def _resolve_dimension(self, model, expected_dim):
        if hasattr(model, "get_dimension"):
            return model.get_dimension()

        internal = getattr(model, "f", None)
        if internal is not None and hasattr(internal, "getDimension"):
            return int(internal.getDimension())

        if expected_dim is not None:
            return expected_dim

        raise RuntimeError(
            "Unable to determine the FastText embedding dimension from the installed package."
        )


def build_embedding_provider(embedding_config, project_root):
    """Instantiate the configured embedding provider."""
    source = embedding_config["source"]
    embedding_dim = embedding_config["embedding_dim"]

    if source == "random":
        return RandomStaticEmbeddingProvider(embedding_dim=embedding_dim, seed=embedding_config["seed"])

    if source == "fasttext_bin":
        relative_path = embedding_config["path"]
        return FastTextBinEmbeddingProvider(
            model_path=project_root / relative_path,
            expected_dim=embedding_dim,
        )

    raise ValueError(f"Unknown embedding source: {source}")
