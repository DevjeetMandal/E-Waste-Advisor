"""
Embedding backends behind one tiny interface.

- "sentence-transformers": the real thing (local, free, CPU-friendly).
- "hash": a deterministic bag-of-words hashing embedder used ONLY so the test
  suite can run without downloading a model. Never use it for the real index.
"""

import hashlib
import re

import numpy as np

import config

_cache = {}


class SentenceTransformerEmbedder:
    name = "sentence-transformers"

    def __init__(self, model_name):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def encode(self, texts):
        vecs = self.model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vecs, dtype=np.float32)


class HashEmbedder:
    name = "hash"
    model_name = "hash-512"
    DIM = 512

    def encode(self, texts):
        out = np.zeros((len(texts), self.DIM), dtype=np.float32)
        for row, text in enumerate(texts):
            for tok in re.findall(r"[a-z0-9]+", text.lower()):
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
                out[row, h % self.DIM] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms


def get_embedder():
    backend = config.EMBED_BACKEND
    if backend not in _cache:
        if backend == "hash":
            _cache[backend] = HashEmbedder()
        else:
            _cache[backend] = SentenceTransformerEmbedder(config.EMBED_MODEL_NAME)
    return _cache[backend]


def identity():
    """String stored alongside the index so a mismatch is caught at query time."""
    e = get_embedder()
    return f"{e.name}:{e.model_name}"
