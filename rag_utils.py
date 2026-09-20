
import json

import numpy as np
from sentence_transformers import SentenceTransformer

CHUNKS_PATH = "data/chunks.json"
EMBEDDINGS_PATH = "data/embeddings.npy"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

_model = None
_chunks = None
_embeddings = None


def _load():
    global _model, _chunks, _embeddings
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL_NAME)
    if _chunks is None:
        with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
            _chunks = json.load(f)
    if _embeddings is None:
        _embeddings = np.load(EMBEDDINGS_PATH)
    return _model, _chunks, _embeddings


def retrieve(query, k=4):
    """Return the top-k most relevant chunks for the query, each with its source."""
    model, chunks, embeddings = _load()
    query_vec = model.encode([query], normalize_embeddings=True)[0]

    # embeddings are already normalized, so dot product == cosine similarity
    scores = embeddings @ query_vec
    top_idx = np.argsort(scores)[::-1][:k]

    results = []
    for i in top_idx:
        results.append(
            {
                "text": chunks[i]["text"],
                "source": chunks[i]["source"],
                "score": float(scores[i]),
            }
        )
    return results


def index_exists():
    import os

    return os.path.exists(CHUNKS_PATH) and os.path.exists(EMBEDDINGS_PATH)
