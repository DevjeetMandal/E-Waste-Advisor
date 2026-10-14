"""
Retrieval over the local index: cosine similarity on normalised vectors.
Results carry source file, page number and score so every answer can cite them.
"""

import collections
import json
import math
import os
import re

import numpy as np

import config
import embedder



_STOP = set("a an and are as at be by for from how i in is it my of on or that the this to what which with do does "
            "old new can should".split())


def _stem(w):
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 4 and w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


def tokenize(text):
    return [_stem(w) for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOP and len(w) > 1]


class BM25:
    """Plain BM25. Needed because dense embeddings blur long item-list tables
    (e.g. a schedule naming 100 appliances), so a query for 'refrigerator' can miss them."""

    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.tf = [collections.Counter(d) for d in docs]
        self.dl = np.array([len(d) for d in docs], dtype=np.float32)
        self.avgdl = float(self.dl.mean()) if len(docs) else 1.0
        df = collections.Counter(t for d in docs for t in set(d))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def scores(self, query_tokens):
        out = np.zeros(len(self.tf), dtype=np.float32)
        for t in set(query_tokens):
            idf = self.idf.get(t)
            if idf is None:
                continue
            for i, tf in enumerate(self.tf):
                f = tf.get(t, 0)
                if f:
                    out[i] += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.dl[i] / self.avgdl))
        return out


class IndexError_(RuntimeError):
    pass


class Index:
    def __init__(self, chunks, vectors, meta):
        self.chunks = chunks
        self.vectors = vectors
        self.meta = meta
        self.bm25 = BM25([tokenize(c["text"]) for c in chunks])

    @classmethod
    def load(cls, data_dir=None):
        data_dir = data_dir or config.DATA_DIR
        paths = [os.path.join(data_dir, n) for n in ("chunks.json", "embeddings.npy", "index_meta.json")]
        if not all(os.path.exists(p) for p in paths):
            raise IndexError_("No retrieval index found. Run `python ingest.py` first.")
        with open(paths[0], "r", encoding="utf-8") as f:
            chunks = json.load(f)
        vectors = np.load(paths[1])
        with open(paths[2], "r", encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("embedder") != embedder.identity():
            raise IndexError_(
                f"Index was built with '{meta.get('embedder')}' but the app is configured for "
                f"'{embedder.identity()}'. Re-run `python ingest.py`."
            )
        return cls(chunks, vectors, meta)

    def search(self, query, k=config.TOP_K):
        """Hybrid search: dense cosine + BM25, merged by reciprocal-rank fusion.
        `score` stays the raw cosine similarity (what the grounding gate uses);
        `fused` is the rank used for ordering; `via` says which retriever found it."""
        n = len(self.chunks)
        qv = embedder.get_embedder().encode([query])[0]
        dense = self.vectors @ qv
        lex = self.bm25.scores(tokenize(query))

        order_d = np.argsort(-dense)
        rank_d = np.empty(n, dtype=int)
        rank_d[order_d] = np.arange(n)
        fused = 1.0 / (config.RRF_K + rank_d + 1)

        rank_l = None
        if lex.max() > 0:
            order_l = np.argsort(-lex)
            rank_l = np.empty(n, dtype=int)
            rank_l[order_l] = np.arange(n)
            mask = lex > 0
            fused[mask] += 1.0 / (config.RRF_K + rank_l[mask] + 1)

        results = []
        for i in np.argsort(-fused)[:k]:
            in_dense = rank_d[i] < 2 * k
            in_lex = rank_l is not None and lex[i] > 0 and rank_l[i] < 2 * k
            via = "both" if (in_dense and in_lex) else "keyword" if in_lex else "semantic"
            results.append({
                "source": self.chunks[i]["source"],
                "page": self.chunks[i].get("page"),
                "text": self.chunks[i]["text"],
                "score": round(float(dense[i]), 4),
                "lex": round(float(lex[i]), 3),
                "fused": round(float(fused[i]), 5),
                "via": via,
            })
        return results


_default = None


def get_index():
    global _default
    if _default is None:
        _default = Index.load()
    return _default


def reset_cache():
    global _default
    _default = None


def index_exists(data_dir=None):
    data_dir = data_dir or config.DATA_DIR
    return all(
        os.path.exists(os.path.join(data_dir, n)) for n in ("chunks.json", "embeddings.npy", "index_meta.json")
    )
