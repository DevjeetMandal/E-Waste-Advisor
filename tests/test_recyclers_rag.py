import json
import os

import numpy as np
import pytest

import config
import rag_utils
import recyclers


def test_city_infers_state_and_prefers_local():
    res = recyclers.find_recyclers(city="Bhubaneswar")
    assert res["state"] == "Odisha"
    assert res["matches"][0]["name"] == "Cosmic Net"


def test_state_alias_orissa():
    assert recyclers.find_recyclers(state="Orissa")["state"] == "Odisha"


def test_uncovered_state_is_honest():
    res = recyclers.find_recyclers(state="Kerala")
    assert res["matches"] == []
    assert "does not yet cover" in res["note"]


def test_unknown_city_in_covered_state_falls_back_with_note():
    res = recyclers.find_recyclers(state="Odisha", city="Puri")
    assert res["matches"] and "No listed recycler mentions" in res["note"]


def test_no_location_means_no_lookup():
    res = recyclers.find_recyclers()
    assert res["matches"] == [] and "No state/city" in res["note"]


def test_all_seed_rows_have_name_address_state():
    for r in recyclers.load_recyclers():
        assert r["name"].strip() and r["address"].strip() and r["state"].strip()


def test_search_returns_scored_cited_chunks(index):
    hits = index.search("lithium battery power bank fire", k=3)
    assert len(hits) == 3
    assert hits[0]["fused"] >= hits[1]["fused"] >= hits[2]["fused"]
    assert all({"source", "page", "text", "score", "lex", "fused", "via"} <= set(h) for h in hits)
    assert "Batteries" in hits[0]["text"]


def test_embedder_mismatch_is_caught(tmp_path):
    import shutil
    for n in ("chunks.json", "embeddings.npy", "index_meta.json"):
        shutil.copy(os.path.join(config.DATA_DIR, n), tmp_path / n)
    meta = json.load(open(tmp_path / "index_meta.json"))
    meta["embedder"] = "sentence-transformers:other-model"
    json.dump(meta, open(tmp_path / "index_meta.json", "w"))
    with pytest.raises(rag_utils.IndexError_, match="Re-run"):
        rag_utils.Index.load(str(tmp_path))
