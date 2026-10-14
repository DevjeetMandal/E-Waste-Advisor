"""Regression tests for the problems found by running the real PDFs."""
import numpy as np
import pytest

import config
import embedder
import ingest
import rag_utils
from agent import assess_grounding
from classify import classify_with_evidence


# ---- ingestion: bilingual Gazette PDF ---------------------------------------------------
def test_garbled_devanagari_is_stripped_and_header_removed():
    hi = "[भाग II—खण् ड 3(i)] भारत का राजपत्र : ऄसाधारण 25 (7) accept waste electrical and electronic equipment"
    assert ingest.clean(hi) == "25 (7) accept waste electrical and electronic equipment"
    en = "22 THE GAZETTE OF INDIA : EXTRAORDINARY [PART II—SEC. 3(i)] end-product and Items"
    assert ingest.clean(en) == "22 end-product and Items"


def test_devanagari_only_page_becomes_empty():
    assert ingest.clean("डाटा प्रोसेससग : मेनफ्र े म, जमनी क ं प्यूटर") == ""


def test_numeric_table_chunks_are_dropped():
    pages = [{"source": "t.pdf", "page": 1,
              "text": "0 0 0 0 25 0 25 50 0 0 0 50 Green 43.4 " * 12},
             {"source": "t.pdf", "page": 2,
              "text": "Every producer shall ensure that discarded electronic equipment is collected and sent to a recycler " * 3}]
    chunks = ingest.build_chunks(pages)
    assert chunks and all(c["page"] == 2 for c in chunks)


def test_repeated_running_headers_are_removed():
    topics = ["batteries", "lamps", "printers", "tablets", "routers", "monitors", "chargers", "cables"]
    pages = [[f"Running header {i}", f"body text discussing {topic} and their handling"]
             for i, topic in enumerate(topics, start=1)]
    cleaned = ingest.drop_repeated_lines(pages)
    assert all(len(p) == 1 and "body text" in p[0] for p in cleaned)


# ---- hybrid retrieval: item-list tables ---------------------------------------------------
class StubEmbedder:
    """Dense search deliberately prefers chunk 0; only BM25 can find the refrigerator list."""
    name, model_name = "stub", "stub"

    def encode(self, texts):
        return np.array([[1.0, 0.0]] * len(texts), dtype=np.float32)


def test_keyword_search_rescues_item_list_chunk(monkeypatch):
    chunks = [
        {"source": "r.pdf", "page": 1, "text": "general rules about disposal of old electronics by households and producers"},
        {"source": "r.pdf", "page": 2, "text": "unrelated rules about penalties and environmental compensation"},
        {"source": "r.pdf", "page": 31, "text": "Schedule I items: refrigerator washing machine television microwave printer scanner"},
    ]
    vectors = np.array([[1.0, 0.0], [0.9, 0.4], [0.0, 1.0]], dtype=np.float32)  # dense ranks chunk 2 LAST
    monkeypatch.setattr(embedder, "get_embedder", lambda: StubEmbedder())
    idx = rag_utils.Index(chunks, vectors, {})
    hits = idx.search("refrigerator disposal", k=2)
    assert any(h["page"] == 31 for h in hits)
    rescued = next(h for h in hits if h["page"] == 31)
    assert rescued["via"] in ("keyword", "both") and rescued["score"] < 0.1


def test_plural_and_ies_stemming():
    assert rag_utils.tokenize("Refrigerators batteries lamps") == ["refrigerator", "battery", "lamp"]


# ---- grounding gate ---------------------------------------------------------------------------
def hit(text, score):
    return {"text": text, "score": score}


def test_device_named_in_passage_rescues_low_similarity():
    cls = classify_with_evidence("refrigerator disposal")
    ok, why, _ = assess_grounding(cls, [hit("Schedule: Refrigerators, washing machines", 0.18)])
    assert ok and "refrigerator" in why


def test_no_device_query_needs_higher_similarity(monkeypatch):
    monkeypatch.setattr(config, "MIN_SCORE", 0.25)
    monkeypatch.setattr(config, "MIN_SCORE_NO_DEVICE", 0.40)
    cls = classify_with_evidence("how to bake bread")
    assert cls["matched"] == []
    assert assess_grounding(cls, [hit("anything", 0.33)])[0] is False
    assert assess_grounding(cls, [hit("anything", 0.45)])[0] is True


def test_device_query_uses_normal_bar(monkeypatch):
    monkeypatch.setattr(config, "MIN_SCORE", 0.25)
    cls = classify_with_evidence("old laptop")
    assert assess_grounding(cls, [hit("unrelated words", 0.30)])[0] is True


def test_short_ambiguous_keywords_cannot_rescue():
    cls = classify_with_evidence("my AC")           # matched keyword is 'ac' (2 letters)
    assert cls["matched"] == ["ac"]
    assert assess_grounding(cls, [hit("AC voltage ratings for tools", 0.05)])[0] is False


def test_empty_evidence_is_not_grounded():
    assert assess_grounding(classify_with_evidence("old laptop"), [])[0] is False
