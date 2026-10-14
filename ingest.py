"""
Builds the retrieval index.

Reads every PDF in data/raw/ page by page (so answers can cite page numbers).
If there are no PDFs it falls back to the placeholder text in sample_data/ so
the pipeline still runs - but never submit work based on the placeholder.

Usage:  python ingest.py
"""

import glob
import json
import os
import re

import numpy as np
from pypdf import PdfReader

import config
import embedder

CHUNK_WORDS = 160
CHUNK_OVERLAP = 30
MIN_CHUNK_WORDS = 15


DEVANAGARI = re.compile(r"[\u0900-\u097F\u200c\u200d]+")
PART_TAG = re.compile(r"\[\s*(?:PART\s*)?II\s*[\u2014\u2013\-]*\s*(?:SEC\.?\s*)?3\s*\(i\)\s*\]", re.I)
GAZETTE = re.compile(r"THE GAZETTE OF INDIA\s*:?\s*EXTRAORDINARY", re.I)
MIN_ALPHA_RATIO = 0.25


def clean(text):
    """Normalise one page of extracted text.

    The Gazette PDFs are bilingual and pypdf mangles the Devanagari (legacy font
    mapping), producing gibberish that embeds close to unrelated English queries.
    We strip it, plus the running page header, before chunking.
    """
    text = DEVANAGARI.sub(" ", text or "")
    text = PART_TAG.sub(" ", text)
    text = GAZETTE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return re.sub(r"^[\s:;,\-\u2013\u2014|]+", "", text)


def alpha_ratio(text):
    chars = [c for c in text if not c.isspace()]
    return sum(c.isalpha() for c in chars) / len(chars) if chars else 0.0


def drop_repeated_lines(pages_lines):
    """Remove running headers/footers: short lines whose digit-normalised form repeats on many pages."""
    n_pages = len(pages_lines)
    if n_pages < 4:
        return pages_lines
    seen = {}
    for lines in pages_lines:
        for key in {re.sub(r"\d+", "#", ln.strip().lower()) for ln in lines if ln.strip()}:
            seen[key] = seen.get(key, 0) + 1
    limit = max(3, int(0.4 * n_pages))
    return [
        [ln for ln in lines
         if len(ln.strip()) > 150 or seen.get(re.sub(r"\d+", "#", ln.strip().lower()), 0) < limit]
        for lines in pages_lines
    ]


def load_pdf_pages(raw_dir):
    pages = []
    for path in sorted(glob.glob(os.path.join(raw_dir, "*.pdf"))):
        reader = PdfReader(path)
        raw_lines = [(p.extract_text() or "").splitlines() for p in reader.pages]
        cleaned_lines = drop_repeated_lines(raw_lines)
        for i, lines in enumerate(cleaned_lines, start=1):
            text = clean(" ".join(lines))
            if len(text.split()) >= MIN_CHUNK_WORDS:
                pages.append({"source": os.path.basename(path), "page": i, "text": text})
    return pages


def load_fallback(path):
    with open(path, "r", encoding="utf-8") as f:
        return [{"source": os.path.basename(path), "page": None, "text": clean(f.read())}]


def chunk_text(text, chunk_words=CHUNK_WORDS, overlap=CHUNK_OVERLAP):
    words = text.split()
    chunks, start = [], 0
    while start < len(words):
        end = start + chunk_words
        piece = " ".join(words[start:end])
        if len(piece.split()) >= MIN_CHUNK_WORDS or not chunks:
            chunks.append(piece)
        if end >= len(words):
            break
        start = end - overlap
    return chunks


def build_chunks(pages):
    out, dropped = [], 0
    for p in pages:
        for piece in chunk_text(p["text"]):
            if alpha_ratio(piece) < MIN_ALPHA_RATIO:  
                print("DROPPED p.%s: %s" % (p["page"], piece[:120]))   
                dropped += 1
                continue
            out.append({"source": p["source"], "page": p["page"], "text": piece})
    build_chunks.last_dropped = dropped
    return out


def build_index(raw_dir=None, fallback=None, data_dir=None):
    raw_dir = raw_dir or config.RAW_DIR
    fallback = fallback or config.FALLBACK_FILE
    data_dir = data_dir or config.DATA_DIR

    pages = load_pdf_pages(raw_dir)
    using_placeholder = not pages
    if using_placeholder:
        print(f"WARNING: no PDFs in {raw_dir}. Using placeholder text - add the real PDFs.")
        pages = load_fallback(fallback)
    else:
        print(f"Read {len(pages)} pages from {len({p['source'] for p in pages})} PDF(s).")

    chunks = build_chunks(pages)
    if not chunks:
        raise RuntimeError("No text could be extracted. Scanned PDFs need OCR first.")

    per_source = {}
    for c in chunks:
        per_source[c["source"]] = per_source.get(c["source"], 0) + 1
    print(f"{len(chunks)} chunks kept ({build_chunks.last_dropped} number-heavy chunks dropped):")
    for src, n in sorted(per_source.items()):
        print(f"   {n:4d}  {src}")
    print(f"Embedding with {embedder.identity()} ...")
    vecs = embedder.get_embedder().encode([c["text"] for c in chunks])

    os.makedirs(data_dir, exist_ok=True)
    with open(os.path.join(data_dir, "chunks.json"), "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=1)
    np.save(os.path.join(data_dir, "embeddings.npy"), vecs)
    with open(os.path.join(data_dir, "index_meta.json"), "w", encoding="utf-8") as f:
        json.dump(
            {"embedder": embedder.identity(), "chunks": len(chunks), "placeholder": using_placeholder,
             "per_source": per_source},
            f,
        )
    print("Index saved.")
    return len(chunks)


if __name__ == "__main__":
    build_index()
