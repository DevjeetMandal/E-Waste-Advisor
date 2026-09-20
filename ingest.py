

import json
import glob
import os

import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

RAW_DIR = "data/raw"
FALLBACK_FILE = "sample_data/sample_rules.txt"
CHUNKS_PATH = "data/chunks.json"
EMBEDDINGS_PATH = "data/embeddings.npy"

CHUNK_WORDS = 180
CHUNK_OVERLAP = 40

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"


def load_text_from_pdfs(raw_dir):
    """Extract raw text from every PDF in raw_dir, tagged by filename."""
    documents = []
    for path in sorted(glob.glob(os.path.join(raw_dir, "*.pdf"))):
        reader = PdfReader(path)
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        if text.strip():
            documents.append({"source": os.path.basename(path), "text": text})
    return documents


def load_fallback_text(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    return [{"source": os.path.basename(path), "text": text}]


def chunk_text(text, chunk_words=CHUNK_WORDS, overlap=CHUNK_OVERLAP):
    """Simple sliding-window word chunking."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_words
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        if end >= len(words):
            break
        start = end - overlap
    return chunks


def build_index():
    documents = load_text_from_pdfs(RAW_DIR)

    if documents:
        print(f"Found {len(documents)} PDF(s) in {RAW_DIR}/ — using these as the knowledge base.")
    else:
        print(
            f"No PDFs found in {RAW_DIR}/. Using placeholder demo text from "
            f"{FALLBACK_FILE}. Add the real official PDFs before your final "
            f"submission."
        )
        documents = load_fallback_text(FALLBACK_FILE)

    all_chunks = []
    for doc in documents:
        for chunk in chunk_text(doc["text"]):
            all_chunks.append({"source": doc["source"], "text": chunk})

    if not all_chunks:
        raise RuntimeError("No text could be extracted — check your source files.")

    print(f"Built {len(all_chunks)} chunks. Embedding with {EMBED_MODEL_NAME}...")
    model = SentenceTransformer(EMBED_MODEL_NAME)
    texts = [c["text"] for c in all_chunks]
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)

    os.makedirs("data", exist_ok=True)
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    np.save(EMBEDDINGS_PATH, np.array(embeddings, dtype=np.float32))

    print(f"Saved {len(all_chunks)} chunks to {CHUNKS_PATH}")
    print(f"Saved embeddings to {EMBEDDINGS_PATH}")


if __name__ == "__main__":
    build_index()
