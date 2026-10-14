"""Central configuration. Everything tunable lives here or in .env."""

import os

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- LLM -------------------------------------------------------------------
# Comma-separated list, tried in order. If Groq retires a model (it happened
# to llama-3.3-70b-versatile), the next one is used automatically.
GROQ_MODELS = [
    m.strip()
    for m in os.getenv("GROQ_MODEL", "openai/gpt-oss-120b,openai/gpt-oss-20b").split(",")
    if m.strip()
]
LLM_BACKEND = os.getenv("EWASTE_LLM", "groq")  # "groq" | "fake" (tests only)
LLM_MAX_TOKENS = int(os.getenv("EWASTE_MAX_TOKENS", "3000"))  # reasoning tokens count too

# --- Retrieval ---------------------------------------------------------------
EMBED_BACKEND = os.getenv("EWASTE_EMBEDDER", "sentence-transformers")  # or "hash" (tests only)
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 4
# Below this best-chunk similarity the app refuses to claim the rules cover the
# device. Calibrate with `python eval.py` - the right value depends on your PDFs.
MIN_SCORE = float(os.getenv("EWASTE_MIN_SCORE", "0.25"))
# Stricter bar when the query names no recognisable device ("how to bake bread").
# Real e-waste questions almost always name a device; vague ones must score higher.
MIN_SCORE_NO_DEVICE = float(os.getenv("EWASTE_MIN_SCORE_NO_DEVICE", "0.40"))
RRF_K = 60  # reciprocal-rank-fusion constant for hybrid search

# --- Agent -------------------------------------------------------------------
MAX_AGENT_STEPS = 4

# --- Vision ------------------------------------------------------------------
YOLO_WEIGHTS = os.getenv("EWASTE_YOLO_WEIGHTS", "yolov8n.pt")
VISION_CONF = float(os.getenv("EWASTE_VISION_CONF", "0.35"))
MAX_UPLOAD_MB = 5
MAX_IMAGE_PIXELS = 25_000_000

# --- Paths -------------------------------------------------------------------
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
CHUNKS_PATH = os.path.join(DATA_DIR, "chunks.json")
EMBEDDINGS_PATH = os.path.join(DATA_DIR, "embeddings.npy")
INDEX_META_PATH = os.path.join(DATA_DIR, "index_meta.json")
DB_PATH = os.path.join(DATA_DIR, "history.db")
RECYCLERS_CSV = os.path.join(DATA_DIR, "recyclers.csv")
FALLBACK_FILE = os.path.join(BASE_DIR, "sample_data", "sample_rules.txt")

LANGUAGES = ["English", "Hindi", "Odia"]
