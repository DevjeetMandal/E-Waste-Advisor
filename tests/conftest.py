import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402

# Everything runs offline: fake LLM, hash embeddings, throwaway data dir.
_TMP = tempfile.mkdtemp(prefix="ewaste_test_")
config.EMBED_BACKEND = "hash"
config.LLM_BACKEND = "fake"
config.DATA_DIR = _TMP
config.DB_PATH = os.path.join(_TMP, "history.db")

import ingest  # noqa: E402

_EMPTY_RAW = tempfile.mkdtemp(prefix="ewaste_raw_")
ingest.build_index(raw_dir=_EMPTY_RAW, data_dir=_TMP)

import pytest  # noqa: E402
import rag_utils  # noqa: E402


@pytest.fixture(scope="session")
def index():
    rag_utils.reset_cache()
    return rag_utils.Index.load(_TMP)
