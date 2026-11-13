"""Settings for the local research library."""

import os
from pathlib import Path

# Folder that contains this project (the parent of the rag/ package).
ROOT = Path(__file__).resolve().parent.parent


def _load_env() -> None:
    path = ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_env()

# The papers you added.
PAPERS_DIR = ROOT / "research_papers"

# The index is two files you can inspect:
#   data/index/chunks.jsonl     the passages, one per line
#   data/index/embeddings.npy   one vector per line of the jsonl file
INDEX_DIR = ROOT / "data" / "index"

# all-minilm turns text into 384 numbers. It is a small Ollama model,
# about 45MB, and runs on the M1 CPU.
# The papers and the question must be embedded with this same model.
EMBEDDING_MODEL = "all-minilm"

# How the papers are cut before they are embedded.
# Size and overlap are in characters (about 150-200 words per chunk).
# Smaller chunks search more precisely and lose surrounding context.
# Larger chunks keep context and get harder to match to a short question.
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# How many passages are given to the language model for one question.
TOP_K = 5
# At most this many passages from the same PDF, so one introduction cannot fill the list.
MAX_PER_PAPER = 2
# Blend of vector similarity and shared words. Higher means names like "Raman" matter more.
LEXICAL_WEIGHT = 0.35
# Page 1 is usually the title, authors, and a sentence that repeats the abstract.
PAGE_ONE_PENALTY = 0.15

NUM_CTX = 8192
# Fuse semantic search with BM25, then ask the local model to reorder the shortlist.
RERANK = True
RRF_K = 60
POOL = 12
WEB_HOST = "127.0.0.1"
WEB_PORT = 8000

# Ollama runs the language model in a separate process, which is the
# comfortable setup on a 2020 MacBook Air (this machine has 16GB).
# qwen3:1.7b is about 1.4GB, already downloaded on this Mac, and answers quickly.
# After the pipeline makes sense you can try a stronger model:
#   ollama pull llama3.1:8b
# and set OLLAMA_MODEL = "llama3.1:8b"
OLLAMA_MODEL = "llama3.1:8b"
OLLAMA_URL = "http://127.0.0.1:11434"

# Writing and reranking go to OpenAI when this is set. Embeddings stay on Ollama.
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_URL = "https://api.openai.com/v1/chat/completions"


def writer_name() -> str:
    if OPENAI_API_KEY:
        return OPENAI_MODEL
    return OLLAMA_MODEL
