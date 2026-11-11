"""Every setting for this project lives here.

Change these numbers, rebuild the index, and ask the same question again.
That is the fastest way to see how a RAG behaves.
"""

from pathlib import Path

# Folder that contains this project (the parent of the rag/ package).
ROOT = Path(__file__).resolve().parent.parent

# The papers you added. The folder name is spelled "reseach_papers" on disk.
PAPERS_DIR = ROOT / "reseach_papers"

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

# Ollama runs the language model in a separate process, which is the
# comfortable setup on a 2020 MacBook Air (this machine has 16GB).
# qwen3:1.7b is about 1.4GB, already downloaded on this Mac, and answers quickly.
# After the pipeline makes sense you can try a stronger model:
#   ollama pull llama3.1:8b
# and set OLLAMA_MODEL = "llama3.1:8b"
OLLAMA_MODEL = "qwen3:1.7b"
OLLAMA_URL = "http://127.0.0.1:11434"
