# Research library

A local assistant for the PDFs in `research_papers`. Search, follow-up questions, paper summaries, and comparisons run on this Mac through Ollama. No API key.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
ollama pull all-minilm
ollama pull qwen3:1.7b
python main.py ingest
```

In PyCharm, point the interpreter at `.venv/bin/python` and run `main.py`.

## Use

```bash
python main.py serve
python main.py
python main.py papers
python main.py summarize Raman
python main.py compare hyperspectral imaging and Raman
python main.py "What methods detect olive oil adulteration?"
```

`serve` opens the library at http://127.0.0.1:8000. The sidebar lists every paper. Click one for a summary. Questions in the box search the library, and a follow-up stays on the previous topic.

In the terminal, the same commands work inside the prompt: `summarize <name>` and `compare <topic>`.

Retrieval mixes meaning with the actual words in your question, skips letting a title page crowd out the rest of a paper, and keeps at most two passages from each PDF. If the model names a file that was not retrieved, that citation is flagged under the answer.

## Rebuild the index

```bash
python main.py ingest
```

Run that again after adding or removing PDFs. The index is `data/index/`.

## Settings

`rag/config.py` holds the chunk size, how many passages to return, and the Ollama model names. A stronger writer is a one-line change after `ollama pull llama3.1:8b`.
