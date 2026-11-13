# Research library

A local assistant for the PDFs in `research_papers`. Search stays on this Mac. When `.env` contains `OPENAI_API_KEY`, writing and reranking use OpenAI.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
ollama pull all-minilm
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

Retrieval uses BM25 and semantic search together (reciprocal rank fusion), then the writer reranks the shortlist. With a key in `.env`, that writer is `gpt-4o-mini`. Title pages and bibliographies are kept out of that shortlist. If the model names a PDF that was not retrieved, the citation is flagged under the answer.

Set `RERANK = False` in `rag/config.py` if you want the faster fusion result without the extra model pass.

## Compare strategies

```bash
python main.py bench
python main.py mode hyde "Which paper uses Raman spectroscopy on olive oil?"
python main.py mode fusion "How is adulteration detected?"
python main.py mode crag "How is adulteration detected?"
python main.py mode raptor "How is hyperspectral imaging used?"
```

`bench` builds five chunk indexes under `data/compare/` and writes `data/compare/report.txt`. The first run embeds those indexes, so leave it for several minutes. It scores whether the expected paper lands in the top 3.

Modes you can pass to `mode`:

| Mode | What it does on this Mac |
| --- | --- |
| semantic | Vector search only |
| bm25 | Word search only |
| rrf | BM25 and vectors fused |
| rerank | Fusion, then the local model reorders the shortlist |
| fusion | Several rewritten queries, then fusion. This is RAG-Fusion |
| decompose | Splits the question, searches each part, fuses |
| stepback | Also searches a broader version of the question |
| hyde | Embeds a hypothetical answer paragraph, then fuses that with BM25 |
| route | Classifies the question, then picks fusion, rerank, or no search |
| crag | Drops the first results if the model says they do not answer, and searches once more |
| adaptive | Routes, then uses fusion for broad questions and CRAG for specific ones |
| raptor | Clusters the library, summarizes each cluster, and searches inside the nearest clusters |

Chunkers in the benchmark:

- `fixed` cuts every 1000 characters with no overlap
- `overlap` takes 5 sentences and keeps 1 in common with the next window
- `structure` splits on section headings such as Methods and Results
- `semantic` starts a new passage when the next sentence drifts away from the previous one
- `contextual` embeds the paper name and section in front of each structure passage

ColBERT is not included. It stores a vector per token and wants PyTorch plus a separate model. On a 16GB M1 Air that fights with Ollama for memory. The fusion-plus-rerank path is the practical substitute.

A full contextual-retrieval pass, where the language model writes a sentence for every passage, is also not the default. There are a few thousand passages, and that would be a few thousand model calls. The benchmark uses the paper name and section as the context line, which is the part that fits this machine. `raptor` is one level of clustering rather than a deep summary tree, for the same reason.

## Rebuild the index

```bash
python main.py ingest
```

Run that again after adding or removing PDFs. The index is `data/index/`.

## Settings

`rag/config.py` holds the chunk size, how many passages to return, and the Ollama model names. A stronger writer is a one-line change after `ollama pull llama3.1:8b`.
