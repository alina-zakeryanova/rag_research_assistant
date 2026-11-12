# Research assistant RAG

A small retrieval-augmented generation project over the papers in `research_papers`. It runs locally on this MacBook Air M1 (16GB). No API key.

The code is the lesson. Each file in `rag/` is one step, and a question prints the passages it retrieved before it prints an answer.

## What RAG is

A language model has not read your PDFs. RAG gives it a chance to, in two moves:

1. **Retrieve.** Search the papers and keep the few passages that look relevant.
2. **Generate.** Hand only those passages to a language model and ask it to answer from them.

```
PDFs  ->  text  ->  chunks  ->  vectors  ->  local index
                                                 |
question  ->  vector  ->  nearest chunks  ->  model  ->  answer
```

You already do a version of this when you search a PDF, copy a paragraph, and paste it into a chat. This project makes that loop repeatable.

## Read the code in this order

| Step | File | What it does |
| --- | --- | --- |
| Settings | `rag/config.py` | Chunk size, how many passages, which models |
| 1. Load | `rag/load.py` | Read each PDF page into text |
| 2. Chunk | `rag/chunk.py` | Cut a page into overlapping passages |
| 3. Embed | `rag/embed.py` | Turn text into 384 numbers, via Ollama |
| 4. Index | `rag/index.py` | Save the passages and the numbers as files |
| 5. Retrieve | `rag/retrieve.py` | Find passages close to the question |
| 6. Answer | `rag/answer.py` | Ask Ollama to write from those passages |
| Models | `rag/ollama_client.py` | The HTTP calls to the local model server |
| Walkthrough | `rag/lesson.py` | The same ideas on a single page, no chat model |
| Run | `main.py` | The commands below |

## Setup

From the project folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

In PyCharm: Settings → Project → Python Interpreter → Add → Existing → `.venv/bin/python`, then run `main.py`.

The models run in [Ollama](https://ollama.com), which is already installed on this Mac. Open the Ollama app once so the server is up, then:

```bash
ollama pull all-minilm
ollama pull qwen3:1.7b
```

`all-minilm` is about 45MB. It only turns text into numbers. It does not write answers.

`qwen3:1.7b` is about 1.4GB. It writes the answer. It is small on purpose: replies come back in a few seconds, and you can read the retrieved passages while you still remember the question. This Mac already has it. When you want sharper answers, `ollama pull llama3.1:8b` and change `OLLAMA_MODEL` in `rag/config.py`.

If a command reports that Ollama is not running, open the Ollama app, or run `ollama serve` in a second terminal.

## Run it

```bash
source .venv/bin/activate

python main.py lesson
python main.py ingest
python main.py "What methods are used to detect olive oil adulteration?"
python main.py
```

- `lesson` uses one page of one paper. It shows the text, a chunk, and two similarity scores. Run this first.
- `ingest` indexes every PDF. The first run loads `all-minilm` into memory and then takes a few minutes. Running it again rebuilds the index from scratch. Open `data/index/chunks.jsonl` afterwards: each line is one passage the search can return.
- A question prints the retrieved passages, then the answer. Start the chat with no question to keep asking.
- `python -m rag.ask --no-llm "your question"` shows retrieval only.
- `python -m rag.ask --prompt "your question"` also prints the exact prompt the model received.

Scores are cosine similarity. For this embedding model, a useful match is often around 0.30 to 0.70. A score near 1.0 would mean the chunk is almost the same text as the question, which is rare.

## What you are looking at in an answer

Above the answer, each line is a passage: a similarity score, a file name, and a page. That list is the retrieval. The model is not allowed to see the rest of the paper. If a citation looks wrong, check it against that list. The list is the ground truth of what entered the prompt.

## Three experiments

Do these after the first index exists. They teach more than reading further.

**1. Retrieval only.** Run the same question with `--no-llm`. Read the five passages and decide whether a person could answer from them. If the passages are wrong, generation cannot save it. Fix retrieval first.

**2. How many passages.** In `rag/config.py`, `TOP_K` is 5. Ask a question, then run it again with `-k 2` and with `-k 8`:

```bash
python -m rag.ask -k 2 "How is hyperspectral imaging used in these papers?"
python -m rag.ask -k 8 "How is hyperspectral imaging used in these papers?"
```

Two passages give a narrower answer. Eight passages give more coverage and more chances for the model to get distracted.

**3. Chunk size.** Set `CHUNK_SIZE` to `300`, run `python main.py ingest`, and ask the same question. Then set it to `2500`, ingest again, and ask again. Short chunks match specific phrases and lose the surrounding method. Long chunks keep context and are harder to match to a short question. Put the value back to `1000` when you are done. Ingest again so the index matches the file.

## When an answer is bad

The failure is usually in a specific step:

- **A paper was skipped at ingest.** The indexer prints `skipped ... (no extractable text)` for scans and image PDFs. This project never sees those pages. The fix is OCR, which is a separate tool in front of `load.py`.
- **The right paper is missing from the retrieved list.** The embedder did not place that passage near the question. Rephrase the question with words the papers use (method names, "adulteration", "hyperspectral"), or raise `TOP_K`.
- **The right passage is there and the answer still ignores it.** That is the generator. A 3B model does this. The passages are still on screen, so you can read them. A larger Ollama model follows the prompt more carefully.
- **The question is not in these papers.** The prompt tells the model to say it cannot find it. If it answers anyway, trust the passage list.

## What this project leaves out

Tutorials often install LangChain and a vector database at this point. LangChain wraps the six steps above. Chroma, FAISS, and Pinecone run the same nearest-neighbour search as `retrieve.py`, which matters once the library is too big to multiply in NumPy. Thirty papers are not that big, so the search stays visible.

The pieces people add next, still around the same six steps: OCR, a PDF parser that respects two-column papers, hybrid search (vectors plus keywords), a reranker, and a list of questions whose answers you already know, so you can tell whether a change helped.

## Layout

```
main.py              commands
rag/                 one file per step
research_papers/     your PDFs
data/index/          chunks.jsonl and embeddings.npy, recreated by ingest
```
