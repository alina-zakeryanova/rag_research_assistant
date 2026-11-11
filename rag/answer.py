"""Step 6. Ask a language model to write the answer from the passages.

This is the "G" in RAG. The model does not see your PDFs. It sees only the
passages retrieve.py returned, pasted into a prompt. If those passages do
not contain the answer, the model is told to say so.

Ollama serves the model at localhost:11434. Install it once with:

    ollama pull qwen3:1.7b
"""

from __future__ import annotations

from dataclasses import dataclass

from rag.config import OLLAMA_MODEL
from rag.ollama_client import chat, installed_models, model_ready
from rag.retrieve import Hit


SYSTEM_PROMPT = """You are a research assistant for a personal library of scientific papers.
Answer only from the passages the user gives you.
Copy the source line as the citation, in this exact form: [filename.pdf, p. 4].
If the passages do not contain the answer, say: I cannot find this in the indexed papers.
Do not invent measurements, methods, datasets, or citations.
Write a short answer. Use a list only when the question asks for several items."""


@dataclass
class Answer:
    text: str
    prompt: str
    model_used: bool
    note: str


def build_prompt(question: str, hits: list[Hit]) -> str:
    blocks = []
    for hit in hits:
        blocks.append(f"Source: {hit.source}, page {hit.page}\n{hit.text}")
    passages = "\n\n".join(blocks)
    return (
        "Passages from the indexed papers:\n\n"
        f"{passages}\n\n"
        f"Question: {question}\n\n"
        "Answer from the passages above."
    )


def generate(question: str, hits: list[Hit], model: str = OLLAMA_MODEL) -> Answer:
    prompt = build_prompt(question, hits)
    if not hits:
        return Answer(
            text="No passages were retrieved, so there is nothing to answer from.",
            prompt=prompt,
            model_used=False,
            note="",
        )

    names = installed_models()
    if names is None:
        return Answer(
            text="",
            prompt=prompt,
            model_used=False,
            note=(
                "Ollama is not running, so this stop is retrieval only. "
                "The passages above are what the model would have seen. "
                "Start the Ollama app, or run `ollama serve` in another terminal."
            ),
        )
    if not model_ready(names, model):
        installed = ", ".join(names) if names else "none"
        return Answer(
            text="",
            prompt=prompt,
            model_used=False,
            note=(
                f"Ollama is running, and it does not have `{model}` yet. "
                f"Installed models: {installed}. "
                f"Download the one this project expects with: ollama pull {model}"
            ),
        )

    try:
        text = chat(model, SYSTEM_PROMPT, prompt)
    except RuntimeError as exc:
        return Answer(
            text="",
            prompt=prompt,
            model_used=False,
            note=f"{exc} The passages above are still valid.",
        )
    return Answer(text=text, prompt=prompt, model_used=True, note="")
