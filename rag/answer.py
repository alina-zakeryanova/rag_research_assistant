"""Write an answer from retrieved passages, and flag citations that were not retrieved."""

from __future__ import annotations

import re
from dataclasses import dataclass

from rag.config import OPENAI_API_KEY, writer_name
from rag.ollama_client import chat, installed_models, model_ready
from rag.hits import Hit

SYSTEM_PROMPT = """You are the research assistant for a personal library of scientific papers.
The only papers you may name are the ones in the Source lines.
Answer only from those passages.
After each claim, cite that Source line as [filename.pdf, p. 4].
Never present a bibliography entry, DOI, or author-year citation from inside a passage as if it were one of the library papers.
If the passages do not contain the answer, say: I cannot find this in the indexed papers.
Do not invent measurements, methods, or datasets.
Write a short answer. Use a list when the question asks for several items."""

SUMMARY_PROMPT = """You keep reading notes on scientific papers.
Summarize the paper from the passages. Cover, in order: the question, the method, the data, and the result.
Cite pages as [filename.pdf, p. 4]. Leave out a section if the passages do not support it.
Do not invent measurements or datasets."""

COMPARE_PROMPT = """You compare scientific papers for a personal library.
Write one short paragraph per paper about how it treats the topic.
Start each paragraph with the file name. Use only the passages.
If a paper's passages do not address the topic, skip that paper."""

_PDF_NAME = re.compile(r"\S+\.pdf", re.IGNORECASE)


@dataclass
class Answer:
    text: str
    prompt: str
    model_used: bool
    note: str


def build_prompt(question: str, hits: list[Hit]) -> str:
    blocks = [f"Source: {hit.source}, page {hit.page}\n{hit.text}" for hit in hits]
    passages = "\n\n".join(blocks)
    return (
        "Passages from the indexed papers:\n\n"
        f"{passages}\n\n"
        f"Question: {question}\n\n"
        "Answer from the passages above."
    )


def citation_warnings(text: str, hits: list[Hit]) -> list[str]:
    allowed = {hit.source.lower() for hit in hits}
    warnings = []
    seen = set()
    for raw in _PDF_NAME.findall(text):
        name = raw.strip(".,;:)][")
        key = name.lower()
        if key in seen or key in allowed:
            continue
        seen.add(key)
        warnings.append(f"{name} is cited, and it was not in the retrieved passages.")
    return warnings


def generate(
    question: str,
    hits: list[Hit],
    model: str = "",
    system: str = SYSTEM_PROMPT,
) -> Answer:
    prompt = build_prompt(question, hits)
    if not hits:
        return Answer(
            text="Nothing in the library matched that.",
            prompt=prompt,
            model_used=False,
            note="",
        )

    if not OPENAI_API_KEY:
        names = installed_models()
        if names is None:
            return Answer(
                text="",
                prompt=prompt,
                model_used=False,
                note="Ollama is not running. Open the Ollama app, or run `ollama serve` in another terminal.",
            )
        if not model_ready(names, model or writer_name()):
            installed = ", ".join(names) if names else "none"
            wanted = model or writer_name()
            return Answer(
                text="",
                prompt=prompt,
                model_used=False,
                note=(
                    f"Ollama does not have `{wanted}` yet (installed: {installed}). "
                    f"Download it with: ollama pull {wanted}"
                ),
            )

    try:
        text = chat(model or writer_name(), system, prompt)
    except RuntimeError as exc:
        return Answer(text="", prompt=prompt, model_used=False, note=str(exc))
    return Answer(text=text, prompt=prompt, model_used=True, note="")
