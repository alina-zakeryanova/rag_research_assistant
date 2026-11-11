"""Ask the index a question and print the RAG trace.

The trace is the lesson. Above every answer you see the passages that were
retrieved. The model was allowed to use those passages and no others.
"""

from __future__ import annotations

import argparse
import textwrap

from rag.answer import generate
from rag.config import OLLAMA_MODEL, TOP_K
from rag.index import indexed_count
from rag.retrieve import Hit, retrieve


EXAMPLES = (
    "What methods are used to detect olive oil adulteration?",
    "How is hyperspectral imaging used in these papers?",
    "Which machine learning models are used for classification?",
)


def _preview(text: str, limit: int = 280) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + "..."


def print_hits(hits: list[Hit]) -> None:
    print()
    print("Retrieved passages")
    print("These are the only pages the language model will see.")
    if not hits:
        print("  none")
        return
    for number, hit in enumerate(hits, start=1):
        print(f"  {number}. similarity {hit.score:.2f}   {hit.source}   p. {hit.page}")
        wrapped = textwrap.fill(
            _preview(hit.text),
            width=88,
            initial_indent="     ",
            subsequent_indent="     ",
        )
        print(wrapped)


def ask_once(question: str, top_k: int = TOP_K, show_prompt: bool = False, use_model: bool = True) -> None:
    print()
    print("Question")
    print(textwrap.fill(question, width=88, initial_indent="  ", subsequent_indent="  "))

    hits = retrieve(question, top_k=top_k)
    print_hits(hits)

    if not use_model:
        print()
        print("Stopped before generation (--no-llm). Retrieval is the first half of RAG.")
        return

    print()
    print(f"Asking {OLLAMA_MODEL}...")
    result = generate(question, hits)
    if show_prompt:
        print()
        print("Prompt sent to the model")
        print(result.prompt)
    if result.note:
        print()
        print(result.note)
    if result.text:
        print()
        print("Answer")
        for line in result.text.splitlines():
            print(f"  {line}" if line else "")
        print()
        print("Check the passages above before trusting a citation. The model can only know what they contain.")


def chat_loop(top_k: int = TOP_K) -> None:
    count = indexed_count()
    print(f"Index: {count} passages. Model: {OLLAMA_MODEL}.")
    print("Example questions:")
    for example in EXAMPLES:
        print(f"  {example}")
    print("Type quit to exit.")

    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not question:
            continue
        if question.lower() in {"quit", "exit", "q"}:
            return
        try:
            ask_once(question, top_k=top_k)
        except Exception as exc:
            print(f"\n{exc}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Ask the research-paper index a question.")
    parser.add_argument("question", nargs="*", help="Question. Omit it to start a prompt.")
    parser.add_argument("-k", "--top-k", type=int, default=TOP_K, help="Passages to retrieve.")
    parser.add_argument("--prompt", action="store_true", help="Print the prompt sent to the model.")
    parser.add_argument("--no-llm", action="store_true", help="Retrieve passages and do not generate.")
    args = parser.parse_args(argv)

    if indexed_count() == 0:
        raise SystemExit("No index yet. From the project folder run: python main.py ingest")

    question = " ".join(args.question).strip()
    if question:
        ask_once(question, top_k=args.top_k, show_prompt=args.prompt, use_model=not args.no_llm)
        return
    chat_loop(top_k=args.top_k)


if __name__ == "__main__":
    main()
