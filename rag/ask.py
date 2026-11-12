"""Terminal interface for the library."""

from __future__ import annotations

import argparse
import textwrap

from rag.assistant import Exchange, answer_question
from rag.config import OLLAMA_MODEL, TOP_K
from rag.index import indexed_count
from rag.library import list_papers


def _preview(text: str, limit: int = 280) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + "..."


def print_papers() -> None:
    papers = list_papers()
    print(f"{len(papers)} papers")
    for paper in papers:
        print(f"  {paper['title']}")
        print(f"    {paper['source']}   {paper['pages']} pages   {paper['chunks']} passages")


def print_exchange(exchange: Exchange) -> None:
    if exchange.search_query and exchange.search_query not in {exchange.question, exchange.question.removeprefix("Compare ")}:
        if exchange.kind == "answer" and exchange.search_query != exchange.question:
            print()
            print(f"Searched: {exchange.search_query}")

    print()
    print("Sources")
    if not exchange.hits:
        print("  none")
    for number, hit in enumerate(exchange.hits, start=1):
        score = f"{hit.score:.2f}  " if hit.score else "     "
        print(f"  {number}. {score}{hit.source}   p. {hit.page}")
        wrapped = textwrap.fill(
            _preview(hit.text),
            width=88,
            initial_indent="     ",
            subsequent_indent="     ",
        )
        print(wrapped)

    if exchange.note:
        print()
        print(exchange.note)
    if exchange.answer:
        print()
        print("Answer" if exchange.kind == "answer" else exchange.kind.capitalize())
        for line in exchange.answer.splitlines():
            print(f"  {line}" if line else "")
    for warning in exchange.warnings:
        print(f"  ! {warning}")


def ask_once(question: str, history: list[dict] | None = None, top_k: int = TOP_K) -> Exchange:
    exchange = answer_question(question, history=history, top_k=top_k)
    print_exchange(exchange)
    return exchange


def chat_loop(top_k: int = TOP_K) -> None:
    papers = list_papers()
    print(f"{len(papers)} papers, {indexed_count()} passages, model {OLLAMA_MODEL}.")
    print("Ask a question, or:")
    print("  summarize <part of a file name>")
    print("  compare <topic>")
    print("Type quit to exit.")

    history: list[dict] = []
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
            exchange = ask_once(question, history=history, top_k=top_k)
        except Exception as exc:
            print(f"\n{exc}")
            continue
        if exchange.answer:
            history.append({"role": "user", "content": question})
            history.append({"role": "assistant", "content": exchange.answer})
            history = history[-8:]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Ask the research library a question.")
    parser.add_argument("question", nargs="*", help="Question. Omit it to start a conversation.")
    parser.add_argument("-k", "--top-k", type=int, default=TOP_K, help="Passages to retrieve.")
    args = parser.parse_args(argv)

    if indexed_count() == 0:
        raise SystemExit("No index yet. From the project folder run: python main.py ingest")

    question = " ".join(args.question).strip()
    if question:
        ask_once(question, top_k=args.top_k)
        return
    chat_loop(top_k=args.top_k)


if __name__ == "__main__":
    main()
