"""Research library over the PDFs in research_papers.

    python main.py
    python main.py serve
    python main.py ingest
    python main.py papers
    python main.py "What methods detect olive oil adulteration?"
"""

from __future__ import annotations

import sys


def main() -> None:
    args = sys.argv[1:]
    command = args[0].lower() if args else "chat"

    if command in {"help", "-h", "--help"}:
        print(__doc__.strip())
        return

    if command == "ingest":
        from rag.ingest import build

        build()
        return

    if command == "serve":
        from rag.server import serve

        serve()
        return

    from rag.index import indexed_count

    if indexed_count() == 0:
        print("No index yet. Building one from the PDFs.\n")
        from rag.ingest import build

        build()

    if command == "papers":
        from rag.ask import print_papers

        print_papers()
        return

    if command == "summarize":
        from rag.ask import print_exchange
        from rag.assistant import summarize_paper

        if len(args) < 2:
            raise SystemExit("Usage: python main.py summarize <part of a file name>")
        print_exchange(summarize_paper(" ".join(args[1:])))
        return

    if command == "compare":
        from rag.ask import print_exchange
        from rag.assistant import compare_topic

        if len(args) < 2:
            raise SystemExit("Usage: python main.py compare <topic>")
        print_exchange(compare_topic(" ".join(args[1:])))
        return

    if command == "chat":
        args = args[1:]

    from rag.ask import ask_once, chat_loop

    question = " ".join(args).strip()
    if question:
        ask_once(question)
        return
    chat_loop()


if __name__ == "__main__":
    main()
