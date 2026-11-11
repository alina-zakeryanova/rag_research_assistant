"""Run this file from PyCharm, or from the project folder in a terminal.

    python main.py lesson
    python main.py ingest
    python main.py
    python main.py "What methods detect olive oil adulteration?"

`lesson` shows the mechanics on a single page.
`ingest` reads every PDF and builds the search index.
A question searches that index and asks the local model to answer.
"""

from __future__ import annotations

import sys


def main() -> None:
    args = sys.argv[1:]
    command = args[0].lower() if args else ""

    if command == "lesson":
        from rag.lesson import run

        run()
        return

    if command == "ingest":
        from rag.ingest import build

        build()
        return

    from rag.ask import ask_once, chat_loop
    from rag.index import indexed_count

    if indexed_count() == 0:
        print("No index yet. Building one from the PDFs first.")
        print("This is the ingest step. Later runs skip it.\n")
        from rag.ingest import build

        build()

    question = " ".join(args).strip()
    if question:
        ask_once(question)
        return
    chat_loop()


if __name__ == "__main__":
    main()
