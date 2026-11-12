"""Local web app for the library. Bind is localhost only."""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from rag.assistant import Exchange, answer_question, compare_topic, summarize_paper
from rag.config import WEB_HOST, WEB_PORT
from rag.index import indexed_count
from rag.library import display_name, list_papers

_STATIC = Path(__file__).resolve().parent / "static" / "index.html"


def _preview(text: str, limit: int = 320) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + "..."


def _payload(exchange: Exchange) -> dict:
    return {
        "question": exchange.question,
        "searchQuery": exchange.search_query,
        "answer": exchange.answer,
        "note": exchange.note,
        "kind": exchange.kind,
        "warnings": exchange.warnings,
        "sources": [
            {
                "source": hit.source,
                "title": display_name(hit.source),
                "page": hit.page,
                "score": round(hit.score, 2),
                "preview": _preview(hit.text),
            }
            for hit in exchange.hits
        ],
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args) -> None:
        return

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self._send(status, body, "application/json; charset=utf-8")

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if length > 200_000:
            raise ValueError("Request is too large.")
        raw = self.rfile.read(length) if length else b"{}"
        data = json.loads(raw.decode() or "{}")
        if not isinstance(data, dict):
            raise ValueError("Expected a JSON object.")
        return data

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path in {"/", "/index.html"}:
            if not _STATIC.exists():
                self._send(500, b"Missing interface file.", "text/plain; charset=utf-8")
                return
            self._send(200, _STATIC.read_bytes(), "text/html; charset=utf-8")
            return
        if path == "/api/papers":
            try:
                papers = list_papers()
            except RuntimeError as exc:
                self._json(400, {"error": str(exc)})
                return
            self._json(200, {"count": indexed_count(), "papers": papers})
            return
        self._json(404, {"error": "Not found."})

    def do_POST(self) -> None:
        path = self.path.split("?", 1)[0]
        try:
            data = self._read_json()
            if path == "/api/chat":
                message = str(data.get("message", "")).strip()
                if not message:
                    raise ValueError("Message is empty.")
                history = data.get("history") or []
                if not isinstance(history, list):
                    raise ValueError("History must be a list.")
                exchange = answer_question(message, history=history[-8:])
            elif path == "/api/summarize":
                exchange = summarize_paper(str(data.get("source", "")))
            elif path == "/api/compare":
                exchange = compare_topic(str(data.get("topic", "")))
            else:
                self._json(404, {"error": "Not found."})
                return
        except (ValueError, RuntimeError, json.JSONDecodeError) as exc:
            self._json(400, {"error": str(exc)})
            return
        except Exception as exc:
            self._json(500, {"error": str(exc)})
            return
        self._json(200, _payload(exchange))


def serve() -> None:
    if indexed_count() == 0:
        raise SystemExit("No index yet. Run: python main.py ingest")
    server = ThreadingHTTPServer((WEB_HOST, WEB_PORT), Handler)
    print(f"Library  http://{WEB_HOST}:{WEB_PORT}", flush=True)
    print("Ctrl-C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()
    finally:
        server.server_close()
