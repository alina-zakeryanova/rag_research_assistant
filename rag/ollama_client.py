"""HTTP client for the local Ollama server."""

from __future__ import annotations

import json
import ssl
import urllib.error
import urllib.request

import certifi

from rag.config import NUM_CTX, OLLAMA_URL, OPENAI_API_KEY, OPENAI_MODEL, OPENAI_URL

_SSL = ssl.create_default_context(cafile=certifi.where())


def installed_models() -> list[str] | None:
    """Model names Ollama has downloaded, or None if it is not running."""
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=3) as response:
            payload = json.loads(response.read().decode())
    except (urllib.error.URLError, TimeoutError, OSError):
        return None
    return [item.get("name", "") for item in payload.get("models", [])]


def model_ready(names: list[str], wanted: str) -> bool:
    for name in names:
        if name == wanted or name == f"{wanted}:latest":
            return True
        # "all-minilm" matches the tag "all-minilm:latest".
        if ":" not in wanted and name.split(":")[0] == wanted:
            return True
    return False


def require_model(wanted: str) -> None:
    names = installed_models()
    if names is None:
        raise SystemExit(
            "Ollama is not running. Open the Ollama app, or run `ollama serve` "
            "in another terminal."
        )
    if not model_ready(names, wanted):
        installed = ", ".join(names) if names else "none"
        raise SystemExit(
            f"Ollama does not have `{wanted}` yet (installed: {installed}).\n"
            f"Download it with: ollama pull {wanted}"
        )


def embed(texts: list[str], model: str) -> list[list[float]]:
    body = json.dumps({"model": model, "input": texts}).encode()
    request = urllib.request.Request(
        f"{OLLAMA_URL}/api/embed",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            payload = json.loads(response.read().decode())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        raise RuntimeError(f"Ollama embeddings failed ({exc.code}): {detail}") from exc
    vectors = payload.get("embeddings")
    if not vectors or len(vectors) != len(texts):
        raise RuntimeError("Ollama returned an unexpected embedding payload.")
    return vectors


def chat(model: str, system: str, user: str) -> str:
    if OPENAI_API_KEY:
        return _openai_chat(system, user)
    body = json.dumps(
        {
            "model": model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "think": False,
            "keep_alive": "10m",
            "options": {"temperature": 0.2, "num_ctx": NUM_CTX},
        }
    ).encode()
    request = urllib.request.Request(
        f"{OLLAMA_URL}/api/chat",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            payload = json.loads(response.read().decode())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        raise RuntimeError(f"Ollama chat failed ({exc.code}): {detail}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(f"Ollama did not return an answer ({exc}).") from exc
    return payload.get("message", {}).get("content", "").strip()


def _openai_chat(system: str, user: str) -> str:
    body = json.dumps(
        {
            "model": OPENAI_MODEL,
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
    ).encode()
    request = urllib.request.Request(
        OPENAI_URL,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAI_API_KEY}",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=60, context=_SSL) as response:
            payload = json.loads(response.read().decode())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        raise RuntimeError(f"OpenAI chat failed ({exc.code}): {detail}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(f"OpenAI did not return an answer ({exc}).") from exc
    choices = payload.get("choices") or []
    if not choices:
        raise RuntimeError("OpenAI returned no choices.")
    return choices[0].get("message", {}).get("content", "").strip()
