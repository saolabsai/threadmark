"""Whole-event ingestion and memory-guided original-source selection."""

from __future__ import annotations

import ast
import hashlib
import inspect
import os
from pathlib import Path
import tempfile
import textwrap
from typing import Any, Callable
from urllib.request import urlopen

from .config import Settings
from .schema import canonical

TokenCounter = Callable[[str], int]
MAX_TOKENIZER_BYTES = 16_000_000


def _encoding_metadata() -> tuple[str, str]:
    """Read the official asset location and digest from installed tiktoken code."""
    from tiktoken_ext.openai_public import o200k_base

    tree = ast.parse(textwrap.dedent(inspect.getsource(o200k_base)))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "load_tiktoken_bpe"
    ]
    try:
        if len(calls) != 1 or not calls[0].args:
            raise ValueError("Unsupported o200k_base constructor")
        url = ast.literal_eval(calls[0].args[0])
        expected_hash = ast.literal_eval(
            next(
                keyword.value
                for keyword in calls[0].keywords
                if keyword.arg == "expected_hash"
            )
        )
    except (ValueError, StopIteration) as exc:
        raise ValueError(
            "Cannot verify the installed o200k_base asset metadata"
        ) from exc
    if (
        not isinstance(url, str)
        or not url.startswith("https://openaipublic.blob.core.windows.net/encodings/")
        or not isinstance(expected_hash, str)
        or len(expected_hash) != 64
        or any(character not in "0123456789abcdef" for character in expected_hash)
    ):
        raise ValueError("Unsupported o200k_base asset metadata")
    return url, expected_hash


def _cache_path(url: str) -> Path:
    if "TIKTOKEN_CACHE_DIR" in os.environ:
        directory = os.environ["TIKTOKEN_CACHE_DIR"]
    elif "DATA_GYM_CACHE_DIR" in os.environ:
        directory = os.environ["DATA_GYM_CACHE_DIR"]
    else:
        directory = os.path.join(tempfile.gettempdir(), "data-gym-cache")
    if not directory:
        raise ValueError(
            "Tokenizer caching is disabled; set TIKTOKEN_CACHE_DIR before running threadmark setup-tokenizer"
        )
    return Path(directory) / hashlib.sha1(url.encode()).hexdigest()


def _verified_asset(path: Path, expected_hash: str) -> bytes:
    try:
        if path.stat().st_size > MAX_TOKENIZER_BYTES:
            raise ValueError("Tokenizer asset is oversized")
        content = path.read_bytes()
    except OSError as exc:
        raise ValueError(
            "Tokenizer cache is unavailable; run threadmark setup-tokenizer explicitly"
        ) from exc
    if hashlib.sha256(content).hexdigest() != expected_hash:
        raise ValueError(
            "Tokenizer cache failed SHA-256 verification; run threadmark setup-tokenizer explicitly"
        )
    return content


def setup_tokenizer() -> dict[str, str]:
    """Explicitly download and verify the tokenizer asset; no model API is used.

    Default token counting never initiates this setup. This method opts in to a
    network request to the official tokenizer asset URL from installed tiktoken.
    """
    url, expected_hash = _encoding_metadata()
    path = _cache_path(url)
    status = "ready"
    try:
        _verified_asset(path, expected_hash)
    except ValueError:
        with urlopen(url, timeout=30) as response:
            content = response.read(MAX_TOKENIZER_BYTES + 1)
        if (
            len(content) > MAX_TOKENIZER_BYTES
            or hashlib.sha256(content).hexdigest() != expected_hash
        ):
            raise ValueError(
                "Downloaded tokenizer asset failed size or SHA-256 verification"
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=path.parent, prefix=path.name + ".", delete=False
            ) as handle:
                temporary_path = Path(handle.name)
                handle.write(content)
            os.replace(temporary_path, path)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
        status = "downloaded"
    return {
        "status": status,
        "encoding": "o200k_base",
        "cache_path": str(path),
        "sha256": expected_hash,
    }


def token_counter() -> TokenCounter:
    """Load the evaluated tokenizer from a verified local cache.

    Missing or corrupt assets fail before tiktoken can download them. Run the
    separate ``setup_tokenizer`` operation to authorize that asset download.
    """
    import tiktoken

    url, expected_hash = _encoding_metadata()
    _verified_asset(_cache_path(url), expected_hash)
    encoding = tiktoken.get_encoding("o200k_base")
    return lambda text: len(encoding.encode(text, disallowed_special=()))


def _tokens(text: str, count_tokens: TokenCounter) -> int:
    count = count_tokens(text)
    if type(count) is not int or count < 0:
        raise ValueError("Token counter must return a nonnegative integer")
    return count


def chunks_for(
    events: list[dict[str, str]], count_tokens: TokenCounter, settings: Settings
) -> list[list[dict[str, str]]]:
    """Keep chronological events intact; reject an event too large to ingest."""
    chunks: list[list[dict[str, str]]] = []
    current: list[dict[str, str]] = []
    for event in events:
        if (
            _tokens(canonical({"events": [event]}), count_tokens)
            > settings.chunk_tokens
        ):
            raise ValueError(
                "One event exceeds the chunk budget; events are never truncated or split"
            )
        trial = current + [event]
        if (
            current
            and _tokens(canonical({"events": trial}), count_tokens)
            > settings.chunk_tokens
        ):
            chunks.append(current)
            current = [event]
        else:
            current = trial
    if current:
        chunks.append(current)
    return chunks


def select_source_events(
    events: list[dict[str, str]],
    query: dict[str, Any],
    state: dict[str, Any],
    count_tokens: TokenCounter,
    settings: Settings,
) -> list[dict[str, str]]:
    """Select a recent whole-event suffix plus citations for requested facts.

    State is an index into the retained archive. Predicted fact values are never
    copied into this packet. Selected events remain in original archive order.
    """
    recent: list[dict[str, str]] = []
    for event in reversed(events):
        trial = [event, *recent]
        if (
            _tokens(canonical({"source_events": trial}), count_tokens)
            > settings.recent_source_tokens
        ):
            break
        recent = trial
    requested = {
        (item["entity_id"], item["field"]) for item in query["requested_facts"]
    }
    selected_ids = {event["event_id"] for event in recent}
    selected_ids.update(
        identifier
        for fact in state["facts"]
        if (fact["entity_id"], fact["field"]) in requested
        for identifier in fact["evidence_event_ids"]
    )
    if not selected_ids <= {event["event_id"] for event in events}:
        raise ValueError("Memory cites a source outside the original event archive")
    selected = [event for event in events if event["event_id"] in selected_ids]
    if (
        len(canonical({"source_events": selected}).encode("utf-8"))
        > settings.max_source_bytes
    ):
        raise ValueError("Selected original sources exceed the bounded byte size")
    return selected
