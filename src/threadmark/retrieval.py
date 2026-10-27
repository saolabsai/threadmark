"""Deterministic lexical retrieval baseline over original workflow events.

This is event-level BM25 with a recent-event window. It uses keyword overlap;
it does not implement semantic/vector retrieval or interpret workflow changes.
No model, memory values, benchmark labels, or expected answers are consulted.
"""

from __future__ import annotations

from collections import Counter
import math
import re
from typing import Any

from .config import Settings
from .context import TokenCounter
from .schema import (
    canonical,
    catalog_keys,
    public_catalog,
    public_events,
    public_query,
)

# Fixed language preprocessing, independent of benchmark labels and domains.
# Underscores are separators so a public field such as start_at is searchable.
_WORDS = re.compile(r"[^\W_]+", re.UNICODE)
_STOP_WORDS = frozenset(
    "a an and are as at be been but by can could did do does for from had has "
    "have how i if in into is it its may of on or our should than that the "
    "their them then there these they this those to was we were what when "
    "where which who will with would you your".split()
)
_K1 = 1.5
_B = 0.75


def _terms(text: str) -> list[str]:
    return [word for word in _WORDS.findall(text.casefold()) if word not in _STOP_WORDS]


def _query_terms(query: dict[str, Any], catalog: list[dict[str, Any]]) -> set[str]:
    entities = {entity["entity_id"]: entity for entity in catalog}
    pieces = [query["text"]]
    for requested in query["requested_facts"]:
        entity = entities[requested["entity_id"]]
        pieces.extend(
            entity[key] for key in ("entity_id", "label", "kind") if key in entity
        )
        field = requested["field"]
        pieces.append(field)
        definition = entity.get("field_definitions", {}).get(field)
        if definition:
            pieces.append(definition)
    # Query term frequency is binary: repeated field labels cannot inflate it.
    return set(_terms(" ".join(pieces)))


def _ranking(events: list[dict[str, str]], terms: set[str]) -> list[int]:
    documents = [
        Counter(_terms(event["text"] + " " + event["sender"])) for event in events
    ]
    lengths = [sum(document.values()) for document in documents]
    average_length = sum(lengths) / len(documents) if documents else 0.0
    if not average_length or not terms:
        return []
    frequencies = Counter(term for document in documents for term in document)
    inverse_frequency = {
        term: math.log(
            1 + (len(documents) - frequencies[term] + 0.5) / (frequencies[term] + 0.5)
        )
        for term in sorted(terms)
    }
    scored = []
    for index, (document, length) in enumerate(zip(documents, lengths)):
        normalization = _K1 * (1 - _B + _B * length / average_length)
        score = sum(
            inverse_frequency[term]
            * document[term]
            * (_K1 + 1)
            / (document[term] + normalization)
            for term in sorted(terms)
            if document[term]
        )
        if score > 0:
            scored.append((score, index))
    # Original array order determines chronology. Prefer the latest event only
    # when lexical scores tie; no special words trigger interpretation rules.
    return [index for _, index in sorted(scored, reverse=True)]


def _token_count(value: str, count_tokens: TokenCounter) -> int:
    count = count_tokens(value)
    if type(count) is not int or count < 0:
        raise ValueError("Token counter must return a nonnegative integer")
    return count


def select_bm25_sources(
    events: list[dict[str, Any]],
    query: dict[str, Any],
    catalog: list[dict[str, Any]],
    count_tokens: TokenCounter,
    settings: Settings,
    top_k: int = 8,
) -> list[dict[str, str]]:
    """Select at most ``top_k`` BM25 hits plus the recent whole-event suffix.

    The corpus contains public event text and sender. Query expansion contains
    only public question text and the requested entities' labels, kinds, field
    names and definitions. Events are indivisible retrieval documents. Positive
    BM25 scores select candidates; equal scores favor later original events.

    The recent suffix uses the same ``recent_source_tokens`` check and token
    envelope as Threadmark's active selector. It can retain a recent clearing or
    correction even without matching keywords. The union is deduplicated in
    chronological order and must fit ``max_source_bytes``; an oversized union
    raises ValueError instead of truncating text or silently dropping a hit.

    This baseline cannot understand synonyms, authority, negation or which
    conflicting event establishes the current value. A reader must interpret
    the returned original evidence. It is a lexical baseline, not a complete
    evaluation of every retrieval-augmented generation system.
    """
    if not isinstance(settings, Settings):
        raise TypeError("settings must be a Settings instance")
    if not callable(count_tokens):
        raise TypeError("count_tokens must be callable")
    if type(top_k) is not int or not 1 <= top_k <= 100:
        raise ValueError("top_k must be an integer between 1 and 100")
    catalog = public_catalog(catalog)
    query = public_query(query, catalog_keys(catalog))
    events = public_events(events)

    selected = set(_ranking(events, _query_terms(query, catalog))[:top_k])
    recent: list[dict[str, str]] = []
    for index in range(len(events) - 1, -1, -1):
        trial = [events[index], *recent]
        if (
            _token_count(canonical({"source_events": trial}), count_tokens)
            > settings.recent_source_tokens
        ):
            break
        recent = trial
        selected.add(index)
    sources = [event for index, event in enumerate(events) if index in selected]
    if (
        len(canonical({"source_events": sources}).encode("utf-8"))
        > settings.max_source_bytes
    ):
        raise ValueError("Selected BM25 sources exceed the bounded byte size")
    return sources
