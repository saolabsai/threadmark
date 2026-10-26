"""Strict public input and model-output validation.

Only explicit public fields reach the model. Annotation, expected-answer and
gold-state fields in benchmark records are intentionally discarded.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import json
from typing import Any, Iterable

from .config import Settings

FactKey = tuple[str, str]
FACT_KEYS = {"entity_id", "field", "value", "evidence_event_ids"}
EVENT_KEYS = ("event_id", "timestamp", "sender", "text")
MAX_FACTS = 32
MAX_EVENTS = 100_000
MAX_ARCHIVE_BYTES = 64_000_000


def canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def _invalid_constant(value: str) -> None:
    raise ValueError(f"Invalid JSON constant: {value}")


def parse_json(value: str) -> Any:
    try:
        return json.loads(
            value, object_pairs_hook=_object, parse_constant=_invalid_constant
        )
    except (json.JSONDecodeError, RecursionError) as exc:
        raise ValueError("Expected valid JSON") from exc


def _string(value: Any, name: str, limit: int) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ValueError(
            f"{name} must be a nonempty string of at most {limit} characters"
        )
    return value


def public_catalog(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not 1 <= len(value) <= MAX_FACTS:
        raise ValueError(
            "A nonempty public entity catalog of at most 32 entities is required"
        )
    catalog: list[dict[str, Any]] = []
    entities: set[str] = set()
    fact_count = 0
    for item in value:
        if not isinstance(item, dict):
            raise ValueError("Catalog entities must be objects")
        identifier = _string(item.get("entity_id"), "entity_id", 128)
        if identifier in entities:
            raise ValueError("Duplicate entity identifier")
        entities.add(identifier)
        names = item.get("fields")
        if not isinstance(names, list) or not 1 <= len(names) <= MAX_FACTS:
            raise ValueError(
                "Catalog fields must be a nonempty list of at most 32 names"
            )
        names = [_string(name, "field", 128) for name in names]
        if len(set(names)) != len(names):
            raise ValueError("Duplicate catalog field")
        fact_count += len(names)
        if fact_count > MAX_FACTS:
            raise ValueError("The bounded workflow supports at most 32 current facts")
        entity: dict[str, Any] = {"entity_id": identifier, "fields": names}
        for key in ("label", "kind", "evidence_contract"):
            if key in item:
                entity[key] = _string(item[key], key, 4096)
        if "field_definitions" in item:
            definitions = item["field_definitions"]
            if not isinstance(definitions, dict) or not set(definitions) <= set(names):
                raise ValueError("Field definitions must describe only catalog fields")
            entity["field_definitions"] = {
                name: _string(description, "field definition", 4096)
                for name, description in definitions.items()
            }
        catalog.append(entity)
    if len(canonical(catalog).encode("utf-8")) > 64_000:
        raise ValueError("Public entity catalog exceeds 64000 bytes")
    return catalog


def catalog_keys(catalog: list[dict[str, Any]]) -> list[FactKey]:
    return [
        (entity["entity_id"], field) for entity in catalog for field in entity["fields"]
    ]


def initial_state(keys: Iterable[FactKey]) -> dict[str, Any]:
    return {
        "facts": [
            {
                "entity_id": entity,
                "field": field,
                "value": None,
                "evidence_event_ids": [],
            }
            for entity, field in keys
        ]
    }


def event_time(timestamp: str) -> datetime:
    try:
        value = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("Event timestamp must be ISO 8601 with a timezone") from exc
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Event timestamp must include a timezone")
    return value


def public_events(
    value: Any, *, prior: list[dict[str, Any]] | None = None
) -> list[dict[str, str]]:
    if not isinstance(value, list) or len(value) > MAX_EVENTS:
        raise ValueError("Events must be a list of at most 100000 records")
    previous = prior or []
    if len(previous) + len(value) > MAX_EVENTS:
        raise ValueError("Event archive exceeds 100000 records")
    seen = {event["event_id"] for event in previous}
    last_time = event_time(previous[-1]["timestamp"]) if previous else None
    events: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict):
            raise ValueError("Events must be objects")
        event = {
            key: _string(item.get(key), key, 64_000 if key == "text" else 1024)
            for key in EVENT_KEYS
        }
        if event["event_id"] in seen:
            raise ValueError("Duplicate event ID")
        seen.add(event["event_id"])
        timestamp = event_time(event["timestamp"])
        if last_time is not None and timestamp < last_time:
            raise ValueError("Events must be in nondecreasing timestamp order")
        last_time = timestamp
        events.append(event)
    if len(canonical(previous + events).encode("utf-8")) > MAX_ARCHIVE_BYTES:
        raise ValueError("Event archive exceeds 64000000 bytes")
    return events


def public_query(value: Any, allowed: Iterable[FactKey]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("Query must be an object")
    text = _string(value.get("text"), "query text", 16_000)
    requested = value.get("requested_facts")
    if not isinstance(requested, list) or not 1 <= len(requested) <= MAX_FACTS:
        raise ValueError("Every query needs 1 to 32 requested facts")
    allowed = set(allowed)
    seen: set[FactKey] = set()
    result = []
    for item in requested:
        if not isinstance(item, dict) or set(item) != {"entity_id", "field"}:
            raise ValueError("Requested facts contain only entity_id and field")
        key = (
            _string(item["entity_id"], "entity_id", 128),
            _string(item["field"], "field", 128),
        )
        if key not in allowed or key in seen:
            raise ValueError("Unknown or duplicate requested fact")
        seen.add(key)
        result.append({"entity_id": key[0], "field": key[1]})
    return {"text": text, "requested_facts": result}


def validate_facts(
    value: Any,
    keys: Iterable[FactKey],
    allowed_evidence: Iterable[str],
    settings: Settings,
    *,
    memory: bool = False,
) -> dict[str, Any]:
    if isinstance(value, str):
        if len(value.encode("utf-8")) > max(settings.max_memory_bytes, 256_000):
            raise ValueError("Model output exceeds the validation byte limit")
        value = parse_json(value)
    if (
        not isinstance(value, dict)
        or set(value) != {"facts"}
        or not isinstance(value["facts"], list)
    ):
        raise ValueError("Output must contain only a facts array")
    keys = list(keys)
    allowed_keys = set(keys)
    allowed_evidence = set(allowed_evidence)
    if len(value["facts"]) != len(keys):
        raise ValueError("Every requested fact must occur exactly once")
    facts: dict[FactKey, dict[str, Any]] = {}
    for fact in value["facts"]:
        if not isinstance(fact, dict) or set(fact) != FACT_KEYS:
            raise ValueError("Invalid fact shape")
        key = (
            _string(fact["entity_id"], "entity_id", 128),
            _string(fact["field"], "field", 128),
        )
        if key not in allowed_keys or key in facts:
            raise ValueError("Unknown or duplicate fact")
        content, evidence = fact["value"], fact["evidence_event_ids"]
        if content is not None and (
            not isinstance(content, str) or len(content) > settings.max_value_chars
        ):
            raise ValueError("Value must be a bounded string or null")
        if (
            not isinstance(evidence, list)
            or len(evidence) > settings.max_evidence_ids
            or any(
                not isinstance(identifier, str) or identifier not in allowed_evidence
                for identifier in evidence
            )
            or len(set(evidence)) != len(evidence)
            or (content is not None and not evidence)
        ):
            raise ValueError(
                "Evidence must cite supplied events; nonnull facts need evidence"
            )
        facts[key] = deepcopy(fact)
    ordered = {"facts": [facts[key] for key in keys]}
    if memory and len(canonical(ordered).encode("utf-8")) > settings.max_memory_bytes:
        raise ValueError("Memory exceeds the bounded byte size")
    return ordered
