"""Request-specific decoding schema for WorkflowMemory updater responses.

This is a structural aid, not a replacement for the reducer's validation.
Whitespace-only strings, duplicate event groups/operations, chronological
reduction, proposal ambiguity, byte caps and semantic correctness remain the
reducer's responsibility. JSON Schema validates objects, not duplicate keys in
serialized JSON; WorkflowMemory's strict JSON parser still checks those keys.

The intentionally small schema vocabulary is supported by LM Format Enforcer's
JSON parser: https://github.com/noamgat/lm-format-enforcer . In particular, its
string parser does not support combining patterns with min/max length. Avoid
patterns here so legitimate quoted and escaped values stay representable.
"""

from __future__ import annotations

from typing import Any

from .schema import public_catalog, public_events
from .workflow_memory import (
    MAX_EXTERNAL_ALIAS_CHARS,
    MAX_PROPOSAL_REFERENCE_CHARS,
)


WORKFLOW_SCHEMA_VERSION = 1


def _string(limit: int) -> dict[str, Any]:
    return {"type": "string", "minLength": 1, "maxLength": limit}


def _nullable_string(limit: int) -> dict[str, Any]:
    return {"anyOf": [_string(limit), {"type": "null"}]}


def _object(properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def workflow_update_schema(
    content: dict[str, Any], *, max_value_chars: int = 256
) -> dict[str, Any]:
    """Constrain one nonempty updater request to its public IDs and targets.

    ``content`` is the exact WorkflowMemory provider payload: entity_catalog,
    reference_context and new_events. The schema never incorporates source
    text, predicted facts, aliases, or expected answers. Unknown proposal labels
    remain valid strings so the reducer can retain unresolved references.

    ``max_value_chars`` must match the consuming WorkflowMemory Settings. Notes
    and reasons keep their independent 1024-character bounds. Empty batches do
    not need a provider call and are rejected here rather than decoded.
    """
    if (
        not isinstance(content, dict)
        or set(content) != {"entity_catalog", "reference_context", "new_events"}
        or not isinstance(content["reference_context"], dict)
    ):
        raise ValueError("Expected an exact WorkflowMemory updater request")
    if type(max_value_chars) is not int or not 1 <= max_value_chars <= 1_000_000:
        raise ValueError("max_value_chars must be an integer from 1 to 1000000")
    catalog = public_catalog(content["entity_catalog"])
    events = public_events(content["new_events"])
    if not events:
        raise ValueError("Workflow decoding requires at least one new event")

    alternatives = []
    for entity in catalog:
        # Keeping each entity and its fields in one alternative prevents a
        # field belonging only to another entity from becoming a valid target.
        for operations in (
            ["set"],
            ["clear"],
            ["propose"],
            ["accept", "reject"],
            ["note"],
        ):
            kind = operations[0]
            value = (
                _string(1024 if kind == "note" else max_value_chars)
                if kind in {"set", "propose", "note"}
                else {"type": "null"}
            )
            proposal = (
                _nullable_string(
                    MAX_EXTERNAL_ALIAS_CHARS
                    if kind == "propose"
                    else MAX_PROPOSAL_REFERENCE_CHARS
                )
                if kind in {"propose", "accept"}
                else {"type": "null"}
            )
            alternatives.append(
                _object(
                    {
                        "op": {"type": "string", "enum": list(operations)},
                        "entity_id": {"type": "string", "enum": [entity["entity_id"]]},
                        "field": (
                            _string(128)
                            if kind == "note"
                            else {"type": "string", "enum": list(entity["fields"])}
                        ),
                        "value": value,
                        "proposal_id": proposal,
                        "reason": _nullable_string(1024),
                    }
                )
            )

    group = _object(
        {
            "event_id": {
                "type": "string",
                "enum": [event["event_id"] for event in events],
            },
            "operations": {
                "type": "array",
                "minItems": 1,
                "maxItems": 128,
                "items": {"anyOf": alternatives},
            },
        }
    )
    return _object(
        {
            "events": {
                "type": "array",
                "minItems": 0,
                "maxItems": len(events),
                "items": group,
            }
        }
    )
