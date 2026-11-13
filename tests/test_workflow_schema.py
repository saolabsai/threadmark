"""Structural decoding constraints, independently checked with JSON Schema."""

from copy import deepcopy

from jsonschema import Draft202012Validator, ValidationError
import pytest

from threadmark import WorkflowMemory
from threadmark.schema import canonical
from threadmark.workflow_schema import workflow_update_schema


def request():
    return {
        "entity_catalog": [
            {"entity_id": "meeting", "fields": ["room", "owner"]},
            {"entity_id": "task", "fields": ["status"]},
        ],
        "reference_context": {"current_state": {}, "open_proposals": []},
        "new_events": [
            {
                "event_id": f"e{index}",
                "timestamp": f"2026-01-0{index}T00:00:00Z",
                "sender": "Coordinator",
                "text": f"Private source text {index}",
            }
            for index in (1, 2)
        ],
    }


def operation(kind="set", value="Pine", **updates):
    return {
        "op": kind,
        "entity_id": "meeting",
        "field": "room",
        "value": value,
        "proposal_id": None,
        "reason": None,
        **updates,
    }


def response(*operations, event_id="e1"):
    return {"events": [{"event_id": event_id, "operations": list(operations)}]}


def validator(**kwargs):
    schema = workflow_update_schema(request(), **kwargs)
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


@pytest.mark.parametrize(
    "value",
    [
        {"events": []},
        response(operation()),
        response(operation("clear", None)),
        response(operation("propose", "Oak", proposal_id="ROOM-A")),
        response(operation("accept", None, proposal_id="UNKNOWN-KEEP-UNRESOLVED")),
        response(operation("reject", None, proposal_id="e0::meeting::room::2")),
        response(operation("accept", None)),
        response(operation("note", "Needs interpreter", field="access")),
        response(operation(reason="The owner requested this change")),
        response(operation(entity_id="task", field="status", value="ready")),
        response(operation(value='A quoted "value" with \\ and\nnewlines')),
        response(operation(value="हिंदी 東京 🧭")),
    ],
)
def test_all_operation_shapes_and_legitimate_values_are_representable(value):
    validator().validate(value)


@pytest.mark.parametrize(
    "value",
    [
        [],
        {"events": [], "answer": "extra"},
        {"events": {}},
        {"events": [{"event_id": "e1", "operations": [], "extra": True}]},
        response(operation(), event_id="old-event-not-in-this-chunk"),
        response(),
        response(operation(op="approve")),
        response(operation(op=True)),
        response(operation(entity_id="unknown")),
        response(operation(field="unknown")),
        response(operation(entity_id="meeting", field="status")),
        response(operation(entity_id="task", field="room")),
        response(operation(value=None)),
        response(operation(value="")),
        response(operation(value=1)),
        response(operation(value=False)),
        response(operation("propose", "")),
        response(operation("clear", "Pine")),
        response(operation("accept", "Pine")),
        response(operation("reject", "Pine")),
        response(operation(proposal_id="not-allowed-on-set")),
        response(operation("clear", None, proposal_id="not-allowed-on-clear")),
        response(operation("note", "Reason", proposal_id="not-allowed-on-note")),
        response(operation("propose", "Pine", proposal_id="")),
        response(operation("accept", None, proposal_id=12)),
        response(operation("reject", None, proposal_id=[])),
        response(operation("note", "Reason", field="")),
        response(operation("note", "Reason", field="x" * 129)),
        response(operation("note", "Reason", entity_id="unknown")),
        response(operation(reason="")),
        response(operation(reason={})),
        response(operation(evidence_event_ids=["e1"])),
    ],
)
def test_schema_rejects_bad_keys_ids_targets_and_value_types(value):
    with pytest.raises(ValidationError):
        validator().validate(value)


@pytest.mark.parametrize(
    "field", ["op", "entity_id", "field", "value", "proposal_id", "reason"]
)
def test_every_operation_key_is_required(field):
    item = operation()
    del item[field]
    with pytest.raises(ValidationError):
        validator().validate(response(item))


def test_operation_and_event_counts_are_bounded():
    schema = validator()
    schema.validate(response(*(operation(value=str(n)) for n in range(128))))
    with pytest.raises(ValidationError):
        schema.validate(response(*(operation(value=str(n)) for n in range(129))))
    two = {
        "events": [
            response(operation())["events"][0],
            response(operation(), event_id="e2")["events"][0],
        ]
    }
    schema.validate(two)
    with pytest.raises(ValidationError):
        schema.validate({"events": two["events"] + [deepcopy(two["events"][0])]})


@pytest.mark.parametrize(
    "item,maximum",
    [
        (operation(), 7),
        (operation("propose", "Pine"), 7),
        (operation("note", "Pine", field="access"), 1024),
    ],
)
def test_runtime_value_limits_and_separate_note_cap(item, maximum):
    schema = validator(max_value_chars=7)
    item["value"] = "x" * maximum
    schema.validate(response(item))
    item["value"] += "x"
    with pytest.raises(ValidationError):
        schema.validate(response(item))


@pytest.mark.parametrize(
    "kind,limit", [("propose", 2048), ("accept", 16000), ("reject", 16000)]
)
def test_proposal_labels_and_canonical_references_have_distinct_limits(kind, limit):
    item = operation(
        kind, "Pine" if kind == "propose" else None, proposal_id="x" * limit
    )
    schema = validator()
    schema.validate(response(item))
    item["proposal_id"] += "x"
    with pytest.raises(ValidationError):
        schema.validate(response(item))


def test_reason_length_bound_is_independent_of_value_bound():
    schema = validator(max_value_chars=4)
    schema.validate(response(operation(reason="x" * 1024)))
    with pytest.raises(ValidationError):
        schema.validate(response(operation(reason="x" * 1025)))


def test_schema_contains_only_public_ids_not_text_values_or_history():
    payload = request()
    payload["entity_catalog"][0]["label"] = "PRIVATE-CATALOG-LABEL"
    payload["reference_context"] = {"current_state": "PRIVATE-PREDICTED-VALUE"}
    payload["new_events"][0]["text"] = "PRIVATE-EVENT-CONTENT"
    schema = workflow_update_schema(payload)
    assert "PRIVATE-" not in canonical(schema)
    assert "meeting" in canonical(schema) and "e1" in canonical(schema)
    original = deepcopy(payload)
    group = schema["properties"]["events"]["items"]
    group["properties"]["event_id"]["enum"].append("fake")
    assert payload == original
    assert "fake" not in canonical(workflow_update_schema(payload))


@pytest.mark.parametrize("value", [0, -1, 1_000_001, True, 1.5, "256", None])
def test_invalid_runtime_value_limits_fail_before_decoding(value):
    with pytest.raises(ValueError, match="max_value_chars"):
        workflow_update_schema(request(), max_value_chars=value)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda r: r.pop("reference_context"),
        lambda r: r.update(question="Wrong provider path"),
        lambda r: r.update(reference_context=[]),
        lambda r: r.update(entity_catalog=[]),
        lambda r: r.update(new_events=[]),
        lambda r: r["new_events"].reverse(),
        lambda r: r["new_events"][1].update(event_id="e1"),
    ],
)
def test_schema_builder_rejects_invalid_or_empty_updater_requests(mutate):
    payload = request()
    mutate(payload)
    with pytest.raises(ValueError):
        workflow_update_schema(payload)


class NoCalls:
    def complete(self, **kwargs):
        raise AssertionError("Schema tests never call a model")


@pytest.mark.parametrize(
    "value,error",
    [
        (response(operation(value="   ")), "nonempty"),
        (response(operation(), operation()), "Duplicate operation"),
        (
            {
                "events": [
                    response(operation())["events"][0],
                    response(operation(value="Oak"))["events"][0],
                ]
            },
            "duplicate event ID",
        ),
    ],
)
def test_schema_acceptance_does_not_replace_stricter_reducer_checks(value, error):
    payload = request()
    validator().validate(value)
    memory = WorkflowMemory(payload["entity_catalog"], NoCalls(), count_tokens=len)
    before = memory.snapshot()
    with pytest.raises(ValueError, match=error):
        memory.apply_operations(payload["new_events"], value)
    assert memory.snapshot() == before


def test_schema_does_not_force_aliases_to_existing_proposals_or_infer_truth():
    payload = request()
    value = response(operation("accept", None, proposal_id="UNSTATED-PROPOSAL"))
    validator().validate(value)
    memory = WorkflowMemory(payload["entity_catalog"], NoCalls(), count_tokens=len)
    memory.apply_operations(payload["new_events"], value)
    unresolved = memory.memory_packet()["unresolved_references"]
    assert unresolved[0]["proposal_id"] == "UNSTATED-PROPOSAL"
    assert unresolved[0]["candidate_proposal_ids"] == []
