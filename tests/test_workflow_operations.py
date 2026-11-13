"""Offline contracts for external journals and opt-in historical context."""

from copy import deepcopy

import pytest

from threadmark import WorkflowMemory
from threadmark.config import Settings
from threadmark.providers import Completion
from threadmark.schema import canonical
from threadmark.workflow_memory import HISTORY_REFERENCE_INSTRUCTIONS


CATALOG = [{"entity_id": "meeting", "fields": ["room", "owner"]}]


def event(number, timestamp=None):
    return {
        "event_id": f"e{number}",
        "timestamp": timestamp or f"2026-01-{number:02d}T00:00:00Z",
        "sender": "Coordinator",
        "text": f"Original source-only text {number}",
    }


def operation(kind, value=None, field="room", alias=None):
    return {
        "op": kind,
        "entity_id": "meeting",
        "field": field,
        "value": value,
        "proposal_id": alias,
        "reason": None,
    }


def journal(*groups):
    return {
        "events": [
            {"event_id": identifier, "operations": operations}
            for identifier, operations in groups
        ]
    }


class RecordedProvider:
    def __init__(self, *outputs):
        self.outputs = list(outputs)
        self.calls = []

    def complete(self, **request):
        self.calls.append(deepcopy(request))
        if not self.outputs:
            raise AssertionError("Unexpected provider call")
        return Completion(text=canonical(self.outputs.pop(0)))


def build(*, provider=None, **kwargs):
    return WorkflowMemory(
        CATALOG,
        provider if provider is not None else RecordedProvider(),
        count_tokens=len,
        **kwargs,
    )


def room(memory):
    return next(f for f in memory.state["facts"] if f["field"] == "room")


def populate_history(memory):
    memory.apply_operations(
        [event(1), event(2), event(3)],
        journal(
            ("e1", [operation("set", "Pine")]),
            ("e2", [operation("set", "Cedar")]),
            ("e3", [operation("set", "Oak")]),
        ),
    )


@pytest.mark.parametrize("as_json", [False, True])
def test_external_operations_need_neither_provider_nor_tokenizer(monkeypatch, as_json):
    def forbidden(*args, **kwargs):
        raise AssertionError("Model-free ingestion must not tokenize or call a model")

    monkeypatch.setattr("threadmark.workflow_memory.token_counter", forbidden)
    provider = RecordedProvider()
    memory = WorkflowMemory(CATALOG, provider, Settings(chunk_tokens=1))
    operations = journal(("e1", [operation("set", "Pine")]))
    result = memory.apply_operations(
        [event(1)], canonical(operations) if as_json else operations
    )
    assert room(memory)["value"] == "Pine"
    assert room(memory)["evidence_event_ids"] == ["e1"]
    assert result == memory.state
    assert provider.calls == []


def test_external_and_recorded_model_paths_use_the_same_reducer():
    events = [event(1), event(2)]
    operations = journal(
        ("e2", [operation("accept", alias="B")]),
        (
            "e1",
            [
                operation("set", "Pine"),
                operation("propose", "Cedar", alias="A"),
                operation("propose", "Oak", alias="B"),
                operation("note", "Interpreter requires audio", field="access"),
            ],
        ),
    )
    provider = RecordedProvider(operations)
    inferred = build(provider=provider)
    external = build()
    inferred.ingest(events)
    external.apply_operations(events, operations)
    assert len(provider.calls) == 1
    assert external.snapshot() == inferred.snapshot()
    assert room(external)["evidence_event_ids"] == ["e1", "e2"]
    assert [p["status"] for p in external.memory_packet()["proposals"]] == [
        "open",
        "accepted",
    ]


def test_external_inputs_and_return_values_are_copied():
    memory = build()
    events = [event(1)]
    operations = journal(("e1", [operation("set", "Pine")]))
    returned = memory.apply_operations(events, operations)
    saved = memory.snapshot()
    events[0]["text"] = "Tampered"
    operations["events"][0]["operations"][0]["value"] = "Tampered"
    returned["facts"][0]["value"] = "Tampered"
    assert memory.snapshot() == saved


def test_external_noops_archive_sources_and_empty_batch_does_not_change_memory():
    memory = build()
    memory.apply_operations([event(1)], {"events": []})
    assert memory.events == [event(1)]
    assert memory.snapshot()["operations"] == []
    saved = memory.snapshot()
    assert memory.apply_operations([], {"events": []}) == memory.state
    assert memory.snapshot() == saved
    with pytest.raises(ValueError):
        memory.apply_operations([], journal(("e1", [operation("set", "Pine")])))
    assert memory.snapshot() == saved


@pytest.mark.parametrize(
    "bad_operations",
    [
        '{"events":[],"events":[]}',
        "not JSON",
        [],
        {"events": [], "extra": True},
        journal(("e3", [operation("set", "Invented event")])),
        journal(("e2", [{**operation("set", "Pine"), "entity_id": "unknown"}])),
        journal(("e2", [operation("set", "Pine", field="unknown")])),
        journal(("e2", [{**operation("set", "Pine"), "evidence_event_ids": ["e1"]}])),
        journal(("e2", [operation("set", "Pine"), operation("set", "Pine")])),
    ],
)
def test_external_invalid_journal_never_commits_new_events(bad_operations):
    memory = build()
    memory.apply_operations([event(1)], journal(("e1", [operation("set", "Oak")])))
    saved = memory.snapshot()
    with pytest.raises(ValueError):
        memory.apply_operations([event(2)], bad_operations)
    assert memory.snapshot() == saved


def test_external_duplicate_and_out_of_order_events_are_rejected():
    memory = build()
    memory.apply_operations([event(2)], journal(("e2", [operation("set", "Oak")])))
    saved = memory.snapshot()
    for events in ([event(2)], [event(1)], [event(3), event(3)]):
        with pytest.raises(ValueError):
            memory.apply_operations(events, {"events": []})
        assert memory.snapshot() == saved


@pytest.mark.parametrize(
    "options,expected",
    [
        ({"max_entries": 1}, "entry cap"),
        ({"max_packet_bytes": 350}, "packet exceeds"),
        ({"settings": Settings(max_memory_bytes=130)}, "Memory exceeds"),
    ],
)
def test_external_cap_failure_rolls_back_entire_transaction(options, expected):
    memory = build(**options)
    saved = memory.snapshot()
    with pytest.raises(ValueError, match=expected):
        memory.apply_operations(
            [event(1), event(2)],
            journal(
                ("e1", [operation("set", "Pine")]),
                ("e2", [operation("set", "Oak")]),
            ),
        )
    assert memory.snapshot() == saved


def test_external_pending_alias_survives_snapshot_and_model_continuation():
    memory = build(reference_history_entries=4)
    memory.apply_operations(
        [event(1)], journal(("e1", [operation("propose", "Oak", alias="ROOM-B")]))
    )
    provider = RecordedProvider(journal(("e2", [operation("accept", alias="ROOM-B")])))
    restored = WorkflowMemory.from_snapshot(
        canonical(memory.snapshot()), provider=provider, count_tokens=len
    )
    assert provider.calls == []
    assert restored.reference_history_entries == 4
    restored.ingest([event(2)])
    assert room(restored) == {
        "entity_id": "meeting",
        "field": "room",
        "value": "Oak",
        "evidence_event_ids": ["e1", "e2"],
    }
    assert restored.snapshot()["schema_version"] == 2


def test_default_reference_context_and_instructions_stay_unchanged():
    provider = RecordedProvider({"events": []})
    memory = build(provider=provider, updater_instructions="Custom extraction prompt")
    populate_history(memory)
    memory.ingest([event(4)])
    assert provider.calls[0]["instructions"] == "Custom extraction prompt"
    assert set(provider.calls[0]["content"]["reference_context"]) == {
        "current_state",
        "open_proposals",
    }


def test_history_context_carries_closed_values_and_claims_without_source_text():
    provider = RecordedProvider({"events": []})
    memory = build(provider=provider, reference_history_entries=10)
    populate_history(memory)
    memory.apply_operations(
        [event(4)], journal(("e4", [operation("note", "Needs badges", field="access")]))
    )
    packet = memory.memory_packet()
    memory.ingest([event(5)])
    request = provider.calls[0]
    reference = request["content"]["reference_context"]
    assert [v["value"] for v in reference["recent_history"]] == ["Cedar", "Pine"]
    assert [n["value"] for n in reference["recent_claims"]] == ["Needs badges"]
    assert reference["history_context_complete"] is True
    assert HISTORY_REFERENCE_INSTRUCTIONS in request["instructions"]
    assert "Original source-only text" not in canonical(reference)
    assert memory.memory_packet() == packet


def test_recent_window_is_shared_across_history_and_claims_without_pruning_memory():
    provider = RecordedProvider({"events": []})
    memory = build(provider=provider, reference_history_entries=2)
    populate_history(memory)
    memory.apply_operations(
        [event(4)],
        journal(
            (
                "e4",
                [
                    operation("note", "Needs badges", field="access"),
                    operation("note", "Needs interpreter", field="access"),
                ],
            )
        ),
    )
    packet = memory.memory_packet()
    memory.ingest([event(5)])
    reference = provider.calls[0]["content"]["reference_context"]
    assert reference["recent_history"] == []
    assert len(reference["recent_claims"]) == 2
    assert reference["history_context_complete"] is False
    assert memory.memory_packet() == packet
    assert len(packet["fact_history"]) == 3
    assert len(packet["explanatory_claims"]) == 2


@pytest.mark.parametrize("fit_metadata", [False, True])
def test_optional_history_cannot_evict_mandatory_reference_or_exceed_byte_cap(
    fit_metadata,
):
    provider = RecordedProvider({"events": []})
    memory = build(provider=provider, reference_history_entries=10)
    populate_history(memory)
    memory.apply_operations(
        [event(4)], journal(("e4", [operation("propose", "Elm", alias="ROOM-E")]))
    )
    packet = memory.memory_packet()
    mandatory = {
        "current_state": memory.state,
        "open_proposals": packet["proposals"],
    }
    bounded = (
        {
            **mandatory,
            "recent_history": [],
            "recent_claims": [],
            "history_context_complete": False,
        }
        if fit_metadata
        else mandatory
    )
    memory.max_reference_bytes = len(canonical(bounded).encode("utf-8"))
    memory.ingest([event(5)])
    reference = provider.calls[0]["content"]["reference_context"]
    assert reference == bounded
    assert len(canonical(reference).encode("utf-8")) <= memory.max_reference_bytes
    assert memory.memory_packet() == packet


def test_oversized_mandatory_reference_still_fails_before_any_provider_call():
    provider = RecordedProvider({"events": []})
    memory = build(
        provider=provider, reference_history_entries=10, max_reference_bytes=300
    )
    memory.apply_operations(
        [event(1)], journal(("e1", [operation("propose", "x" * 200)]))
    )
    saved = memory.snapshot()
    with pytest.raises(ValueError, match="reference-context cap"):
        memory.ingest([event(2)])
    assert provider.calls == []
    assert memory.snapshot() == saved


def test_recent_history_orders_timezones_by_instant_not_timestamp_text():
    provider = RecordedProvider({"events": []})
    memory = build(provider=provider, reference_history_entries=1)
    memory.apply_operations(
        [
            event(1, "2026-01-01T00:00:00Z"),
            event(2, "2026-01-01T05:30:00+03:00"),
            event(3, "2026-01-01T03:00:00Z"),
        ],
        journal(
            ("e1", [operation("set", "Pine")]),
            ("e2", [operation("set", "Cedar")]),
            ("e3", [operation("set", "Oak")]),
        ),
    )
    memory.ingest([event(4)])
    reference = provider.calls[0]["content"]["reference_context"]
    assert [v["value"] for v in reference["recent_history"]] == ["Cedar"]
    assert reference["history_context_complete"] is False


def test_equal_timestamp_history_uses_committed_event_order():
    provider = RecordedProvider({"events": []})
    memory = build(provider=provider, reference_history_entries=1)
    timestamp = "2026-01-01T00:00:00Z"
    memory.apply_operations(
        [event(n, timestamp) for n in (1, 2, 3)],
        journal(
            (
                "e1",
                [
                    operation("set", "Pine"),
                    operation("note", "Earlier claim", field="access"),
                ],
            ),
            ("e2", [operation("set", "Cedar")]),
            ("e3", [operation("set", "Oak")]),
        ),
    )
    memory.ingest([event(4)])
    reference = provider.calls[0]["content"]["reference_context"]
    assert [v["value"] for v in reference["recent_history"]] == ["Cedar"]
    assert reference["recent_claims"] == []


def test_equal_timestamp_history_sees_prior_chunks_before_transaction_commit():
    timestamp = "2026-01-01T00:00:00Z"
    events = [event(n, timestamp) for n in (1, 2, 3, 4)]
    limit = max(len(canonical({"events": [item]})) for item in events)
    provider = RecordedProvider(
        journal(
            (
                "e1",
                [
                    operation("set", "Pine"),
                    operation("note", "Earlier claim", field="access"),
                ],
            )
        ),
        journal(("e2", [operation("set", "Cedar")])),
        journal(("e3", [operation("set", "Oak")])),
        {"events": []},
    )
    memory = build(
        provider=provider,
        settings=Settings(chunk_tokens=limit),
        reference_history_entries=1,
    )
    memory.ingest(events)
    assert len(provider.calls) == 4
    reference = provider.calls[3]["content"]["reference_context"]
    assert [v["value"] for v in reference["recent_history"]] == ["Cedar"]
    assert reference["recent_claims"] == []
    assert memory.events == events


def test_same_event_closed_versions_use_operation_order_for_recency():
    provider = RecordedProvider({"events": []})
    memory = build(provider=provider, reference_history_entries=1)
    memory.apply_operations(
        [event(1)],
        journal(
            (
                "e1",
                [
                    operation("set", "Pine"),
                    operation("set", "Cedar"),
                    operation("set", "Oak"),
                ],
            )
        ),
    )
    memory.ingest([event(2)])
    reference = provider.calls[0]["content"]["reference_context"]
    assert [v["value"] for v in reference["recent_history"]] == ["Cedar"]


@pytest.mark.parametrize("bad_limit", [-1, 257, True, 1.0, "1", None])
def test_history_window_limit_is_strict_and_snapshot_revalidated(bad_limit):
    with pytest.raises(ValueError, match="reference_history_entries"):
        build(reference_history_entries=bad_limit)
    saved = build().snapshot()
    saved["limits"]["reference_history_entries"] = bad_limit
    with pytest.raises(ValueError, match="reference_history_entries"):
        WorkflowMemory.from_snapshot(saved, provider=RecordedProvider())


def test_legacy_schema_one_restores_with_history_disabled_and_no_calls():
    memory = build(
        updater_instructions="Legacy updater", reader_instructions="Legacy reader"
    )
    populate_history(memory)
    legacy = memory.snapshot()
    legacy["schema_version"] = 1
    del legacy["limits"]["reference_history_entries"]
    provider = RecordedProvider()
    restored = WorkflowMemory.from_snapshot(canonical(legacy), provider=provider)
    assert provider.calls == []
    assert restored.reference_history_entries == 0
    assert restored.memory_packet() == memory.memory_packet()
    assert restored.snapshot()["schema_version"] == 2
    assert restored.snapshot()["prompts"] == legacy["prompts"]
    restored.apply_operations(
        [event(4)], journal(("e4", [operation("set", "Elm")]))
    )
    assert room(restored)["value"] == "Elm"
    assert room(memory)["value"] == "Oak"


@pytest.mark.parametrize("version", [1, 2])
def test_snapshot_history_configuration_cannot_be_silently_dropped(version):
    saved = build(reference_history_entries=4).snapshot()
    saved["schema_version"] = version
    if version == 2:
        del saved["limits"]["reference_history_entries"]
    with pytest.raises(ValueError, match="snapshot"):
        WorkflowMemory.from_snapshot(saved, provider=RecordedProvider())
