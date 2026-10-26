"""Offline tests of state isolation, provenance and failure behavior."""

from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace
import json

import pytest

from threadmark import Settings, Threadmark
from threadmark.context import chunks_for
from threadmark.prompts import READER_INSTRUCTIONS, UPDATER_INSTRUCTIONS
from threadmark.schema import canonical, catalog_keys, public_catalog, validate_facts

CATALOG = [
    {
        "entity_id": "launch",
        "label": "Launch",
        "fields": ["owner"],
        "field_definitions": {"owner": "Exact current owner, or null."},
    }
]
QUERY = {
    "text": "Who owns the launch?",
    "requested_facts": [{"entity_id": "launch", "field": "owner"}],
}


def event(identifier="e1", minute=0, text="Ada owns the launch."):
    return {
        "event_id": identifier,
        "timestamp": f"2026-10-27T10:{minute:02}:00+00:00",
        "sender": "Coordinator",
        "text": text,
    }


def facts(value="Ada", evidence=None):
    return {
        "facts": [
            {
                "entity_id": "launch",
                "field": "owner",
                "value": value,
                "evidence_event_ids": ["e1"] if evidence is None else evidence,
            }
        ]
    }


class FakeProvider:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def complete(self, **request):
        self.calls.append(deepcopy(request))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return SimpleNamespace(
            text=response if isinstance(response, str) else json.dumps(response),
            usage={},
            elapsed_seconds=0,
            model="fake",
            response_id="fake",
        )


def engine(provider, **settings):
    return Threadmark(CATALOG, provider, Settings(**settings), count_tokens=len)


def test_public_inputs_strip_annotations_and_do_not_alias():
    catalog = deepcopy(CATALOG)
    catalog[0]["gold_state"] = "HIDDEN_CATALOG"
    source = event()
    source["annotation"] = "HIDDEN_EVENT"
    query = {**QUERY, "expected": "HIDDEN_ANSWER", "query_id": "metadata-only"}
    provider = FakeProvider(facts(), facts())
    memory = Threadmark(catalog, provider, count_tokens=len)
    catalog[0]["fields"].append("injected")
    memory.ingest([source])
    source["text"] = "Mutated after ingest"
    memory.answer(query)
    requests = json.dumps(provider.calls)
    assert "HIDDEN_" not in requests
    assert "metadata-only" not in requests
    assert "injected" not in requests
    assert memory.events[0]["text"] == "Ada owns the launch."


def test_state_archive_catalog_and_snapshot_are_defensive_copies():
    memory = engine(FakeProvider(facts()))
    memory.ingest([event()])
    memory.state["facts"][0]["value"] = "changed"
    memory.events[0]["text"] = "changed"
    memory.entity_catalog[0]["fields"].append("changed")
    snapshot = memory.snapshot()
    snapshot["state"]["facts"][0]["value"] = "changed"
    assert memory.state == facts()
    assert memory.events == [event()]
    assert memory.entity_catalog == CATALOG


def test_active_reader_uses_original_evidence_not_predicted_values():
    provider = FakeProvider(facts("PREDICTION_MUST_NOT_REACH_READER"), facts())
    memory = engine(provider)
    memory.ingest([event()])
    assert memory.answer(QUERY) == facts()
    reader = provider.calls[-1]
    assert reader["content"]["context"] == {"source_events": [event()]}
    assert "PREDICTION_MUST_NOT_REACH_READER" not in json.dumps(reader)


def test_both_arms_have_identical_reader_instructions_and_output_cap():
    provider = FakeProvider(facts(), facts(), facts())
    memory = engine(provider)
    memory.ingest([event()])
    memory.answer(QUERY, mode="active_memory")
    memory.answer(QUERY, mode="full_history")
    updater, active, full = provider.calls
    assert updater["instructions"] == UPDATER_INSTRUCTIONS
    assert active["instructions"] == full["instructions"] == READER_INSTRUCTIONS
    assert active["max_output_tokens"] == full["max_output_tokens"] == 256
    assert active["content"]["requested_facts"] == full["content"]["requested_facts"]
    assert full["content"]["context"] == {"events": [event()]}


def test_multi_chunk_ingest_is_atomic_even_after_a_charged_call():
    first, second = event("e1", 0), event("e2", 1)
    cap = max(len(canonical({"events": [item]})) for item in (first, second))
    provider = FakeProvider(facts(), ValueError("provider failed after first charge"))
    memory = engine(provider, chunk_tokens=cap)
    with pytest.raises(ValueError, match="provider failed"):
        memory.ingest([first, second])
    assert len(provider.calls) == 2
    assert memory.events == []
    assert memory.state == facts(None, [])


def test_invalid_later_chunk_output_does_not_commit_any_new_events():
    first, second = event("e1", 0), event("e2", 1)
    cap = max(len(canonical({"events": [item]})) for item in (first, second))
    provider = FakeProvider(facts(), facts("Ada", ["future-event"]))
    memory = engine(provider, chunk_tokens=cap)
    with pytest.raises(ValueError, match="Evidence"):
        memory.ingest([first, second])
    assert memory.events == []
    assert memory.state == facts(None, [])
    assert len(provider.calls) == 2


def test_duplicate_and_out_of_order_events_fail_before_provider_call():
    provider = FakeProvider(facts())
    memory = engine(provider)
    for events in ([event(), event()], [event("later", 2), event("earlier", 1)]):
        with pytest.raises(ValueError):
            memory.ingest(events)
    assert provider.calls == []
    memory.ingest([event()])
    with pytest.raises(ValueError, match="Duplicate"):
        memory.ingest([event()])
    assert len(provider.calls) == 1


def test_timezone_aware_order_is_compared_as_an_instant():
    first = event()
    first["timestamp"] = "2026-10-27T11:00:00+01:00"
    second = event("e2", 1)
    provider = FakeProvider(facts())
    memory = engine(provider)
    memory.ingest([first, second])
    naive = event("e3", 2)
    naive["timestamp"] = "2026-10-27T10:02:00"
    with pytest.raises(ValueError, match="timezone"):
        memory.ingest([naive])


def test_recent_suffix_and_cited_sources_preserve_original_order():
    sources = [event("e1", 0), event("e2", 1), event("e3", 2)]
    recent_cap = len(canonical({"source_events": [sources[-1]]}))
    memory = engine(FakeProvider(facts()), recent_source_tokens=recent_cap)
    memory.ingest(sources)
    assert [item["event_id"] for item in memory.evidence(QUERY)] == ["e1", "e3"]


def test_source_union_byte_limit_rejects_instead_of_truncating():
    provider = FakeProvider(facts())
    memory = engine(provider, max_source_bytes=10)
    memory.ingest([event()])
    with pytest.raises(ValueError, match="bounded byte"):
        memory.answer(QUERY)
    assert len(provider.calls) == 1


def test_active_reader_cannot_cite_an_omitted_archive_event():
    sources = [event("e1", 0), event("e2", 1), event("e3", 2)]
    recent_cap = len(canonical({"source_events": [sources[-1]]}))
    provider = FakeProvider(facts(), facts("Ada", ["e2"]))
    memory = engine(provider, recent_source_tokens=recent_cap)
    memory.ingest(sources)
    with pytest.raises(ValueError, match="Evidence"):
        memory.answer(QUERY)


def test_oversized_event_fails_before_provider_call():
    provider = FakeProvider()
    memory = engine(provider, chunk_tokens=10)
    with pytest.raises(ValueError, match="never truncated"):
        memory.ingest([event()])
    assert provider.calls == []
    assert memory.events == []


@pytest.mark.parametrize(
    "output",
    [
        '{"facts":[],"facts":[]}',
        {"facts": []},
        {
            "facts": [
                {
                    "entity_id": "launch",
                    "field": "owner",
                    "value": "Ada",
                    "evidence_event_ids": [],
                }
            ]
        },
        facts("Ada", ["unknown"]),
        facts("Ada", ["e1", "e1"]),
        {
            "facts": [
                {
                    "entity_id": "launch",
                    "field": "owner",
                    "value": float("nan"),
                    "evidence_event_ids": ["e1"],
                }
            ]
        },
        {
            "facts": [
                {
                    "entity_id": "launch",
                    "field": "owner",
                    "value": "Ada",
                    "evidence_event_ids": ["e1"],
                    "extra": True,
                }
            ]
        },
    ],
)
def test_model_outputs_are_strictly_validated(output):
    with pytest.raises(ValueError):
        validate_facts(output, [("launch", "owner")], {"e1"}, Settings())


def test_snapshot_round_trip_uses_no_provider_calls():
    memory = engine(FakeProvider(facts()))
    memory.ingest([event()])
    provider = FakeProvider()
    restored = Threadmark.from_snapshot(
        json.dumps(memory.snapshot()), provider=provider, count_tokens=len
    )
    assert restored.snapshot() == memory.snapshot()
    assert provider.calls == []
    assert restored.evidence(QUERY) == [event()]


def test_full_history_archive_load_needs_no_updater_call():
    provider = FakeProvider(facts())
    memory = Threadmark.from_events(CATALOG, [event()], provider, count_tokens=len)
    assert provider.calls == []
    assert memory.state == facts(None, [])
    assert memory.answer(QUERY, mode="full_history") == facts()
    assert len(provider.calls) == 1
    assert provider.calls[0]["instructions"] == READER_INSTRUCTIONS
    with pytest.raises(ValueError, match="no ingested memory"):
        memory.answer(QUERY)
    restored = Threadmark.from_snapshot(
        memory.snapshot(), provider=provider, count_tokens=len
    )
    with pytest.raises(ValueError, match="no ingested memory"):
        restored.evidence(QUERY)


def test_archive_only_baseline_can_ingest_loaded_archive_explicitly():
    provider = FakeProvider(facts(), facts())
    memory = Threadmark.from_events(CATALOG, [event()], provider, count_tokens=len)
    assert memory.ingest([]) == facts()
    assert memory.events == [event()]
    assert memory.answer(QUERY) == facts()
    assert provider.calls[0]["content"]["new_events"] == [event()]
    assert memory.snapshot()["baseline_only"] is False


def test_baseline_can_answer_when_an_event_exceeds_updater_chunk_limit():
    provider = FakeProvider(facts())
    memory = Threadmark.from_events(
        CATALOG, [event()], provider, Settings(chunk_tokens=10), count_tokens=len
    )
    restored = Threadmark.from_snapshot(
        memory.snapshot(), provider=provider, count_tokens=len
    )
    assert restored.answer(QUERY, mode="full_history") == facts()
    with pytest.raises(ValueError, match="never truncated"):
        restored.ingest([])
    assert restored.snapshot()["baseline_only"] is True
    assert len(provider.calls) == 1


@pytest.mark.parametrize(
    "damage",
    ["missing-setting", "future-citation", "duplicate-event", "version", "extra-key"],
)
def test_invalid_snapshots_are_rejected_without_provider_calls(damage):
    memory = engine(FakeProvider(facts()))
    memory.ingest([event()])
    snapshot = memory.snapshot()
    if damage == "missing-setting":
        del snapshot["settings"]["model"]
    elif damage == "future-citation":
        snapshot["state"]["facts"][0]["evidence_event_ids"] = ["future"]
    elif damage == "duplicate-event":
        snapshot["events"].append(event())
    elif damage == "version":
        snapshot["schema_version"] = True
    else:
        snapshot["gold_state"] = facts()
    provider = FakeProvider()
    with pytest.raises(ValueError):
        Threadmark.from_snapshot(snapshot, provider=provider, count_tokens=len)
    assert provider.calls == []


def test_provider_configuration_and_settings_validation():
    provider = FakeProvider()
    provider.model = "different-model"
    with pytest.raises(ValueError, match="Provider model"):
        engine(provider)
    with pytest.raises(ValueError):
        Settings(chunk_tokens=True)
    with pytest.raises(FrozenInstanceError):
        Settings().model = "changed"
    assert replace(Settings(), chunk_tokens=3000).chunk_tokens == 3000


def test_catalog_bounds_and_empty_ingest():
    with pytest.raises(ValueError, match="32 current facts"):
        public_catalog(
            [
                {"entity_id": "a", "fields": [str(index) for index in range(32)]},
                {"entity_id": "b", "fields": ["extra"]},
            ]
        )
    with pytest.raises(ValueError):
        public_catalog([{"entity_id": [], "fields": ["owner"]}])
    provider = FakeProvider()
    memory = engine(provider)
    assert memory.ingest([]) == facts(None, [])
    assert provider.calls == []
    assert catalog_keys(memory.entity_catalog) == [("launch", "owner")]


def test_chunking_keeps_events_whole():
    sources = [event("e1", 0), event("e2", 1)]
    cap = max(len(canonical({"events": [item]})) for item in sources)
    assert chunks_for(sources, len, Settings(chunk_tokens=cap)) == [
        [sources[0]],
        [sources[1]],
    ]
