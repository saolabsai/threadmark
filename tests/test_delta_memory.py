"""Offline tests of sparse extraction, chronological reduction and isolation."""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from threadmark import Settings
from threadmark.delta_memory import DELTA_UPDATER_INSTRUCTIONS, EventDeltaMemory
from threadmark.schema import canonical

CATALOG = [
    {"entity_id": "alpha", "label": "Alpha review", "fields": ["owner", "location"]}
]
QUERY = {
    "text": "Who owns Alpha?",
    "requested_facts": [{"entity_id": "alpha", "field": "owner"}],
}


def event(identifier="e1", minute=0, text="Ada owns Alpha."):
    return {
        "event_id": identifier,
        "timestamp": f"2026-10-28T10:{minute:02}:00+00:00",
        "sender": "Coordinator",
        "text": text,
    }


def update(identifier, field="owner", value="Ada", entity="alpha"):
    return {
        "event_id": identifier,
        "facts": [{"entity_id": entity, "field": field, "value": value}],
    }


def output(*updates):
    return {"updates": list(updates)}


class Provider:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def complete(self, **kwargs):
        self.calls.append(deepcopy(kwargs))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return SimpleNamespace(
            text=response if isinstance(response, str) else json.dumps(response)
        )


def memory(provider, settings=None, catalog=CATALOG, **kwargs):
    return EventDeltaMemory(catalog, provider, settings, count_tokens=len, **kwargs)


def indexed(state):
    return {(fact["entity_id"], fact["field"]): fact for fact in state["facts"]}


def test_partial_updates_preserve_prior_fields_and_explicit_null_clears():
    first = update("e1")
    first["facts"].append(
        {"entity_id": "alpha", "field": "location", "value": "Library"}
    )
    provider = Provider(output(first), output(update("e2", "location", None)))
    tracker = memory(provider)
    tracker.ingest([event()])
    tracker.ingest([event("e2", 1, "There is no current location for Alpha.")])
    state = indexed(tracker.state)
    assert state[("alpha", "owner")]["value"] == "Ada"
    assert state[("alpha", "owner")]["evidence_event_ids"] == ["e1"]
    assert state[("alpha", "location")]["value"] is None
    assert state[("alpha", "location")]["evidence_event_ids"] == ["e2"]


def test_out_of_order_model_updates_are_reduced_in_original_event_order():
    provider = Provider(
        output(
            update("e3", value=None),
            update("e1", value="Ada"),
            update("e2", value="Bo"),
        )
    )
    tracker = memory(provider)
    tracker.ingest([event("e1", 0), event("e2", 1), event("e3", 2)])
    fact = indexed(tracker.state)[("alpha", "owner")]
    assert fact["value"] is None
    assert fact["evidence_event_ids"] == ["e3"]


def test_same_fields_on_multiple_entities_remain_separate():
    catalog = CATALOG + [{"entity_id": "beta", "fields": ["owner"]}]
    changes = update("e1")
    changes["facts"].append({"entity_id": "beta", "field": "owner", "value": "Bo"})
    tracker = memory(Provider(output(changes)), catalog=catalog)
    tracker.ingest([event()])
    assert indexed(tracker.state)[("alpha", "owner")]["value"] == "Ada"
    assert indexed(tracker.state)[("beta", "owner")]["value"] == "Bo"


def test_pending_delta_preserves_unmodified_approved_counterpart():
    catalog = [{"entity_id": "alpha", "fields": ["approved", "pending", "decision"]}]
    pending = update("e2", "pending", "USD 20.00")
    pending["facts"].append(
        {"entity_id": "alpha", "field": "decision", "value": "approval"}
    )
    tracker = memory(
        Provider(output(update("e1", "approved", "USD 10.00")), output(pending)),
        catalog=catalog,
    )
    tracker.ingest([event()])
    tracker.ingest([event("e2", 1, "A new amount is pending approval.")])
    state = indexed(tracker.state)
    assert state[("alpha", "approved")]["value"] == "USD 10.00"
    assert state[("alpha", "pending")]["value"] == "USD 20.00"
    assert state[("alpha", "decision")]["value"] == "approval"


def test_payload_contains_only_public_catalog_and_new_events_never_prior_state_or_gold():
    catalog = deepcopy(CATALOG)
    catalog[0]["gold_state"] = "PRIVATE_CATALOG_GOLD"
    source = event()
    source["gold_updates"] = "PRIVATE_EVENT_GOLD"
    provider = Provider(
        output(update("e1", value="PRIVATE_PRIOR_PREDICTION")), output()
    )
    tracker = memory(provider, catalog=catalog)
    tracker.ingest([source])
    tracker.ingest([event("e2", 1, "An unrelated background note.")])
    assert set(provider.calls[0]["content"]) == {"entity_catalog", "new_events"}
    assert "PRIVATE_" not in json.dumps(provider.calls)
    assert provider.calls[1]["content"]["new_events"] == [
        event("e2", 1, "An unrelated background note.")
    ]
    assert "current_memory" not in provider.calls[1]["content"]


def test_later_chunk_failure_preserves_committed_prior_state_and_archive_without_retry():
    sources = [event("e1", 1), event("e2", 2)]
    cap = max(
        len(canonical({"events": [source]})) for source in [event("e0", 0), *sources]
    )
    provider = Provider(
        output(update("e0", value="Original")),
        output(update("e1", value="Ada")),
        ValueError("later chunk failed"),
    )
    tracker = memory(provider, Settings(chunk_tokens=cap))
    tracker.ingest([event("e0", 0)])
    before_state, before_archive = tracker.state, tracker.events
    with pytest.raises(ValueError, match="later chunk"):
        tracker.ingest(sources)
    assert tracker.state == before_state
    assert tracker.events == before_archive
    assert len(provider.calls) == 3


def test_all_nonoperative_batch_retains_state_and_archives_events():
    tracker = memory(Provider(output()))
    tracker.ingest([event()])
    assert all(
        fact["value"] is None and fact["evidence_event_ids"] == []
        for fact in tracker.state["facts"]
    )
    assert tracker.events == [event()]


@pytest.mark.parametrize(
    "invalid",
    [
        {"updates": [update("unseen")]},
        {"updates": [update("e1"), update("e1")]},
        {"updates": [{"event_id": "e1", "facts": [update("e1")["facts"][0]] * 2}]},
        {"updates": [update("e1", field="unknown")]},
        {"updates": [update("e1", entity="unknown")]},
        {"updates": [update("e1", value=12)]},
        {"updates": [update("e1", value="x" * 257)]},
        {"updates": [{"event_id": "e1", "facts": []}]},
        {"updates": [None]},
        {"updates": [], "gold": True},
        '{"updates":[],"updates":[]}',
        {
            "updates": [
                {
                    "event_id": "e1",
                    "facts": [
                        {
                            "entity_id": "alpha",
                            "field": "owner",
                            "value": "Ada",
                            "evidence_event_ids": ["invented"],
                        }
                    ],
                }
            ]
        },
    ],
)
def test_invalid_delta_schema_never_commits_partial_state(invalid):
    provider = Provider(invalid)
    tracker = memory(provider)
    before = tracker.state
    with pytest.raises(ValueError):
        tracker.ingest([event()])
    assert tracker.state == before
    assert tracker.events == []
    assert len(provider.calls) == 1


def test_extractor_cannot_update_old_or_future_event_outside_current_chunk():
    sources = [event("e1", 0), event("e2", 1)]
    cap = max(len(canonical({"events": [source]})) for source in sources)
    tracker = memory(Provider(output(update("e2"))), Settings(chunk_tokens=cap))
    with pytest.raises(ValueError, match="event ID"):
        tracker.ingest(sources)
    assert tracker.events == []


def test_memory_cap_failure_is_atomic_even_after_extraction_call():
    initial = memory(Provider()).state
    cap = len(canonical(initial).encode("utf-8")) + 10
    provider = Provider(output(update("e1", value="x" * 256)))
    tracker = memory(provider, Settings(max_memory_bytes=cap))
    with pytest.raises(ValueError, match="Memory exceeds"):
        tracker.ingest([event()])
    assert tracker.events == []
    assert tracker.state == initial
    assert len(provider.calls) == 1


def test_input_order_duplicates_and_oversized_event_fail_before_provider_call():
    provider = Provider()
    tracker = memory(provider)
    for events in ([event(), event()], [event("e2", 2), event("e1", 1)]):
        with pytest.raises(ValueError):
            tracker.ingest(events)
    with pytest.raises(ValueError, match="never truncated"):
        memory(provider, Settings(chunk_tokens=1)).ingest([event()])
    assert provider.calls == []


def test_state_lookup_is_explicit_key_routing_without_reader_or_retrieval():
    provider = Provider(output(update("e1", value="UNVERIFIED_PREDICTION")))
    tracker = memory(provider)
    tracker.ingest([event()])
    query = {
        **QUERY,
        "text": "An unrelated free-form question.",
        "expected": "PRIVATE_GOLD",
    }
    answer = tracker.answer(query)
    assert answer["facts"][0]["value"] == "UNVERIFIED_PREDICTION"
    assert answer["facts"][0]["evidence_event_ids"] == ["e1"]
    assert len(provider.calls) == 1
    answer["facts"][0]["value"] = "mutated"
    tracker.events[0]["text"] = "mutated"
    assert tracker.state["facts"][0]["value"] == "UNVERIFIED_PREDICTION"
    assert tracker.events == [event()]
    with pytest.raises(ValueError, match="only mode"):
        tracker.answer(QUERY, mode="active_memory")
    assert not hasattr(tracker, "snapshot")


def test_empty_ingest_and_custom_instructions_need_no_hidden_fallback():
    provider = Provider(output())
    tracker = memory(provider, updater_instructions="CUSTOM DELTA\n")
    before = tracker.state
    assert tracker.ingest([]) == before
    assert provider.calls == []
    tracker.ingest([event()])
    assert provider.calls[0]["instructions"] == "CUSTOM DELTA\n"


def test_exported_prompt_matches_research_source_exactly():
    source = (
        Path(__file__).resolve().parents[1]
        / "research/prompts/event_delta_updater_v1.txt"
    )
    assert source.read_text(encoding="utf-8").strip() == DELTA_UPDATER_INSTRUCTIONS
