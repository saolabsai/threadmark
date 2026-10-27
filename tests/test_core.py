"""Offline tests of state isolation, provenance and failure behavior."""

from copy import deepcopy
from dataclasses import FrozenInstanceError, asdict, replace
from importlib.resources import files
from types import SimpleNamespace
import json

import pytest

from threadmark import Settings, Threadmark
from threadmark.context import chunks_for
from threadmark.prompts import READER_INSTRUCTIONS, UPDATER_INSTRUCTIONS, prompt_profile
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


def test_bm25_archive_baseline_uses_the_same_reader_without_updater_calls():
    provider = FakeProvider(facts(), facts())
    memory = Threadmark.from_events(CATALOG, [event()], provider, count_tokens=len)
    assert memory.answer(QUERY, mode="bm25") == facts()
    assert memory.answer(QUERY, mode="full_history") == facts()
    assert len(provider.calls) == 2
    bm25, full = provider.calls
    assert bm25["instructions"] == full["instructions"] == READER_INSTRUCTIONS
    assert bm25["max_output_tokens"] == full["max_output_tokens"] == 256
    assert bm25["content"]["context"] == {"source_events": [event()]}
    assert memory.state == facts(None, [])
    assert memory.snapshot()["baseline_only"] is True


def test_bm25_ignores_predicted_state_values_and_private_query_annotations():
    provider = FakeProvider(facts("PRIVATE_PREDICTED_VALUE"), facts())
    memory = engine(provider)
    memory.ingest([event()])
    query = {
        **QUERY,
        "expected": "PRIVATE_GOLD_VALUE",
        "annotation": "PRIVATE_ANNOTATION",
    }
    assert memory.answer(query, mode="bm25") == facts()
    reader = provider.calls[-1]
    assert reader["content"]["context"] == {"source_events": [event()]}
    assert "PRIVATE_" not in json.dumps(reader)


def test_bm25_cannot_cite_an_archive_event_outside_its_selected_packet():
    sources = [event(f"e{index + 1}", index) for index in range(9)]
    provider = FakeProvider(facts("Ada", ["e1"]))
    memory = Threadmark.from_events(
        CATALOG, sources, provider, Settings(recent_source_tokens=1), count_tokens=len
    )
    with pytest.raises(ValueError, match="Evidence"):
        memory.answer(QUERY, mode="bm25")
    supplied = provider.calls[-1]["content"]["context"]["source_events"]
    assert [source["event_id"] for source in supplied] == [
        f"e{index}" for index in range(2, 10)
    ]


def test_invalid_answer_mode_never_calls_provider():
    provider = FakeProvider()
    memory = engine(provider)
    with pytest.raises(ValueError, match="mode must be"):
        memory.answer(QUERY, mode="arbitrary-mode")
    assert provider.calls == []


def test_state_mode_returns_unverified_predicted_facts_without_reader_or_retrieval():
    provider = FakeProvider(facts("UNVERIFIED_PREDICTION"))
    memory = engine(provider)
    memory.ingest([event()])

    def forbidden_counter(_):
        raise AssertionError("Direct state lookup must not tokenize or retrieve")

    memory._count_tokens = forbidden_counter
    result = memory.answer(QUERY, mode="state")
    assert result == facts("UNVERIFIED_PREDICTION")
    assert len(provider.calls) == 1  # Only the updater is charged/called.
    result["facts"][0]["value"] = "mutated"
    assert memory.state == facts("UNVERIFIED_PREDICTION")


def test_state_mode_routes_only_explicit_public_requested_keys_not_freeform_text():
    provider = FakeProvider(facts())
    memory = engine(provider)
    memory.ingest([event()])
    query = {
        "text": "A freeform question about an unrelated refund.",
        "requested_facts": [{"entity_id": "launch", "field": "owner"}],
        "expected": "PRIVATE_ANNOTATION",
    }
    assert memory.answer(query, mode="state") == facts()
    assert len(provider.calls) == 1


def test_state_mode_rejects_archive_only_baselines_without_implicit_ingestion():
    provider = FakeProvider()
    memory = Threadmark.from_events(CATALOG, [event()], provider, count_tokens=len)
    with pytest.raises(ValueError, match="no ingested memory"):
        memory.answer(QUERY, mode="state")
    assert provider.calls == []


def test_state_mode_restores_from_snapshot_without_provider_calls():
    memory = engine(FakeProvider(facts()))
    memory.ingest([event()])
    provider = FakeProvider()
    restored = Threadmark.from_snapshot(
        memory.snapshot(), provider=provider, count_tokens=len
    )
    assert restored.answer(QUERY, mode="state") == facts()
    assert provider.calls == []


def test_state_mode_revalidates_citation_membership_against_original_archive():
    memory = engine(FakeProvider(facts()))
    memory.ingest([event()])
    # Simulate accidental internal corruption after a valid ingestion.
    memory._state["facts"][0]["evidence_event_ids"] = ["future-event"]
    with pytest.raises(ValueError, match="Evidence"):
        memory.answer(QUERY, mode="state")


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


def test_custom_prompts_are_bound_to_instance_and_dispatch_all_reader_modes():
    provider = FakeProvider(facts(), facts(), facts(), facts())
    memory = Threadmark(
        CATALOG,
        provider,
        count_tokens=len,
        updater_instructions="CUSTOM UPDATER\n",
        reader_instructions="CUSTOM READER\n",
    )
    memory.ingest([event()])
    for mode in ("active_memory", "full_history", "bm25"):
        memory.answer(QUERY, mode=mode)
    assert provider.calls[0]["instructions"] == "CUSTOM UPDATER\n"
    assert all(call["instructions"] == "CUSTOM READER\n" for call in provider.calls[1:])
    assert memory.snapshot()["prompts"] == {
        "updater": "CUSTOM UPDATER\n",
        "reader": "CUSTOM READER\n",
    }


def test_default_snapshot_retains_exact_legacy_schema_and_prompts():
    memory = engine(FakeProvider())
    assert memory.snapshot() == {
        "schema_version": 1,
        "baseline_only": False,
        "entity_catalog": CATALOG,
        "settings": asdict(Settings()),
        "events": [],
        "state": facts(None, []),
    }
    provider = FakeProvider(facts(), facts())
    restored = Threadmark.from_snapshot(
        memory.snapshot(), provider=provider, count_tokens=len
    )
    restored.ingest([event()])
    restored.answer(QUERY)
    assert provider.calls[0]["instructions"] == UPDATER_INSTRUCTIONS
    assert provider.calls[1]["instructions"] == READER_INSTRUCTIONS


def test_custom_snapshot_round_trip_restores_prompts_for_future_updates_and_answers():
    memory = Threadmark(
        CATALOG,
        FakeProvider(facts()),
        count_tokens=len,
        updater_instructions="Selected updater",
        reader_instructions="Selected reader",
    )
    memory.ingest([event()])
    snapshot = memory.snapshot()
    assert snapshot["schema_version"] == 2
    provider = FakeProvider(facts("Bo", ["e2"]), facts("Bo", ["e2"]))
    restored = Threadmark.from_snapshot(
        json.dumps(snapshot), provider=provider, count_tokens=len
    )
    assert restored.snapshot() == snapshot
    assert provider.calls == []
    restored.ingest([event("e2", 1, "Bo now owns the launch.")])
    assert restored.answer(QUERY) == facts("Bo", ["e2"])
    assert [call["instructions"] for call in provider.calls] == [
        "Selected updater",
        "Selected reader",
    ]


@pytest.mark.parametrize(
    "damage",
    [
        "missing-prompts",
        "missing-reader",
        "extra-key",
        "blank",
        "oversized",
        "nonstring",
        "legacy-extra",
    ],
)
def test_invalid_custom_snapshot_prompt_metadata_is_rejected(damage):
    memory = Threadmark(
        CATALOG,
        FakeProvider(),
        count_tokens=len,
        updater_instructions="Selected updater",
        reader_instructions="Selected reader",
    )
    snapshot = memory.snapshot()
    if damage == "missing-prompts":
        del snapshot["prompts"]
    elif damage == "missing-reader":
        del snapshot["prompts"]["reader"]
    elif damage == "extra-key":
        snapshot["prompts"]["hidden"] = "extra"
    elif damage == "blank":
        snapshot["prompts"]["reader"] = " \n"
    elif damage == "oversized":
        snapshot["prompts"]["updater"] = "x" * 16001
    elif damage == "nonstring":
        snapshot["prompts"]["updater"] = None
    else:
        snapshot["schema_version"] = 1
    provider = FakeProvider()
    with pytest.raises(ValueError):
        Threadmark.from_snapshot(snapshot, provider=provider, count_tokens=len)
    assert provider.calls == []


@pytest.mark.parametrize("value", ["", " \n", None, 12, "x" * 16001])
def test_custom_constructor_prompt_validation(value):
    with pytest.raises(ValueError, match="nonempty string"):
        Threadmark(
            CATALOG, FakeProvider(), count_tokens=len, updater_instructions=value
        )
    with pytest.raises(ValueError, match="nonempty string"):
        Threadmark(CATALOG, FakeProvider(), count_tokens=len, reader_instructions=value)


def test_packaged_profile_matches_normalized_research_copy_and_is_defensive():
    raw = json.loads(
        files("threadmark")
        .joinpath("data", "current-facts-v2.json")
        .read_text(encoding="utf-8")
    )
    profile = prompt_profile()
    assert profile == {key: text.strip() for key, text in raw.items()}
    assert set(profile) == {"updater_instructions", "reader_instructions"}
    assert profile["updater_instructions"] != UPDATER_INSTRUCTIONS
    profile["reader_instructions"] = "mutated"
    assert prompt_profile()["reader_instructions"] == raw["reader_instructions"].strip()
    with pytest.raises(ValueError, match="Unknown"):
        prompt_profile("../not-a-profile")


def test_profile_can_be_used_with_archive_baseline_and_snapshot_restores_it():
    provider = FakeProvider(facts())
    memory = Threadmark.from_events(
        CATALOG, [event()], provider, count_tokens=len, **prompt_profile()
    )
    assert memory.snapshot()["schema_version"] == 2
    restored = Threadmark.from_snapshot(
        memory.snapshot(), provider=provider, count_tokens=len
    )
    restored.answer(QUERY, mode="full_history")
    assert provider.calls[0]["instructions"] == prompt_profile()["reader_instructions"]
