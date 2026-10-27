"""Offline retrieval tests use no model output or benchmark answer labels."""

from copy import deepcopy

import pytest

from threadmark import Settings
from threadmark.retrieval import select_bm25_sources
from threadmark.schema import canonical

CATALOG = [
    {
        "entity_id": "maple",
        "label": "Maple review",
        "kind": "meeting",
        "fields": ["owner", "location"],
        "field_definitions": {
            "owner": "Current coordinator name.",
            "location": "Approved venue label.",
        },
    },
    {
        "entity_id": "heron",
        "label": "Heron review",
        "kind": "meeting",
        "fields": ["owner"],
        "field_definitions": {"owner": "Current facilitator name."},
    },
]
QUERY = {
    "text": "Who owns the Maple review?",
    "requested_facts": [{"entity_id": "maple", "field": "owner"}],
}


def event(identifier: str, minute: int, text: str) -> dict[str, str]:
    return {
        "event_id": identifier,
        "timestamp": f"2026-10-28T10:{minute:02}:00+00:00",
        "sender": "Coordinator",
        "text": text,
    }


def selected_ids(events, query=QUERY, *, settings=None, top_k=1, catalog=CATALOG):
    sources = select_bm25_sources(
        events,
        query,
        catalog,
        len,
        settings or Settings(recent_source_tokens=1),
        top_k,
    )
    return [source["event_id"] for source in sources]


def test_bm25_finds_relevant_older_event_instead_of_neutral_padding():
    events = [
        event("e1", 0, "Maple review coordinator Ada."),
        event("e2", 1, "This paragraph discusses unrelated cafeteria blueprints."),
        event("e3", 2, "Archive storage and miscellaneous notes."),
    ]
    assert selected_ids(events) == ["e1"]


def test_requested_entity_and_field_definitions_augment_generic_question():
    events = [
        event("e1", 0, "Maple review coordinator Ada."),
        event("e2", 1, "Heron review facilitator Bo."),
        event("e3", 2, "Another coordinator describes unrelated work."),
    ]
    query = {
        "text": "Who is responsible?",
        "requested_facts": [{"entity_id": "heron", "field": "owner"}],
    }
    assert selected_ids(events, query) == ["e2"]


def test_multiple_requested_entities_can_supply_separate_original_events():
    events = [
        event("e1", 0, "Maple review coordinator Ada."),
        event("e2", 1, "Heron review facilitator Bo."),
        event("e3", 2, "Unrelated astronomy background."),
    ]
    query = {
        "text": "Who coordinates Maple review and Heron review?",
        "requested_facts": [
            {"entity_id": "maple", "field": "owner"},
            {"entity_id": "heron", "field": "owner"},
        ],
    }
    assert selected_ids(events, query, top_k=2) == ["e1", "e2"]


def test_recent_clearing_is_included_without_semantic_interpretation():
    events = [
        event("e1", 0, "Maple review coordinator Ada."),
        event("e2", 1, "The venue carpet was repaired."),
        event("e3", 2, "That assignment is now cleared."),
    ]
    cap = len(canonical({"source_events": [events[-1]]}))
    assert selected_ids(events, settings=Settings(recent_source_tokens=cap)) == [
        "e1",
        "e3",
    ]


def test_gold_annotations_do_not_change_rank_or_appear_in_sources():
    events = [
        event("e1", 0, "Maple review coordinator Ada."),
        event("e2", 1, "Unrelated astronomy background."),
    ]
    expected = selected_ids(events)
    annotated = deepcopy(events)
    annotated[1]["gold_state"] = "Maple review coordinator owner name" * 50
    annotated[1]["is_correct_evidence"] = True
    query = {
        **QUERY,
        "expected": "Unrelated astronomy background",
        "gold_event_id": "e2",
    }
    catalog = deepcopy(CATALOG)
    catalog[0]["gold_answer"] = "astronomy"
    sources = select_bm25_sources(
        annotated, query, catalog, len, Settings(recent_source_tokens=1), 1
    )
    assert [source["event_id"] for source in sources] == expected
    assert all(
        set(source) == {"event_id", "timestamp", "sender", "text"} for source in sources
    )


def test_selection_is_chronological_deduplicated_and_defensive():
    events = [event("e1", 0, "Maple owner Ada."), event("e2", 1, "Maple owner Bo.")]
    sources = select_bm25_sources(events, QUERY, CATALOG, len, Settings(), 2)
    assert [source["event_id"] for source in sources] == ["e1", "e2"]
    sources[0]["text"] = "mutated"
    assert events[0]["text"] == "Maple owner Ada."


def test_equal_bm25_scores_prefer_latest_original_event_deterministically():
    events = [
        event("e1", 0, "Maple review owner Ada."),
        event("e2", 1, "Maple review owner Ada."),
    ]
    assert selected_ids(events) == ["e2"]
    assert selected_ids(events) == ["e2"]


def test_source_byte_cap_fails_without_truncating_or_dropping_hits():
    events = [event("e1", 0, "Maple review owner Ada.")]
    exact_bytes = len(canonical({"source_events": events}).encode("utf-8"))
    assert selected_ids(events, settings=Settings(max_source_bytes=exact_bytes)) == [
        "e1"
    ]
    with pytest.raises(ValueError, match="bounded byte"):
        selected_ids(events, settings=Settings(max_source_bytes=exact_bytes - 1))


def test_recent_window_does_not_split_oversized_latest_event_or_fill_older_gaps():
    events = [
        event("e1", 0, "Neutral note."),
        event("e2", 1, "Unrelated latest entry " * 50),
    ]
    query = {
        "text": "zebras",
        "requested_facts": [{"entity_id": "heron", "field": "owner"}],
    }
    # Sender is deliberately neutral so a query/catalog term cannot score it.
    for source in events:
        source["sender"] = "Recorder"
    assert (
        selected_ids(events, query, settings=Settings(recent_source_tokens=150)) == []
    )


def test_empty_archive_and_no_lexical_matches_are_valid():
    assert selected_ids([]) == []
    assert (
        selected_ids(
            [event("e1", 0, "Astronomy")],
            catalog=[{"entity_id": "zebra", "fields": ["striping"]}],
            query={
                "text": "stripes",
                "requested_facts": [{"entity_id": "zebra", "field": "striping"}],
            },
        )
        == []
    )


@pytest.mark.parametrize("top_k", [0, -1, True, 1.5, 101])
def test_top_k_is_bounded_and_requires_an_integer(top_k):
    with pytest.raises(ValueError, match="top_k"):
        selected_ids([], top_k=top_k)


def test_duplicate_or_out_of_order_event_ids_are_rejected():
    duplicate = [event("e1", 0, "Maple"), event("e1", 1, "Maple")]
    with pytest.raises(ValueError, match="Duplicate"):
        selected_ids(duplicate)
    with pytest.raises(ValueError, match="timestamp order"):
        selected_ids([event("e2", 2, "Maple"), event("e1", 1, "Maple")])


def test_invalid_token_counter_fails_before_returning_a_source_packet():
    with pytest.raises(ValueError, match="nonnegative integer"):
        select_bm25_sources(
            [event("e1", 0, "Maple")], QUERY, CATALOG, lambda _: True, Settings()
        )
