"""Offline checks of fact history, decision provenance and portable workflow memory."""

from copy import deepcopy
import json

import pytest

from threadmark import WorkflowMemory
from threadmark.workflow_memory import proposal_identifier
from threadmark.config import Settings
from threadmark.providers import Completion
from threadmark.schema import canonical

CATALOG = [
    {"entity_id": "p", "label": "Atlas", "fields": ["owner", "scope", "status"]},
    {"entity_id": "q", "label": "Beacon", "fields": ["owner", "scope"]},
]


def event(n, text=None):
    return {
        "event_id": f"e{n}",
        "timestamp": f"2026-01-{n:02d}T00:00:00+00:00",
        "sender": "Fictional coordinator",
        "text": text or f"Public fixture record {n}.",
    }


def op(kind, field="scope", value=None, entity="p", proposal=None, reason=None):
    return {
        "op": kind,
        "entity_id": entity,
        "field": field,
        "value": value,
        "proposal_id": proposal,
        "reason": reason,
    }


def output(*groups):
    return {
        "events": [
            {"event_id": identifier, "operations": operations}
            for identifier, operations in groups
        ]
    }


class Scripted:
    def __init__(self, *outputs):
        self.outputs = list(outputs)
        self.calls = []

    def complete(self, **kwargs):
        self.calls.append(deepcopy(kwargs))
        value = self.outputs.pop(0)
        if isinstance(value, BaseException):
            raise value
        return Completion(text=value if isinstance(value, str) else json.dumps(value))


def memory(*outputs, **kwargs):
    provider = Scripted(*outputs)
    return (
        WorkflowMemory(
            CATALOG,
            provider,
            Settings(chunk_tokens=2000),
            count_tokens=lambda text: len(text),
            **kwargs,
        ),
        provider,
    )


def value(m, field="scope", entity="p"):
    return next(
        f for f in m.state["facts"] if (f["entity_id"], f["field"]) == (entity, field)
    )


def test_versions_preserve_old_owner_and_close_only_that_field():
    m, _ = memory(
        output(
            ("e1", [op("set", "owner", "Mira"), op("set", value="CSV")]),
            ("e2", [op("set", "owner", "Noah")]),
        )
    )
    m.ingest([event(1), event(2)])
    versions = m.memory_packet()["fact_history"]
    assert value(m, "owner")["value"] == "Noah"
    assert value(m)["value"] == "CSV"
    assert [v["value"] for v in versions] == ["Mira", "CSV", "Noah"]
    assert versions[0]["valid_to"] == event(2)["timestamp"]
    assert versions[1]["valid_to"] is None
    assert versions[2]["supersedes_version_id"] == versions[0]["version_id"]


def test_clear_is_a_versioned_tombstone_not_history_deletion():
    m, _ = memory(output(("e1", [op("set", value="CSV")]), ("e2", [op("clear")])))
    m.ingest([event(1), event(2)])
    assert value(m) == {
        "entity_id": "p",
        "field": "scope",
        "value": None,
        "evidence_event_ids": ["e2"],
    }
    assert [v["value"] for v in m.memory_packet()["fact_history"]] == ["CSV", None]


def test_same_batch_proposal_acceptance_copies_value_with_two_sources():
    m, _ = memory(
        output(("e1", [op("propose", value="encrypted JSON")]), ("e2", [op("accept")]))
    )
    m.ingest([event(1), event(2)])
    assert value(m)["value"] == "encrypted JSON"
    assert value(m)["evidence_event_ids"] == ["e1", "e2"]
    assert m.memory_packet()["proposals"][0]["status"] == "accepted"
    assert m.memory_packet()["fact_history"][0]["proposal_id"] == proposal_identifier(
        "e1", "p", "scope"
    )


def test_cross_ingestion_reference_context_has_pending_value_but_not_archive():
    identifier = proposal_identifier("e1", "p", "scope")
    m, provider = memory(
        output(("e1", [op("propose", value="encrypted JSON")])),
        output(("e2", [op("accept", proposal=identifier)])),
    )
    m.ingest([event(1, "Raw private fixture text not repeated to the resolver")])
    m.ingest([event(2)])
    content = provider.calls[1]["content"]
    assert set(content) == {"entity_catalog", "reference_context", "new_events"}
    assert (
        content["reference_context"]["open_proposals"][0]["value"] == "encrypted JSON"
    )
    assert "Raw private fixture text" not in canonical(content)
    assert "question" not in content and "expected" not in content
    assert value(m)["evidence_event_ids"] == ["e1", "e2"]


def test_ambiguous_acceptance_does_not_guess_or_change_current_value():
    m, _ = memory(
        output(
            ("e1", [op("set", value="CSV"), op("propose", value="JSON")]),
            ("e2", [op("propose", value="Parquet")]),
            ("e3", [op("accept")]),
        )
    )
    m.ingest([event(1), event(2), event(3)])
    assert value(m)["value"] == "CSV"
    unresolved = m.memory_packet()["unresolved_references"]
    assert len(unresolved) == 1 and len(unresolved[0]["candidate_proposal_ids"]) == 2
    assert all(p["status"] == "open" for p in m.memory_packet()["proposals"])


def test_exact_reference_resolves_one_of_multiple_proposals():
    identifier = proposal_identifier("e2", "p", "scope")
    m, _ = memory(
        output(
            ("e1", [op("propose", value="JSON")]),
            ("e2", [op("propose", value="Parquet")]),
            ("e3", [op("accept", proposal=identifier)]),
        )
    )
    m.ingest([event(1), event(2), event(3)])
    assert value(m)["value"] == "Parquet"
    assert [p["status"] for p in m.memory_packet()["proposals"]] == ["open", "accepted"]


def test_same_batch_explicit_earlier_reference_is_not_replaced_by_latest_proposal():
    identifier = proposal_identifier("e1", "p", "scope")
    m, _ = memory(
        output(
            ("e1", [op("propose", value="JSON")]),
            ("e2", [op("propose", value="Parquet")]),
            ("e3", [op("accept", proposal=identifier)]),
        )
    )
    m.ingest([event(1), event(2), event(3)])
    assert value(m)["value"] == "JSON"
    assert value(m)["evidence_event_ids"] == ["e1", "e3"]
    assert [p["status"] for p in m.memory_packet()["proposals"]] == ["accepted", "open"]


def test_rejection_preserves_approved_value_and_resolution_evidence():
    m, _ = memory(
        output(
            ("e1", [op("set", value="CSV"), op("propose", value="JSON")]),
            ("e2", [op("reject")]),
        )
    )
    m.ingest([event(1), event(2)])
    assert value(m)["value"] == "CSV"
    proposal = m.memory_packet()["proposals"][0]
    assert proposal["status"] == "rejected" and proposal["resolution_event_id"] == "e2"
    assert len(m.memory_packet()["fact_history"]) == 1


def test_other_entity_proposal_cannot_resolve_or_overwrite_requested_entity():
    identifier = proposal_identifier("e1", "q", "scope")
    m, _ = memory(
        output(
            ("e1", [op("propose", value="JSON", entity="q")]),
            ("e2", [op("accept", proposal=identifier)]),
        )
    )
    m.ingest([event(1), event(2)])
    assert value(m)["value"] is None and value(m, entity="q")["value"] is None
    assert m.diagnostics["unresolved_reference_count"] == 1


def test_unknown_proposal_id_is_preserved_as_unresolved_not_fabricated():
    m, _ = memory(output(("e1", [op("accept", proposal="invented-proposal")])))
    m.ingest([event(1)])
    assert value(m)["value"] is None
    assert (
        m.memory_packet()["unresolved_references"][0]["proposal_id"]
        == "invented-proposal"
    )


def test_general_explanatory_claim_not_limited_to_catalog_fields():
    m, _ = memory(
        output(
            (
                "e1",
                [
                    op(
                        "note",
                        field="depends_on_vendor_access",
                        value="Credential rotation must finish before delivery resumes",
                    )
                ],
            )
        )
    )
    m.ingest([event(1, "RAW_RECORD_SENTINEL")])
    packet = m.memory_packet("Why can delivery not resume?")
    assert packet["explanatory_claims"][0]["predicate"] == "depends_on_vendor_access"
    assert packet["explanatory_claims"][0]["evidence_event_ids"] == ["e1"]
    assert "RAW_RECORD_SENTINEL" not in canonical(packet)
    assert "source_events" not in canonical(packet)


def test_stated_reason_links_to_the_changed_version():
    m, _ = memory(
        output(
            (
                "e1",
                [
                    op(
                        "set",
                        field="status",
                        value="active",
                        reason="Vendor access was restored",
                    )
                ],
            )
        )
    )
    m.ingest([event(1)])
    packet = m.memory_packet()
    assert (
        packet["explanatory_claims"][0]["related_version_id"]
        == packet["fact_history"][0]["version_id"]
    )
    assert packet["explanatory_claims"][0]["value"] == "Vendor access was restored"


def test_later_invalid_chunk_rolls_back_all_state_and_archived_events():
    e1, e2 = event(1, "a" * 100), event(2, "b" * 100)
    limit = max(len(canonical({"events": [e]})) for e in (e1, e2))
    provider = Scripted(
        output(("e1", [op("set", value="JSON")])), {"events": [], "extra": "bad"}
    )
    m = WorkflowMemory(
        CATALOG, provider, Settings(chunk_tokens=limit), count_tokens=len
    )
    before = m.memory_packet()
    with pytest.raises(ValueError):
        m.ingest([e1, e2])
    assert len(provider.calls) == 2 and m.events == [] and m.memory_packet() == before


@pytest.mark.parametrize(
    "operation",
    [
        op("set", field="not_catalog", value="bad"),
        op("set", value=42),
        op("clear", value="bad"),
        op("note", value="x", entity="unknown"),
        {**op("set", value="x"), "extra": True},
    ],
)
def test_bad_operation_shapes_fail_atomically(operation):
    m, _ = memory(output(("e1", [operation])))
    with pytest.raises(ValueError):
        m.ingest([event(1)])
    assert m.events == [] and m.diagnostics["version_count"] == 0


def test_event_groups_are_reduced_in_original_order_not_model_array_order():
    m, _ = memory(
        output(
            ("e2", [op("set", value="latest")]), ("e1", [op("set", value="earlier")])
        )
    )
    m.ingest([event(1), event(2)])
    assert value(m)["value"] == "latest"


def test_prior_reference_cap_rejects_before_a_second_call_without_dropping_proposal():
    provider = Scripted(output(("e1", [op("propose", value="x" * 200)])))
    m = WorkflowMemory(
        [{"entity_id": "p", "fields": ["scope"]}],
        provider,
        count_tokens=len,
        max_reference_bytes=300,
    )
    m.ingest([event(1)])
    before = m.memory_packet()
    with pytest.raises(ValueError, match="reference-context"):
        m.ingest([event(2)])
    assert (
        len(provider.calls) == 1 and m.memory_packet() == before and len(m.events) == 1
    )


def test_packet_cap_fails_instead_of_silently_truncating_history():
    m, _ = memory(output(("e1", [op("set", value="JSON")])), max_packet_bytes=700)
    with pytest.raises(ValueError, match="packet"):
        m.ingest([event(1)])
    assert m.events == [] and m.diagnostics["version_count"] == 0


def test_returned_objects_cannot_mutate_memory():
    m, _ = memory(output(("e1", [op("set", value="JSON")])))
    m.ingest([event(1)])
    packet = m.memory_packet()
    packet["fact_history"][0]["value"] = "tampered"
    state = m.state
    state["facts"][0]["value"] = "tampered"
    archived = m.events
    archived[0]["text"] = "tampered"
    assert (
        value(m)["value"] == "JSON"
        and m.memory_packet()["fact_history"][0]["value"] == "JSON"
    )
    assert m.events[0]["text"] != "tampered"


def test_proposal_identifier_escapes_delimiters_to_avoid_collisions():
    assert proposal_identifier("a::b", "c", "d") != proposal_identifier(
        "a", "b::c", "d"
    )


@pytest.mark.parametrize("resolution", ["accept", "reject"])
def test_exact_external_alias_resolves_same_batch_with_canonical_link(resolution):
    m, _ = memory(
        output(
            (
                "e1",
                [op("set", value="CSV"), op("propose", value="JSON", proposal="PX-17")],
            ),
            ("e2", [op(resolution, proposal="PX-17")]),
        )
    )
    m.ingest([event(1), event(2)])
    packet = m.memory_packet()
    proposal = packet["proposals"][0]
    assert proposal["proposal_id"] == proposal_identifier("e1", "p", "scope")
    assert proposal["external_alias"] == "PX-17"
    assert proposal["status"] == ("accepted" if resolution == "accept" else "rejected")
    assert proposal["resolution_event_id"] == "e2"
    assert packet["unresolved_references"] == []
    if resolution == "accept":
        assert value(m)["value"] == "JSON"
        assert value(m)["evidence_event_ids"] == ["e1", "e2"]
        assert packet["fact_history"][-1]["proposal_id"] == proposal["proposal_id"]
    else:
        assert value(m)["value"] == "CSV"
        assert len(packet["fact_history"]) == 1


def test_external_alias_survives_cross_ingestion_in_bounded_prior_context():
    m, provider = memory(
        output(("e1", [op("propose", value="JSON", proposal="PX-17")])),
        output(("e2", [op("accept", proposal="PX-17")])),
    )
    m.ingest([event(1)])
    m.ingest([event(2)])
    pending = provider.calls[1]["content"]["reference_context"]["open_proposals"]
    assert pending[0]["proposal_id"] == proposal_identifier("e1", "p", "scope")
    assert pending[0]["external_alias"] == "PX-17"
    assert value(m)["value"] == "JSON"
    assert value(m)["evidence_event_ids"] == ["e1", "e2"]


def test_external_alias_is_scoped_to_entity_and_field():
    m, _ = memory(
        output(
            ("e1", [op("propose", value="JSON", proposal="PX-17")]),
            ("e2", [op("propose", value="Parquet", entity="q", proposal="PX-17")]),
            ("e3", [op("propose", field="owner", value="Mira", proposal="PX-17")]),
            ("e4", [op("accept", proposal="PX-17")]),
        )
    )
    m.ingest([event(n) for n in range(1, 5)])
    assert value(m)["value"] == "JSON"
    assert value(m, entity="q")["value"] is None
    assert value(m, field="owner")["value"] is None
    assert [p["status"] for p in m.memory_packet()["proposals"]] == [
        "accepted",
        "open",
        "open",
    ]
    assert m.diagnostics["unresolved_reference_count"] == 0


@pytest.mark.parametrize("resolution", ["accept", "reject"])
def test_duplicate_open_aliases_remain_unresolved_without_recency_guess(resolution):
    m, _ = memory(
        output(
            ("e1", [op("propose", value="JSON", proposal="PX-17")]),
            ("e2", [op("propose", value="Parquet", proposal="PX-17")]),
            ("e3", [op(resolution, proposal="PX-17")]),
        )
    )
    m.ingest([event(1), event(2), event(3)])
    packet = m.memory_packet()
    assert value(m)["value"] is None
    assert [p["status"] for p in packet["proposals"]] == ["open", "open"]
    unresolved = packet["unresolved_references"][0]
    assert unresolved["proposal_id"] == "PX-17"
    assert unresolved["candidate_proposal_ids"] == [
        proposal_identifier("e1", "p", "scope"),
        proposal_identifier("e2", "p", "scope"),
    ]


def test_unknown_or_case_different_alias_does_not_use_unique_open_fallback():
    m, _ = memory(
        output(
            ("e1", [op("propose", value="JSON", proposal="PX-17")]),
            ("e2", [op("accept", proposal="px-17")]),
        )
    )
    m.ingest([event(1), event(2)])
    packet = m.memory_packet()
    assert value(m)["value"] is None
    assert packet["proposals"][0]["status"] == "open"
    assert packet["unresolved_references"][0]["candidate_proposal_ids"] == []


def test_closed_duplicate_alias_is_not_a_candidate_for_later_open_proposal():
    first = proposal_identifier("e1", "p", "scope")
    m, _ = memory(
        output(
            ("e1", [op("propose", value="JSON", proposal="PX-17")]),
            ("e2", [op("reject", proposal=first)]),
            ("e3", [op("propose", value="Parquet", proposal="PX-17")]),
            ("e4", [op("accept", proposal="PX-17")]),
        )
    )
    m.ingest([event(n) for n in range(1, 5)])
    assert value(m)["value"] == "Parquet"
    assert value(m)["evidence_event_ids"] == ["e3", "e4"]
    assert [p["status"] for p in m.memory_packet()["proposals"]] == [
        "rejected",
        "accepted",
    ]


def test_alias_canonical_collision_is_ambiguous_without_canonical_precedence():
    first = proposal_identifier("e1", "p", "scope")
    m, _ = memory(
        output(
            ("e1", [op("propose", value="JSON")]),
            ("e2", [op("propose", value="Parquet", proposal=first)]),
            ("e3", [op("accept", proposal=first)]),
        )
    )
    m.ingest([event(1), event(2), event(3)])
    packet = m.memory_packet()
    assert value(m)["value"] is None
    assert [p["status"] for p in packet["proposals"]] == ["open", "open"]
    assert len(packet["unresolved_references"][0]["candidate_proposal_ids"]) == 2


def test_one_proposal_matching_own_alias_and_canonical_id_counts_once():
    first = proposal_identifier("e1", "p", "scope")
    m, _ = memory(
        output(
            ("e1", [op("propose", value="JSON", proposal=first)]),
            ("e2", [op("accept", proposal=first)]),
        )
    )
    m.ingest([event(1), event(2)])
    assert value(m)["value"] == "JSON"
    assert m.memory_packet()["unresolved_references"] == []


def test_proposal_and_alias_acceptance_in_one_event_preserve_one_evidence_source():
    m, _ = memory(
        output(
            (
                "e1",
                [
                    op("propose", value="JSON", proposal="PX-17"),
                    op("accept", proposal="PX-17"),
                ],
            )
        )
    )
    m.ingest([event(1)])
    assert value(m)["value"] == "JSON"
    assert value(m)["evidence_event_ids"] == ["e1"]


@pytest.mark.parametrize("proposal", ["", " " * 4, 17, True, [], "x" * 2049])
def test_propose_alias_is_a_bounded_nonempty_string_or_null(proposal):
    m, _ = memory(output(("e1", [op("propose", value="JSON", proposal=proposal)])))
    with pytest.raises(ValueError):
        m.ingest([event(1)])
    assert m.events == [] and m.diagnostics["proposal_count"] == 0


@pytest.mark.parametrize("kind", ["set", "clear", "note"])
def test_external_alias_does_not_relax_unrelated_operation_shapes(kind):
    m, _ = memory(
        output(
            (
                "e1",
                [op(kind, value=None if kind == "clear" else "JSON", proposal="PX-17")],
            )
        )
    )
    with pytest.raises(ValueError):
        m.ingest([event(1)])
    assert m.events == []


def decision_memory():
    """Build a journal with history, an open alias, notes and an unresolved ID."""
    m, provider = memory(
        output(
            (
                "e1",
                [op("set", "owner", "Mira"), op("set", value="CSV")],
            ),
            (
                "e2",
                [
                    op("set", "owner", "Noah", reason="Mira moved teams"),
                    op("propose", value="JSON", proposal="AP-71"),
                    op("note", "delivery_blocker", "Vendor access is pending"),
                ],
            ),
            ("e3", [op("accept", proposal="UNKNOWN")]),
        )
    )
    m.ingest([event(1), event(2), event(3)])
    return m, provider


@pytest.mark.parametrize("as_text", [False, True])
def test_snapshot_roundtrip_preserves_history_pending_alias_notes_and_continuation(
    as_text,
):
    original, _ = decision_memory()
    saved = original.snapshot()
    if as_text:
        saved = canonical(saved)
    continuation = Scripted(output(("e4", [op("accept", proposal="AP-71")])))
    restored = WorkflowMemory.from_snapshot(
        saved, provider=continuation, count_tokens=len
    )
    assert continuation.calls == []
    assert restored.events == original.events
    assert restored.state == original.state
    assert restored.memory_packet() == original.memory_packet()
    assert restored.diagnostics == original.diagnostics
    restored.ingest([event(4)])
    assert value(restored)["value"] == "JSON"
    assert value(restored)["evidence_event_ids"] == ["e2", "e4"]
    assert value(original)["value"] == "CSV"
    assert restored.memory_packet()["proposals"][0]["status"] == "accepted"
    assert restored.diagnostics["unresolved_reference_count"] == 1
    assert len(continuation.calls) == 1


def test_snapshot_is_a_replay_journal_without_a_trusted_derived_cache():
    m, _ = decision_memory()
    saved = m.snapshot()
    assert set(saved) == {
        "schema",
        "schema_version",
        "entity_catalog",
        "settings",
        "limits",
        "prompts",
        "events",
        "operations",
    }
    assert saved["schema"] == "threadmark-workflow-memory"
    assert saved["schema_version"] == 1
    assert [group["event_id"] for group in saved["operations"]] == ["e1", "e2", "e3"]
    assert "current_state" not in saved and "fact_history" not in saved


def test_snapshot_and_restore_do_not_alias_caller_objects():
    m, _ = decision_memory()
    saved = m.snapshot()
    provider = Scripted()
    restored = WorkflowMemory.from_snapshot(saved, provider=provider, count_tokens=len)
    saved["operations"][0]["operations"][0]["value"] = "tampered"
    saved["events"][0]["text"] = "tampered"
    saved["entity_catalog"][0]["fields"].append("tampered")
    restored.entity_catalog[0]["fields"].append("another mutation")
    assert m.memory_packet() == restored.memory_packet()
    assert restored.events[0]["text"] != "tampered"
    assert "tampered" not in restored.entity_catalog[0]["fields"]
    assert "another mutation" not in restored.entity_catalog[0]["fields"]
    assert provider.calls == []


def test_snapshot_preserves_custom_limits_settings_and_prompts():
    settings = Settings(chunk_tokens=2500, reader_max_output_tokens=777)
    m = WorkflowMemory(
        CATALOG,
        Scripted(output(("e1", [op("set", value="CSV")]))),
        settings,
        count_tokens=len,
        max_reference_bytes=9000,
        max_packet_bytes=100000,
        max_entries=900,
        updater_instructions="Extract typed workflow changes.",
        reader_instructions="Answer only from the supplied memory.",
    )
    m.ingest([event(1)])
    reader = Scripted({"answer": "CSV", "evidence_event_ids": ["e1"]})
    restored = WorkflowMemory.from_snapshot(
        m.snapshot(), provider=reader, count_tokens=len
    )
    assert restored.settings == settings
    assert restored.max_reference_bytes == 9000
    assert restored.max_packet_bytes == 100000
    assert restored.max_entries == 900
    assert restored.snapshot()["prompts"] == m.snapshot()["prompts"]
    restored.answer("Which scope is approved?")
    assert reader.calls[0]["instructions"] == "Answer only from the supplied memory."
    assert reader.calls[0]["max_output_tokens"] == 777


@pytest.mark.parametrize("bad_version", [True, 0, 2, "1", None])
def test_snapshot_rejects_unknown_or_noninteger_schema_versions(bad_version):
    m, _ = decision_memory()
    saved = m.snapshot()
    saved["schema_version"] = bad_version
    provider = Scripted()
    with pytest.raises(ValueError):
        WorkflowMemory.from_snapshot(saved, provider=provider, count_tokens=len)
    assert provider.calls == []


def test_snapshot_rejects_missing_operations_and_injected_derived_cache():
    m, _ = decision_memory()
    missing = m.snapshot()
    missing.pop("operations")
    injected = m.snapshot()
    injected["current_state"] = {"facts": []}
    for saved in (missing, injected):
        with pytest.raises(ValueError):
            WorkflowMemory.from_snapshot(saved, provider=Scripted(), count_tokens=len)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda s: s["settings"].pop("chunk_tokens"),
        lambda s: s["settings"].update(unknown=True),
        lambda s: s["settings"].update(max_value_chars=0),
        lambda s: s["limits"].update(max_packet_bytes=True),
        lambda s: s["limits"].update(max_packet_bytes=1),
        lambda s: s["limits"].update(max_entries=1),
        lambda s: s["limits"].update(unknown=1),
        lambda s: s["prompts"].update(updater=""),
        lambda s: s["prompts"].update(reader=42),
        lambda s: s["prompts"].update(extra="not permitted"),
    ],
)
def test_snapshot_revalidates_configuration_and_replay_bounds(mutate):
    m, _ = decision_memory()
    saved = m.snapshot()
    mutate(saved)
    provider = Scripted()
    with pytest.raises((TypeError, ValueError)):
        WorkflowMemory.from_snapshot(saved, provider=provider, count_tokens=len)
    assert provider.calls == []


@pytest.mark.parametrize(
    "mutate",
    [
        lambda s: s["events"].reverse(),
        lambda s: s["events"].append(deepcopy(s["events"][0])),
        lambda s: s["events"][0].update(timestamp="2026-01-01T00:00:00"),
        lambda s: s["operations"][0].update(event_id="invented"),
        lambda s: s["operations"].append(deepcopy(s["operations"][0])),
        lambda s: s["operations"].reverse(),
        lambda s: s["operations"][0]["operations"][0].update(entity_id="unknown"),
        lambda s: s["operations"][0]["operations"][0].update(field="unknown"),
        lambda s: s["operations"][0]["operations"][0].update(evidence_event_ids=["e3"]),
        lambda s: s["operations"][0]["operations"][0].update(op="invented"),
    ],
)
def test_snapshot_rejects_invalid_events_and_operation_journals(mutate):
    m, _ = decision_memory()
    saved = m.snapshot()
    mutate(saved)
    provider = Scripted()
    with pytest.raises(ValueError):
        WorkflowMemory.from_snapshot(saved, provider=provider, count_tokens=len)
    assert provider.calls == []


@pytest.mark.parametrize("raw", ["{", '{"schema":1,"schema":2}', "[]", "null"])
def test_snapshot_json_validation_fails_without_a_provider_call(raw):
    provider = Scripted()
    with pytest.raises(ValueError):
        WorkflowMemory.from_snapshot(raw, provider=provider, count_tokens=len)
    assert provider.calls == []


def test_late_failure_preserves_the_existing_snapshot_and_journal():
    provider = Scripted(
        output(("e1", [op("propose", value="JSON", proposal="AP-71")])),
        output(("e2", [op("accept", proposal="AP-71")])),
        {"events": [], "extra": "bad"},
    )
    e1, e2, e3 = event(1, "a" * 100), event(2, "b" * 100), event(3, "c" * 100)
    chunk_limit = max(len(canonical({"events": [e]})) for e in (e1, e2, e3))
    m = WorkflowMemory(
        CATALOG, provider, Settings(chunk_tokens=chunk_limit), count_tokens=len
    )
    m.ingest([e1])
    before = m.snapshot()
    with pytest.raises(ValueError):
        m.ingest([e2, e3])
    assert m.snapshot() == before
    assert m.memory_packet()["proposals"][0]["status"] == "open"
    assert len(provider.calls) == 3


def test_freeform_answer_receives_history_and_notes_without_original_record_text():
    m, writer = decision_memory()
    reader = Scripted(
        {
            "answer": "Mira owned it; Noah took over because Mira moved teams.",
            "evidence_event_ids": ["e1", "e2"],
        }
    )
    question = (
        "Who owned Atlas immediately before Noah took over, and why did it change?"
    )
    result = m.answer(question, reader_provider=reader)
    assert result["answer"] == "Mira owned it; Noah took over because Mira moved teams."
    content = canonical(reader.calls[0]["content"])
    assert question in content
    assert "Mira moved teams" in content
    assert "Mira" in content and "Noah" in content
    assert "Public fixture record" not in content
    assert "requested_facts" not in content
    assert len(writer.calls) == 1
    assert len(reader.calls) == 1


def test_freeform_reader_can_cite_the_proposal_and_acceptance_chain():
    m, _ = memory(
        output(
            ("e1", [op("propose", value="JSON", proposal="AP-71")]),
            ("e2", [op("accept", proposal="AP-71")]),
        )
    )
    m.ingest([event(1), event(2)])
    reader = Scripted({"answer": "JSON", "evidence_event_ids": ["e1", "e2"]})
    assert m.answer("What was approved?", reader_provider=reader) == {
        "answer": "JSON",
        "evidence_event_ids": ["e1", "e2"],
    }


def test_reader_does_not_accept_an_archived_id_that_only_appears_inside_a_value():
    m, _ = memory(output(("e1", [op("note", "untrusted_label", "e2")])))
    m.ingest([event(1), event(2)])
    reader = Scripted({"answer": "Some claim", "evidence_event_ids": ["e2"]})
    before = m.snapshot()
    with pytest.raises(ValueError):
        m.answer("What does the record say?", reader_provider=reader)
    assert m.snapshot() == before
    assert len(reader.calls) == 1


@pytest.mark.parametrize(
    "answer",
    [
        {"answer": "CSV", "evidence_event_ids": []},
        {"answer": "CSV", "evidence_event_ids": ["invented"]},
        {"answer": "CSV", "evidence_event_ids": ["e1", "e1"]},
        {"answer": "CSV", "evidence_event_ids": "e1"},
        {"answer": None, "evidence_event_ids": ["e1"]},
        {"answer": 17, "evidence_event_ids": ["e1"]},
        {"answer": "", "evidence_event_ids": ["e1"]},
        {"answer": "CSV", "evidence_event_ids": ["e1"], "extra": True},
        "not JSON",
        '{"answer":"CSV","answer":"JSON","evidence_event_ids":["e1"]}',
    ],
)
def test_reader_rejects_bad_shapes_or_unavailable_citations_without_mutating_memory(
    answer,
):
    m, _ = memory(output(("e1", [op("set", value="CSV")])))
    m.ingest([event(1)])
    before = m.snapshot()
    reader = Scripted(answer)
    with pytest.raises(ValueError):
        m.answer("Which scope is approved?", reader_provider=reader)
    assert m.snapshot() == before
    assert len(reader.calls) == 1


def test_genuine_unknown_answer_and_default_reader_provider():
    m, provider = memory({"answer": None, "evidence_event_ids": []})
    assert m.answer("What is the unrecorded budget?") == {
        "answer": None,
        "evidence_event_ids": [],
    }
    assert len(provider.calls) == 1


@pytest.mark.parametrize("question", ["", " " * 4, None, 7, "x" * 16001])
def test_invalid_question_fails_before_reader_call(question):
    m, provider = memory()
    with pytest.raises(ValueError):
        m.answer(question)
    assert provider.calls == []


def test_empty_ingestion_uses_no_provider_calls_or_journal_entries():
    m, provider = memory()
    before = m.snapshot()
    m.ingest([])
    assert m.snapshot() == before
    assert provider.calls == []


@pytest.mark.parametrize("section", ["events", "entity_catalog"])
def test_snapshot_rejects_silently_normalized_extra_public_fields(section):
    m, _ = decision_memory()
    saved = m.snapshot()
    saved[section][0]["annotation"] = "must not be silently dropped"
    with pytest.raises(ValueError):
        WorkflowMemory.from_snapshot(saved, provider=Scripted(), count_tokens=len)


def test_restored_archive_rejects_replayed_event_before_another_paid_call():
    m, _ = decision_memory()
    provider = Scripted()
    restored = WorkflowMemory.from_snapshot(
        m.snapshot(), provider=provider, count_tokens=len
    )
    before = restored.snapshot()
    with pytest.raises(ValueError, match="Duplicate event"):
        restored.ingest([event(3)])
    assert restored.snapshot() == before
    assert provider.calls == []


def test_reader_exception_has_no_automatic_retry_or_memory_changes():
    m, _ = decision_memory()
    before = m.snapshot()
    reader = Scripted(RuntimeError("simulated provider failure"))
    with pytest.raises(RuntimeError, match="simulated provider failure"):
        m.answer("Who owns Atlas?", reader_provider=reader)
    assert len(reader.calls) == 1
    assert m.snapshot() == before


@pytest.mark.parametrize("resolution", ["accept", "reject"])
@pytest.mark.parametrize(
    "event_id,entity,field,reference_cap",
    [
        ("/" * 700, "p", "scope", 16000),
        ("🧭" * 1024, "🧭" * 128, "🧭" * 128, 100000),
    ],
)
def test_long_generated_canonical_reference_resolves_exact_open_proposal_and_restores(
    resolution, event_id, entity, field, reference_cap
):
    """Generated IDs can exceed the separate 2048-character external-alias cap."""
    first = {**event(1), "event_id": event_id}
    provider = Scripted(
        output(
            (
                event_id,
                [
                    op("set", field, "CSV", entity),
                    op("propose", field, "JSON", entity),
                ],
            ),
            ("e2", [op("propose", field, "Parquet", entity)]),
        )
    )
    m = WorkflowMemory(
        [{"entity_id": entity, "fields": [field]}],
        provider,
        Settings(chunk_tokens=5000),
        count_tokens=len,
        max_reference_bytes=reference_cap,
    )
    m.ingest([first, event(2)])
    proposals = m.memory_packet()["proposals"]
    identifier = proposals[0]["proposal_id"]
    assert len(identifier) > 2048
    assert all(p["status"] == "open" and p["external_alias"] is None for p in proposals)

    # Restore before resolution too: the ID is generated by the public packet,
    # and two open targets make a null/unique-proposal fallback insufficient.
    resolver = Scripted(
        output(("e3", [op(resolution, field, entity=entity, proposal=identifier)]))
    )
    restored = WorkflowMemory.from_snapshot(
        m.snapshot(), provider=resolver, count_tokens=len
    )
    assert resolver.calls == []
    restored.ingest([event(3)])
    packet = restored.memory_packet()
    assert packet["unresolved_references"] == []
    assert [p["status"] for p in packet["proposals"]] == [
        "accepted" if resolution == "accept" else "rejected",
        "open",
    ]
    current = value(restored, field, entity)
    assert current["value"] == ("JSON" if resolution == "accept" else "CSV")
    assert current["evidence_event_ids"] == (
        [event_id, "e3"] if resolution == "accept" else [event_id]
    )
    replay = WorkflowMemory.from_snapshot(
        canonical(restored.snapshot()), provider=Scripted(), count_tokens=len
    )
    assert replay.memory_packet() == packet
