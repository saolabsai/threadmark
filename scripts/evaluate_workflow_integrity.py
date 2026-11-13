"""Offline recorded-operation integrity checks; never an LM accuracy benchmark.

Run from a checkout with Python and Threadmark's ordinary dependencies:
  python scripts/evaluate_workflow_integrity.py --output runs/integrity-new
  python scripts/evaluate_workflow_integrity.py --baseline-root ../baseline --output runs/integrity-paired

Each arm runs in an isolated subprocess against a frozen source copy. No model,
tokenizer download, API, or paid fallback is used. Expected extractor operations
are supplied as recorded responses, never inserted into provider request inputs.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SCOPE = "offline recorded-extractor reducer integrity; not live extraction or answer accuracy"


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def operation(
    kind, field="venue", value=None, *, entity="review", alias=None, reason=None
):
    return dict(
        op=kind,
        entity_id=entity,
        field=field,
        value=value,
        proposal_id=alias,
        reason=reason,
    )


def make_case(identifier, description, records, checks, **options):
    events, operations = [], {}
    for index, (text, ops) in enumerate(records, 1):
        event_id = f"{identifier}-{index:04d}"
        events.append(
            dict(
                event_id=event_id,
                timestamp=(
                    datetime(2026, 11, 12, tzinfo=timezone.utc)
                    + timedelta(minutes=index)
                ).isoformat(),
                sender="Fictional coordinator",
                text=text,
            )
        )
        operations[event_id] = deepcopy(ops)
    return dict(
        id=identifier,
        description=description,
        catalog=[
            dict(
                entity_id="review",
                label="Orchard review",
                fields=["owner", "venue", "deadline"],
            )
        ],
        events=events,
        recorded_operations=operations,
        checks=checks,
        **options,
    )


def build_cases():
    """Independent fictional cases, not the prior training/test scaffold."""
    o = operation
    cases = [
        make_case(
            "ordered-handoffs",
            "Keep three ordered assignments inside one record",
            [
                (
                    "Orchard ownership starts with Ada, passes to Ben, then to Cora in that order.",
                    [
                        o("set", "owner", "Ada"),
                        o("set", "owner", "Ben"),
                        o("set", "owner", "Cora"),
                    ],
                )
            ],
            dict(
                fields=[["review", "owner", "Cora"]],
                history_values=["Ada", "Ben", "Cora"],
            ),
        ),
        make_case(
            "parallel-proposals",
            "Same-record proposals have distinct identities and exact alias resolution",
            [
                (
                    "For Orchard, ROOM-11 proposes Birch Hall and ROOM-12 proposes Elm Hall; approve ROOM-12 now.",
                    [
                        o("propose", value="Birch Hall", alias="ROOM-11"),
                        o("propose", value="Elm Hall", alias="ROOM-12"),
                        o("accept", alias="ROOM-12"),
                    ],
                )
            ],
            dict(
                fields=[["review", "venue", "Elm Hall"]],
                proposals=[
                    ["ROOM-11", "Birch Hall", "open"],
                    ["ROOM-12", "Elm Hall", "accepted"],
                ],
                history_count=1,
            ),
        ),
        make_case(
            "parallel-notes",
            "Keep two dependencies with the same predicate",
            [
                (
                    "Orchard depends on room access and separately on the interpreter booking.",
                    [
                        o("note", "dependency", "Room access must be confirmed"),
                        o(
                            "note",
                            "dependency",
                            "The interpreter booking must be confirmed",
                        ),
                    ],
                )
            ],
            dict(
                notes=[
                    ["dependency", "Room access must be confirmed"],
                    ["dependency", "The interpreter booking must be confirmed"],
                ],
                history_count=0,
            ),
        ),
        make_case(
            "clear-one-field",
            "Clearing deadline preserves owner and venue",
            [
                (
                    "Dana owns Orchard, in Ash Hall, due on 22 November.",
                    [
                        o("set", "owner", "Dana"),
                        o("set", value="Ash Hall"),
                        o("set", "deadline", "22 November"),
                    ],
                ),
                (
                    "Withdraw only the Orchard deadline; the room and owner remain approved.",
                    [o("clear", "deadline")],
                ),
            ],
            dict(
                fields=[
                    ["review", "owner", "Dana"],
                    ["review", "venue", "Ash Hall"],
                    ["review", "deadline", None],
                ],
                history_count=4,
            ),
        ),
        make_case(
            "ambiguous-alias",
            "Duplicate open aliases cannot silently select the newest",
            [
                (
                    "Orchard is approved for Ash Hall. Offer Birch Hall under ROOM-X.",
                    [
                        o("set", value="Ash Hall"),
                        o("propose", value="Birch Hall", alias="ROOM-X"),
                    ],
                ),
                (
                    "A second Orchard offer under the same ROOM-X label names Cedar Hall.",
                    [o("propose", value="Cedar Hall", alias="ROOM-X")],
                ),
                ("Accept ROOM-X for Orchard.", [o("accept", alias="ROOM-X")]),
            ],
            dict(
                fields=[["review", "venue", "Ash Hall"]],
                unresolved_candidates=[2],
                proposals=[
                    ["ROOM-X", "Birch Hall", "open"],
                    ["ROOM-X", "Cedar Hall", "open"],
                ],
            ),
        ),
        make_case(
            "unknown-reference",
            "Unknown explicit alias cannot use unique-open fallback",
            [
                (
                    "Orchard remains in Ash Hall; ROOM-K proposes Cedar Hall.",
                    [
                        o("set", value="Ash Hall"),
                        o("propose", value="Cedar Hall", alias="ROOM-K"),
                    ],
                ),
                (
                    "Approve ROOM-M for Orchard, without supplying any proposal for that label.",
                    [o("accept", alias="ROOM-M")],
                ),
            ],
            dict(
                fields=[["review", "venue", "Ash Hall"]],
                unresolved_candidates=[0],
            ),
        ),
        make_case(
            "entity-scope",
            "Same external alias on two entities stays scoped",
            [
                (
                    "Orchard ROOM-S proposes Birch Hall; the separate Lakeside review ROOM-S proposes Elm Hall.",
                    [
                        o("propose", value="Birch Hall", alias="ROOM-S"),
                        o(
                            "propose",
                            value="Elm Hall",
                            entity="lakeside",
                            alias="ROOM-S",
                        ),
                    ],
                ),
                (
                    "Accept ROOM-S only for Lakeside.",
                    [o("accept", entity="lakeside", alias="ROOM-S")],
                ),
            ],
            dict(fields=[["review", "venue", None], ["lakeside", "venue", "Elm Hall"]]),
            extra_entity=True,
        ),
        make_case(
            "late-chunk-rollback",
            "A malformed second chunk preserves the entire prior snapshot",
            [
                ("Eli owns Orchard.", [o("set", "owner", "Eli")]),
                ("Orchard venue is now Willow Hall.", [o("set", value="Willow Hall")]),
                (
                    "Orchard deadline is 27 November.",
                    [o("set", "deadline", "27 November")],
                ),
            ],
            dict(
                fields=[["review", "owner", "Eli"], ["review", "venue", None]],
                history_count=1,
                archive_count=1,
            ),
            batches=[[0], [1, 2]],
            fail_event_index=2,
            one_event_chunks=True,
        ),
        make_case(
            "snapshot-continuation",
            "Restore pending aliases and notes, then accept and correct",
            [
                (
                    "Faye owns Orchard. ROOM-T proposes Maple Hall; catering needs a vegan count.",
                    [
                        o("set", "owner", "Faye"),
                        o("propose", value="Maple Hall", alias="ROOM-T"),
                        o("note", "dependency", "Catering needs a vegan count"),
                    ],
                ),
                (
                    "Approve ROOM-T for Orchard because the hall has wheelchair access.",
                    [
                        o(
                            "accept",
                            alias="ROOM-T",
                            reason="The hall has wheelchair access",
                        )
                    ],
                ),
                ("Gita now owns Orchard, replacing Faye.", [o("set", "owner", "Gita")]),
            ],
            dict(
                fields=[["review", "owner", "Gita"], ["review", "venue", "Maple Hall"]],
                history_count=3,
                notes=[
                    ["dependency", "Catering needs a vegan count"],
                    ["reason_for_venue", "The hall has wheelchair access"],
                ],
            ),
            batches=[[0], [1], [2]],
            restore_after=0,
        ),
        make_case(
            "chronological-groups",
            "Provider group order cannot reorder source chronology",
            [
                (
                    "Orchard deadline is 20 November.",
                    [o("set", "deadline", "20 November")],
                ),
                (
                    "Correct Orchard deadline to 24 November.",
                    [o("set", "deadline", "24 November")],
                ),
                (
                    "The latest Orchard deadline is 26 November.",
                    [o("set", "deadline", "26 November")],
                ),
            ],
            dict(
                fields=[["review", "deadline", "26 November"]],
                history_values=["20 November", "24 November", "26 November"],
            ),
            reverse_groups=True,
        ),
        make_case(
            "closed-reference",
            "Rejected proposal stays closed and approval evidence is paired",
            [
                (
                    "ROOM-U proposes Aspen Hall for Orchard.",
                    [o("propose", value="Aspen Hall", alias="ROOM-U")],
                ),
                ("Reject ROOM-U.", [o("reject", alias="ROOM-U")]),
                (
                    "Accept ROOM-U without a new proposal.",
                    [o("accept", alias="ROOM-U")],
                ),
                (
                    "ROOM-V proposes Beech Hall.",
                    [o("propose", value="Beech Hall", alias="ROOM-V")],
                ),
                (
                    "Approve ROOM-V because it has the required capacity.",
                    [
                        o(
                            "accept",
                            alias="ROOM-V",
                            reason="It has the required capacity",
                        )
                    ],
                ),
            ],
            dict(
                fields=[["review", "venue", "Beech Hall"]],
                unresolved_candidates=[0],
                proposals=[
                    ["ROOM-U", "Aspen Hall", "rejected"],
                    ["ROOM-V", "Beech Hall", "accepted"],
                ],
                history_count=1,
                final_evidence_indices=[3, 4],
            ),
        ),
    ]
    long_records = []
    for index in range(1000):
        depot = f"depot-{index % 8 + 1}"
        batch = f"DISPATCH-{index + 1:04d}"
        long_records.append(
            (
                f"Depot {index % 8 + 1} has approved outbound batch {batch}. It replaces that depot's previously approved dispatch batch; other depots keep their current batch.",
                [o("set", "approved_batch", batch, entity=depot)],
            )
        )
    long_case = make_case(
        "long-dispatch",
        "1000 chronological unique operative dispatch updates, not filler",
        long_records,
        dict(
            history_count=1000,
            archive_count=1000,
            fields=[
                [f"depot-{i + 1}", "approved_batch", f"DISPATCH-{993 + i:04d}"]
                for i in range(8)
            ],
        ),
        max_packet_bytes=2_000_000,
    )
    long_case["catalog"] = [
        dict(
            entity_id=f"depot-{i + 1}",
            label=f"Depot {i + 1}",
            fields=["approved_batch"],
        )
        for i in range(8)
    ]
    cases.append(long_case)
    for case in cases:
        if case.get("extra_entity"):
            case["catalog"].append(
                dict(entity_id="lakeside", label="Lakeside review", fields=["venue"])
            )
    return cases


class RecordedExtractor:
    """An oracle response fixture, explicitly not an inference implementation."""

    def __init__(self, case, completion_type):
        self.case, self.completion_type, self.requests = case, completion_type, []
        self.allowed = {}

    def complete(self, **kwargs):
        if set(kwargs) != {"instructions", "content", "max_output_tokens"}:
            raise AssertionError("Unexpected provider request fields")
        content = kwargs["content"]
        if set(content) != {"entity_catalog", "reference_context", "new_events"}:
            raise AssertionError(
                "Oracle, questions, or other data leaked into provider inputs"
            )
        if content["entity_catalog"] != self.case["catalog"]:
            raise AssertionError("Unexpected catalog")
        for event in content["new_events"]:
            if event != self.allowed.get(event["event_id"]):
                raise AssertionError(
                    "New event was changed or is outside the current ingest"
                )
            if set(event) != {"event_id", "timestamp", "sender", "text"}:
                raise AssertionError("Expected answers leaked into event fields")
        self.requests.append(deepcopy(kwargs))
        failure = self.case.get("fail_event_index")
        if (
            failure is not None
            and self.case["events"][failure] in content["new_events"]
        ):
            return self.completion_type(text='{"events":[],"unexpected":true}')
        groups = [
            dict(
                event_id=e["event_id"],
                operations=deepcopy(self.case["recorded_operations"][e["event_id"]]),
            )
            for e in content["new_events"]
        ]
        if self.case.get("reverse_groups"):
            groups.reverse()
        return self.completion_type(text=canonical({"events": groups}))


def check_packet(memory, case):
    packet, checks = memory.memory_packet(), case["checks"]
    facts = {(f["entity_id"], f["field"]): f for f in packet["current_state"]["facts"]}
    for entity, field, expected in checks.get("fields", []):
        assert (
            facts[entity, field]["value"] == expected
        ), f"Current value mismatch: {entity}/{field}"
    history = packet["fact_history"]
    if "history_values" in checks:
        assert [v["value"] for v in history] == checks[
            "history_values"
        ], "History order/content differs"
    if "history_count" in checks:
        assert len(history) == checks["history_count"], "History count differs"
    if "archive_count" in checks:
        assert len(memory.events) == checks["archive_count"], "Archive count differs"
    else:
        assert memory.events == case["events"], "Original archive changed"
    if "proposals" in checks:
        assert [
            [p["external_alias"], p["value"], p["status"]] for p in packet["proposals"]
        ] == checks["proposals"], "Proposal resolution differs"
    if "notes" in checks:
        assert [
            [n["predicate"], n["value"]] for n in packet["explanatory_claims"]
        ] == checks["notes"], "Explanatory content was lost"
    assert [
        len(u["candidate_proposal_ids"]) for u in packet["unresolved_references"]
    ] == checks.get("unresolved_candidates", []), "Unresolved references differ"
    if "final_evidence_indices" in checks:
        assert history[-1]["evidence_event_ids"] == [
            case["events"][i]["event_id"] for i in checks["final_evidence_indices"]
        ], "Approval chain evidence differs"
    for key, identity in [
        ("fact_history", "version_id"),
        ("proposals", "proposal_id"),
        ("explanatory_claims", "claim_id"),
    ]:
        ids = [x[identity] for x in packet[key]]
        assert len(ids) == len(set(ids)), f"Duplicate identity in {key}"
    events = {e["event_id"]: e for e in memory.events}
    expected_notes = []
    for event in memory.events:
        for op in case["recorded_operations"][event["event_id"]]:
            if op["op"] == "note":
                expected_notes.append(
                    [
                        op["entity_id"],
                        op["field"],
                        op["value"],
                        [event["event_id"]],
                        event["timestamp"],
                    ]
                )
            if op["reason"] is not None:
                expected_notes.append(
                    [
                        op["entity_id"],
                        "reason_for_" + op["field"],
                        op["reason"],
                        [event["event_id"]],
                        event["timestamp"],
                    ]
                )
    assert [
        [
            n["entity_id"],
            n["predicate"],
            n["value"],
            n["evidence_event_ids"],
            n["observed_at"],
        ]
        for n in packet["explanatory_claims"]
    ] == expected_notes, "Note content, order, or provenance differs"
    previous = {}
    for version in history:
        assert (
            version["evidence_event_ids"]
            and set(version["evidence_event_ids"]) <= events.keys()
        ), "Unavailable evidence"
        assert (
            version["observed_at"]
            == events[version["evidence_event_ids"][-1]]["timestamp"]
        ), "Wrong observation timestamp"
        target = (version["entity_id"], version["field"])
        old = previous.get(target)
        if old:
            assert (
                version["supersedes_version_id"] == old["version_id"]
            ), "Broken version chain"
            assert old["valid_to"] == version["observed_at"], "Broken version interval"
            assert (
                old["superseded_by_event_id"] == version["evidence_event_ids"][-1]
            ), "Broken superseding evidence"
        previous[target] = version
    for target, version in previous.items():
        assert version["valid_to"] is None, "Current version unexpectedly closed"
        assert (
            facts[target]["value"] == version["value"]
            and facts[target]["evidence_event_ids"] == version["evidence_event_ids"]
        ), "Current fact differs from latest version"
    return packet


def evaluate_case(case, memory_type, settings_type, completion_type):
    provider = RecordedExtractor(case, completion_type)
    limit = 4000
    if case.get("one_event_chunks"):
        limit = max(len(canonical({"events": [e]})) for e in case["events"])
    memory = memory_type(
        case["catalog"],
        provider,
        settings_type(chunk_tokens=limit),
        count_tokens=len,
        max_packet_bytes=case.get("max_packet_bytes", 256_000),
    )
    rolled_back = False
    for number, indices in enumerate(
        case.get("batches", [list(range(len(case["events"])))])
    ):
        batch = [case["events"][i] for i in indices]
        provider.allowed = {e["event_id"]: e for e in batch}
        before = memory.snapshot()
        expected_failure = case.get("fail_event_index") in indices
        before_calls = len(provider.requests)
        try:
            memory.ingest(batch)
        except ValueError:
            if not expected_failure:
                raise
            assert (
                memory.snapshot() == before
            ), "Failed ingestion committed partial changes"
            assert (
                len(provider.requests) - before_calls == 2
            ), "Failure did not occur after a successful first chunk"
            rolled_back = True
        else:
            assert not expected_failure, "Malformed provider output was accepted"
        if case.get("restore_after") == number:
            snapshot = memory.snapshot()
            memory = memory_type.from_snapshot(
                snapshot, provider=provider, count_tokens=len
            )
            assert memory.snapshot() == snapshot, "Snapshot replay changed state"
    packet = check_packet(memory, case)
    snapshot = memory.snapshot()
    call_count = len(provider.requests)
    restored = memory_type.from_snapshot(snapshot, provider=provider, count_tokens=len)
    assert (
        restored.memory_packet() == packet and restored.events == memory.events
    ), "Final snapshot replay differs"
    assert len(provider.requests) == call_count, "Snapshot replay invoked provider"
    return dict(
        input_events=len(case["events"]),
        committed_events=len(memory.events),
        recorded_provider_calls=call_count,
        history_entries=len(packet["fact_history"]),
        packet_bytes=len(canonical(packet).encode()),
        rollback_verified=rolled_back,
        snapshot_replay_verified=True,
        request_input_isolation_verified=True,
        packet_sha256=hashlib.sha256(canonical(packet).encode()).hexdigest(),
        provider_requests=provider.requests,
    )


def run_worker(source_root, cases):
    sys.path.insert(0, str(source_root / "src"))
    import threadmark
    from threadmark import WorkflowMemory, Settings, Completion

    if not Path(threadmark.__file__).resolve().is_relative_to(source_root.resolve()):
        raise RuntimeError("Worker imported the wrong Threadmark source")
    rows = []
    for case in cases:
        started = time.perf_counter()
        try:
            observed = evaluate_case(case, WorkflowMemory, Settings, Completion)
            row = dict(case_id=case["id"], passed=True, observed=observed)
        except Exception as exc:
            row = dict(
                case_id=case["id"],
                passed=False,
                error_type=type(exc).__name__,
                error=str(exc)[:1000],
            )
        row["offline_wall_seconds"] = time.perf_counter() - started
        rows.append(row)
    return dict(
        scope=SCOPE,
        cases=rows,
        passed=sum(r["passed"] for r in rows),
        total=len(rows),
        live_model_calls=0,
        api_calls=0,
        offline_time_is_model_latency=False,
    )


def snapshot_sources(root, destination):
    root = root.resolve(strict=True)
    if not (root / "src/threadmark/workflow_memory.py").is_file():
        raise ValueError("Source root must contain src/threadmark/workflow_memory.py")
    records = []
    paths = [
        p
        for p in (root / "src/threadmark").rglob("*")
        if p.is_file() and p.suffix in {".py", ".json", ".typed"}
    ]
    paths += [root / "pyproject.toml"]
    for path in sorted(paths):
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError(
                "Source snapshots cannot contain external paths or symlinks"
            )
        relative = path.relative_to(root)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
        records.append(
            dict(
                path=relative.as_posix(),
                sha256=digest(target),
                bytes=target.stat().st_size,
            )
        )
    return dict(
        files=records,
        manifest_sha256=hashlib.sha256(canonical(records).encode()).hexdigest(),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--source-root", type=Path, default=ROOT)
    parser.add_argument("--baseline-root", type=Path)
    parser.add_argument("--_worker", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--_cases", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--_result", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args._worker:
        write_json(
            args._result,
            run_worker(
                args._worker, json.loads(args._cases.read_text(encoding="utf-8"))
            ),
        )
        return 0
    if args.output is None:
        parser.error("--output requires a fresh local directory")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    cases = build_cases()
    write_json(output / "fictional-cases.json", cases)
    script = output / "evaluator.py"
    script.write_bytes(Path(__file__).read_bytes())
    arms = {"candidate": args.source_root}
    if args.baseline_root:
        arms["baseline"] = args.baseline_root
    sources = {
        arm: snapshot_sources(path, output / "sources" / arm)
        for arm, path in arms.items()
    }
    protocol = dict(
        scope=SCOPE,
        fixture_sha256=digest(output / "fictional-cases.json"),
        evaluator_sha256=digest(script),
        sources=sources,
        count=len(cases),
        long_case_events=1000,
        counter="Python string length in characters for offline chunking; not model tokens",
        packet_cap_long_case=2_000_000,
        no_expected_answers_in_request_inputs=True,
        api_calls=0,
        model_calls=0,
        acceptance="All predeclared checks pass per case; failing cases remain in denominator. No inference accuracy or latency claim.",
    )
    write_json(output / "protocol.json", protocol)
    results = {}
    for arm in arms:
        result_path = output / f"{arm}.json"
        env = {
            k: v for k, v in os.environ.items() if k not in {"PYTHONPATH", "PYTHONHOME"}
        }
        process = subprocess.run(
            [
                sys.executable,
                "-I",
                "-B",
                str(script),
                "--_worker",
                str(output / "sources" / arm),
                "--_cases",
                str(output / "fictional-cases.json"),
                "--_result",
                str(result_path),
            ],
            capture_output=True,
            text=True,
            env=env,
            timeout=180,
        )
        if process.returncode:
            raise RuntimeError(f"{arm} worker failed: {process.stderr[-2000:]}")
        results[arm] = json.loads(result_path.read_text(encoding="utf-8"))
    summary = dict(
        scope=SCOPE,
        protocol_sha256=digest(output / "protocol.json"),
        results={
            arm: {
                "passed": r["passed"],
                "total": r["total"],
                "sha256": digest(output / f"{arm}.json"),
            }
            for arm, r in results.items()
        },
        model_calls=0,
        api_calls=0,
    )
    if "baseline" in results:
        old = {r["case_id"]: r["passed"] for r in results["baseline"]["cases"]}
        summary["paired_changes"] = [
            dict(
                case_id=r["case_id"], baseline=old[r["case_id"]], candidate=r["passed"]
            )
            for r in results["candidate"]["cases"]
        ]
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
