"""Public export privacy and independent scoring checks; no models or networking."""

from __future__ import annotations

from copy import deepcopy
from importlib.util import module_from_spec, spec_from_file_location
from io import BytesIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import zipfile

SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts" / "prepare_comparison_evidence.py"
)
SPEC = spec_from_file_location("comparison_evidence", SCRIPT)
export = module_from_spec(SPEC)
SPEC.loader.exec_module(export)


def fact(field="owner", value="Mira", evidence="e1"):
    return {
        "entity_id": "item-1",
        "field": field,
        "value": value,
        "evidence_event_ids": [] if evidence is None else [evidence],
    }


def envelope(*facts):
    return {"facts": list(facts)}


def exact_score(correct=True):
    return {
        "correct_fields": int(correct),
        "planned_fields": 1,
        "exact_values": correct,
        "exact_values_and_evidence": correct,
        "correct_fields_with_evidence": int(correct),
    }


def rag_fixture(*, unavailable=False):
    owners = envelope(fact())
    dates = envelope(fact("start_at", "2027-01-01", "e2"))
    case = {
        "id": "workflow-one",
        "synthetic": True,
        "domain": "meeting",
        "size_band": "8192",
        "entity_catalog": [
            {
                "entity_id": "item-1",
                "label": "meeting one",
                "kind": "meeting",
                "fields": ["owner", "start_at"],
            }
        ],
        "events": [
            {"event_id": "e1", "text": "Mira owns the meeting."},
            {"event_id": "e2", "text": "The agreed date is 2027-01-01."},
        ],
        "queries": [
            {
                "query_id": "q1",
                "text": "Who owns the meeting?",
                "requested_facts": [{"entity_id": "item-1", "field": "owner"}],
                "expected": owners,
            },
            {
                "query_id": "q2",
                "text": "What is the date?",
                "requested_facts": [{"entity_id": "item-1", "field": "start_at"}],
                "expected": dates,
            },
        ],
        "gold_state": envelope(fact(), fact("start_at", "2027-01-01", "e2")),
    }
    dataset = {"manifest": {"synthetic": True}, "workflows": [case]}
    protocol = {
        "schema": "fixture-protocol",
        "model": "reader",
        "seed": 2026102801,
        "arms": list(export.ARMS),
        "sources": {},
        "settings": {"max_value_chars": 256, "max_evidence_ids": 2},
    }
    report = {
        "status": "halted" if unavailable else "completed",
        "backend": "openai",
        "model": "reader",
        "queries": [],
        "workflows": [],
        "calls": [],
        "api_calls": 1,
        "api_charge_usd": 0,
        "run_budget": {"private_hold": 0.75},
        "summary": {"arms": {}},
    }
    requests, responses = [], []

    def call(stage, answer=None, query_id=None):
        identifier = f"request-{len(requests)}"
        request = {
            "request_id": identifier,
            "model": "writer" if stage == "ingest" else "reader",
            "arguments": {"content": {"events": case["events"]}},
        }
        metadata = {
            "workflow_id": case["id"],
            "stage": stage,
            "request_started": True,
            "status": "request_failed" if unavailable else "completed",
            "model": request["model"],
            "usage": None if unavailable else {"input_tokens": 10, "output_tokens": 5},
            "estimated_cost_usd": None if unavailable else 0.001,
            "budget_cost_usd": None if unavailable else 0.001,
            "response_id": "resp_abcdefghijklmnop",
        }
        response = {
            "request_id": identifier,
            "status": "failed" if unavailable else "completed",
        }
        if query_id:
            metadata["query_id"] = query_id
        if not unavailable:
            response.update(text=json.dumps(answer), usage=metadata["usage"])
        requests.append(request)
        responses.append(response)
        report["calls"].append(metadata)

    state = envelope(fact(value="Wrong"), fact("start_at", "2027-01-01", "e2"))
    call("ingest", state)
    report["workflows"] = [
        {
            "workflow_id": case["id"],
            "status": "ingest_failed" if unavailable else "ingested",
        }
    ]
    if unavailable:
        report["workflows"][0][
            "error"
        ] = "OpenAI HTTP 403: project proj_A0123456789 has no model access"
    else:
        report["workflows"][0]["state"] = state
    for query in case["queries"]:
        for arm in export.ARMS:
            correct = not (
                query["query_id"] == "q1" and arm in ("active_memory", "state")
            )
            answer = deepcopy(query["expected"])
            if not correct:
                answer["facts"][0]["value"] = "Wrong"
            row = {
                "workflow_id": case["id"],
                "query_id": query["query_id"],
                "arm": arm,
                "domain": case["domain"],
                "size_band": case["size_band"],
                "status": "not_run" if unavailable else "completed",
            }
            if not unavailable:
                row.update(
                    answer=answer,
                    score=exact_score(correct),
                    expected=query["expected"],
                    source_ids=[] if arm == "state" else ["e1", "e2"],
                )
                if arm != "state":
                    row.update(
                        raw_answer=json.dumps(answer),
                        support_complete=True,
                        support_ids_found=1,
                        support_ids_required=1,
                    )
                    call(arm, answer, query["query_id"])
            report["queries"].append(row)
    report["api_calls"] = len(report["calls"])
    report["api_charge_usd"] = 0 if unavailable else len(report["calls"]) * 0.001
    for arm in export.ARMS:
        correct = 0 if unavailable else (1 if arm in ("active_memory", "state") else 2)
        report["summary"]["arms"][arm] = {
            "planned": 2,
            "completed": 0 if unavailable else 2,
            "correct": correct,
            "exact_evidence_correct": correct,
        }
    return report, dataset, protocol, requests, responses


def write_run(directory: Path, *, unavailable=False):
    directory.mkdir(parents=True)
    report, dataset, protocol, requests, responses = rag_fixture(
        unavailable=unavailable
    )
    source_name = "src/threadmark/engine.py"
    source_data = b"def example_runtime():\n    return 'recorded evidence'\n"
    source_path = directory / "source" / source_name
    source_path.parent.mkdir(parents=True)
    source_path.write_bytes(source_data)
    protocol["sources"][source_name] = export.sha(source_data)
    dataset_bytes, protocol_bytes = export.encoded(dataset), export.encoded(protocol)
    report.update(
        dataset_sha256=export.sha(dataset_bytes),
        protocol_sha256=export.sha(protocol_bytes),
    )
    (directory / "dataset.json").write_bytes(dataset_bytes)
    (directory / "protocol.json").write_bytes(protocol_bytes)
    (directory / "report.json").write_bytes(export.encoded(report))
    for filename, values in (
        ("requests.jsonl", requests),
        ("responses.jsonl", responses),
    ):
        (directory / filename).write_bytes(
            b"".join((json.dumps(row) + "\n").encode() for row in values)
        )
    return report


def write_delta_run(directory: Path, arms=("state",)):
    write_run(directory)
    report, dataset, protocol = (
        json.loads((directory / name).read_text())
        for name in ("report.json", "dataset.json", "protocol.json")
    )
    requests = [
        json.loads(line)
        for line in (directory / "requests.jsonl").read_text().splitlines()
    ]
    responses = [
        json.loads(line)
        for line in (directory / "responses.jsonl").read_text().splitlines()
    ]
    selected = [
        (call, request, response)
        for call, request, response in zip(report["calls"], requests, responses)
        if call["stage"] == "ingest" or call["stage"] in arms
    ]
    report["calls"] = [row[0] for row in selected]
    requests, responses = [row[1] for row in selected], [row[2] for row in selected]
    report["arms"] = protocol["arms"] = list(arms)
    report["update_strategy"] = protocol["update_strategy"] = "event_delta"
    report["queries"] = [row for row in report["queries"] if row["arm"] in arms]
    report["summary"]["arms"] = {arm: report["summary"]["arms"][arm] for arm in arms}
    report["api_calls"] = len(report["calls"])
    report["api_charge_usd"] = len(report["calls"]) * 0.001
    case = dataset["workflows"][0]
    old_id, case["id"] = case["id"], "delta-workflow-one"
    for row in report["queries"] + report["calls"] + report["workflows"]:
        if row.get("workflow_id") == old_id:
            row["workflow_id"] = case["id"]
    dataset_bytes = export.encoded(dataset)
    (directory / "dataset.json").write_bytes(dataset_bytes)
    report["dataset_sha256"] = export.sha(dataset_bytes)
    protocol["seed"] = 2026102802
    requests[0]["arguments"]["content"] = {
        "entity_catalog": case["entity_catalog"],
        "new_events": case["events"],
    }
    responses[0]["text"] = json.dumps(
        {
            "updates": [
                {
                    "event_id": "e2",
                    "facts": [
                        {
                            "entity_id": "item-1",
                            "field": "start_at",
                            "value": "2027-01-01",
                        }
                    ],
                },
                {
                    "event_id": "e1",
                    "facts": [
                        {"entity_id": "item-1", "field": "owner", "value": "Wrong"}
                    ],
                },
            ]
        }
    )
    source = b"def sparse_delta_runtime():\n    return 'per-event assignments'\n"
    name = "src/threadmark/delta_memory.py"
    (directory / "source" / name).write_bytes(source)
    protocol["sources"][name] = export.sha(source)
    protocol_bytes = export.encoded(protocol)
    (directory / "protocol.json").write_bytes(protocol_bytes)
    report["protocol_sha256"] = export.sha(protocol_bytes)
    (directory / "report.json").write_bytes(export.encoded(report))
    for filename, rows in (
        ("requests.jsonl", requests),
        ("responses.jsonl", responses),
    ):
        (directory / filename).write_text(
            "".join(json.dumps(row) + "\n" for row in rows)
        )
    return report, dataset, protocol


class PublicEvidenceTests(unittest.TestCase):
    def test_local_compatibility_artifact_is_one_fixture_not_a_benchmark(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "smoke.json"
            provider = SCRIPT.parents[1] / "src/threadmark/local_provider.py"
            answer = envelope(
                {
                    "entity_id": "project-cedar",
                    "field": "owner",
                    "value": "Sam",
                    "evidence_event_ids": ["e1"],
                }
            )
            smoke = {
                "status": "generation_completed",
                "api_calls": 0,
                "generation_attempts": 1,
                "automatic_retries": 0,
                "calls": [{"status": "completed"}],
                "completion": {
                    "status": "completed",
                    "text": json.dumps(answer),
                    "usage": {"input_tokens": 409, "output_tokens": 35},
                },
                "content": {
                    "context": {
                        "source_events": [{"event_id": "e1", "text": "Sam is owner."}]
                    }
                },
                "parsed_facts": answer,
                "provider_source_sha256": export.sha(provider.read_bytes()),
            }
            path.write_bytes(export.encoded(smoke))
            evidence = export.Evidence()
            result = export._compatibility_smoke(evidence, path)
            self.assertEqual(result["generation_attempts"], 1)
            self.assertEqual(result["api_calls"], 0)
            self.assertIn("not a benchmark", result["scope"])
            self.assertEqual(result["input_tokens"], 409)
            export.verify_archive(evidence.archive(kind="compatibility-test"))
            smoke["automatic_retries"] = 1
            path.write_bytes(export.encoded(smoke))
            with self.assertRaisesRegex(ValueError, "no API/retry"):
                export._compatibility_smoke(export.Evidence(), path)

    def test_delta_reducer_uses_event_order_clears_and_preserves_untouched_fields(self):
        catalog = [
            {"entity_id": "a", "fields": ["owner", "date"]},
            {"entity_id": "b", "fields": ["owner"]},
        ]
        events = [{"event_id": f"e{i}", "text": str(i)} for i in range(1, 5)]
        batches = [
            {
                "new_events": events[:2],
                "text": json.dumps(
                    {
                        "updates": [
                            {
                                "event_id": "e2",
                                "facts": [
                                    {"entity_id": "a", "field": "owner", "value": "Ivo"}
                                ],
                            },
                            {
                                "event_id": "e1",
                                "facts": [
                                    {
                                        "entity_id": "a",
                                        "field": "owner",
                                        "value": "Mira",
                                    },
                                    {
                                        "entity_id": "a",
                                        "field": "date",
                                        "value": "Monday",
                                    },
                                ],
                            },
                        ]
                    }
                ),
            },
            {
                "new_events": events[2:],
                "text": json.dumps(
                    {
                        "updates": [
                            {
                                "event_id": "e4",
                                "facts": [
                                    {"entity_id": "a", "field": "date", "value": None}
                                ],
                            },
                            {
                                "event_id": "e3",
                                "facts": [
                                    {"entity_id": "b", "field": "owner", "value": "Zoe"}
                                ],
                            },
                        ]
                    }
                ),
            },
        ]
        result = export.replay_event_deltas(catalog, events, batches, {})
        values = {
            (fact["entity_id"], fact["field"]): fact
            for fact in result["state"]["facts"]
        }
        self.assertEqual(values[("a", "owner")]["value"], "Ivo")
        self.assertEqual(values[("a", "owner")]["evidence_event_ids"], ["e2"])
        self.assertIsNone(values[("a", "date")]["value"])
        self.assertEqual(values[("a", "date")]["evidence_event_ids"], ["e4"])
        self.assertEqual(values[("b", "owner")]["value"], "Zoe")
        self.assertFalse(result["gold_used_for_replay"])
        noop = export.replay_event_deltas(
            catalog, events, [{"new_events": events, "text": '{"updates":[]}'}], {}
        )
        self.assertTrue(
            all(
                fact["value"] is None and fact["evidence_event_ids"] == []
                for fact in noop["state"]["facts"]
            )
        )

    def test_delta_schema_rejects_duplicate_keys_unknown_routes_and_future_events(self):
        catalog = [{"entity_id": "a", "fields": ["owner"]}]
        events = [{"event_id": f"e{i}", "text": str(i)} for i in range(1, 4)]
        good = {
            "event_id": "e1",
            "facts": [{"entity_id": "a", "field": "owner", "value": "Mira"}],
        }
        bad_outputs = [
            '{"updates":[],"updates":[]}',
            json.dumps({"sparseupdates": []}),
            json.dumps({"updates": [good, good]}),
            json.dumps({"updates": [{**good, "event_id": "e3"}]}),
            json.dumps({"updates": [{**good, "facts": []}]}),
            json.dumps({"updates": [{**good, "facts": good["facts"] * 2}]}),
            json.dumps(
                {
                    "updates": [
                        {
                            **good,
                            "facts": [
                                {"entity_id": "a", "field": "unknown", "value": "X"}
                            ],
                        }
                    ]
                }
            ),
            json.dumps(
                {
                    "updates": [
                        {
                            **good,
                            "facts": [
                                {"entity_id": "a", "field": "owner", "value": True}
                            ],
                        }
                    ]
                }
            ),
            json.dumps(
                {
                    "updates": [
                        {
                            **good,
                            "facts": [
                                {**good["facts"][0], "evidence_event_ids": ["e1"]}
                            ],
                        }
                    ]
                }
            ),
        ]
        for output in bad_outputs:
            with self.subTest(output_kind=output[:25]), self.assertRaises(ValueError):
                export.replay_event_deltas(
                    catalog,
                    events,
                    [
                        {"new_events": events[:2], "text": output},
                        {"new_events": events[2:], "text": '{"updates":[]}'},
                    ],
                    {},
                )
        with self.assertRaisesRegex(ValueError, "memory limit"):
            export.replay_event_deltas(
                catalog,
                events,
                [{"new_events": events, "text": json.dumps({"updates": [good]})}],
                {"max_memory_bytes": 5},
            )
        with self.assertRaisesRegex(ValueError, "in order"):
            export.replay_event_deltas(
                catalog,
                events,
                [{"new_events": events[::-1], "text": '{"updates":[]}'}],
                {},
            )

    def test_delta_subset_export_replays_raw_updates_and_keeps_prior_register_evidence(
        self,
    ):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            prior = root / "rag-final-20261028"
            write_run(prior)
            delta = root / "rag-delta-final-20261028"
            report, dataset, protocol = write_delta_run(
                delta, arms=("full_history", "bm25", "dense", "state")
            )
            audit = export.recompute_rag(report, dataset, protocol)
            self.assertEqual(
                set(audit["arms"]), {"full_history", "bm25", "dense", "state"}
            )
            self.assertEqual(audit["paired_vs_threadmark"], {})
            self.assertEqual(
                audit["paired_by_context_arm"]["state"]["dense"]["aggregate"][
                    "baseline_only"
                ],
                1,
            )
            evidence = export.Evidence()
            result = export._rag_run(evidence, delta, "final", final=True)
            self.assertEqual(
                result["audit"]["delta_replay"]["delta-workflow-one"]["status"],
                "verified_committed_state",
            )
            export.prepare(
                rag_run=prior,
                destination=root / "public",
                rag_prefix="rag-register-comparison-2026-10-28",
            )
            prior_bytes = (
                root / "public/rag-register-comparison-2026-10-28.json"
            ).read_bytes()
            export.prepare(rag_run=delta, destination=root / "public")
            summary = json.loads(
                (root / f"public/rag-comparison-{export.DATE}.json").read_text()
            )
            self.assertEqual(
                summary["final"]["method"]["update_strategy"], "event_delta"
            )
            self.assertIn("prior_register_final", summary["selection_provenance"])
            self.assertEqual(
                prior_bytes,
                (root / "public/rag-register-comparison-2026-10-28.json").read_bytes(),
            )
            with zipfile.ZipFile(
                root / f"public/rag-comparison-{export.DATE}-evidence.zip"
            ) as archive:
                self.assertIn("prior-register-final/report.json", archive.namelist())
                self.assertIn(
                    "final/source/src/threadmark/delta_memory.py", archive.namelist()
                )
            response_lines = (delta / "responses.jsonl").read_text().splitlines()
            response = json.loads(response_lines[0])
            sparse = json.loads(response["text"])
            sparse["updates"][1]["facts"][0]["value"] = "Mira"
            response["text"] = json.dumps(sparse)
            response_lines[0] = json.dumps(response)
            (delta / "responses.jsonl").write_text("\n".join(response_lines) + "\n")
            with self.assertRaisesRegex(ValueError, "Final predicted state"):
                export._rag_run(export.Evidence(), delta, "final", final=True)

    def test_state_only_development_has_explicit_scope_and_prefix_is_a_safe_stem(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            report, dataset, protocol = write_delta_run(root / "delta-dev")
            audit = export.recompute_rag(report, dataset, protocol)
            self.assertEqual(set(audit["arms"]), {"state"})
            self.assertIn("no reader arm", audit["quality_assessment"])
            self.assertEqual(audit["paired_by_context_arm"], {"state": {}})
            with self.assertRaisesRegex(ValueError, "filename stem"):
                export.prepare(
                    rag_run=root / "delta-dev",
                    destination=root / "public",
                    rag_prefix="../private",
                )

    def test_delta_export_rejects_missing_or_boolean_prior_seed(self):
        for seed in (None, True):
            with (
                self.subTest(seed_type=type(seed).__name__),
                TemporaryDirectory() as temporary,
            ):
                root = Path(temporary)
                prior = root / "rag-final-20261028"
                write_run(prior)
                delta = root / "rag-delta-final-20261028"
                write_delta_run(delta)
                protocol = json.loads((prior / "protocol.json").read_text())
                if seed is None:
                    protocol.pop("seed")
                else:
                    protocol["seed"] = seed
                data = export.encoded(protocol)
                (prior / "protocol.json").write_bytes(data)
                report = json.loads((prior / "report.json").read_text())
                report["protocol_sha256"] = export.sha(data)
                (prior / "report.json").write_bytes(export.encoded(report))
                with self.assertRaisesRegex(ValueError, "explicit distinct integer"):
                    export.prepare(rag_run=delta, destination=root / "public")
                self.assertFalse((root / "public").exists())

    def test_redaction_removes_private_values_but_retains_attempts_and_numeric_evidence(
        self,
    ):
        original = {
            "calls": [
                {"estimated_cost_usd": None, "request_started": True, "usage": None}
            ],
            "api_charge_usd": 0,
            "run_budget": {"hold": 99},
            "api_key": "sk-proj-" + "x" * 40,
            "error": "HTTP403 project proj_ABCDEFGHIJK cannot access model",
            "base": r"C:\Users\Private Person\models\weights",
            "credential": "Bearer " + "x" * 30,
            "download": "https://us.aws.cdn.hf.co/file?X-Amz-Signature=secret&X-Amz-Credential=id",
            "official": "https://huggingface.co/author/model/resolve/revision/model.safetensors",
        }
        public = export.public_copy(original)
        export.scan_public(public)
        self.assertEqual(public["calls"], original["calls"])
        self.assertEqual(public["known_estimated_api_cost_usd"], 0)
        self.assertNotIn("api_charge_usd", public)
        self.assertNotIn("run_budget", public)
        self.assertNotIn("api_key", public)
        self.assertEqual(public["official"], original["official"])
        self.assertIn("HTTP403", public["error"])
        self.assertNotIn("proj_", json.dumps(public))
        self.assertNotIn("X-Amz", json.dumps(public))

    def test_strict_values_citations_and_both_state_and_reader_regressions_are_recomputed(
        self,
    ):
        report, dataset, protocol, _, _ = rag_fixture()
        audit = export.recompute_rag(report, dataset, protocol)
        self.assertEqual(audit["arms"]["full_history"]["correct"], 2)
        self.assertEqual(audit["arms"]["state"]["correct"], 1)
        self.assertEqual(audit["memory_state"]["correct_fields"], 1)
        for arm in ("state", "active_memory"):
            pair = audit["paired_by_context_arm"][arm]["dense"]
            self.assertEqual(pair["aggregate"]["baseline_only"], 1)
            self.assertEqual(
                pair["by_domain"]["meeting"]["regressions"][0]["query_id"], "q1"
            )
            self.assertEqual(pair["by_size_band"]["8192"]["planned"], 2)
            self.assertEqual(pair["by_workflow"]["workflow-one"]["both_correct"], 1)
        with self.assertRaisesRegex(ValueError, "strings or null"):
            export.score_facts(envelope(fact(value=True)), envelope(fact(value="True")))
        cited = export.score_facts(
            envelope(fact(evidence="e2")), envelope(fact()), allowed_ids={"e1", "e2"}
        )
        self.assertTrue(cited["exact_values"])
        self.assertFalse(cited["exact_values_and_evidence"])
        with self.assertRaisesRegex(ValueError, "outside"):
            export.score_facts(
                envelope(fact(evidence="e3")), envelope(fact()), allowed_ids={"e1"}
            )
        with self.assertRaisesRegex(ValueError, "require source"):
            export.score_facts(envelope(fact(evidence=None)), envelope(fact()))

    def test_corrupted_saved_score_and_missing_denominator_are_rejected(self):
        report, dataset, protocol, _, _ = rag_fixture()
        report["queries"][0]["score"]["exact_values"] = False
        with self.assertRaisesRegex(ValueError, "recomputed"):
            export.recompute_rag(report, dataset, protocol)
        report, dataset, protocol, _, _ = rag_fixture()
        report["queries"].pop()
        with self.assertRaisesRegex(ValueError, "denominator"):
            export.recompute_rag(report, dataset, protocol)

    def test_export_preserves_unknown_sol_attempt_and_has_verified_distinct_hashes(
        self,
    ):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "rag-final-20261028"
            write_run(run)
            dev = root / "rag-dev-sol-writer-20261028"
            original = write_run(dev, unavailable=True)
            canonical_before = (dev / "report.json").read_bytes()
            outputs = export.prepare(rag_run=run, destination=root / "public")
            archive_path = (
                root / "public" / f"rag-comparison-{export.DATE}-evidence.zip"
            )
            payload = archive_path.read_bytes()
            manifest = export.verify_archive(payload)
            summary = json.loads(
                (root / "public" / f"rag-comparison-{export.DATE}.json").read_text()
            )
            failure = summary["development"][0]
            self.assertEqual(failure["status"], "halted")
            self.assertIn("not evaluable", failure["audit"]["quality_assessment"])
            self.assertEqual(failure["audit"]["cost"]["unknown_charge_attempts"], 1)
            self.assertEqual(
                failure["audit"]["cost"]["known_estimated_api_cost_usd"], 0
            )
            self.assertIsNone(failure["audit"]["cost"]["actual_total_charge_usd"])
            with zipfile.ZipFile(BytesIO(payload)) as archive:
                report = json.loads(
                    archive.read("development/rag-dev-sol-writer-20261028/report.json")
                )
                self.assertEqual(len(report["calls"]), len(original["calls"]))
                self.assertIsNone(report["calls"][0]["estimated_cost_usd"])
                self.assertEqual(report["known_estimated_api_cost_usd"], 0)
                self.assertNotIn("run_budget", report)
                self.assertTrue(
                    all(
                        info.date_time == (1980, 1, 1, 0, 0, 0)
                        for info in archive.infolist()
                    )
                )
                self.assertFalse(
                    any(
                        name.endswith((".pt", ".safetensors"))
                        for name in archive.namelist()
                    )
                )
            entry = next(
                row
                for row in manifest["files"]
                if row["member"]
                == "development/rag-dev-sol-writer-20261028/report.json"
            )
            self.assertEqual(entry["canonical_sha256"], export.sha(canonical_before))
            self.assertNotEqual(entry["canonical_sha256"], entry["public_sha256"])
            self.assertEqual(canonical_before, (dev / "report.json").read_bytes())
            self.assertEqual(len(outputs), 2)
            with self.assertRaises(FileExistsError):
                export.prepare(rag_run=run, destination=root / "public")
            export.prepare(rag_run=run, destination=root / "public-second")
            self.assertEqual(
                payload, (root / "public-second" / archive_path.name).read_bytes()
            )

    def test_canonical_source_and_journal_tampering_fail_before_publication(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "final"
            write_run(run)
            source = run / "source/src/threadmark/engine.py"
            saved = source.read_bytes()
            source.write_bytes(saved + b"# changed\n")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                export.prepare(rag_run=run, destination=root / "public")
            self.assertFalse((root / "public").exists())
            source.write_bytes(saved)
            responses = (run / "responses.jsonl").read_text().splitlines()
            row = json.loads(responses[1])
            row["text"] = json.dumps(envelope(fact(value="falsified")))
            responses[1] = json.dumps(row)
            (run / "responses.jsonl").write_text("\n".join(responses) + "\n")
            with self.assertRaisesRegex(ValueError, "raw response journal"):
                export.prepare(rag_run=run, destination=root / "public")

    def test_archive_integrity_and_credential_scanner_fail_closed(self):
        evidence = export.Evidence()
        evidence.add_json("report.json", {"correct": 3, "planned": 4})
        payload = evidence.archive(kind="test")
        with zipfile.ZipFile(BytesIO(payload)) as original:
            buffer = BytesIO()
            with zipfile.ZipFile(buffer, "w") as altered:
                for name in original.namelist():
                    altered.writestr(
                        name, b"{}" if name == "report.json" else original.read(name)
                    )
        with self.assertRaisesRegex(ValueError, "integrity"):
            export.verify_archive(buffer.getvalue())
        for private in (
            r"C:\Users\person\private.json",
            "https://host/file?Signature=bad",
            "proj_secret123",
            "hf_" + "x" * 30,
        ):
            with self.subTest(private_kind=private[:5]), self.assertRaises(ValueError):
                export.scan_public(private)
        with self.assertRaisesRegex(ValueError, "(?i)scoring-only"):
            export._no_gold({"content": {"gold_state": {"facts": []}}})

    def test_backbone_replay_uses_predicted_effects_and_checks_control_and_pair_scores(
        self,
    ):
        examples = []
        for index, day in enumerate(("Monday", "Tuesday")):
            examples.append(
                {
                    "id": f"case-{index}",
                    "synthetic": True,
                    "answer": day,
                    "events": [f"confirm-{day}", f"pending-{day}"],
                    "event_labels": [
                        {"action": "SET", "day": day, "state": day},
                        {"action": "KEEP", "day": None, "state": day},
                    ],
                    "pair_id": "pair-one",
                    "pair_variant": index,
                    "cohort": "direct",
                }
            )
        report = {
            "planned_cases_per_arm": 2,
            "arms": {},
            "paired": {
                "planned_cases": 2,
                "both_correct": 1,
                "candidate_only_correct": 1,
                "reference_only_correct": 0,
                "both_wrong": 0,
                "improvements": ["case-0"],
                "regressions": [],
                "accuracy_difference_candidate_minus_reference": 0.5,
                "counterfactual_pairs": 1,
                "candidate_both_variants_correct": 1,
                "reference_both_variants_correct": 0,
            },
        }
        for arm in ("candidate", "reference"):
            final = {
                "case-0": "Monday" if arm == "candidate" else "Wednesday",
                "case-1": "Tuesday",
            }
            predictions, event_records, records = [], [], []
            for index, case in enumerate(examples):
                predicted_day = final[case["id"]]
                for event_index, (outcome, action) in enumerate(
                    ((predicted_day, "SET"), ("KEEP", "KEEP"))
                ):
                    label = case["event_labels"][event_index]
                    predictions.append(
                        {"text": case["events"][event_index], "outcome": outcome}
                    )
                    event_records.append(
                        {
                            "id": case["id"],
                            "event_index": event_index,
                            "expected_action": action,
                            "predicted_action": action,
                            "expected_day": label["day"],
                            "predicted_day": predicted_day if action == "SET" else None,
                            "expected_state": case["answer"],
                            "predicted_state": predicted_day,
                        }
                    )
                donor = examples[(index + 1) % len(examples)]
                for mode, predicted in {
                    "active_memory": predicted_day,
                    "zero_memory": "UNKNOWN",
                    "swapped_memory": final[donor["id"]],
                    "wrong_memory": final[donor["id"]],
                }.items():
                    records.append(
                        {
                            "id": case["id"],
                            "mode": mode,
                            "expected": case["answer"],
                            "generated": predicted,
                            "correct": predicted == case["answer"],
                        }
                    )
            report["arms"][arm] = {
                "status": "completed",
                "frozen_base_unchanged": True,
                "base_has_no_gradients": True,
                "event_predictions": predictions,
                "test": {
                    "event_records": event_records,
                    "records": records,
                    "event_metrics": {
                        "events": 4,
                        "action_correct": 4,
                        "state_correct": 4 if arm == "candidate" else 2,
                        "set_events": 2,
                        "set_day_correct": 2 if arm == "candidate" else 1,
                    },
                },
            }
        audit = export.recompute_backbone(report, {"examples": examples})
        self.assertEqual(audit["arms"]["candidate"]["correct"], 2)
        self.assertEqual(audit["arms"]["reference"]["correct"], 1)
        self.assertEqual(audit["paired"]["candidate_both_variants_correct"], 1)
        changed = deepcopy(report)
        changed["arms"]["candidate"]["test"]["records"][0]["correct"] = False
        with self.assertRaisesRegex(ValueError, "differs from replay"):
            export.recompute_backbone(changed, {"examples": examples})
        changed_examples = deepcopy(examples)
        changed_examples[0]["event_labels"][0]["state"] = "Friday"
        with self.assertRaisesRegex(ValueError, "annotation"):
            export.recompute_backbone(report, {"examples": changed_examples})

    def test_training_chain_preserves_parent_phases_and_rejects_broken_initialization(
        self,
    ):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            identity = {
                "frozen_parameter_sha256": "f" * 64,
                "checkpoint_sha256": "b" * 64,
                "training_chain": [],
            }
            for index, epochs in enumerate((3, 2)):
                directory = root / f"phase-{index}"
                directory.mkdir()
                config = {
                    "epochs": epochs,
                    "batch_size": 2,
                    "rank": 8,
                    "learning_rate": 0.0001,
                    "seed": 20261028,
                    "train_count": 256,
                    "augment_events": 0 if index == 0 else 896,
                }
                report = {
                    "status": "development_only_completed",
                    "test_used_for_optimization": False,
                    "frozen_parameter_sha256_before": "f" * 64,
                    "config": config,
                    "selected_epoch": 1,
                    "trajectory": [{"epoch": epoch} for epoch in range(1, epochs + 1)],
                    "initialized_checkpoint_sha256": None if index == 0 else "a" * 64,
                }
                report_data = export.encoded(report)
                (directory / "report.json").write_bytes(report_data)
                training_data = export.encoded({"synthetic": True, "examples": []})
                (directory / "training-data.json").write_bytes(training_data)
                identity["training_chain"].append(
                    {
                        "run_directory": str(directory),
                        "config": config,
                        "development_report_sha256": export.sha(report_data),
                        "selected_epoch": 1,
                        "checkpoint_sha256": ("a" if index == 0 else "b") * 64,
                        "training_data_files": {
                            "training-data.json": {"sha256": export.sha(training_data)}
                        },
                    }
                )
            identity["development_report_sha256"] = identity["training_chain"][-1][
                "development_report_sha256"
            ]
            evidence = export.Evidence()
            phases = export._training_chain_evidence(evidence, identity, "candidate")
            self.assertEqual([phase["settings"]["epochs"] for phase in phases], [3, 2])
            self.assertEqual(len(evidence.members), 4)
            export.verify_archive(evidence.archive(kind="training-test"))
            broken = deepcopy(identity)
            broken["training_chain"][0]["checkpoint_sha256"] = "c" * 64
            with self.assertRaisesRegex(ValueError, "parent"):
                export._training_chain_evidence(export.Evidence(), broken, "candidate")


if __name__ == "__main__":
    unittest.main()
