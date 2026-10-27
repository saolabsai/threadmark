"""Prevent false success claims in the standalone local comparison report."""

import importlib.util
import json
from pathlib import Path
from copy import deepcopy
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location(
    "compare_local",
    Path(__file__).resolve().parents[1] / "scripts" / "compare_local.py",
)
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)


def test_failed_answers_stay_in_denominator_and_ingestion_is_charged_to_active():
    rows = []
    for arm in comparison.ARMS:
        row = {"workflow_id": "w", "query_id": "q", "arm": arm, "status": "completed"}
        row["score"] = {
            "exact_values": True,
            "exact_values_and_evidence": arm == "full_history",
        }
        if arm == "active_memory":
            row.update(status="failed")
            row.pop("score")
        rows.append(row)
    calls = [
        {
            "stage": "ingest",
            "usage": {"input_tokens": 500, "output_tokens": 60},
            "elapsed_seconds": 3,
        }
    ]
    calls += [
        {
            "stage": arm,
            "usage": {"input_tokens": 100, "output_tokens": 20},
            "elapsed_seconds": 1,
        }
        for arm in comparison.ARMS
    ]
    result = comparison.summary({"queries": rows, "calls": calls})
    active = result["arms"]["active_memory"]
    assert (active["planned"], active["completed"], active["correct"]) == (1, 0, 0)
    assert active["lifecycle_input_tokens"] == 600
    assert active["lifecycle_output_tokens"] == 80
    assert active["lifecycle_provider_seconds"] == 4
    assert result["arms"]["bm25"]["lifecycle_input_tokens"] == 100
    assert result["arms"]["full_history"]["exact_evidence_correct"] == 1
    assert result["arms"]["bm25"]["exact_evidence_correct"] == 0
    assert result["paired_vs_threadmark"]["bm25"]["baseline_only"] == 1


def test_unattempted_answers_do_not_disappear_from_report():
    rows = [
        {"workflow_id": "w", "query_id": "q", "arm": arm, "status": "not_run"}
        for arm in comparison.ARMS
    ]
    result = comparison.summary({"queries": rows, "calls": []})
    for arm in comparison.ARMS:
        assert result["arms"][arm]["planned"] == 1
        assert result["arms"][arm]["accuracy"] == 0
    assert result["paired_vs_threadmark"]["full_history"]["both_wrong"] == 1


@pytest.mark.parametrize("gold", ["Sam", "EVALUATOR_LABEL_MUST_NOT_ENTER_INFERENCE"])
def test_all_modes_run_without_leaking_evaluator_labels(monkeypatch, tmp_path, gold):
    """Exercise metadata/public-query separation through the actual CLI loop."""
    fixture = {
        "id": "w",
        "domain": "meeting",
        "size_band": "2048",
        "entity_catalog": [
            {"entity_id": "meeting-a", "label": "Launch", "fields": ["owner"]}
        ],
        "events": [
            {
                "event_id": "e1",
                "timestamp": "2026-10-28T10:00:00+00:00",
                "sender": "Lee",
                "text": "Sam owns Launch.",
            }
        ],
        "queries": [
            {
                "query_id": "q1",
                "text": "Who owns Launch?",
                "requested_facts": [{"entity_id": "meeting-a", "field": "owner"}],
                "expected": {
                    "facts": [
                        {
                            "entity_id": "meeting-a",
                            "field": "owner",
                            "value": gold,
                            "evidence_event_ids": ["e1"],
                        }
                    ]
                },
            }
        ],
    }
    observed = []

    class Provider:
        def __init__(self, path, *, model, device):
            self.model, self.reasoning_effort, self.calls = model, "none", []

        def count_tokens(self, text):
            return len(text)

        def load(self):
            return {"model_load_seconds": 0}

        def complete(self, **kwargs):
            observed.append(deepcopy(kwargs))
            usage = {"input_tokens": 10, "output_tokens": 5}
            self.calls.append(
                {
                    "usage": usage,
                    "elapsed_seconds": 1,
                    "request_started": True,
                    "budget_cost_usd": None,
                }
            )
            return SimpleNamespace(
                status="completed",
                usage=usage,
                text=json.dumps(
                    {
                        "facts": [
                            {
                                "entity_id": "meeting-a",
                                "field": "owner",
                                "value": "Sam",
                                "evidence_event_ids": ["e1"],
                            }
                        ]
                    }
                ),
            )

    class Dense:
        last_selection = {"index_cache_hit": False}

        def __init__(self, *args, **kwargs):
            pass

        def load(self):
            return {}

        def select(self, events, *args):
            return deepcopy(events)

    base = tmp_path / "base"
    base.mkdir()
    (base / "config.json").write_text("{}")
    output = tmp_path / "result"
    monkeypatch.setattr(comparison, "LocalTransformersProvider", Provider)
    monkeypatch.setattr(comparison, "LocalDenseRetriever", Dense)
    monkeypatch.setattr(comparison, "workflow", lambda *args: deepcopy(fixture))
    monkeypatch.setattr(comparison.importlib.metadata, "version", lambda name: "test")
    monkeypatch.setattr(
        "sys.argv",
        [
            "compare_local",
            "--base",
            str(base),
            "--dense-base",
            str(base),
            "--output",
            str(output),
            "--domains",
            "meeting",
            "--bands",
            "2048",
        ],
    )
    comparison.main()
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "completed"
    assert len(report["queries"]) == len(comparison.ARMS)
    assert all(row["query_id"] == "q1" for row in report["queries"])
    assert all(row["status"] == "completed" for row in report["queries"])
    assert all(
        row["score"]["exact_values"] == (gold == "Sam") for row in report["queries"]
    )
    # One updater and five readers; direct-state lookup must make no call.
    assert len(observed) == 6
    assert all(
        "expected" not in json.dumps(call) and "query_id" not in json.dumps(call)
        for call in observed
    )
    assert "EVALUATOR_LABEL" not in json.dumps(observed)
    assert len((output / "requests.jsonl").read_text().splitlines()) == 6
    assert len((output / "responses.jsonl").read_text().splitlines()) == 6
