"""Checks of the offline harness and fixed fictional integrity cases."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys

import pytest

from threadmark import Completion, Settings, WorkflowMemory

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/evaluate_workflow_integrity.py"
SPEC = importlib.util.spec_from_file_location("workflow_integrity_eval", SCRIPT)
EVAL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVAL)


def test_fixture_scope_has_twelve_cases_and_one_thousand_unique_updates():
    cases = EVAL.build_cases()
    assert len(cases) == 12 and len({c["id"] for c in cases}) == 12
    long_case = next(c for c in cases if c["id"] == "long-dispatch")
    assert len(long_case["events"]) == 1000
    assert len({e["event_id"] for e in long_case["events"]}) == 1000
    values = [ops[0]["value"] for ops in long_case["recorded_operations"].values()]
    assert len(values) == len(set(values)) == 1000
    assert all("replaces" in e["text"] for e in long_case["events"])
    assert sum(len(e["fields"]) for e in long_case["catalog"]) == 8


@pytest.mark.parametrize("case", EVAL.build_cases(), ids=lambda case: case["id"])
def test_recorded_integrity_cases(case):
    result = EVAL.evaluate_case(case, WorkflowMemory, Settings, Completion)
    assert result["snapshot_replay_verified"]
    assert result["request_input_isolation_verified"]
    assert result["recorded_provider_calls"] > 0
    for request in result["provider_requests"]:
        assert set(request["content"]) == {
            "entity_catalog",
            "reference_context",
            "new_events",
        }


def test_recorded_provider_rejects_oracle_and_future_event_input():
    case = EVAL.build_cases()[3]
    provider = EVAL.RecordedExtractor(case, Completion)
    provider.allowed = {case["events"][0]["event_id"]: case["events"][0]}
    request = dict(
        instructions="Fixture protocol",
        max_output_tokens=1000,
        content=dict(
            entity_catalog=case["catalog"],
            reference_context={},
            new_events=[case["events"][0]],
        ),
    )
    leaked = deepcopy(request)
    leaked["content"]["expected"] = case["checks"]
    with pytest.raises(AssertionError, match="leaked"):
        provider.complete(**leaked)
    future = deepcopy(request)
    future["content"]["new_events"] = [case["events"][1]]
    with pytest.raises(AssertionError, match="outside"):
        provider.complete(**future)
    assert not provider.requests


def test_packet_checks_detect_wrong_state_and_missing_explanatory_content():
    case = EVAL.build_cases()[2]
    provider = EVAL.RecordedExtractor(case, Completion)
    memory = WorkflowMemory(
        case["catalog"], provider, Settings(chunk_tokens=4000), count_tokens=len
    )
    provider.allowed = {e["event_id"]: e for e in case["events"]}
    memory.ingest(case["events"])
    wrong = deepcopy(case)
    wrong["checks"]["notes"].pop()
    with pytest.raises(AssertionError, match="content"):
        EVAL.check_packet(memory, wrong)


def test_source_snapshot_and_subprocess_report_are_explicitly_offline(
    tmp_path, monkeypatch
):
    # Test the real subprocess/import boundary with two source copies; no model.
    root = SCRIPT.parents[1]
    output = tmp_path / "result"
    assert (
        EVAL.main(
            [
                "--source-root",
                str(root),
                "--baseline-root",
                str(root),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    protocol = json.loads((output / "protocol.json").read_text(encoding="utf-8"))
    result = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert protocol["model_calls"] == protocol["api_calls"] == 0
    assert result["scope"] == EVAL.SCOPE
    assert result["results"]["candidate"]["total"] == 12
    assert result["results"]["candidate"]["passed"] == 12
    assert all(r["baseline"] == r["candidate"] for r in result["paired_changes"])
    assert protocol["sources"]["candidate"] == protocol["sources"]["baseline"]
    with pytest.raises(FileExistsError):
        EVAL.main(["--output", str(output)])
