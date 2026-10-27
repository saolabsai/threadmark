"""Offline CLI regressions: unused ingestion and unsupported writer settings."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from threadmark.providers import Completion

ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "src/threadmark").is_dir()
)
REVIEW_COPY = Path(__file__).with_name("compare_local.fixed.py")


@pytest.fixture
def driver():
    path = REVIEW_COPY if REVIEW_COPY.is_file() else ROOT / "scripts/compare_local.py"
    spec = importlib.util.spec_from_file_location("comparison_driver_regression", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT = ROOT
    return module


@pytest.fixture
def fixture_run(tmp_path, monkeypatch, driver):
    facts = {
        "facts": [
            {
                "entity_id": "cedar",
                "field": "owner",
                "value": "Sam",
                "evidence_event_ids": ["e1"],
            }
        ]
    }
    source = {
        "manifest": {"synthetic": True, "split": "dev"},
        "workflows": [
            {
                "id": "offline-fixture",
                "synthetic": True,
                "domain": "project",
                "size_band": "2048",
                "entity_catalog": [{"entity_id": "cedar", "fields": ["owner"]}],
                "events": [
                    {
                        "event_id": "e1",
                        "timestamp": "2026-10-28T09:00:00+00:00",
                        "sender": "Fictional coordinator",
                        "text": "Sam is the approved owner of Cedar.",
                    }
                ],
                "gold_state": deepcopy(facts),
                "queries": [
                    {
                        "query_id": "q1",
                        "text": "Who owns Cedar?",
                        "requested_facts": [{"entity_id": "cedar", "field": "owner"}],
                        "expected": deepcopy(facts),
                    }
                ],
            }
        ],
    }
    data = tmp_path / "fixture.json"
    data.write_text(json.dumps(source), encoding="utf-8")
    base = tmp_path / "base"
    base.mkdir()
    (base / "config.json").write_text("{}", encoding="utf-8")
    constructed = []

    class OfflineProvider:
        def __init__(self, path, *, model, device):
            self.model = model
            self.reasoning_effort = "none"
            self.calls = []
            constructed.append(self)

        def count_tokens(self, text):
            return len(text)

        def load(self):
            return {"fixture_provider": True, "model_load_seconds": 0}

        def complete(self, **kwargs):
            self.calls.append(
                {
                    "status": "completed",
                    "request_started": True,
                    "usage": {"input_tokens": 10, "output_tokens": 5},
                    "elapsed_seconds": 0,
                    "estimated_cost_usd": None,
                    "budget_cost_usd": None,
                }
            )
            return Completion(
                text=json.dumps(facts), model=self.model, usage=self.calls[-1]["usage"]
            )

    def forbidden_dense_load():
        raise AssertionError("These selected modes must not load dense weights")

    monkeypatch.setattr(driver, "LocalTransformersProvider", OfflineProvider)
    monkeypatch.setattr(
        driver.importlib.metadata, "version", lambda name: "offline-fixture"
    )
    monkeypatch.setattr(
        driver,
        "LocalDenseRetriever",
        lambda *args, **kwargs: SimpleNamespace(load=forbidden_dense_load),
    )
    output = tmp_path / "output"

    def execute(*arms):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "compare_local.py",
                "--base",
                str(base),
                "--dense-base",
                str(tmp_path / "unused-dense"),
                "--output",
                str(output),
                "--dataset",
                str(data),
                "--split",
                "dev",
                "--arms",
                *arms,
            ],
        )
        driver.main()
        return json.loads((output / "report.json").read_text(encoding="utf-8"))

    return SimpleNamespace(
        execute=execute, constructed=constructed, output=output, driver=driver
    )


def test_baseline_only_does_not_construct_memory_or_issue_updater_calls(
    fixture_run, monkeypatch
):
    def forbidden_memory(*args, **kwargs):
        raise AssertionError("A baseline-only comparison must not construct memory")

    monkeypatch.setattr(fixture_run.driver, "Threadmark", forbidden_memory)
    report = fixture_run.execute("full_history", "recent_only", "bm25")
    assert report["status"] == "completed"
    assert len(fixture_run.constructed[0].calls) == 3
    assert {row["stage"] for row in report["calls"]} == {
        "full_history",
        "recent_only",
        "bm25",
    }
    assert report["workflows"][0]["status"] == "ingestion_not_required"
    assert report["workflows"][0]["ingest_wall_seconds"] == 0
    for arm in ("full_history", "recent_only", "bm25"):
        item = report["summary"]["arms"][arm]
        assert item["planned"] == item["completed"] == item["correct"] == 1
        assert item["lifecycle_input_tokens"] == item["reader_input_tokens"] == 10
        assert item["ingest_wall_seconds"] == 0


def test_state_mode_still_ingests_once_and_needs_no_reader_call(fixture_run):
    report = fixture_run.execute("full_history", "state")
    assert report["status"] == "completed"
    assert report["workflows"][0]["status"] == "ingested"
    assert len(fixture_run.constructed[0].calls) == 2
    assert [row["stage"] for row in report["calls"]] == ["ingest", "full_history"]
    state = next(row for row in report["queries"] if row["arm"] == "state")
    assert state["status"] == "completed"
    assert state["score"]["exact_values"] is True
    assert state["reader_calls"] == 0
    assert report["summary"]["arms"]["state"]["lifecycle_input_tokens"] == 10


@pytest.mark.parametrize("effort", ["none", "low", "medium"])
def test_standalone_updater_effort_is_rejected_before_creating_a_run(
    tmp_path, monkeypatch, capsys, driver, effort
):
    def forbidden_provider(*args, **kwargs):
        raise AssertionError("Invalid CLI must not construct a hosted provider")

    monkeypatch.setattr(driver, "OpenAIProvider", forbidden_provider)
    output = tmp_path / "output"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_local.py",
            "--backend",
            "openai",
            "--dense-base",
            str(tmp_path / "unused-dense"),
            "--output",
            str(output),
            "--updater-effort",
            effort,
        ],
    )
    with pytest.raises(SystemExit) as error:
        driver.main()
    assert error.value.code == 2
    assert (
        "--updater-effort requires an explicit --updater-model"
        in capsys.readouterr().err
    )
    assert not output.exists()
