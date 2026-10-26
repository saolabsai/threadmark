from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from threadmark.cli import RecordedDemoProvider
from threadmark.datasets import demo_dataset, recorded_demo_responses
from threadmark.evaluation import (
    _summary,
    inspect_dataset,
    run_comparison,
    score_facts,
    validate_dataset,
)


def test_malformed_reported_usage_remains_unknown_without_counting_local_denials():
    calls = [
        {
            "stage": "updater",
            "request_started": True,
            "usage": {"input_tokens": None, "output_tokens": 10},
            "budget_cost_usd": None,
        },
        {"stage": "updater", "request_started": False, "usage": None},
    ]
    stage = _summary([], calls)["cost_by_stage"]["updater"]
    assert stage["completion_attempts"] == 2
    assert stage["requests_attempted"] == 1
    assert stage["usage_unknown_requests"] == 1


def counter(text):
    return 1


class RecordingProvider(RecordedDemoProvider):
    def __init__(self):
        super().__init__()
        self.requests = []

    def complete(self, **kwargs):
        self.requests.append(deepcopy(kwargs))
        result = super().complete(**kwargs)
        result.usage = {"input_tokens": 10, "output_tokens": 5}
        result.estimated_cost_usd = 0.01
        result.budget_cost_usd = 0.012
        return result


def test_paired_evaluation_counts_updater_cost_and_all_planned_queries():
    provider = RecordingProvider()
    report = run_comparison(demo_dataset(), provider, count_tokens=counter)
    assert report["status"] == "completed"
    assert len(provider.requests) == 5
    assert report["summary"]["full_history"]["correct_queries"] == 2
    assert report["summary"]["active_memory"]["planned_queries"] == 2
    assert report["summary"]["active_memory"]["query_accuracy_over_planned"] == 1
    assert report["summary"]["cost_by_stage"]["updater"][
        "known_api_cost_usd"
    ] == pytest.approx(0.01)
    assert report["summary"]["lifecycle_cost_usd"]["full_history"] == pytest.approx(
        0.02
    )
    assert report["summary"]["lifecycle_cost_usd"]["active_memory"] == pytest.approx(
        0.03
    )


def test_labels_and_construction_annotations_never_enter_inference():
    dataset = demo_dataset()
    workflow = dataset["workflows"][0]
    marker = "EXPECTED_LABEL_MUST_NEVER_REACH_PROVIDER"
    workflow["gold_state"] = marker
    workflow["events"][0]["construction_annotation"] = marker
    workflow["queries"][0]["expected"]["facts"][0]["value"] = marker
    workflow["queries"][0]["construction_annotation"] = marker
    provider = RecordingProvider()
    report = run_comparison(dataset, provider, count_tokens=counter)
    sent = json.dumps(provider.requests)
    assert marker not in sent
    assert '"expected"' not in sent
    assert '"gold_state"' not in sent
    assert report["summary"]["full_history"]["correct_queries"] == 1


def test_ingestion_failure_preserves_baseline_and_failed_denominators():
    class FailedUpdater:
        def __init__(self):
            self.calls_made = 0

        def complete(self, **kwargs):
            self.calls_made += 1
            if self.calls_made == 1:
                raise ValueError("malformed updater output")
            responses = recorded_demo_responses()
            answer = responses[1] if self.calls_made == 2 else responses[4]
            return SimpleNamespace(text=json.dumps(answer))

    provider = FailedUpdater()
    snapshots = []
    report = run_comparison(
        demo_dataset(), provider, count_tokens=counter, on_progress=snapshots.append
    )
    assert snapshots[0]["summary"]["active_memory"]["planned_queries"] == 2
    assert snapshots[0]["summary"]["full_history"]["planned_queries"] == 2
    assert report["summary"]["full_history"]["correct_queries"] == 2
    assert report["summary"]["active_memory"]["completed_queries"] == 0
    assert report["summary"]["active_memory"]["query_accuracy_over_planned"] == 0
    assert (
        sum(
            row["status"] == "skipped_ingestion_failure"
            for row in report["query_results"]
        )
        == 2
    )
    assert provider.calls_made == 3  # no retries of the failed updater


def test_reader_validation_failure_does_not_disappear_from_denominator():
    responses = recorded_demo_responses()
    responses[1] = {"facts": []}
    report = run_comparison(
        demo_dataset(), RecordedDemoProvider(responses), count_tokens=counter
    )
    assert report["summary"]["full_history"]["planned_queries"] == 2
    assert report["summary"]["full_history"]["completed_queries"] == 1
    assert report["summary"]["full_history"]["query_accuracy_over_planned"] == 0.5
    assert report["summary"]["active_memory"]["correct_queries"] == 2


@pytest.mark.parametrize("transport_started", [False, True])
def test_local_budget_denials_are_not_counted_as_unknown_transport_usage(
    transport_started,
):
    from threadmark.providers import ProviderError

    class StoppedProvider:
        def __init__(self):
            self.calls = []
            self.attempts = 0

        def complete(self, **kwargs):
            self.attempts += 1
            if self.attempts == 1 and transport_started:
                self.calls.append(
                    {"request_started": True, "usage": None, "status": "request_failed"}
                )
                raise ProviderError("Uncertain transport", request_started=True)
            raise ProviderError(
                "Local budget or provider-halted rejection", request_started=False
            )

    report = run_comparison(demo_dataset(), StoppedProvider(), count_tokens=counter)
    costs = report["summary"]["cost_by_stage"]
    assert costs["updater"]["completion_attempts"] == 1
    assert costs["updater"]["requests_attempted"] == int(transport_started)
    assert costs["updater"]["usage_unknown_requests"] == int(transport_started)
    assert costs["full_history"]["completion_attempts"] == 2
    assert costs["full_history"]["requests_attempted"] == 0
    assert costs["full_history"]["usage_unknown_requests"] == 0
    assert report["summary"]["full_history"]["planned_queries"] == 2
    assert report["summary"]["active_memory"]["planned_queries"] == 2


def test_exact_values_and_citations_are_separate_without_normalization():
    expected = demo_dataset()["workflows"][0]["queries"][0]["expected"]
    observed = deepcopy(expected)
    observed["facts"][0]["evidence_event_ids"] = ["e1"]
    score = score_facts(observed, expected)
    assert score["exact_values"] is True
    assert score["exact_values_and_evidence"] is False
    observed["facts"][0]["value"] = " Leon Park "
    assert score_facts(observed, expected)["exact_values"] is False


def test_inspection_is_offline_and_does_not_mutate_the_input():
    dataset = demo_dataset()
    original = deepcopy(dataset)
    plan = inspect_dataset(dataset, count_tokens=counter)
    assert plan["planned_queries_per_arm"] == 2
    assert plan["planned_reader_requests"] == 4
    assert plan["planned_updater_requests"] == 1
    assert dataset == original
    assert "expected" not in json.dumps(plan)


@pytest.mark.parametrize(
    "change",
    ["duplicate_workflow", "duplicate_query", "missing_label", "foreign_label_source"],
)
def test_invalid_benchmark_labels_are_rejected_before_inference(change):
    dataset = demo_dataset()
    workflow = dataset["workflows"][0]
    if change == "duplicate_workflow":
        dataset["workflows"].append(deepcopy(workflow))
    elif change == "duplicate_query":
        workflow["queries"].append(deepcopy(workflow["queries"][0]))
    elif change == "missing_label":
        del workflow["queries"][0]["expected"]
    else:
        workflow["queries"][0]["expected"]["facts"][0]["evidence_event_ids"] = [
            "foreign"
        ]
    with pytest.raises(ValueError):
        validate_dataset(dataset)
