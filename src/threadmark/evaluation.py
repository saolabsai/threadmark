"""Small, sequential paired evaluation with explicit planned denominators.

This is a portable evaluation tool, not the original concurrent research runner.
Labels are used only after inference to score the returned facts.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

from .config import Settings
from .schema import (
    catalog_keys,
    public_catalog,
    public_events,
    public_query,
    validate_facts,
)

ARMS = ("full_history", "active_memory")


def load_dataset(path: str | Path, settings: Settings | None = None) -> dict[str, Any]:
    """Read and validate an annotated synthetic or user-authored dataset."""
    with Path(path).open(encoding="utf-8-sig") as handle:
        value = json.load(handle)
    return validate_dataset(value, settings)


def validate_dataset(value: Any, settings: Settings | None = None) -> dict[str, Any]:
    """Validate labels separately and return a defensive copy.

    Each query needs an ``expected.facts`` envelope to make comparison scores
    meaningful. Applications without labels should use ``Threadmark.answer``.
    """
    settings = settings or Settings()
    if (
        not isinstance(value, dict)
        or not isinstance(value.get("workflows"), list)
        or not value["workflows"]
    ):
        raise ValueError("Dataset needs a nonempty workflows array")
    seen = set()
    for workflow in value["workflows"]:
        if not isinstance(workflow, dict):
            raise ValueError("Workflows must be objects")
        identifier = workflow.get("id")
        if (
            not isinstance(identifier, str)
            or not identifier.strip()
            or identifier in seen
        ):
            raise ValueError("Workflow IDs must be unique nonempty strings")
        seen.add(identifier)
        catalog = public_catalog(workflow.get("entity_catalog"))
        events = public_events(workflow.get("events"))
        queries = workflow.get("queries")
        if not events or not isinstance(queries, list) or not queries:
            raise ValueError("Each workflow needs events and queries")
        query_ids = set()
        for query in queries:
            public = public_query(query, catalog_keys(catalog))
            query_id = query.get("query_id")
            if (
                not isinstance(query_id, str)
                or not query_id.strip()
                or query_id in query_ids
            ):
                raise ValueError(
                    "Query IDs must be unique nonempty strings within a workflow"
                )
            query_ids.add(query_id)
            keys = [
                (fact["entity_id"], fact["field"]) for fact in public["requested_facts"]
            ]
            validate_facts(
                query.get("expected"),
                keys,
                [event["event_id"] for event in events],
                settings,
            )
    return deepcopy(value)


def public_workflow(workflow: dict[str, Any]) -> dict[str, Any]:
    """The only workflow values allowed into inference; never return labels."""
    catalog = public_catalog(workflow["entity_catalog"])
    return {
        "entity_catalog": catalog,
        "events": public_events(workflow["events"]),
        "queries": [
            public_query(query, catalog_keys(catalog)) for query in workflow["queries"]
        ],
    }


def inspect_dataset(
    dataset: dict[str, Any],
    settings: Settings | None = None,
    count_tokens: Callable[[str], int] | None = None,
) -> dict[str, Any]:
    """Validate and plan chunks without constructing a provider or making calls."""
    from .context import chunks_for, token_counter

    settings = settings or Settings()
    dataset = validate_dataset(dataset, settings)
    count_tokens = count_tokens or token_counter()
    workflows = []
    for workflow in dataset["workflows"]:
        public = public_workflow(workflow)
        chunks = chunks_for(public["events"], count_tokens, settings)
        workflows.append(
            {
                "workflow_id": workflow["id"],
                "domain": workflow.get("domain"),
                "events": len(public["events"]),
                "facts": len(catalog_keys(public["entity_catalog"])),
                "queries": len(public["queries"]),
                "updater_requests": len(chunks),
                "reader_requests": 2 * len(public["queries"]),
                "chunks": [[event["event_id"] for event in chunk] for chunk in chunks],
            }
        )
    payload = json.dumps(
        dataset, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return {
        "schema": "threadmark-plan-v1",
        "dataset_sha256": hashlib.sha256(payload).hexdigest(),
        "dataset_hash_encoding": "UTF-8 JSON with sorted keys and compact separators; not raw file bytes.",
        "settings": asdict(settings),
        "workflows": workflows,
        "planned_queries_per_arm": sum(row["queries"] for row in workflows),
        "planned_updater_requests": sum(row["updater_requests"] for row in workflows),
        "planned_reader_requests": sum(row["reader_requests"] for row in workflows),
        "execution": "Sequential workflows; both readers run after full ingestion. No automatic retries.",
        "scope": "Chunk counts are a request plan, not a price estimate or a latency prediction.",
    }


def score_facts(actual: dict[str, Any], expected: dict[str, Any]) -> dict[str, Any]:
    """Score exact values and exact citation sets, without output normalization."""
    actual_by_key = {
        (fact["entity_id"], fact["field"]): fact for fact in actual["facts"]
    }
    values = evidence = 0
    for fact in expected["facts"]:
        observed = actual_by_key.get((fact["entity_id"], fact["field"]))
        correct = observed is not None and observed["value"] == fact["value"]
        values += int(correct)
        evidence += int(
            correct
            and set(observed["evidence_event_ids"]) == set(fact["evidence_event_ids"])
        )
    planned = len(expected["facts"])
    return {
        "correct_fields": values,
        "planned_fields": planned,
        "exact_values": values == planned,
        "exact_values_and_evidence": evidence == planned,
        "correct_fields_with_evidence": evidence,
    }


class _TrackedProvider:
    def __init__(self, provider: Any) -> None:
        self.provider = provider
        self.calls: list[dict[str, Any]] = []
        self.context: dict[str, str] = {}
        for attribute in ("model", "reasoning_effort"):
            if hasattr(provider, attribute):
                setattr(self, attribute, getattr(provider, attribute))

    def complete(self, **kwargs: Any) -> Any:
        prior = len(getattr(self.provider, "calls", []))
        try:
            result = self.provider.complete(**kwargs)
        except Exception as exc:
            native = getattr(self.provider, "calls", [])
            row = deepcopy(native[-1]) if len(native) > prior else {}
            # Do not copy provider exception strings, credentials, or tracebacks.
            row.pop("error", None)
            row.setdefault(
                "request_started", bool(getattr(exc, "request_started", False))
            )
            row.update(self.context, status="failed", error_type=type(exc).__name__)
            self.calls.append(row)
            raise
        row = {
            "status": getattr(result, "status", "completed"),
            "request_started": getattr(result, "request_started", True),
            "usage": deepcopy(getattr(result, "usage", None)),
            "elapsed_seconds": getattr(result, "elapsed_seconds", None),
            "model": getattr(result, "model", None),
            "estimated_cost_usd": getattr(result, "estimated_cost_usd", None),
            "budget_cost_usd": getattr(result, "budget_cost_usd", None),
        }
        row.update(self.context)
        self.calls.append(row)
        return result


def _summary(
    results: list[dict[str, Any]], calls: list[dict[str, Any]]
) -> dict[str, Any]:
    summary = {}
    for arm in ARMS:
        rows = [row for row in results if row["arm"] == arm]
        completed = [row for row in rows if row["status"] == "completed"]
        correct = sum(row["score"]["exact_values"] for row in completed)
        evidence = sum(row["score"]["exact_values_and_evidence"] for row in completed)
        planned_fields = sum(row["planned_fields"] for row in rows)
        correct_fields = sum(row["score"]["correct_fields"] for row in completed)
        summary[arm] = {
            "planned_queries": len(rows),
            "completed_queries": len(completed),
            "correct_queries": correct,
            "exact_queries_with_evidence": evidence,
            "query_accuracy_over_planned": correct / len(rows) if rows else None,
            "correct_fields": correct_fields,
            "planned_fields": planned_fields,
            "field_accuracy_over_planned": (
                correct_fields / planned_fields if planned_fields else None
            ),
        }
    costs = {}
    for stage in ("updater", *ARMS):
        stage_calls = [row for row in calls if row["stage"] == stage]
        costs[stage] = {
            "completion_attempts": len(stage_calls),
            "requests_attempted": sum(
                bool(row.get("request_started")) for row in stage_calls
            ),
            "known_api_cost_usd": sum(
                row.get("estimated_cost_usd") or 0 for row in stage_calls
            ),
            "known_budget_cost_usd": sum(
                row.get("budget_cost_usd") or 0 for row in stage_calls
            ),
            "usage_unknown_requests": sum(
                bool(row.get("request_started")) and row.get("budget_cost_usd") is None
                for row in stage_calls
            ),
        }
    summary["cost_by_stage"] = costs
    summary["lifecycle_cost_usd"] = {
        "full_history": costs["full_history"]["known_api_cost_usd"],
        "active_memory": costs["updater"]["known_api_cost_usd"]
        + costs["active_memory"]["known_api_cost_usd"],
    }
    return summary


def run_comparison(
    dataset: dict[str, Any],
    provider: Any,
    *,
    settings: Settings | None = None,
    count_tokens: Callable[[str], int] | None = None,
    on_progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Run a paired evaluation. Failed and skipped queries stay in denominators.

    The caller supplies a budget-enforcing provider. Ingestion errors make that
    workflow's active queries unavailable; baseline queries are still attempted.
    Provider budget guards remain authoritative, including unknown-usage holds.
    """
    from .engine import Threadmark

    settings = settings or Settings()
    dataset = validate_dataset(dataset, settings)
    tracked = _TrackedProvider(provider)
    report: dict[str, Any] = {
        "schema": "threadmark-comparison-v1",
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "settings": asdict(settings),
        "method": "Memory-guided original evidence; sequential portable evaluation.",
        "dataset_manifest": deepcopy(dataset.get("manifest", {})),
        "query_results": [],
        "workflow_results": [],
        "calls": tracked.calls,
        "notes": [
            "All planned queries remain in accuracy denominators, including failures.",
            "Exact values are scored separately from exact citation sets.",
            "Active lifecycle cost includes every attempted updater and active reader request.",
            "Known cost is incomplete when usage is unknown; consult the budget ledger.",
            "Latency is not directly comparable with the original concurrent benchmark.",
        ],
    }

    def checkpoint() -> None:
        report["summary"] = _summary(report["query_results"], tracked.calls)
        if on_progress:
            on_progress(deepcopy(report))

    # Install planned rows before the first request, so an interrupted run retains
    # all planned denominators in the last checkpoint.
    rows = {}
    for workflow in dataset["workflows"]:
        for query in workflow["queries"]:
            for arm in ARMS:
                row = {
                    "workflow_id": workflow["id"],
                    "query_id": query["query_id"],
                    "arm": arm,
                    "status": "not_run",
                    "planned_fields": len(query["requested_facts"]),
                    "expected": deepcopy(query["expected"]),
                }
                rows[(workflow["id"], query["query_id"], arm)] = row
                report["query_results"].append(row)
    checkpoint()
    for workflow_index, workflow in enumerate(dataset["workflows"]):
        public = public_workflow(workflow)
        engine = Threadmark(
            public["entity_catalog"],
            tracked,
            settings=settings,
            count_tokens=count_tokens,
        )
        baseline = Threadmark.from_events(
            public["entity_catalog"],
            public["events"],
            tracked,
            settings=settings,
            count_tokens=count_tokens,
        )
        workflow_result = {"workflow_id": workflow["id"], "ingestion_status": "not_run"}
        report["workflow_results"].append(workflow_result)
        tracked.context = {"workflow_id": workflow["id"], "stage": "updater"}
        try:
            engine.ingest(public["events"])
            workflow_result.update(
                ingestion_status="completed", final_state=engine.state
            )
        except Exception as exc:
            workflow_result.update(
                ingestion_status="failed", error_type=type(exc).__name__
            )
        checkpoint()
        for query_index, (query, public_query_value) in enumerate(
            zip(workflow["queries"], public["queries"])
        ):
            arms = (
                ARMS
                if (workflow_index + query_index) % 2 == 0
                else tuple(reversed(ARMS))
            )
            for arm in arms:
                row = rows[(workflow["id"], query["query_id"], arm)]
                if (
                    arm == "active_memory"
                    and workflow_result["ingestion_status"] != "completed"
                ):
                    row.update(status="skipped_ingestion_failure")
                    checkpoint()
                    continue
                # Baseline construction never runs the updater and remains usable
                # even when the active path failed partway through ingestion.
                reader = baseline if arm == "full_history" else engine
                tracked.context = {
                    "workflow_id": workflow["id"],
                    "query_id": query["query_id"],
                    "stage": arm,
                }
                try:
                    answer = reader.answer(public_query_value, mode=arm)
                    row.update(
                        status="completed",
                        answer=answer,
                        score=score_facts(answer, query["expected"]),
                    )
                except Exception as exc:
                    row.update(status="failed", error_type=type(exc).__name__)
                checkpoint()
    report["status"] = "completed"
    report["completed_at"] = datetime.now(timezone.utc).isoformat()
    checkpoint()
    return report
