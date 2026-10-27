"""Export verified, redacted synthetic comparisons without inference or networking.

Canonical inputs are read only. Outputs contain no model/adapter tensors, ledgers,
credential files or signed download URLs. All scores are recomputed from rows.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
from io import BytesIO
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any
from urllib.parse import parse_qsl, urlsplit
import zipfile

DATE = "2026-10-28"
POLICY = "threadmark-comparison-public-evidence-v1"
DEV_RUNS = (
    "rag-dev-v2-20261028-ready",
    "rag-dev-updater-v2-20261028",
    "rag-dev-low-20261028",
    "rag-dev-sol-writer-20261028",
    "rag-dev-luna-medium-writer-20261028",
    "rag-delta-dev-20261028",
)
ARMS = ("full_history", "recent_only", "bm25", "dense", "active_memory", "state")
DAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
OUTCOMES = (*DAYS, "CLEAR", "KEEP")
REMOVED = {
    "run_budget",
    "budget",
    "ledger",
    "project_budget",
    "project_budget_before",
    "project_budget_after",
    "run_budget_usd",
    "max_cost_usd",
    "budget_cost_usd",
    "known_budget_cost_usd",
    "reservation_id",
    "response_id",
    "pending_reservation_ids",
    "unknown_reservation_ids",
    "project_budget_snapshot_at_run_end",
}
SECRET_FIELD = re.compile(
    r"^(?:api[_-]?key|authorization|cookie|password|passwd|secret|client[_-]?secret|"
    r"access[_-]?token|refresh[_-]?token|session[_-]?token)$",
    re.I,
)
SECRET = re.compile(
    r"\b(?:sk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{20,}|hf_[A-Za-z0-9]{20,}|"
    r"Bearer\s+[A-Za-z0-9._-]{12,}|(?:AKIA|ASIA)[A-Z0-9]{16})\b",
    re.I,
)
PROJECT = re.compile(r"\bproj_[A-Za-z0-9_-]+\b")
PROVIDER_ID = re.compile(r"\b(?:req_|resp_)[A-Za-z0-9_-]{10,}\b")
ABSOLUTE = re.compile(
    r"(?<![A-Za-z0-9_])(?:[A-Za-z]:[\\/]|\\\\[^\\\s]+\\|file://|"
    r"/(?:Users|home|mnt|tmp|workspace|opt)/)[^\r\n\"'<>]*"
)
URL = re.compile(r"https?://[^\s\"'<>]+")
SIGNED_KEYS = {
    "signature",
    "policy",
    "key-pair-id",
    "token",
    "access_token",
    "auth",
    "authorization",
}
GOLD_FIELDS = {
    "expected",
    "gold_state",
    "gold_updates",
    "event_labels",
    "event_kinds",
    "construction_metrics",
    "counterfactual_event_index",
    "scenario_variant",
}
RAG_SOURCE_PREFIXES = ("src/threadmark/", "scripts/")
BACKBONE_SOURCES = {
    "compare_backbones.py",
    "robust_state_data_v3.py",
    "decoder_bridge.py",
    "lora_state_memory.py",
    "train_lora_state.py",
    "neural_state_memory.py",
    "train_state_memory.py",
    "run_experiment.py",
    "robust_state_data.py",
    "augment_event_data.py",
}
TRAINING_FIELDS = (
    "epochs",
    "batch_size",
    "rank",
    "learning_rate",
    "seed",
    "train_count",
    "augment_events",
)


def encoded(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode()


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load(path: Path) -> Any:
    return strict_json(path.read_text(encoding="utf-8"))


def strict_json(value: str) -> Any:
    def object_pairs(pairs):
        result = {}
        for key, item in pairs:
            if key in result:
                raise ValueError("Duplicate JSON object key")
            result[key] = item
        return result

    return json.loads(value, object_pairs_hook=object_pairs)


def _signed_url(value: str) -> bool:
    parsed = urlsplit(value)
    keys = {key.lower() for key, _ in parse_qsl(parsed.query, keep_blank_values=True)}
    return bool(
        parsed.username
        or parsed.password
        or keys & SIGNED_KEYS
        or any(key.startswith(("x-amz-", "x-goog-", "x-xet-")) for key in keys)
        or (
            parsed.query
            and (parsed.hostname or "").endswith((".cdn.hf.co", ".xethub.hf.co"))
        )
    )


def redact_text(value: str) -> str:
    value = URL.sub(
        lambda match: (
            "[redacted signed URL]" if _signed_url(match.group()) else match.group()
        ),
        value,
    )
    value = SECRET.sub("[redacted credential]", value)
    value = PROJECT.sub("[redacted project ID]", value)
    value = PROVIDER_ID.sub("[redacted provider ID]", value)
    return ABSOLUTE.sub("[redacted local path]", value)


def public_copy(value: Any, changes: Counter | None = None, path: str = "$") -> Any:
    """Preserve task arrays/numerics; remove private accounting and redact strings."""
    changes = changes if changes is not None else Counter()
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            child = f"{path}.{key}"
            if key in REMOVED or SECRET_FIELD.fullmatch(key):
                changes[(child, "removed private field")] += 1
                continue
            new_key = (
                "known_estimated_api_cost_usd"
                if key == "api_charge_usd"
                else redact_text(key)
            )
            if new_key != key:
                changes[
                    (
                        child,
                        (
                            "renamed cost estimate"
                            if key == "api_charge_usd"
                            else "redacted key"
                        ),
                    )
                ] += 1
            if new_key in result:
                raise ValueError(
                    f"Redaction would collide with another field at {path}"
                )
            result[new_key] = public_copy(item, changes, child)
        return result
    if isinstance(value, list):
        return [
            public_copy(item, changes, f"{path}[{i}]") for i, item in enumerate(value)
        ]
    if isinstance(value, str):
        redacted = redact_text(value)
        if redacted != value:
            changes[(re.sub(r"\[\d+\]", "[*]", path), "redacted string")] += 1
        return redacted
    return value


def scan_public(value: Any, path: str = "$") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key in REMOVED or SECRET_FIELD.fullmatch(key):
                raise ValueError(f"Private field remains at {path}")
            scan_public(key, path)
            scan_public(item, f"{path}.{key}")
    elif isinstance(value, list):
        for i, item in enumerate(value):
            scan_public(item, f"{path}[{i}]")
    elif isinstance(value, str):
        if (
            SECRET.search(value)
            or PROJECT.search(value)
            or PROVIDER_ID.search(value)
            or ABSOLUTE.search(value)
        ):
            raise ValueError(f"Private value remains at {path}")
        if any(_signed_url(match.group()) for match in URL.finditer(value)):
            raise ValueError(f"Signed URL remains at {path}")
        if "-----BEGIN" in value and "PRIVATE KEY-----" in value:
            raise ValueError(f"Private key remains at {path}")


def _fact_map(
    value: Any,
    *,
    allowed_ids: set[str] | None = None,
    max_value_chars: int | None = None,
    max_evidence_ids: int | None = None,
) -> dict:
    if (
        not isinstance(value, dict)
        or set(value) != {"facts"}
        or not isinstance(value["facts"], list)
    ):
        raise ValueError("Invalid facts object")
    result = {}
    for fact in value["facts"]:
        if not isinstance(fact, dict) or set(fact) != {
            "entity_id",
            "field",
            "value",
            "evidence_event_ids",
        }:
            raise ValueError("Invalid fact fields")
        if any(
            not isinstance(fact[key], str) or not fact[key]
            for key in ("entity_id", "field")
        ):
            raise ValueError("Invalid fact routing key")
        if fact["value"] is not None and not isinstance(fact["value"], str):
            raise ValueError("Fact values must be strings or null")
        if (
            max_value_chars is not None
            and isinstance(fact["value"], str)
            and len(fact["value"]) > max_value_chars
        ):
            raise ValueError("Fact value exceeds locked length limit")
        ids = fact["evidence_event_ids"]
        if (
            not isinstance(ids, list)
            or any(not isinstance(x, str) or not x for x in ids)
            or len(set(ids)) != len(ids)
        ):
            raise ValueError("Invalid evidence ID list")
        if fact["value"] is not None and not ids:
            raise ValueError("Non-null fact values require source evidence")
        if allowed_ids is not None and not set(ids) <= allowed_ids:
            raise ValueError("Answer cites evidence outside its supplied context")
        if max_evidence_ids is not None and len(ids) > max_evidence_ids:
            raise ValueError("Evidence count exceeds locked limit")
        key = (fact["entity_id"], fact["field"])
        if key in result:
            raise ValueError("Duplicate fact routing key")
        result[key] = fact
    return result


def score_facts(
    answer: Any,
    expected: Any,
    *,
    allowed_ids: set[str] | None = None,
    max_value_chars: int | None = None,
    max_evidence_ids: int | None = None,
) -> dict:
    actual = _fact_map(
        answer,
        allowed_ids=allowed_ids,
        max_value_chars=max_value_chars,
        max_evidence_ids=max_evidence_ids,
    )
    gold = _fact_map(expected)
    if set(actual) != set(gold):
        raise ValueError("Answer does not cover exactly the requested facts")
    correct = sum(actual[key]["value"] == fact["value"] for key, fact in gold.items())
    cited = sum(
        actual[key]["value"] == fact["value"]
        and set(actual[key]["evidence_event_ids"]) == set(fact["evidence_event_ids"])
        for key, fact in gold.items()
    )
    return {
        "correct_fields": correct,
        "planned_fields": len(gold),
        "exact_values": correct == len(gold),
        "exact_values_and_evidence": cited == len(gold),
        "correct_fields_with_evidence": cited,
    }


def _no_gold(value: Any) -> None:
    if isinstance(value, dict):
        if set(value) & GOLD_FIELDS:
            raise ValueError("Scoring-only fields occur in a recorded model request")
        for item in value.values():
            _no_gold(item)
    elif isinstance(value, list):
        for item in value:
            _no_gold(item)


def recompute_rag(report: dict, dataset: dict, protocol: dict) -> dict:
    """Recompute query, field, citation, source-coverage and paired results."""
    workflows = dataset["workflows"]
    cases = {case["id"]: case for case in workflows}
    if (
        len(cases) != len(workflows)
        or not cases
        or any(case.get("synthetic") is not True for case in workflows)
    ):
        raise ValueError("Expected unique synthetic workflow IDs")
    arms = tuple(protocol["arms"])
    if not arms or not set(arms) <= set(ARMS) or len(set(arms)) != len(arms):
        raise ValueError("Unexpected comparison arms")
    if tuple(report.get("arms", arms)) != arms:
        raise ValueError("Report arm order differs from locked protocol")
    queries = {
        (case["id"], query["query_id"]): query
        for case in workflows
        for query in case["queries"]
    }
    if len(queries) != sum(len(case["queries"]) for case in workflows):
        raise ValueError("Duplicate query IDs")
    rows = {}
    correctness = {}
    trace_lookup = {row["workflow_id"]: row for row in report["workflows"]}
    limits = {
        key: protocol.get("settings", {}).get(key)
        for key in ("max_value_chars", "max_evidence_ids")
    }
    totals = {arm: Counter(planned=len(queries)) for arm in arms}
    for row in report["queries"]:
        key = (row["workflow_id"], row["query_id"], row["arm"])
        if key in rows or key[:2] not in queries or key[2] not in arms:
            raise ValueError("Duplicate or unexpected query result")
        rows[key] = row
        if row.get("domain") != cases[key[0]]["domain"] or str(
            row.get("size_band")
        ) != str(cases[key[0]]["size_band"]):
            raise ValueError("Result domain or size band differs from dataset")
        expected = queries[key[:2]]["expected"]
        requested = {
            (x["entity_id"], x["field"]) for x in queries[key[:2]]["requested_facts"]
        }
        if requested != set(_fact_map(expected)):
            raise ValueError("Gold answer and public routing keys disagree")
        if "expected" in row and row["expected"] != expected:
            raise ValueError("Result embeds different scoring labels than dataset")
        arm = key[2]
        item = totals[arm]
        item["planned_fields"] += len(requested)
        correct = False
        if row["status"] == "completed":
            archive_ids = {event["event_id"] for event in cases[key[0]]["events"]}
            source_ids = row.get("source_ids", [])
            if (
                len(source_ids) != len(set(source_ids))
                or not set(source_ids) <= archive_ids
            ):
                raise ValueError("Invalid selected-source IDs")
            answer = row["answer"]
            if arm == "state":
                trace = trace_lookup.get(key[0], {})
                if trace.get("status") != "ingested":
                    raise ValueError(
                        "Completed state lookup has no committed ingestion"
                    )
                memory = _fact_map(trace["state"])
                if any(
                    memory.get(fact_key) != fact
                    for fact_key, fact in _fact_map(answer).items()
                ):
                    raise ValueError(
                        "State lookup answer differs from predicted memory"
                    )
            if arm != "state":
                parsed = strict_json(row["raw_answer"])
                if parsed != answer:
                    raise ValueError(
                        "Raw model answer differs from saved parsed answer"
                    )
            computed = score_facts(
                answer,
                expected,
                allowed_ids=archive_ids if arm == "state" else set(source_ids),
                **limits,
            )
            if encoded(row.get("score")) != encoded(computed):
                raise ValueError(
                    "Saved query score does not match independently recomputed score"
                )
            correct = computed["exact_values"]
            item.update(
                completed=1,
                correct=int(correct),
                exact_evidence_correct=int(computed["exact_values_and_evidence"]),
                correct_fields=computed["correct_fields"],
                correct_fields_with_evidence=computed["correct_fields_with_evidence"],
            )
        if "support_complete" in row:
            required = {
                event_id
                for fact in expected["facts"]
                for event_id in fact["evidence_event_ids"]
            }
            found = required & set(row["source_ids"])
            expected_support = (
                required <= set(row["source_ids"]),
                len(found),
                len(required),
            )
            if (
                row["support_complete"],
                row["support_ids_found"],
                row["support_ids_required"],
            ) != expected_support:
                raise ValueError(
                    "Saved source coverage does not match selected evidence"
                )
            item.update(
                source_packets_available=1,
                source_complete_packets=int(expected_support[0]),
            )
        correctness[key] = correct
    expected_keys = {(wid, qid, arm) for wid, qid in queries for arm in arms}
    if set(rows) != expected_keys:
        raise ValueError("Report does not retain every planned query denominator")
    summaries = {}
    for arm, counts in totals.items():
        summaries[arm] = {
            key: counts[key]
            for key in (
                "planned",
                "completed",
                "correct",
                "exact_evidence_correct",
                "planned_fields",
                "correct_fields",
                "correct_fields_with_evidence",
                "source_packets_available",
                "source_complete_packets",
            )
        }
        summaries[arm]["accuracy_over_planned"] = counts["correct"] / counts["planned"]
        summaries[arm]["field_accuracy_over_planned"] = (
            counts["correct_fields"] / counts["planned_fields"]
        )
    paired = {}
    for baseline in ("full_history", "recent_only", "bm25", "dense"):
        if "active_memory" not in arms or baseline not in arms:
            continue
        counts, regressions, improvements = Counter(), [], []
        for wid, qid in queries:
            a, b = (
                correctness[(wid, qid, "active_memory")],
                correctness[(wid, qid, baseline)],
            )
            counts[
                (
                    "both_correct"
                    if a and b
                    else (
                        "threadmark_only"
                        if a
                        else "baseline_only" if b else "both_wrong"
                    )
                )
            ] += 1
            if b and not a:
                regressions.append({"workflow_id": wid, "query_id": qid})
            if a and not b:
                improvements.append({"workflow_id": wid, "query_id": qid})
        paired[baseline] = {
            **{
                key: counts[key]
                for key in (
                    "both_correct",
                    "threadmark_only",
                    "baseline_only",
                    "both_wrong",
                )
            },
            "regressions": regressions,
            "improvements": improvements,
        }

    def matrix(candidate: str, baseline: str, selected: list[tuple]) -> dict:
        counts, regressions, improvements = Counter(), [], []
        for wid, qid in selected:
            a, b = correctness[(wid, qid, candidate)], correctness[(wid, qid, baseline)]
            counts[
                (
                    "both_correct"
                    if a and b
                    else (
                        "threadmark_only"
                        if a
                        else "baseline_only" if b else "both_wrong"
                    )
                )
            ] += 1
            identity = {
                "workflow_id": wid,
                "query_id": qid,
                "domain": cases[wid]["domain"],
                "size_band": str(cases[wid]["size_band"]),
            }
            if b and not a:
                regressions.append(identity)
            if a and not b:
                improvements.append(identity)
        return {
            "planned": len(selected),
            **{
                key: counts[key]
                for key in (
                    "both_correct",
                    "threadmark_only",
                    "baseline_only",
                    "both_wrong",
                )
            },
            "regressions": regressions,
            "improvements": improvements,
        }

    grouped_pairs = {}
    all_queries = list(queries)
    for candidate in ("active_memory", "state"):
        if candidate not in arms:
            continue
        grouped_pairs[candidate] = {}
        for baseline in ("full_history", "recent_only", "bm25", "dense"):
            if baseline not in arms:
                continue
            groups = {"aggregate": matrix(candidate, baseline, all_queries)}
            for label, attribute in (
                ("by_workflow", "id"),
                ("by_domain", "domain"),
                ("by_size_band", "size_band"),
            ):
                selected_groups = defaultdict(list)
                for wid, qid in all_queries:
                    selected_groups[str(cases[wid][attribute])].append((wid, qid))
                groups[label] = {
                    name: matrix(candidate, baseline, selected)
                    for name, selected in sorted(selected_groups.items())
                }
            grouped_pairs[candidate][baseline] = groups
    traces = {row["workflow_id"]: row for row in report["workflows"]}
    if len(traces) != len(report["workflows"]) or not set(traces) <= set(cases):
        raise ValueError("Duplicate or unknown workflow trace")
    state = Counter(planned_workflows=len(cases))
    for wid, case in cases.items():
        expected = case["gold_state"]
        state["planned_fields"] += len(_fact_map(expected))
        trace = traces.get(wid, {})
        if trace.get("status") == "ingested":
            computed = score_facts(
                trace["state"],
                expected,
                allowed_ids={e["event_id"] for e in case["events"]},
                **limits,
            )
            state.update(
                ingested_workflows=1,
                correct_fields=computed["correct_fields"],
                exact_workflows=int(computed["exact_values"]),
                correct_fields_with_evidence=computed["correct_fields_with_evidence"],
            )
    cost = cost_scope(report.get("calls", []))
    if (
        report.get("backend") == "openai"
        and report.get("api_calls") != cost["started_requests"]
    ):
        raise ValueError("Reported API attempt count differs from recorded calls")
    recorded = report.get("summary", {})
    for arm, result in summaries.items():
        for key in ("planned", "completed", "correct", "exact_evidence_correct"):
            if recorded.get("arms", {}).get(arm, {}).get(key) != result[key]:
                raise ValueError("Saved aggregate differs from raw query rows")
    for arm, values in paired.items():
        saved = recorded.get("paired_vs_threadmark", {}).get(arm)
        if saved is not None and any(
            saved.get(key) != values[key]
            for key in (
                "both_correct",
                "threadmark_only",
                "baseline_only",
                "both_wrong",
            )
        ):
            raise ValueError("Saved paired aggregate differs from query rows")
    for baseline, groups in grouped_pairs.get("state", {}).items():
        saved = recorded.get("paired_vs_state", {}).get(baseline)
        if saved is not None and any(
            saved.get(key) != groups["aggregate"][key]
            for key in (
                "both_correct",
                "threadmark_only",
                "baseline_only",
                "both_wrong",
            )
        ):
            raise ValueError("Saved state paired aggregate differs from query rows")
    return {
        "schema": "threadmark-independent-rag-audit-v1",
        "workflow_count": len(cases),
        "planned_queries_per_arm": len(queries),
        "arms": summaries,
        "paired_vs_threadmark": paired,
        "paired_by_context_arm": grouped_pairs,
        "memory_state": dict(state),
        "cost": cost,
        "lifecycle_cost_by_arm": {
            arm: cost_scope(
                [
                    call
                    for call in report.get("calls", [])
                    if call.get("stage") == arm
                    or (
                        arm in ("active_memory", "state")
                        and call.get("stage") == "ingest"
                    )
                ]
            )
            for arm in arms
        },
        "quality_assessment": (
            "not evaluable: no completed reader answers; availability failure is not a quality judgment"
            if not any(totals[arm]["completed"] for arm in arms)
            else (
                "direct predicted-state outputs scored; no reader arm included"
                if arms == ("state",)
                else "synthetic answers scored over all planned queries"
            )
        ),
        "notes": [
            "State lookup and evidence-assisted reading are distinct arms.",
            "Known costs are usage-based estimates, not a provider invoice. Unknown attempts are not zero charges.",
        ],
    }


def cost_scope(calls: list[dict]) -> dict:
    started = [row for row in calls if row.get("request_started")]
    unknown = [row for row in started if row.get("estimated_cost_usd") is None]
    return {
        "recorded_completion_attempts": len(calls),
        "started_requests": len(started),
        "known_estimated_api_cost_usd": sum(
            row["estimated_cost_usd"]
            for row in started
            if row.get("estimated_cost_usd") is not None
        ),
        "unknown_charge_attempts": len(unknown),
        "actual_total_charge_usd": None,
        "charge_completeness": (
            "unknown: one or more attempted requests have no established usage"
            if unknown
            else "all recorded request estimates known; invoice unverified"
        ),
    }


def replay_event_deltas(
    catalog: list[dict], events: list[dict], batches: list[dict], settings: dict
) -> dict:
    """Replay extracted assignments using public order only; no gold input exists."""
    keys, entities = [], set()
    for entity in catalog:
        identifier, fields = entity["entity_id"], entity["fields"]
        if not isinstance(identifier, str) or not identifier or identifier in entities:
            raise ValueError("Invalid or duplicate catalog entity")
        entities.add(identifier)
        if (
            not isinstance(fields, list)
            or not fields
            or any(not isinstance(field, str) or not field for field in fields)
            or len(fields) != len(set(fields))
        ):
            raise ValueError("Invalid or duplicate catalog field")
        keys.extend((identifier, field) for field in fields)
    if not keys:
        raise ValueError("Delta replay requires a nonempty public catalog")
    event_ids = [event["event_id"] for event in events]
    if any(
        not isinstance(identifier, str) or not identifier for identifier in event_ids
    ) or len(set(event_ids)) != len(event_ids):
        raise ValueError("Invalid or duplicate public event ID")
    current = {
        key: {
            "entity_id": key[0],
            "field": key[1],
            "value": None,
            "evidence_event_ids": [],
        }
        for key in keys
    }
    cursor, updates_count, assignments = 0, 0, 0
    max_chars, max_memory = settings.get("max_value_chars", 256), settings.get(
        "max_memory_bytes", 12000
    )
    for batch in batches:
        chunk = batch["new_events"]
        if (
            not isinstance(chunk, list)
            or not chunk
            or chunk != events[cursor : cursor + len(chunk)]
        ):
            raise ValueError(
                "Delta request batches must partition original events in order"
            )
        positions = {event["event_id"]: index for index, event in enumerate(chunk)}
        raw = batch["text"]
        if not isinstance(raw, str) or len(raw.encode("utf-8")) > 4_000_000:
            raise ValueError("Invalid bounded raw delta output")
        value = strict_json(raw)
        if (
            not isinstance(value, dict)
            or set(value) != {"updates"}
            or not isinstance(value["updates"], list)
            or len(value["updates"]) > len(chunk)
        ):
            raise ValueError("Delta output must contain only bounded updates")
        seen_events = set()
        for update in value["updates"]:
            if not isinstance(update, dict) or set(update) != {"event_id", "facts"}:
                raise ValueError("Invalid delta update shape")
            identifier = update["event_id"]
            if (
                not isinstance(identifier, str)
                or identifier not in positions
                or identifier in seen_events
            ):
                raise ValueError("Unknown, old, future or duplicate delta event ID")
            seen_events.add(identifier)
            facts = update["facts"]
            if not isinstance(facts, list) or not 1 <= len(facts) <= len(keys):
                raise ValueError("Delta update facts must be bounded and nonempty")
            seen_keys = set()
            for fact in facts:
                if not isinstance(fact, dict) or set(fact) != {
                    "entity_id",
                    "field",
                    "value",
                }:
                    raise ValueError("Invalid delta fact shape")
                entity, field = fact["entity_id"], fact["field"]
                if not isinstance(entity, str) or not isinstance(field, str):
                    raise ValueError("Delta fact keys must be strings")
                key = (entity, field)
                if key not in current or key in seen_keys:
                    raise ValueError("Unknown or duplicate delta fact key")
                seen_keys.add(key)
                content = fact["value"]
                if content is not None and (
                    not isinstance(content, str) or len(content) > max_chars
                ):
                    raise ValueError("Delta values must be bounded strings or null")
        for update in sorted(
            value["updates"], key=lambda row: positions[row["event_id"]]
        ):
            updates_count += 1
            for fact in update["facts"]:
                current[(fact["entity_id"], fact["field"])] = {
                    **fact,
                    "evidence_event_ids": [update["event_id"]],
                }
                assignments += 1
        state = {"facts": [current[key] for key in keys]}
        memory_bytes = len(
            json.dumps(
                state,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode()
        )
        if memory_bytes > max_memory:
            raise ValueError("Replayed delta state exceeds locked memory limit")
        cursor += len(chunk)
    if cursor != len(events):
        raise ValueError("Delta journals do not cover all ingested original events")
    return {
        "state": {"facts": [current[key] for key in keys]},
        "batch_count": len(batches),
        "processed_event_count": cursor,
        "extracted_event_updates": updates_count,
        "assignments": assignments,
        "ordering": "Original public event order; model update-array order is ignored.",
        "gold_used_for_replay": False,
    }


class Evidence:
    def __init__(self) -> None:
        self.members: dict[str, bytes] = {}
        self.origins: list[dict] = []
        self.changes: Counter = Counter()

    def add(
        self,
        name: str,
        original: bytes,
        *,
        jsonl: bool = False,
        text: bool = False,
        source_reference: str | None = None,
    ) -> None:
        pure = PurePosixPath(name)
        if pure.is_absolute() or ".." in pure.parts or name in self.members:
            raise ValueError("Unsafe or duplicate archive member")
        if pure.suffix not in (".json", ".jsonl", ".py", ".md", ".txt"):
            raise ValueError("Only allowlisted text evidence may be exported")
        if jsonl:
            values = [
                json.loads(line)
                for line in original.decode().splitlines()
                if line.strip()
            ]
            public = public_copy(values, self.changes, name)
            scan_public(public)
            output = b"".join(
                (
                    json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False)
                    + "\n"
                ).encode()
                for row in public
            )
        elif text:
            source_text = original.decode("utf-8")
            public = redact_text(source_text)
            if public != source_text:
                self.changes[(name, "redacted source text")] += 1
            scan_public(public)
            output = public.encode()
        else:
            public = public_copy(json.loads(original), self.changes, name)
            scan_public(public)
            output = encoded(public)
        self.members[name] = output
        self.origins.append(
            {
                "member": name,
                "source_reference": source_reference or name,
                "canonical_sha256": sha(original),
                "canonical_bytes": len(original),
                "public_sha256": sha(output),
                "public_bytes": len(output),
            }
        )

    def add_json(self, name: str, value: Any) -> None:
        self.add(name, encoded(value))

    def archive(self, *, kind: str) -> bytes:
        manifest = {
            "schema": "threadmark-comparison-evidence-manifest-v1",
            "kind": kind,
            "redaction_policy": POLICY,
            "files": sorted(self.origins, key=lambda row: row["member"]),
            "redactions": [
                {"field_path": path, "action": action, "occurrences": count}
                for (path, action), count in sorted(self.changes.items())
            ],
            "notes": [
                "Canonical hashes identify immutable pre-redaction sources; public hashes verify the exported members.",
                "The manifest does not hash itself. All other archive members are listed exactly once.",
                "No model/adapter tensors, credential files, ledgers, or signed CDN URLs are included.",
                "Task arrays, failed attempts and numeric evidence remain intact; private operational fields and strings are redacted.",
                "The historical Sol access failure establishes availability only, with unknown charge; it does not measure model quality.",
            ],
        }
        scan_public(manifest)
        members = {**self.members, "manifest.json": encoded(manifest)}
        output = BytesIO()
        with zipfile.ZipFile(
            output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
        ) as archive:
            for name, payload in sorted(members.items()):
                info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.create_system = 3
                info.external_attr = 0o100644 << 16
                archive.writestr(
                    info, payload, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9
                )
        return output.getvalue()


def _check_hash(path: Path, expected: str) -> bytes:
    data = path.read_bytes()
    if sha(data) != expected:
        raise ValueError(f"Canonical input hash mismatch for {path.name}")
    return data


def _source_name(name: str, *, backbone: bool = False) -> str:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or path.suffix != ".py" or "\\" in name:
        raise ValueError("Invalid runtime source name")
    if backbone and name not in BACKBONE_SOURCES:
        raise ValueError("Unexpected backbone runtime source")
    if not backbone and not name.startswith(RAG_SOURCE_PREFIXES):
        raise ValueError("Unexpected RAG runtime source")
    return name


def _rag_run(evidence: Evidence, run: Path, prefix: str, *, final: bool) -> dict:
    report, dataset, protocol = (
        load(run / name) for name in ("report.json", "dataset.json", "protocol.json")
    )
    if report.get("status") not in (
        ("completed",) if final else ("completed", "halted", "interrupted")
    ):
        raise ValueError(
            "Only a completed final or terminal development run may be exported"
        )
    _check_hash(run / "dataset.json", report["dataset_sha256"])
    _check_hash(run / "protocol.json", report["protocol_sha256"])
    audit = recompute_rag(report, dataset, protocol)
    journals = {}
    for name in ("requests.jsonl", "responses.jsonl"):
        data = (run / name).read_bytes()
        journals[name] = [
            json.loads(line) for line in data.decode().splitlines() if line.strip()
        ]
    requests, responses = journals["requests.jsonl"], journals["responses.jsonl"]
    ids = [row["request_id"] for row in requests]
    if (
        len(set(ids)) != len(ids)
        or ids != [row["request_id"] for row in responses]
        or len(ids) != len(report["calls"])
    ):
        raise ValueError(
            "Request/response journals do not cover every completion attempt in order"
        )
    for request in requests:
        _no_gold(request["arguments"])
    query_rows = {
        (row["workflow_id"], row["query_id"], row["arm"]): row
        for row in report["queries"]
    }
    last_updates = {}
    delta_batches = defaultdict(list)
    strategy = protocol.get("update_strategy", "register")
    if (
        strategy not in ("register", "event_delta")
        or report.get("update_strategy", strategy) != strategy
    ):
        raise ValueError("Unknown or mismatched locked update strategy")
    if (
        strategy == "event_delta"
        and "src/threadmark/delta_memory.py" not in protocol["sources"]
    ):
        raise ValueError("Delta comparison must retain its frozen delta runtime source")
    cases = {case["id"]: case for case in dataset["workflows"]}
    for call, request, response in zip(report["calls"], requests, responses):
        if call["stage"] == "ingest" and strategy == "event_delta":
            content = request["arguments"]["content"]
            if (
                set(content) != {"entity_catalog", "new_events"}
                or content["entity_catalog"]
                != cases[call["workflow_id"]]["entity_catalog"]
            ):
                raise ValueError(
                    "Delta model input must contain only its original public catalog and new events"
                )
            if "text" in response:
                delta_batches[call["workflow_id"]].append(
                    {"new_events": content["new_events"], "text": response["text"]}
                )
        if "text" not in response:
            continue
        if response.get("usage") != call.get("usage"):
            raise ValueError("Raw response usage differs from recorded call usage")
        if call["stage"] == "ingest" and strategy == "register":
            last_updates[call["workflow_id"]] = response["text"]
        elif call["stage"] in ARMS and call["stage"] != "state":
            key = (call["workflow_id"], call["query_id"], call["stage"])
            if query_rows[key].get("raw_answer") != response["text"]:
                raise ValueError("Saved query answer differs from raw response journal")
    delta_replays = {}
    for trace in report["workflows"]:
        if trace["status"] == "ingested":
            if strategy == "event_delta":
                case = cases[trace["workflow_id"]]
                replay = replay_event_deltas(
                    case["entity_catalog"],
                    case["events"],
                    delta_batches[trace["workflow_id"]],
                    protocol.get("settings", {}),
                )
                final_state = replay["state"]
                delta_replays[trace["workflow_id"]] = {
                    "status": "verified_committed_state",
                    **replay,
                }
            else:
                final_state = strict_json(last_updates[trace["workflow_id"]])
            if final_state != trace["state"]:
                raise ValueError(
                    "Final predicted state differs from last raw updater response"
                )
        elif strategy == "event_delta":
            delta_replays[trace["workflow_id"]] = {
                "status": "ingestion_failed_no_committed_state",
                "recorded_batches_with_outputs": len(
                    delta_batches[trace["workflow_id"]]
                ),
                "gold_used_for_replay": False,
            }
    if strategy == "event_delta":
        audit["delta_replay"] = delta_replays
    for name in (
        "report.json",
        "dataset.json",
        "protocol.json",
        "requests.jsonl",
        "responses.jsonl",
    ):
        evidence.add(
            f"{prefix}/{name}", (run / name).read_bytes(), jsonl=name.endswith(".jsonl")
        )
    for name, expected in protocol["sources"].items():
        name = _source_name(name)
        evidence.add(
            f"{prefix}/source/{name}",
            _check_hash(run / "source" / name, expected),
            text=True,
        )
    evidence.add_json(f"{prefix}/independent-audit.json", audit)
    return {
        "run": run.name,
        "status": report["status"],
        "reader_model": report["model"],
        "updater_model": protocol.get("updater_model", report["model"]),
        "audit": audit,
        "method": {
            "update_strategy": strategy,
            "seed": protocol.get("seed"),
            "settings": protocol.get("settings", {}),
            "updater_reasoning_effort": protocol.get("updater_reasoning_effort"),
            "domains": protocol.get("domains"),
            "bands_native_tokens": protocol.get("bands_native_tokens"),
            "per_domain_band": protocol.get("per_domain_band"),
            "dense_model": protocol.get("dense_model"),
            "dense_method": protocol.get("dense_method"),
        },
        "canonical_report_sha256": sha((run / "report.json").read_bytes()),
        "canonical_protocol_sha256": sha((run / "protocol.json").read_bytes()),
        "canonical_dataset_sha256": sha((run / "dataset.json").read_bytes()),
    }


def _replay(outcomes: list[str]) -> tuple[str, list[str]]:
    state, prefixes = "UNKNOWN", []
    for outcome in outcomes:
        if outcome not in OUTCOMES:
            raise ValueError("Invalid predicted weekday effect")
        if outcome != "KEEP":
            state = "UNKNOWN" if outcome == "CLEAR" else outcome
        prefixes.append(state)
    return state, prefixes


def recompute_backbone(report: dict, dataset: dict) -> dict:
    examples = dataset["examples"]
    cases = {row["id"]: row for row in examples}
    if len(cases) != len(examples) or len(examples) != report["planned_cases_per_arm"]:
        raise ValueError("Backbone case count/identity mismatch")
    by_arm = {}
    final_states = {}
    for arm in ("candidate", "reference"):
        result = report["arms"][arm]
        if (
            result.get("status") != "completed"
            or result.get("frozen_base_unchanged") is not True
            or result.get("base_has_no_gradients") is not True
        ):
            raise ValueError("Incomplete backbone arm or failed frozen-base invariant")
        predictions = {
            row["text"]: row["outcome"] for row in result["event_predictions"]
        }
        if len(predictions) != len(result["event_predictions"]) or set(predictions) != {
            text for row in examples for text in row["events"]
        }:
            raise ValueError("Backbone event predictions do not exactly cover inputs")
        finals, event_rows, metrics = {}, [], Counter()
        for case in examples:
            if (
                case.get("synthetic") is not True
                or case["answer"] not in (*DAYS, "UNKNOWN")
                or len(case["events"]) != len(case["event_labels"])
            ):
                raise ValueError(
                    "Invalid synthetic backbone example or annotation length"
                )
            outcomes = [predictions[text] for text in case["events"]]
            final, states = _replay(outcomes)
            finals[case["id"]] = final
            gold_final, gold_prefixes = _replay(
                [
                    label["day"] if label["action"] == "SET" else label["action"]
                    for label in case["event_labels"]
                ]
            )
            if gold_final != case["answer"] or gold_prefixes != [
                label["state"] for label in case["event_labels"]
            ]:
                raise ValueError("Invalid backbone final annotation")
            for index, (outcome, state, label) in enumerate(
                zip(outcomes, states, case["event_labels"])
            ):
                action, day = ("SET", outcome) if outcome in DAYS else (outcome, None)
                metrics.update(
                    events=1,
                    action_correct=int(action == label["action"]),
                    state_correct=int(state == label["state"]),
                )
                if label["action"] == "SET":
                    metrics.update(
                        set_events=1, set_day_correct=int(day == label["day"])
                    )
                event_rows.append(
                    {
                        "id": case["id"],
                        "event_index": index,
                        "expected_action": label["action"],
                        "predicted_action": action,
                        "expected_day": label["day"],
                        "predicted_day": day,
                        "expected_state": label["state"],
                        "predicted_state": state,
                    }
                )
        if (
            event_rows != result["test"]["event_records"]
            or dict(metrics) != result["test"]["event_metrics"]
        ):
            raise ValueError(
                "Backbone prefix/event evidence differs from replayed predictions"
            )
        records = {}
        counts = Counter()
        for row in result["test"]["records"]:
            key = (row["id"], row["mode"])
            if key in records or key[0] not in cases:
                raise ValueError("Duplicate or unknown backbone result")
            records[key] = row
        for index, case in enumerate(examples):
            wrong = next(
                other for other in examples if other["answer"] != case["answer"]
            )
            expected_controls = {
                "active_memory": finals[case["id"]],
                "zero_memory": "UNKNOWN",
                "swapped_memory": finals[examples[(index + 1) % len(examples)]["id"]],
                "wrong_memory": finals[wrong["id"]],
            }
            for mode, predicted in expected_controls.items():
                row = records[(case["id"], mode)]
                correct = predicted == case["answer"]
                if (
                    row["generated"] != predicted
                    or row["expected"] != case["answer"]
                    or row["correct"] is not correct
                ):
                    raise ValueError(
                        "Backbone answer/control score differs from replay"
                    )
                counts[mode] += correct
        if len(records) != 4 * len(examples):
            raise ValueError("Unexpected backbone control rows")
        saved_scores = result["test"].get("scores")
        if saved_scores is not None:
            for mode in (
                "active_memory",
                "zero_memory",
                "swapped_memory",
                "wrong_memory",
            ):
                saved = saved_scores[mode]
                if (
                    saved["correct"] != counts[mode]
                    or saved["total"] != len(examples)
                    or abs(saved["accuracy"] - counts[mode] / len(examples)) > 1e-12
                ):
                    raise ValueError(
                        "Saved backbone aggregate differs from replayed rows"
                    )
        by_arm[arm] = {
            "planned_cases": len(examples),
            "correct": counts["active_memory"],
            "accuracy": counts["active_memory"] / len(examples),
            "controls_correct": dict(counts),
            "event_metrics": dict(metrics),
        }
        final_states[arm] = finals
    pairs = defaultdict(list)
    paired = Counter(planned_cases=len(examples))
    regressions, improvements = [], []
    for case in examples:
        a = final_states["candidate"][case["id"]] == case["answer"]
        b = final_states["reference"][case["id"]] == case["answer"]
        paired[
            (
                "both_correct"
                if a and b
                else (
                    "candidate_only_correct"
                    if a
                    else "reference_only_correct" if b else "both_wrong"
                )
            )
        ] += 1
        pairs[case["pair_id"]].append(case)
        if a and not b:
            improvements.append(case["id"])
        if b and not a:
            regressions.append(case["id"])
    if any(
        len(group) != 2 or {row["pair_variant"] for row in group} != {0, 1}
        for group in pairs.values()
    ):
        raise ValueError("Backbone cases must retain complete counterfactual pairs")
    computed = {
        key: paired[key]
        for key in (
            "planned_cases",
            "both_correct",
            "candidate_only_correct",
            "reference_only_correct",
            "both_wrong",
        )
    }
    computed.update(
        improvements=improvements,
        regressions=regressions,
        accuracy_difference_candidate_minus_reference=by_arm["candidate"]["accuracy"]
        - by_arm["reference"]["accuracy"],
        counterfactual_pairs=len(pairs),
        candidate_both_variants_correct=sum(
            all(final_states["candidate"][row["id"]] == row["answer"] for row in group)
            for group in pairs.values()
        ),
        reference_both_variants_correct=sum(
            all(final_states["reference"][row["id"]] == row["answer"] for row in group)
            for group in pairs.values()
        ),
    )
    for key, value in computed.items():
        saved = report["paired"].get(key)
        if isinstance(value, float):
            if abs(saved - value) > 1e-12:
                raise ValueError("Backbone paired accuracy mismatch")
        elif saved != value:
            raise ValueError("Backbone paired results mismatch")
    return {
        "schema": "threadmark-independent-backbone-audit-v1",
        "arms": by_arm,
        "paired": computed,
        "scope": "Weekday/CLEAR/KEEP classifier component; no RAG or reader generation.",
        "api_calls": 0,
    }


def _training_chain_evidence(
    evidence: Evidence, identity: dict, arm: str
) -> list[dict]:
    phases = identity.get("training_chain", [])
    public_phases = []
    previous_checkpoint = None
    for index, phase in enumerate(phases):
        directory = Path(phase["run_directory"])
        data = _check_hash(
            directory / "report.json", phase["development_report_sha256"]
        )
        report = json.loads(data)
        if (
            report.get("status") != "development_only_completed"
            or report.get("test_used_for_optimization") is not False
            or report.get("test")
            or report.get("frozen_parameter_sha256_before")
            != identity["frozen_parameter_sha256"]
        ):
            raise ValueError(
                "Invalid completed development phase in initialization chain"
            )
        if report.get("initialized_checkpoint_sha256") != previous_checkpoint:
            raise ValueError(
                "Training phase initialization hash does not match its parent"
            )
        if (
            report["config"] != phase["config"]
            or report["selected_epoch"] != phase["selected_epoch"]
        ):
            raise ValueError("Training phase settings differ from locked recipe")
        prefix = f"training/{arm}/phase-{index+1:02d}"
        evidence.add(f"{prefix}/report.json", data)
        file_hashes = {}
        for name, record in phase["training_data_files"].items():
            if name not in ("training-data.json", "training-augmentation.json"):
                raise ValueError("Unexpected training phase evidence file")
            evidence.add(
                f"{prefix}/{name}", _check_hash(directory / name, record["sha256"])
            )
            file_hashes[name] = record["sha256"]
        public_phases.append(
            {
                "phase": index + 1,
                "settings": {key: report["config"].get(key) for key in TRAINING_FIELDS},
                "completed_epochs": len(report.get("trajectory", [])),
                "selected_epoch": report["selected_epoch"],
                "checkpoint_sha256": phase["checkpoint_sha256"],
                "initialized_checkpoint_sha256": report.get(
                    "initialized_checkpoint_sha256"
                ),
                "development_report_sha256": phase["development_report_sha256"],
                "training_data_sha256": file_hashes,
            }
        )
        previous_checkpoint = phase["checkpoint_sha256"]
    if phases and (
        previous_checkpoint != identity["checkpoint_sha256"]
        or phases[-1]["development_report_sha256"]
        != identity["development_report_sha256"]
    ):
        raise ValueError("Training chain does not end at the selected checkpoint")
    return public_phases


def _backbone_run(evidence: Evidence, run: Path) -> dict:
    report, dataset, recipe = (
        load(run / name)
        for name in ("report.json", "final-test-data.json", "locked-recipe.json")
    )
    if report.get("status") != "completed":
        raise ValueError("Backbone comparison must be completed")
    _check_hash(run / "final-test-data.json", report["dataset_sha256"])
    _check_hash(run / "locked-recipe.json", report["recipe_sha256"])
    if dataset["manifest"]["final_recipe_sha256"] != report["recipe_sha256"]:
        raise ValueError("Backbone dataset is not bound to its pre-generation recipe")
    audit = recompute_backbone(report, dataset)
    for name in ("report.json", "final-test-data.json", "locked-recipe.json"):
        evidence.add(f"final/{name}", (run / name).read_bytes())
    for name, expected in recipe["source_hashes"].items():
        name = _source_name(name, backbone=True)
        snapshot = run / "source" / name
        source = snapshot if snapshot.is_file() else run.parents[1] / name
        evidence.add(f"final/source/{name}", _check_hash(source, expected), text=True)
    selected_models = {}
    for arm, identity in recipe["arms"].items():
        if arm not in ("candidate", "reference"):
            raise ValueError("Unexpected backbone identity")
        training_run = Path(identity["run_directory"])
        training_bytes = _check_hash(
            training_run / "report.json", identity["development_report_sha256"]
        )
        evidence.add(f"training/{arm}/report.json", training_bytes)
        training = json.loads(training_bytes)
        selected_models[arm] = {
            key: identity.get(key)
            for key in (
                "model_type",
                "hidden_size",
                "selected_epoch",
                "rank",
                "checkpoint_sha256",
                "frozen_parameter_sha256",
                "tokenizer_render_signature",
            )
        }
        selected_models[arm]["training_settings"] = {
            key: training.get("config", {}).get(key) for key in TRAINING_FIELDS
        }
        selected_models[arm]["training_chain"] = _training_chain_evidence(
            evidence, identity, arm
        )
        model_manifest = Path(identity["base_directory"]) / "download-manifest.json"
        if model_manifest.is_file():
            provenance = load(model_manifest)
            repository, revision = provenance.get("repository", ""), provenance.get(
                "revision", ""
            )
            weight = identity["base_files"].get("model.safetensors", {})
            if (
                provenance.get("status") != "verified_complete"
                or not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository)
                or not re.fullmatch(r"[0-9a-f]{40}", revision)
                or provenance.get("expected_weight_sha256") != weight.get("sha256")
                or provenance.get("expected_weight_bytes") != weight.get("bytes")
            ):
                raise ValueError(
                    "Model provenance does not match locked weight fingerprint"
                )
            selected_models[arm]["model_source"] = {
                "repository": repository,
                "revision": revision,
                "weight_sha256": weight["sha256"],
                "weight_bytes": weight["bytes"],
                "canonical_url": f"https://huggingface.co/{repository}/resolve/{revision}/model.safetensors",
            }
        for name, record in identity["training_data_files"].items():
            if name not in ("training-data.json", "training-augmentation.json"):
                raise ValueError("Unexpected training evidence file")
            evidence.add(
                f"training/{arm}/{name}",
                _check_hash(training_run / name, record["sha256"]),
            )
    evidence.add_json("final/independent-audit.json", audit)
    chains = {arm: model["training_chain"] for arm, model in selected_models.items()}
    matched_training = {
        "requested_schedules_match": bool(chains["candidate"] and chains["reference"])
        and [phase["settings"] for phase in chains["candidate"]]
        == [phase["settings"] for phase in chains["reference"]],
        "training_data_bytes_match": bool(chains["candidate"] and chains["reference"])
        and [phase["training_data_sha256"] for phase in chains["candidate"]]
        == [phase["training_data_sha256"] for phase in chains["reference"]],
        "note": "Matched requested schedules do not require the development-selected epoch to be identical for different backbones.",
    }
    return {
        "status": "completed",
        "audit": audit,
        "canonical_report_sha256": sha((run / "report.json").read_bytes()),
        "canonical_recipe_sha256": sha((run / "locked-recipe.json").read_bytes()),
        "limitations": report.get("limitations", []),
        "checkpoint_tensors_included": False,
        "selected_models": selected_models,
        "matched_training": matched_training,
        "reproduction_note": "Checkpoint hashes identify selected local adapters; adapter/base tensors are not redistributed in this evidence archive.",
    }


def verify_archive(payload: bytes) -> dict:
    """Check public member hashes and the allowlisted archive boundary."""
    with zipfile.ZipFile(BytesIO(payload)) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Duplicate ZIP members")
        manifest = json.loads(archive.read("manifest.json"))
        scan_public(manifest)
        expected = {row["member"]: row for row in manifest["files"]}
        if set(names) != set(expected) | {"manifest.json"}:
            raise ValueError("ZIP member list differs from integrity manifest")
        for name, row in expected.items():
            data = archive.read(name)
            if len(data) != row["public_bytes"] or sha(data) != row["public_sha256"]:
                raise ValueError("Public member integrity mismatch")
            if PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts:
                raise ValueError("Unsafe ZIP member name")
            if PurePosixPath(name).suffix not in (
                ".json",
                ".jsonl",
                ".py",
                ".md",
                ".txt",
            ):
                raise ValueError("Unexpected binary/publication member")
            if name.endswith(".json"):
                scan_public(json.loads(data))
            elif name.endswith(".jsonl"):
                for line in data.decode("utf-8").splitlines():
                    if line.strip():
                        scan_public(json.loads(line))
            else:
                scan_public(data.decode("utf-8"))
        return manifest


def _write_new(path: Path, data: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(data)


def _compatibility_smoke(evidence: Evidence, path: Path) -> dict:
    data = path.read_bytes()
    smoke = strict_json(data.decode())
    if (
        smoke.get("status") != "generation_completed"
        or smoke.get("api_calls") != 0
        or smoke.get("generation_attempts") != 1
        or smoke.get("automatic_retries") != 0
        or len(smoke.get("calls", [])) != 1
        or smoke["completion"].get("status") != "completed"
    ):
        raise ValueError(
            "Expected one completed local compatibility attempt with no API/retry"
        )
    parsed = strict_json(smoke["completion"]["text"])
    if parsed != smoke["parsed_facts"]:
        raise ValueError("Compatibility raw response differs from parsed facts")
    expected = {
        "facts": [
            {
                "entity_id": "project-cedar",
                "field": "owner",
                "value": "Sam",
                "evidence_event_ids": ["e1"],
            }
        ]
    }
    allowed_ids = {
        event["event_id"] for event in smoke["content"]["context"]["source_events"]
    }
    result = score_facts(parsed, expected, allowed_ids=allowed_ids)
    if not result["exact_values_and_evidence"]:
        raise ValueError(
            "Compatibility answer does not match its single public fixture"
        )
    evidence.add("compatibility/local-provider-smoke.json", data)
    provider = Path(__file__).resolve().parents[1] / "src/threadmark/local_provider.py"
    evidence.add(
        "compatibility/source/local_provider.py",
        _check_hash(provider, smoke["provider_source_sha256"]),
        text=True,
    )
    return {
        "canonical_artifact_sha256": sha(data),
        "generation_attempts": 1,
        "api_calls": 0,
        "input_tokens": smoke["completion"]["usage"]["input_tokens"],
        "output_tokens": smoke["completion"]["usage"]["output_tokens"],
        "fixture_match": True,
        "scope": "One fictional local causal-generation compatibility fixture only; not a benchmark, training or tuning input.",
        "hardware_cost_usd": None,
    }


def _backbone_product(run: Path, compatibility_artifact: Path | None = None) -> tuple:
    evidence = Evidence()
    summary = {
        "schema": "threadmark-public-backbone-comparison-v1",
        "date": DATE,
        **_backbone_run(evidence, run),
        "redaction_policy": POLICY,
    }
    if compatibility_artifact is not None:
        summary["compatibility_smoke"] = _compatibility_smoke(
            evidence, compatibility_artifact
        )
    evidence.add_json("summary.json", summary)
    return f"backbone-comparison-{DATE}", summary, evidence.archive(kind="backbone")


def _publish_products(products: list[tuple], destination: Path) -> list[dict]:
    outputs = []
    for stem, summary, archive in products:
        verify_archive(archive)
        for suffix in (".json", "-evidence.zip"):
            if (destination / f"{stem}{suffix}").exists():
                raise FileExistsError(
                    "Public output already exists; never overwrite prior evidence"
                )
        public = public_copy(summary)
        scan_public(public)
        outputs.append((destination / f"{stem}.json", encoded(public)))
        outputs.append((destination / f"{stem}-evidence.zip", archive))
    destination.mkdir(parents=True, exist_ok=True)
    for path, data in outputs:
        _write_new(path, data)
    return [
        {"filename": path.name, "bytes": len(data), "sha256": sha(data)}
        for path, data in outputs
    ]


def prepare(
    *,
    rag_run: Path | None = None,
    destination: Path,
    backbone_run: Path | None = None,
    rag_prefix: str = f"rag-comparison-{DATE}",
    compatibility_artifact: Path | None = None,
) -> list[dict]:
    """Build all artifacts in memory; refuse overwrites and unfinished final runs."""
    if (rag_run and rag_run.resolve() == destination.resolve()) or (
        backbone_run and backbone_run.resolve() == destination.resolve()
    ):
        raise ValueError("Destination must differ from canonical run directories")
    if compatibility_artifact is not None and backbone_run is None:
        raise ValueError("Compatibility evidence requires a backbone comparison")
    if rag_run is None:
        if backbone_run is None:
            raise ValueError("Provide a RAG run, a backbone run, or both")
        return _publish_products(
            [_backbone_product(backbone_run, compatibility_artifact)], destination
        )
    if (
        not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}", rag_prefix)
        or ".." in rag_prefix
    ):
        raise ValueError("RAG prefix must be a safe filename stem")
    products = []
    rag_evidence = Evidence()
    final = _rag_run(rag_evidence, rag_run, "final", final=True)
    development = []
    for name in DEV_RUNS:
        directory = rag_run.parent / name
        if directory.is_dir():
            development.append(
                _rag_run(rag_evidence, directory, f"development/{name}", final=False)
            )
    rag_summary = {
        "schema": "threadmark-public-rag-comparison-v1",
        "date": DATE,
        "final": final,
        "development": development,
        "development_discovery": {
            "allowlisted_runs": list(DEV_RUNS),
            "included": [row["run"] for row in development],
            "not_found": [
                name for name in DEV_RUNS if not (rag_run.parent / name).is_dir()
            ],
        },
        "redaction_policy": POLICY,
        "notes": [
            "Development trials are preserved separately and are not pooled into final accuracy.",
            "All costs are known usage-based estimates; unknown attempted charges remain explicitly unknown.",
            "Original source archives remain retained; this is not a general RAG replacement result.",
        ],
    }
    if final["method"]["update_strategy"] == "event_delta":
        prior = rag_run.parent / "rag-final-20261028"
        if not prior.is_dir() or prior.resolve() == rag_run.resolve():
            raise ValueError(
                "Delta final export requires the preserved prior register comparison"
            )
        prior_result = _rag_run(rag_evidence, prior, "prior-register-final", final=True)
        prior_seed, final_seed = prior_result["method"].get("seed"), final[
            "method"
        ].get("seed")
        if (
            prior_result["canonical_dataset_sha256"]
            == final["canonical_dataset_sha256"]
            or type(prior_seed) is not int
            or type(final_seed) is not int
            or prior_seed == final_seed
        ):
            raise ValueError(
                "Delta final requires distinct datasets and explicit distinct integer seeds"
            )
        rag_summary["selection_provenance"] = {
            "prior_register_final": prior_result,
            "prior_public_summary_filename": f"rag-register-comparison-{DATE}.json",
            "note": "Prior register results informed the delta hypothesis. Delta development selected the strategy before its separate fresh test. The experiments use distinct seeded instances from shared families and are not paired across strategies.",
        }
    rag_evidence.add_json("summary.json", rag_summary)
    products.append((rag_prefix, rag_summary, rag_evidence.archive(kind="rag")))
    if backbone_run is not None:
        products.append(_backbone_product(backbone_run, compatibility_artifact))
    return _publish_products(products, destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rag-run", type=Path)
    parser.add_argument("--backbone-run", type=Path)
    parser.add_argument(
        "--compatibility-artifact",
        type=Path,
        help="One local provider fixture to append to backbone evidence, not a benchmark",
    )
    parser.add_argument("--destination", type=Path, default=Path("benchmarks"))
    parser.add_argument(
        "--rag-prefix",
        default=f"rag-comparison-{DATE}",
        help="Filename stem for a distinct immutable RAG experiment export",
    )
    args = parser.parse_args()
    outputs = prepare(
        rag_run=args.rag_run,
        backbone_run=args.backbone_run,
        destination=args.destination,
        rag_prefix=args.rag_prefix,
        compatibility_artifact=args.compatibility_artifact,
    )
    print(
        json.dumps(
            {"status": "public_evidence_prepared", "outputs": outputs, "api_calls": 0},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
