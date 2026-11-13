"""Frozen offline active-context accounting with application-supplied operations.

No extraction, answer accuracy, billed model calls, or network access. The
default tokenizer loads only verified cached o200k data. Missing cache fails.
Run --prepare-only first to inspect the frozen cases, or omit it to freeze and
then evaluate in one invocation. Existing frozen inputs must match exactly.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from html import escape
import importlib.metadata
import json
from pathlib import Path
import platform
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from threadmark import ActiveContext, ContextBudgetExceeded, WorkflowMemory
from threadmark.active_context import ACTIVE_CONTEXT_INSTRUCTIONS
from threadmark.context import token_counter
from threadmark.schema import canonical

SETTINGS = dict(
    max_input_tokens=6144,
    history_entries=8,
    claim_entries=4,
    recent_events=0,
    search_events=0,
    repetitions=3,
    ingestion_batch_events=64,
    tokenizer="o200k_base",
    network_allowed=False,
    model_calls_allowed=False,
    request_renderer="canonical JSON instructions/content envelope",
    baseline="same envelope and instructions; catalog, question and full original event archive",
    memory_limits=dict(max_packet_bytes=2_000_000, max_entries=10_000),
)
QUESTION = (
    "For the review, give the current owner, approved room and deadline; "
    "also identify the earlier approved Room B, pending proposal and unresolved decision."
)
BACKGROUND = (
    "Background observation only: the archive team reviewed inventory labels, "
    "checked the corridor map, compared stationery samples and discussed the "
    "weather. This entry introduces no instruction or approved change for the "
    "review. The unrelated log keeps its original wording for later inspection. "
) * 3


class NoModelCalls:
    calls = 0

    def complete(self, **kwargs):
        self.calls += 1
        raise RuntimeError("Offline evaluation forbids model calls")


def digest_bytes(value):
    return sha256(value).hexdigest()


def operation(kind, field, value=None, *, entity="review", proposal_id=None):
    return dict(
        op=kind,
        entity_id=entity,
        field=field,
        value=value,
        proposal_id=proposal_id,
        reason=None,
    )


def scenario(name, count, *, growing_history=False, total_facts=3):
    catalog = [dict(entity_id="review", fields=["owner", "room", "deadline"])]
    if total_facts > 3:
        catalog.append(
            dict(
                entity_id="archive",
                fields=[f"field_{i}" for i in range(total_facts - 3)],
            )
        )
    events, groups = [], []
    start = datetime(2026, 11, 11, 8, tzinfo=timezone.utc)

    def add(text, operations=()):
        identifier = f"{name}-e{len(events) + 1:05d}"
        events.append(
            dict(
                event_id=identifier,
                timestamp=(start + timedelta(seconds=len(events))).isoformat(),
                sender="Synthetic application",
                text=text,
            )
        )
        if operations:
            groups.append(dict(event_id=identifier, operations=list(operations)))
        return identifier

    first = add(
        "Ada owns the review in Room A; its deadline is Friday.",
        [
            operation("set", "owner", "Ada"),
            operation("set", "room", "Room A"),
            operation("set", "deadline", "Friday"),
        ],
    )
    proposal_b = add(
        "Propose Room B for the review as ROOM-B.",
        [operation("propose", "room", "Room B", proposal_id="ROOM-B")],
    )
    accepted_b = add(
        "Approve ROOM-B for the review.",
        [operation("accept", "room", proposal_id="ROOM-B")],
    )
    owner = add(
        "Bo now owns the review following Ada's handoff.",
        [operation("set", "owner", "Bo")],
    )
    cleared = add(
        "Clear the review deadline; no replacement is set.",
        [operation("clear", "deadline")],
    )
    proposal_d = add(
        "Propose Room D for the review as ROOM-D.",
        [operation("propose", "room", "Room D", proposal_id="ROOM-D")],
    )
    accepted_d = add(
        "Approve ROOM-D as the review's replacement room.",
        [operation("accept", "room", proposal_id="ROOM-D")],
    )
    add(
        "The review requires live captioning.",
        [operation("note", "requirement", "Live captioning is required")],
    )
    unresolved = add(
        "Approve the unknown review room proposal MISSING.",
        [operation("accept", "room", proposal_id="MISSING")],
    )
    pending = add(
        "Propose Room E for the review as PENDING; it is not approved.",
        [operation("propose", "room", "Room E", proposal_id="PENDING")],
    )
    for index in range(total_facts - 3):
        add(
            f"Archive field {index} is value {index}. " + BACKGROUND,
            [operation("set", f"field_{index}", f"value-{index}", entity="archive")],
        )
    while len(events) < count:
        index = len(events)
        if growing_history:
            add(
                f"Archive field 0 changes to phase {index}. " + BACKGROUND,
                [operation("set", "field_0", f"phase-{index}", entity="archive")],
            )
        else:
            add(f"Background log entry {index}. " + BACKGROUND)
    return dict(
        name=name,
        family=(
            "growing_history"
            if growing_history
            else ("growing_current_facts" if total_facts > 3 else "fixed_state")
        ),
        catalog=catalog,
        events=events,
        operations=dict(events=groups),
        question=QUESTION,
        requested_facts=[
            dict(entity_id="review", field=field)
            for field in ("owner", "room", "deadline")
        ],
        required_event_ids=[first, proposal_b, accepted_b],
        max_input_tokens=SETTINGS["max_input_tokens"],
        expected_outcome="compiled",
        compiler_options={
            key: SETTINGS[key]
            for key in (
                "history_entries",
                "claim_entries",
                "recent_events",
                "search_events",
            )
        },
        selection_mode="explicit three-field scope with caller-pinned historical source IDs",
        expected=dict(
            facts=[
                dict(
                    entity_id="review",
                    field="owner",
                    value="Bo",
                    evidence_event_ids=[owner],
                ),
                dict(
                    entity_id="review",
                    field="room",
                    value="Room D",
                    evidence_event_ids=[proposal_d, accepted_d],
                ),
                dict(
                    entity_id="review",
                    field="deadline",
                    value=None,
                    evidence_event_ids=[cleared],
                ),
            ],
            historical_fact=dict(
                entity_id="review",
                field="room",
                value="Room B",
                evidence_event_ids=[proposal_b, accepted_b],
            ),
            proposal_records=[
                dict(external_alias="ROOM-D", status="accepted", value="Room D"),
                dict(external_alias="PENDING", status="open", value="Room E"),
            ],
            unresolved_event_id=unresolved,
            source_ids=[
                first,
                proposal_b,
                accepted_b,
                owner,
                cleared,
                proposal_d,
                accepted_d,
                unresolved,
                pending,
            ],
        ),
    )


def scenarios():
    values = [scenario(f"fixed-{count}", count) for count in (32, 128, 512)]
    values += [
        scenario(f"history-{count}", count, growing_history=True, total_facts=4)
        for count in (32, 128, 512)
    ]
    values += [
        scenario(f"facts-{facts}", 128, total_facts=facts) for facts in (8, 16, 32)
    ]
    oversized = scenario("oversized-essential", 10)
    oversized["events"][3]["text"] += " " + BACKGROUND * 60
    oversized["expected_outcome"] = "budget_exceeded"
    oversized["family"] = "mandatory_source_overflow"
    values.append(oversized)
    all_facts = scenario("all-facts-pressure", 128, total_facts=32)
    all_facts["requested_facts"] = None
    all_facts["expected_outcome"] = "budget_exceeded"
    all_facts["family"] = "mandatory_state_overflow"
    values.append(all_facts)
    automatic = scenario("unpinned-history-128", 128)
    automatic.update(
        required_event_ids=[],
        family="unpinned_history",
        compiler_options=dict(
            history_entries=8, claim_entries=8, recent_events=2, search_events=4
        ),
        selection_mode="explicit three-field scope; no source pins; default optional settings",
    )
    automatic["expected"]["source_ids"] = automatic["expected"]["source_ids"][1:]
    values.append(automatic)
    relevant = scenario("relevant-history-512", 512)
    relevant.update(
        required_event_ids=[],
        family="growing_relevant_history",
        compiler_options=dict(
            history_entries=8, claim_entries=8, recent_events=2, search_events=4
        ),
        selection_mode="explicit three-field scope; no source pins; default optional settings",
    )
    old_owner_source = relevant["expected"]["facts"][0]["evidence_event_ids"][0]
    for index, event in enumerate(relevant["events"][10:], start=10):
        owner = (
            "Bo" if index == len(relevant["events"]) - 1 else f"Interim owner {index}"
        )
        event["text"] = (
            f"The review owner changes to {owner}. This is an operative owner handoff; the approved room and cleared deadline stay unchanged."
        )
        relevant["operations"]["events"].append(
            dict(
                event_id=event["event_id"],
                operations=[operation("set", "owner", owner)],
            )
        )
    latest_source = relevant["events"][-1]["event_id"]
    relevant["expected"]["facts"][0]["evidence_event_ids"] = [latest_source]
    relevant["expected"]["source_ids"] = [
        identifier
        for identifier in relevant["expected"]["source_ids"][1:]
        if identifier != old_owner_source
    ] + [latest_source]
    values.append(relevant)
    return values


def frozen_inputs(output):
    payload = dict(schema_version=1, settings=SETTINGS, scenarios=scenarios())
    raw = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    path = output / "scenarios.json"
    if path.exists() and path.read_bytes() != raw:
        raise ValueError("Frozen scenarios differ; choose a new --output directory")
    output.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_bytes(raw)
    source_paths = [Path(__file__), *sorted((ROOT / "src/threadmark").glob("*.py"))]
    sources = {
        str(path.relative_to(ROOT)).replace("\\", "/"): digest_bytes(path.read_bytes())
        for path in source_paths
    }
    manifest = dict(
        schema_version=1,
        scenarios_sha256=digest_bytes(raw),
        source_sha256=sources,
        settings=SETTINGS,
    )
    manifest_path = output / "manifest.json"
    if (
        manifest_path.exists()
        and json.loads(manifest_path.read_text(encoding="utf-8")) != manifest
    ):
        raise ValueError(
            "Frozen source/settings differ; choose a new --output directory"
        )
    if not manifest_path.exists():
        manifest_path.write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        for path in source_paths:
            target = output / "sources" / path.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())
    # Evaluation reads the frozen file, never a regenerated in-memory fixture.
    frozen = output / "scenarios.json"
    if digest_bytes(frozen.read_bytes()) != manifest["scenarios_sha256"]:
        raise ValueError("Frozen scenario digest differs")
    return json.loads(frozen.read_text(encoding="utf-8")), manifest


def matches(actual, expected):
    return all(actual.get(key) == value for key, value in expected.items())


def measure_case(case, count_tokens, artifact_dir):
    provider = NoModelCalls()
    memory = WorkflowMemory(case["catalog"], provider, **SETTINGS["memory_limits"])
    operations = {group["event_id"]: group for group in case["operations"]["events"]}
    started_cpu, started_wall = time.process_time(), time.perf_counter()
    batch_size = SETTINGS["ingestion_batch_events"]
    for start in range(0, len(case["events"]), batch_size):
        batch = case["events"][start : start + batch_size]
        memory.apply_operations(
            batch,
            dict(
                events=[
                    operations[event["event_id"]]
                    for event in batch
                    if event["event_id"] in operations
                ]
            ),
        )
    ingest_cpu, ingest_wall = (
        time.process_time() - started_cpu,
        time.perf_counter() - started_wall,
    )
    baseline_content = dict(
        entity_catalog=case["catalog"],
        question=case["question"],
        context=dict(source_events=case["events"]),
    )
    baseline = canonical(
        dict(instructions=ACTIVE_CONTEXT_INSTRUCTIONS, content=baseline_content)
    )
    baseline_tokens = count_tokens(baseline)
    before = digest_bytes(canonical(memory.snapshot()).encode("utf-8"))
    result = dict(
        name=case["name"],
        family=case["family"],
        expected_outcome=case["expected_outcome"],
        selection_mode=case["selection_mode"],
        compiler_options=case["compiler_options"],
        event_count=len(case["events"]),
        stored_current_facts=len(memory.state["facts"]),
        stored_history_records=len(memory.memory_packet()["fact_history"]),
        full_history_input_tokens=baseline_tokens,
        max_input_tokens=case["max_input_tokens"],
        ingestion_cpu_seconds=ingest_cpu,
        ingestion_wall_seconds=ingest_wall,
        model_calls=0,
        measured_answer_accuracy=None,
        measured_extractor_accuracy=None,
    )
    cpu_times, wall_times, requests, failures = [], [], [], []
    for _ in range(SETTINGS["repetitions"]):
        cpu, wall = time.process_time(), time.perf_counter()
        try:
            compiled = ActiveContext(memory).compile(
                case["question"],
                max_input_tokens=case["max_input_tokens"],
                count_tokens=count_tokens,
                requested_facts=case["requested_facts"],
                required_event_ids=case["required_event_ids"],
                **case["compiler_options"],
            )
            requests.append(compiled)
        except ContextBudgetExceeded as exc:
            failures.append(
                dict(required_tokens=exc.required_tokens, budget=exc.budget)
            )
        cpu_times.append(time.process_time() - cpu)
        wall_times.append(time.perf_counter() - wall)
    result.update(
        compilation_cpu_seconds=dict(
            samples=cpu_times,
            median=statistics.median(cpu_times),
            maximum=max(cpu_times),
        ),
        compilation_wall_seconds=dict(
            samples=wall_times,
            median=statistics.median(wall_times),
            maximum=max(wall_times),
        ),
    )
    checks = dict(
        no_provider_calls=provider.calls == 0,
        memory_unchanged=before
        == digest_bytes(canonical(memory.snapshot()).encode("utf-8")),
    )
    if failures:
        result.update(
            outcome="budget_exceeded",
            budget_failure=failures[0],
            input_tokens=None,
            saved_input_tokens=None,
            input_token_reduction_fraction=None,
            coverage=None,
        )
        checks.update(
            expected_outcome=case["expected_outcome"] == "budget_exceeded",
            deterministic_failure=len(failures) == SETTINGS["repetitions"]
            and all(item == failures[0] for item in failures),
            required_exceeds_budget=failures[0]["required_tokens"]
            > failures[0]["budget"],
        )
    else:
        compiled = requests[0]
        context, expected = compiled.content["context"], case["expected"]
        source_map = {event["event_id"]: event for event in context["source_events"]}
        original_map = {event["event_id"]: event for event in case["events"]}
        checks.update(
            expected_outcome=case["expected_outcome"] == "compiled",
            exact_rendered_count=count_tokens(compiled.rendered_request)
            == compiled.input_tokens,
            within_budget=compiled.input_tokens <= case["max_input_tokens"],
            deterministic_request=all(
                item.rendered_request == compiled.rendered_request for item in requests
            ),
            current_values_and_evidence=all(
                any(
                    matches(actual, fact)
                    for actual in context["current_state"]["facts"]
                )
                for fact in expected["facts"]
            ),
            earlier_approved_value=any(
                matches(actual, expected["historical_fact"])
                for actual in context["fact_history"]
            ),
            pending_and_active_approval=all(
                any(matches(actual, proposal) for actual in context["proposals"])
                for proposal in expected["proposal_records"]
            ),
            unresolved_decision=any(
                item["event_id"] == expected["unresolved_event_id"]
                for item in context["unresolved_references"]
            ),
            expected_sources_present=all(
                identifier in source_map for identifier in expected["source_ids"]
            ),
            source_records_whole_and_unchanged=all(
                original_map.get(identifier) == event
                for identifier, event in source_map.items()
            ),
            semantic_verification_not_claimed=compiled.coverage[
                "semantic_completeness_verified"
            ]
            is False,
        )
        result.update(
            outcome="compiled",
            input_tokens=compiled.input_tokens,
            saved_input_tokens=baseline_tokens - compiled.input_tokens,
            input_token_reduction_fraction=1 - compiled.input_tokens / baseline_tokens,
            coverage=compiled.coverage,
        )
        artifact_dir.mkdir(parents=True, exist_ok=True)
        (artifact_dir / f'{case["name"]}-request.json').write_text(
            compiled.rendered_request + "\n", encoding="utf-8"
        )
    result["checks"] = checks
    result["checks_passed"] = all(checks.values())
    return result


def render_report(report):
    rows = []
    for result in report["cases"]:
        coverage = result.get("coverage") or {}
        source = coverage.get("source_events", {})
        sources = (
            f'{source.get("included", "—")}/{source.get("total", "—")}'
            if source
            else "Not sent"
        )
        saving = (
            f'{result["input_token_reduction_fraction"]:.1%}'
            if result["input_token_reduction_fraction"] is not None
            else "No request"
        )
        active = (
            str(result["input_tokens"])
            if result["input_tokens"] is not None
            else "Refused"
        )
        cells = [
            result["name"],
            result["event_count"],
            result["stored_history_records"],
            result["full_history_input_tokens"],
            active,
            saving,
            sources,
            f'{result["compilation_cpu_seconds"]["median"] * 1000:.2f} ms',
            f'{result["compilation_wall_seconds"]["median"] * 1000:.2f} ms',
            "Pass" if result["checks_passed"] else "FAIL",
        ]
        rows.append(
            "<tr>"
            + "".join(f"<td>{escape(str(cell))}</td>" for cell in cells)
            + "</tr>"
        )
    headers = [
        "Scenario",
        "Events",
        "Stored history",
        "Full-history tokens",
        "Compiled tokens",
        "Input reduction",
        "Sources included",
        "Median compile CPU",
        "Median compile wall",
        "Checks",
    ]
    return (
        """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Threadmark active context — offline accounting</title><style>
body{font:16px/1.65 system-ui,sans-serif;background:#f2f5f3;color:#183a32;margin:0}main{max-width:1100px;margin:auto;padding:28px 18px 60px}section{background:white;border:1px solid #d8e1dd;border-radius:14px;padding:26px;margin:22px 0}h1{font-size:2rem;line-height:1.2}h2{font-size:1.35rem}p{max-width:84ch}.notice{background:#fff3d7;border-left:4px solid #b88620;padding:16px}.table{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px}th,td{text-align:left;padding:12px;border-bottom:1px solid #d8e1dd}th{background:#e9f1ed}code{overflow-wrap:anywhere}a{color:#12665c}.tag{font-weight:700;color:#12665c}@media(max-width:600px){section{padding:16px}h1{font-size:1.65rem}th,td{padding:9px}.table{margin:0 -4px}}
</style><main><p class="tag">Local engineering check · 11 November 2026</p><h1>Budgeted context from persistent workflow memory</h1><p><strong>Explicit field scope and caller-pinned history sources in the primary cases; two additional cases test lexical selection without source pins.</strong></p>
<section><p class="notice"><strong>No model accuracy was measured.</strong> These synthetic cases supply known application operations directly. Passing checks establish selected-record integrity and request accounting for these cases; they do not establish extraction accuracy, answer accuracy or production readiness.</p>
<p>Scenarios and settings were frozen before compilation. The same stored memory can supply different readers without replaying ingestion. The tokenizer was cached <code>o200k_base</code>; requests use a canonical JSON envelope, not a provider-native chat template or a billing counter.</p>
<p><strong>0 model calls · 0 billed API calls · no network or cloud operations.</strong> Compilation and ingestion CPU time are measured; electricity and local hardware cost are unpriced. Input-token reduction is not an end-to-end cost or latency saving.</p></section>
<section><h2>Frozen cases and observed accounting</h2><p>The cap is 6,144 tokens for the complete rendered request. The primary fixed-state cases add background records; their growing-history counterparts add updates only to an unrelated archive field, a favorable explicit-scope case. Current-fact cases grow from 8 to 32 fields but explicitly request the same three review fields. Those primary cases pin known historical source IDs; they do not test automatic semantic retrieval. Mandatory-source and all-facts pressure cases must refuse instead of truncating or calling a fallback.</p><p>The two unpinned cases use default optional settings (8 history, 8 claims, 2 recent sources, 4 lexical search candidates). One adds only background. The other adds 502 operative owner updates inside the requested review scope, finishes with owner Bo and still asks for the earlier approved Room B. Neither supplies historical source IDs to the compiler; both retain explicit field scope.</p><div class="table"><table><thead><tr>"""
        + "".join(f"<th>{header}</th>" for header in headers)
        + "</tr></thead><tbody>"
        + "".join(rows)
        + """</tbody></table></div>
<p>CPU and elapsed wall figures are the median of three complete compiler constructions and compilations after memory ingestion. Windows process CPU samples can quantize to zero or 15.625 ms; a zero sample does not mean instantaneous work. Raw samples and ingestion timings are saved in results.json. Omitted sources remain in the retained archive. Coverage counts describe included stored records and explicitly leave semantic completeness unverified.</p></section>
<section><h2>What the checks cover</h2><p>The compiled cases check the latest owner, the replacement room approval with both source records, an explicitly cleared deadline, an earlier approved value, an open proposal, an unresolved reference, and unmodified whole original sources. They also check the exact rendered token count, the cap, deterministic output, an unchanged memory snapshot and zero provider calls.</p>
<p>Fixed current state with repeated background is a favorable compression case. Growing historical state and explicit scope expose a different tradeoff: bounded requests omit stored records. Compression is not guaranteed for small histories or large mandatory sources. No fixed-size representation can promise lossless answers to arbitrary future history questions. Required evidence that cannot fit produces a budget error.</p>
<p>The failed November 10 extractor/repair study remains unchanged: the repair regressed on reused DEV, neither candidate was promoted, and its fresh held test remained sealed. This offline compiler check does not repair or supersede those accuracy results.</p></section>
<section><h2>Reproduce and inspect</h2><p><a href="manifest.json">Frozen source/settings manifest</a> · <a href="scenarios.json">Frozen supplied operations and expected records</a> · <a href="results.json">All counts, checks and CPU samples</a></p><p>Manifest SHA-256: <code>"""
        + escape(report["manifest_sha256"])
        + "</code></p><p>Scenario SHA-256: <code>"
        + escape(report["scenarios_sha256"])
        + "</code></p><p>Protocol note: the first frozen run is preserved one directory above this run. Its intended oversized-source fixture was too small: the mandatory bundle fit, optional history was omitted, and the expected-overflow check failed. A separate corrected-pressure run is preserved in the adjacent final directory. This final-with-unpinned run preserves that corrected fixture and freezes both additional no-pin cases before measurement. No earlier result was removed and no model was tuned.</p></section></main></html>"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "runs/active-context-20261111/final-with-unpinned",
    )
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    payload, manifest = frozen_inputs(output)
    if args.prepare_only:
        print(
            json.dumps(
                dict(
                    status="frozen",
                    output=str(output),
                    scenarios=len(payload["scenarios"]),
                    scenarios_sha256=manifest["scenarios_sha256"],
                )
            )
        )
        return
    if (output / "results.json").exists():
        raise ValueError(
            "Results already exist; preserve this run and choose a new --output directory"
        )
    count_tokens = token_counter()
    report = dict(
        schema_version=1,
        measured_at_utc=datetime.now(timezone.utc).isoformat(),
        scope="offline application-operation selection and token accounting; no model accuracy",
        settings=payload["settings"],
        scenarios_sha256=manifest["scenarios_sha256"],
        manifest_sha256=digest_bytes((output / "manifest.json").read_bytes()),
        environment=dict(
            python=sys.version,
            platform=platform.platform(),
            tiktoken=importlib.metadata.version("tiktoken"),
        ),
        model_calls=0,
        billed_model_calls=0,
        cloud_calls=0,
        network_calls=0,
        api_cost_usd=0,
        local_compute_cost_usd=None,
        extractor_accuracy=None,
        reader_accuracy=None,
        cases=[
            measure_case(case, count_tokens, output / "requests")
            for case in payload["scenarios"]
        ],
    )
    report["status"] = (
        "pass" if all(case["checks_passed"] for case in report["cases"]) else "fail"
    )
    (output / "results.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "index.html").write_text(render_report(report), encoding="utf-8")
    print(
        json.dumps(
            dict(
                status=report["status"],
                output=str(output),
                cases=len(report["cases"]),
                failed_cases=[
                    case["name"]
                    for case in report["cases"]
                    if not case["checks_passed"]
                ],
                model_calls=0,
            )
        )
    )
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
