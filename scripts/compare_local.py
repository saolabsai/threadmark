"""Fresh comparison of full history, BM25, dense retrieval, and Threadmark.

Local inference is the default. Hosted inference requires explicit prices, an
existing project budget, a separate run cap, and an environment API key. This is
a synthetic task comparison, not proof of a general replacement for RAG. The
protocol is written before generating test data.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import statistics
import time

from threadmark.config import Settings
from threadmark.context import select_source_events, token_counter
from threadmark.dense_retrieval import LocalDenseRetriever
from threadmark.datasets import Thread, render_events
from threadmark.engine import Threadmark
from threadmark.evaluation import public_workflow, score_facts
from threadmark.local_provider import LocalTransformersProvider
from threadmark.prompts import READER_INSTRUCTIONS, UPDATER_INSTRUCTIONS
from threadmark.providers import ProviderError, OpenAIProvider
from threadmark.retrieval import select_bm25_sources
from threadmark.schema import canonical, catalog_keys, initial_state, validate_facts

ARMS = ("full_history", "recent_only", "bm25", "dense", "active_memory", "state")
ROOT = Path(__file__).resolve().parents[1]


class JournalProvider:
    """Save public request arguments and raw outputs, including failed attempts."""

    def __init__(self, provider, directory, updater_instructions=None):
        self.provider, self.directory = provider, directory
        self.request_counter = 0
        self.updater_instructions = updater_instructions

    def __getattr__(self, name):
        return getattr(self.provider, name)

    def append(self, name, value):
        with (self.directory / name).open("a", encoding="utf-8") as handle:
            handle.write(canonical(value) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def complete(self, **kwargs):
        if (
            self.updater_instructions is not None
            and kwargs.get("instructions") == UPDATER_INSTRUCTIONS
        ):
            kwargs = {**kwargs, "instructions": self.updater_instructions}
        identifier = f"request-{self.request_counter}"
        self.request_counter += 1
        self.append(
            "requests.jsonl",
            {
                "request_id": identifier,
                "model": (
                    self.provider.model_for(kwargs["instructions"])
                    if callable(getattr(self.provider, "model_for", None))
                    else self.provider.model
                ),
                "arguments": kwargs,
            },
        )
        try:
            response = self.provider.complete(**kwargs)
        except BaseException as exc:
            self.append(
                "responses.jsonl",
                {
                    "request_id": identifier,
                    "status": "failed",
                    "error_type": type(exc).__name__,
                },
            )
            raise
        self.append(
            "responses.jsonl",
            {
                "request_id": identifier,
                "status": response.status,
                "text": response.text,
                "usage": response.usage,
            },
        )
        return response


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def write(path: Path, data: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def workflow(domain, target, variant, seed, counter, split="test"):
    """Reuse disclosed scenario families with fresh instances and native sizes.

    The old generator's published bands remain untouched. This adapter adds a
    4096-native-token band suitable for an 8192-context reader. It neither
    inspects correctness nor resamples on model outcomes.
    """
    identifier = f"local-{split}-{seed}-{domain}-{target}-{variant}"
    case_seed = int(hashlib.sha256(identifier.encode()).hexdigest()[:16], 16)
    middle_count = 0
    for _ in range(100):
        thread = Thread(identifier, domain, split, random.Random(case_seed), counter)
        thread.initialize()
        for step in range(middle_count):
            if step == middle_count // 2:
                owner = thread.people[7]
                thread.emit(
                    f"Responsibility for {thread.labels[1]} has been transferred to {owner}. "
                    f"{owner} has accepted the owner role, replacing the initial assignment. "
                    f"This decision applies only to {thread.labels[1]}.",
                    [(1, "owner", owner)],
                )
            thread.middle(step)
        thread.finalize(variant)
        case = thread.finish(target, variant)
        count = counter(render_events(case["events"]))
        if target * 0.9 <= count <= target * 1.1:
            return case
        if count > target * 1.1:
            raise ValueError(f"Cannot fit {identifier} in its native-token band")
        middle_count += max(1, int((target * 0.95 - count) / 160))
    raise ValueError("Workflow geometry did not converge")


def summary(report):
    result = {}
    arms = tuple(report.get("arms", ARMS))
    update_calls = [call for call in report["calls"] if call["stage"] == "ingest"]
    for arm in arms:
        rows = [row for row in report["queries"] if row["arm"] == arm]
        calls = [call for call in report["calls"] if call["stage"] == arm]
        lifecycle = calls + (update_calls if arm in ("active_memory", "state") else [])
        complete = [row for row in rows if row["status"] == "completed"]
        result[arm] = {
            "planned": len(rows),
            "completed": len(complete),
            "correct": sum(
                row.get("score", {}).get("exact_values", False) for row in rows
            ),
            "exact_evidence_correct": sum(
                row.get("score", {}).get("exact_values_and_evidence", False)
                for row in rows
            ),
            "accuracy": sum(
                row.get("score", {}).get("exact_values", False) for row in rows
            )
            / len(rows),
            "reader_input_tokens": sum(
                (call.get("usage") or {}).get("input_tokens") or 0 for call in calls
            ),
            "lifecycle_input_tokens": sum(
                (call.get("usage") or {}).get("input_tokens") or 0 for call in lifecycle
            ),
            "lifecycle_output_tokens": sum(
                (call.get("usage") or {}).get("output_tokens") or 0
                for call in lifecycle
            ),
            "lifecycle_provider_seconds": sum(
                call.get("elapsed_seconds") or 0 for call in lifecycle
            ),
            "selection_seconds": sum(row.get("selection_seconds", 0) for row in rows),
            "query_wall_seconds": sum(row.get("wall_seconds", 0) for row in rows),
            "median_completed_query_wall_seconds": (
                statistics.median(
                    [row["wall_seconds"] for row in complete if "wall_seconds" in row]
                )
                if any("wall_seconds" in row for row in complete)
                else None
            ),
            "ingest_wall_seconds": (
                sum(
                    row.get("ingest_wall_seconds", 0)
                    for row in report.get("workflows", [])
                )
                if arm in ("active_memory", "state")
                else 0
            ),
            "source_complete_packets": sum(
                row.get("support_complete", False) for row in rows
            ),
            "source_packets_available": sum("support_complete" in row for row in rows),
            "known_api_cost_usd": sum(
                call.get("estimated_cost_usd") or 0 for call in lifecycle
            ),
            "unknown_usage_calls": sum(
                report.get("backend") == "openai"
                and call.get("request_started", False)
                and call.get("budget_cost_usd") is None
                for call in lifecycle
            ),
        }
    comparisons = {}
    for target in ("active_memory", "state"):
        paired = {}
        for baseline in ("full_history", "bm25", "dense", "recent_only"):
            if target not in arms or baseline not in arms:
                continue
            pairs = {}
            for row in report["queries"]:
                pairs.setdefault((row["workflow_id"], row["query_id"]), {})[
                    row["arm"]
                ] = bool(row.get("score", {}).get("exact_values", False))
            paired[baseline] = {
                "both_correct": sum(p[baseline] and p[target] for p in pairs.values()),
                "threadmark_only": sum(
                    not p[baseline] and p[target] for p in pairs.values()
                ),
                "baseline_only": sum(
                    p[baseline] and not p[target] for p in pairs.values()
                ),
                "both_wrong": sum(
                    not p[baseline] and not p[target] for p in pairs.values()
                ),
            }
        comparisons[target] = paired
    return {
        "arms": result,
        "paired_vs_threadmark": comparisons.get("active_memory", {}),
        "paired_vs_state": comparisons.get("state", {}),
    }


def markdown(report):
    lines = [
        "# Local model and retrieval comparison",
        "",
        f"Model: `{report['model']}`. Status: **{report['status']}**.",
        "",
        "This synthetic experiment selects from five reader context strategies and a direct predicted-state lookup with no reader call. "
        "It is not a general replacement-for-RAG result. BM25 is lexical retrieval; dense retrieval uses a local MiniLM encoder, with no reranker.",
        "",
        "Configured modes: " + ", ".join(report.get("arms", ARMS)) + ".",
        "",
        "| Context | Exact answers / planned | Available | Reader input tokens | Lifecycle input tokens |",
        "|---|---:|---:|---:|---:|",
    ]
    for arm, item in report["summary"]["arms"].items():
        lines.append(
            f"| {arm} | {item['correct']}/{item['planned']} | {item['completed']}/{item['planned']} | {item['reader_input_tokens']} | {item['lifecycle_input_tokens']} |"
        )
    lines.extend(
        [
            "",
            "Lifecycle accounting includes every attempted updater call for both Threadmark variants. The state arm returns unverified predicted facts directly; requested field keys are supplied as public query routing. It is not arbitrary document question answering. "
            "Unavailable answers stay incorrect. Models receive only public catalogs, events and questions; labels are used afterward for scoring.",
            "",
            (
                "All reader inference is local: no model API charges were incurred. "
                if report.get("backend", "local") == "local"
                else "Reader inference uses the official Responses API with an explicit project and per-run budget. "
            )
            + "GPU, energy and engineering costs are unpriced. Dense encoder token work is not included in the reader/updater token columns; its indexing and selection time is recorded separately. "
            "Recorded local latency is not comparable with the historical concurrent API benchmark. "
            "Small models can fail strict JSON or state tracking; these failures remain in the results.",
            "",
            "The workflow families are shared with earlier synthetic studies, with fresh instance seeds. "
            "The 2048/4096 bands use the reader's native tokenizer; none of the full prompts is silently truncated. "
            "Longer prompts exceeding model context are recorded as failures. Full original archives remain stored.",
            "All questions occur after complete ingestion. Shared cold model loading is recorded separately; lifecycle timings include updater, selection, and reader work. This is final-state indexing, not a test of interleaved queries.",
            "",
            "See `protocol.json`, `dataset.json`, and `report.json` for source hashes, original answers, failures, selected sources, and paired regressions.",
            "",
        ]
    )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path)
    parser.add_argument("--backend", choices=["local", "openai"], default="local")
    parser.add_argument(
        "--strategy", choices=["register", "event_delta"], default="register"
    )
    parser.add_argument("--arms", nargs="+", choices=ARMS)
    parser.add_argument("--dense-base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model")
    parser.add_argument("--project-budget", type=Path)
    parser.add_argument("--run-budget-usd", type=float, default=1.0)
    parser.add_argument(
        "--run-budget-ledger",
        type=Path,
        help="Optional shared cap across development and final runs; its existing cap cannot change",
    )
    parser.add_argument("--input-price", type=float)
    parser.add_argument("--cached-input-price", type=float)
    parser.add_argument("--cache-write-price", type=float)
    parser.add_argument("--output-price", type=float)
    parser.add_argument("--updater-model")
    parser.add_argument("--updater-effort", choices=["none", "low", "medium"])
    parser.add_argument("--updater-input-price", type=float)
    parser.add_argument("--updater-cached-input-price", type=float)
    parser.add_argument("--updater-cache-write-price", type=float)
    parser.add_argument("--updater-output-price", type=float)
    parser.add_argument("--chunk-tokens", type=int, default=1500)
    parser.add_argument(
        "--reasoning-effort", choices=["none", "low", "medium"], default="none"
    )
    parser.add_argument("--reader-output-tokens", type=int, default=512)
    parser.add_argument("--updater-output-tokens", type=int, default=2048)
    parser.add_argument(
        "--reader-instructions",
        type=Path,
        help="A frozen development-selected prompt applied identically to all reader arms",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--updater-instructions",
        type=Path,
        help="Frozen ingestion instructions selected on development data",
    )
    parser.add_argument("--seed", type=int, default=2026102801)
    parser.add_argument("--split", choices=["dev", "test"], default="test")
    parser.add_argument(
        "--domains",
        nargs="+",
        choices=["meeting", "support", "project"],
        default=["meeting", "support", "project"],
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        help="Reuse an identical previously generated public test fixture across reader backbones",
    )
    parser.add_argument("--bands", type=int, nargs="+", default=[2048, 4096])
    parser.add_argument("--per-cell", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=8)
    args = parser.parse_args()
    arms = tuple(args.arms or ARMS)
    if len(set(arms)) != len(arms):
        parser.error("Mode names must be unique")
    if args.strategy == "event_delta" and "active_memory" in arms:
        parser.error(
            "The event-delta experiment currently evaluates direct state only; choose a subset of modes excluding active_memory"
        )
    if args.output.exists():
        parser.error("Use a fresh output directory; attempts are never overwritten")
    if (
        args.per_cell < 1
        or any(not 512 <= band <= 65536 for band in args.bands)
        or len(set(args.bands)) != len(args.bands)
        or len(set(args.domains)) != len(args.domains)
        or not 1 <= args.top_k <= 100
    ):
        parser.error("Invalid test dimensions")
    if args.backend == "local" and args.base is None:
        parser.error("Local inference requires --base")
    if args.backend == "local" and args.reasoning_effort != "none":
        parser.error(
            "The local adapter currently evaluates non-thinking generation only"
        )
    if (
        not 16 <= args.reader_output_tokens <= 16384
        or not 16 <= args.updater_output_tokens <= 16384
    ):
        parser.error("Output allowances must be between 16 and 16384")
    if args.updater_model and args.backend != "openai":
        parser.error(
            "A separately routed updater currently requires the hosted backend"
        )
    if args.updater_effort is not None and not args.updater_model:
        parser.error("--updater-effort requires an explicit --updater-model")
    prices = None
    updater_prices = None
    if args.backend == "openai":
        from threadmark.budget import BudgetLedger, PairedBudgetLedger, Prices

        if args.project_budget is None or any(
            value is None
            for value in (
                args.input_price,
                args.cached_input_price,
                args.cache_write_price,
                args.output_price,
            )
        ):
            parser.error(
                "Hosted inference requires an existing project budget and all four prices"
            )
        prices = Prices(
            args.input_price,
            args.cached_input_price,
            args.cache_write_price,
            args.output_price,
        )
    if args.updater_model:
        if any(
            value is None
            for value in (
                args.updater_input_price,
                args.updater_cached_input_price,
                args.updater_cache_write_price,
                args.updater_output_price,
            )
        ):
            parser.error("A separate updater requires all four of its own prices")
        updater_prices = Prices(
            args.updater_input_price,
            args.updater_cached_input_price,
            args.updater_cache_write_price,
            args.updater_output_price,
        )
    args.model = args.model or (
        "HuggingFaceTB/SmolLM2-1.7B-Instruct"
        if args.backend == "local"
        else "gpt-5.6-luna"
    )
    reader_instructions = (
        args.reader_instructions.read_text(encoding="utf-8").strip()
        if args.reader_instructions
        else READER_INSTRUCTIONS
    )
    if not reader_instructions or len(reader_instructions) > 16_000:
        parser.error(
            "Reader instructions must be nonempty and at most 16000 characters"
        )
    updater_instructions = (
        args.updater_instructions.read_text(encoding="utf-8").strip()
        if args.updater_instructions
        else UPDATER_INSTRUCTIONS
    )
    if args.strategy == "event_delta":
        from threadmark.delta_memory import EventDeltaMemory, DELTA_UPDATER_INSTRUCTIONS

        engine_class = EventDeltaMemory
        if args.updater_instructions is None:
            updater_instructions = DELTA_UPDATER_INSTRUCTIONS
    else:
        engine_class = Threadmark
    if not updater_instructions or len(updater_instructions) > 16_000:
        parser.error(
            "Updater instructions must be nonempty and at most 16000 characters"
        )
    args.output.mkdir(parents=True)
    if args.backend == "local":
        provider = LocalTransformersProvider(
            args.base, model=args.model, device=args.device
        )
        counter = provider.count_tokens
    else:
        project_budget = BudgetLedger.open_existing(args.project_budget)
        run_budget = BudgetLedger(
            args.run_budget_ledger or args.output / "run-budget.json",
            args.run_budget_usd,
        )
        paired_budget = PairedBudgetLedger(project_budget, run_budget)
        provider = OpenAIProvider(
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            prices=prices,
            budget=paired_budget,
        )
        counter = token_counter()
        if args.updater_model:
            from threadmark.routed_provider import RoleRoutedProvider

            updater = OpenAIProvider(
                model=args.updater_model,
                reasoning_effort=args.updater_effort or args.reasoning_effort,
                prices=updater_prices,
                budget=paired_budget,
            )
            provider = RoleRoutedProvider(
                provider,
                updater,
                updater_instructions={UPDATER_INSTRUCTIONS, updater_instructions},
            )
    dense = LocalDenseRetriever(args.dense_base, device="cpu")
    settings = Settings(
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        chunk_tokens=args.chunk_tokens,
        updater_max_output_tokens=args.updater_output_tokens,
        reader_max_output_tokens=args.reader_output_tokens,
        recent_source_tokens=1024,
    )
    import threadmark.datasets as generator
    import threadmark.local_provider as local_module
    import threadmark.retrieval as retrieval_module
    import threadmark.dense_retrieval as dense_module

    protocol = {
        "schema": "threadmark-local-comparison-v1",
        "locked_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "backend": args.backend,
        "settings": asdict(settings),
        "seed": args.seed,
        "split": args.split,
        "domains": args.domains,
        "bands_native_tokens": args.bands,
        "per_domain_band": args.per_cell,
        "top_k": args.top_k,
        "arms": list(arms),
        "update_strategy": args.strategy,
        "arm_order": "Rotate by question and workflow index",
        "selection_basis": "Configuration selected from preserved development runs; an event-delta follow-up may also use prior register-run diagnostics. This test's fresh seed is not used for tuning.",
        "sources": {
            str(path.relative_to(ROOT)).replace("\\", "/"): digest(path)
            for path in [
                Path(__file__).resolve(),
                Path(generator.__file__),
                Path(local_module.__file__),
                Path(retrieval_module.__file__),
                Path(dense_module.__file__),
                ROOT / "src/threadmark/prompts.py",
                ROOT / "src/threadmark/engine.py",
                ROOT / "src/threadmark/context.py",
                ROOT / "src/threadmark/schema.py",
                ROOT / "src/threadmark/providers.py",
                ROOT / "src/threadmark/budget.py",
                ROOT / "src/threadmark/_ledger.py",
                ROOT / "src/threadmark/evaluation.py",
                ROOT / "src/threadmark/config.py",
                ROOT / "src/threadmark/routed_provider.py",
            ]
            + (
                [ROOT / "src/threadmark/delta_memory.py"]
                if args.strategy == "event_delta"
                else []
            )
        },
        "base_config_sha256": digest(args.base / "config.json") if args.base else None,
        "base_asset_sha256": {
            path.name: digest(path)
            for path in (sorted(args.base.iterdir()) if args.base else [])
            if path.suffix in (".safetensors", ".json", ".jinja", ".txt")
            and path.is_file()
        },
        "runtime_versions": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "tokenizers", "safetensors")
        },
        "device": args.device,
        "dense_asset_sha256": {
            str(path.relative_to(args.dense_base)).replace("\\", "/"): digest(path)
            for path in sorted(args.dense_base.rglob("*"))
            if path.suffix in (".safetensors", ".json", ".txt") and path.is_file()
        },
        "dense_model": "sentence-transformers/all-MiniLM-L6-v2",
        "dense_method": "Maximum native-chunk cosine; complete text; mean pooling; top K plus recent originals; CPU indexing cached across questions",
        "prices_usd_per_million": asdict(prices) if prices else None,
        "updater_model": args.updater_model or args.model,
        "updater_reasoning_effort": args.updater_effort or args.reasoning_effort,
        "updater_prices_usd_per_million": (
            asdict(updater_prices)
            if updater_prices
            else (asdict(prices) if prices else None)
        ),
        "reader_instructions": reader_instructions,
        "updater_instructions": updater_instructions,
        "updater_instructions_sha256": hashlib.sha256(
            updater_instructions.encode()
        ).hexdigest(),
        "reader_instructions_sha256": hashlib.sha256(
            reader_instructions.encode()
        ).hexdigest(),
        "run_budget_usd": args.run_budget_usd if prices else None,
        "input_dataset_sha256": digest(args.dataset) if args.dataset else None,
        "labels": "Never passed to provider; exact values and citation sets scored after inference",
        "acceptance": "Report every planned answer and paired regression; no equal accuracy or RAG replacement claim from a lexical synthetic baseline alone",
        "paid_calls": args.backend == "openai",
    }
    write(args.output / "protocol.json", protocol)
    import shutil

    for relative in protocol["sources"]:
        destination = args.output / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    protocol_sha = digest(args.output / "protocol.json")
    if args.dataset:
        from threadmark.evaluation import load_dataset

        reused_dataset = load_dataset(args.dataset, settings)
        workflows = reused_dataset["workflows"]
    else:
        workflows = [
            workflow(domain, band, index, args.seed, counter, args.split)
            for domain in args.domains
            for band in args.bands
            for index in range(args.per_cell)
        ]
    dataset = {
        "manifest": {
            "protocol_sha256": protocol_sha,
            "synthetic": True,
            "seed": args.seed,
            "split": args.split,
            "shared_scenario_families": True,
            "reused_input_manifest": (
                reused_dataset.get("manifest", {}) if args.dataset else None
            ),
            "generation_scope": (
                "Reused exact fixture; input manifest is authoritative about its original generation and split"
                if args.dataset
                else "New instances under this locked protocol"
            ),
        },
        "workflows": workflows,
    }
    write(args.output / "dataset.json", dataset)
    report = {
        "schema": "threadmark-local-comparison-v1",
        "status": "running",
        "model": args.model,
        "backend": args.backend,
        "arms": list(arms),
        "update_strategy": args.strategy,
        "protocol_sha256": protocol_sha,
        "dataset_sha256": digest(args.output / "dataset.json"),
        "api_calls": 0,
        "api_charge_usd": 0,
        "hardware_cost_usd": None,
        "queries": [],
        "workflows": [],
        "calls": [],
    }
    rows = {}
    provider = JournalProvider(provider, args.output, updater_instructions)
    for case in workflows:
        for query in case["queries"]:
            for arm in arms:
                row = {
                    "workflow_id": case["id"],
                    "domain": case["domain"],
                    "size_band": case["size_band"],
                    "query_id": query["query_id"],
                    "arm": arm,
                    "status": "not_run",
                }
                report["queries"].append(row)
                rows[(case["id"], query["query_id"], arm)] = row

    def checkpoint():
        if args.backend == "openai":
            report["api_calls"] = sum(
                bool(call.get("request_started")) for call in report["calls"]
            )
            report["api_charge_usd"] = sum(
                call.get("estimated_cost_usd") or 0 for call in report["calls"]
            )
            report["run_budget"] = run_budget.snapshot()
        report["summary"] = summary(report)
        write(args.output / "report.json", report)
        (args.output / "report.md").write_text(markdown(report), encoding="utf-8")

    checkpoint()
    try:
        report["shared_model_load"] = (
            provider.load()
            if args.backend == "local"
            else {
                "backend": "hosted",
                "cold_model_loading": "provider_managed_unmeasured",
            }
        )
        if "dense" in arms:
            report["shared_dense_model_load"] = dense.load()
        checkpoint()
        needs_memory = any(arm in ("active_memory", "state") for arm in arms)
        for wi, case in enumerate(workflows):
            public = public_workflow(case)
            engine = (
                engine_class(
                    public["entity_catalog"],
                    provider,
                    settings,
                    counter,
                    updater_instructions=updater_instructions,
                )
                if needs_memory
                else None
            )
            trace = {"workflow_id": case["id"], "status": "ingest_failed"}
            before = len(provider.calls)
            started = time.perf_counter()
            fatal = None
            try:
                if needs_memory:
                    engine.ingest(public["events"])
                    trace.update(status="ingested", state=engine.state)
                else:
                    trace.update(status="ingestion_not_required")
            except Exception as exc:
                trace["error"] = str(exc)[:500]
                if isinstance(exc, ProviderError) and exc.fatal:
                    fatal = exc
            finally:
                trace["ingest_wall_seconds"] = (
                    time.perf_counter() - started if needs_memory else 0.0
                )
                report["calls"].extend(
                    {**call, "workflow_id": case["id"], "stage": "ingest"}
                    for call in provider.calls[before:]
                )
                report["workflows"].append(trace)
                checkpoint()
            if fatal:
                raise fatal
            for qi, query in enumerate(public["queries"]):
                query_id = case["queries"][qi]["query_id"]
                labels = case["queries"][qi]["expected"]
                offset = (wi + qi) % len(arms)
                for arm in arms[offset:] + arms[:offset]:
                    row = rows[(case["id"], query_id, arm)]
                    before = len(provider.calls)
                    started = time.perf_counter()
                    fatal = None
                    try:
                        if arm == "state":
                            if trace["status"] != "ingested":
                                raise ValueError(
                                    "Ingestion failed; direct state answer unavailable"
                                )
                            answer = engine.answer(query, mode="state")
                            row.update(
                                status="completed",
                                answer=answer,
                                score=score_facts(answer, labels),
                                selection_seconds=time.perf_counter() - started,
                                source_ids=[],
                                raw_source_events_read=0,
                                reader_calls=0,
                            )
                            continue
                        if arm == "active_memory":
                            if trace["status"] != "ingested":
                                raise ValueError(
                                    "Ingestion failed; active answer unavailable"
                                )
                            sources = engine.evidence(query)
                        elif arm == "full_history":
                            sources = public["events"]
                        elif arm == "bm25":
                            sources = select_bm25_sources(
                                public["events"],
                                query,
                                public["entity_catalog"],
                                counter,
                                settings,
                                args.top_k,
                            )
                        elif arm == "dense":
                            sources = dense.select(
                                public["events"],
                                query,
                                public["entity_catalog"],
                                counter,
                                settings,
                                args.top_k,
                            )
                            row["dense_selection"] = dense.last_selection
                        else:
                            sources = select_source_events(
                                public["events"],
                                query,
                                initial_state(catalog_keys(public["entity_catalog"])),
                                counter,
                                settings,
                            )
                        row["selection_seconds"] = time.perf_counter() - started
                        selected_ids = {event["event_id"] for event in sources}
                        required = {
                            event_id
                            for fact in labels["facts"]
                            for event_id in fact["evidence_event_ids"]
                        }
                        row.update(
                            source_ids=[event["event_id"] for event in sources],
                            support_complete=required <= selected_ids,
                            support_ids_found=len(required & selected_ids),
                            support_ids_required=len(required),
                        )
                        response = provider.complete(
                            instructions=reader_instructions,
                            content={
                                "entity_catalog": public["entity_catalog"],
                                "context": {"source_events": sources},
                                "query": query["text"],
                                "requested_facts": query["requested_facts"],
                            },
                            max_output_tokens=settings.reader_max_output_tokens,
                        )
                        row["raw_answer"] = response.text
                        answer = validate_facts(
                            response.text,
                            [
                                (f["entity_id"], f["field"])
                                for f in query["requested_facts"]
                            ],
                            selected_ids,
                            settings,
                        )
                        row.update(
                            status="completed",
                            answer=answer,
                            score=score_facts(answer, labels),
                        )
                    except Exception as exc:
                        row.update(status="failed", error=str(exc)[:500])
                        if isinstance(exc, ProviderError) and exc.fatal:
                            fatal = exc
                    finally:
                        row["wall_seconds"] = time.perf_counter() - started
                        row["expected"] = labels
                        report["calls"].extend(
                            {
                                **call,
                                "workflow_id": case["id"],
                                "query_id": query_id,
                                "stage": arm,
                            }
                            for call in provider.calls[before:]
                        )
                        checkpoint()
                    if fatal:
                        raise fatal
                    print(
                        json.dumps(
                            {
                                "workflow": wi + 1,
                                "total_workflows": len(workflows),
                                "query": qi + 1,
                                "arm": arm,
                                "status": row["status"],
                                "correct": row.get("score", {}).get(
                                    "exact_values", False
                                ),
                            }
                        ),
                        flush=True,
                    )
        report["status"] = "completed"
    except BaseException as exc:
        report["status"] = (
            "interrupted"
            if isinstance(exc, (KeyboardInterrupt, SystemExit))
            else "halted"
        )
        checkpoint()
        raise
    checkpoint()
    print(json.dumps(report["summary"]), flush=True)


if __name__ == "__main__":
    main()
