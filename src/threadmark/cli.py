"""Command-line entry point. Demo, inspect, generation and preparation are offline."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from getpass import getpass
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

from .config import Settings
from .evaluation import inspect_dataset, load_dataset, run_comparison


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


class RecordedDemoProvider:
    """Offline scripted responses, never a live model or benchmark result."""

    def __init__(self, responses: list[dict[str, Any]] | None = None) -> None:
        from .datasets import recorded_demo_responses

        self.responses = list(
            responses if responses is not None else recorded_demo_responses()
        )
        self.position = 0

    def complete(
        self, *, instructions: str, content: Any, max_output_tokens: int
    ) -> Any:
        if self.position >= len(self.responses):
            raise ValueError("Offline demonstration exhausted its scripted responses")
        response = self.responses[self.position]
        self.position += 1
        return SimpleNamespace(
            text=json.dumps(response),
            usage=None,
            elapsed_seconds=0.0,
            model="scripted-offline-demonstration",
            response_id=None,
            status="completed",
            estimated_cost_usd=None,
            budget_cost_usd=None,
            request_started=False,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="threadmark",
        description="SaoLabs AI Threadmark: memory-guided workflow evidence.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "demo", help="Run a scripted offline demonstration; no API key or network."
    )
    commands.add_parser(
        "setup-tokenizer",
        help="Explicitly download and verify the official tokenizer cache once; no API key or charge.",
    )
    inspect = commands.add_parser(
        "inspect", help="Validate annotated input and show a request plan offline."
    )
    inspect.add_argument("--input", type=Path, required=True)
    compare = commands.add_parser(
        "compare", help="Compare full history with memory-guided evidence."
    )
    compare.add_argument("--input", type=Path, required=True)
    compare.add_argument(
        "--output",
        type=Path,
        required=True,
        help="A new output directory; never overwritten.",
    )
    compare.add_argument(
        "--prepare-only",
        action="store_true",
        help="Write a plan without a key or network.",
    )
    compare.add_argument("--model", default=Settings().model)
    compare.add_argument("--reasoning-effort", default=Settings().reasoning_effort)
    compare.add_argument(
        "--budget-usd", type=float, help="Required total budget cap for this paid run."
    )
    for flag in ("input", "cached-input", "cache-write", "output"):
        compare.add_argument(
            f"--{flag}-usd-per-million",
            type=float,
            help="Required current USD price per million tokens; verify with your provider.",
        )
    generate = commands.add_parser(
        "generate",
        help="Generate new synthetic workflows offline; not frozen benchmark reproduction.",
    )
    generate.add_argument("--split", choices=("dev", "test"), default="dev")
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument(
        "--protocol-sha256",
        help="Required 64-character locked protocol hash for test generation.",
    )
    generate.add_argument("--seed", type=int, default=20261027)
    generate.add_argument("--per-cell", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "demo":
            from .datasets import demo_dataset

            # Byte counting is conservative and deterministic for this small
            # scripted example, and never downloads tokenizer cache data.
            report = run_comparison(
                demo_dataset(),
                RecordedDemoProvider(),
                count_tokens=lambda text: len(text.encode("utf-8")),
            )
            print(
                json.dumps(
                    {
                        "demonstration": "SCRIPTED OFFLINE OUTPUTS: not measured model accuracy.",
                        "network_calls": 0,
                        "planned_queries_per_arm": 2,
                        "answers": [
                            {
                                "query_id": row["query_id"],
                                "arm": row["arm"],
                                "answer": row.get("answer"),
                            }
                            for row in report["query_results"]
                        ],
                    },
                    indent=2,
                )
            )
            return 0
        if args.command == "setup-tokenizer":
            from .context import setup_tokenizer

            print(json.dumps(setup_tokenizer(), indent=2))
            return 0
        if args.command == "generate":
            from .context import token_counter
            from .datasets import generate_dataset

            if args.output.exists():
                raise FileExistsError("Refusing to overwrite an existing dataset")
            if args.split == "test":
                import re

                if not re.fullmatch(r"[0-9a-f]{64}", args.protocol_sha256 or ""):
                    raise ValueError(
                        "Test generation requires a locked protocol SHA256"
                    )
            data = generate_dataset(
                args.split,
                token_counter(),
                args.protocol_sha256,
                args.seed,
                args.per_cell,
            )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            # Exclusive creation prevents races from overwriting another dataset.
            with args.output.open("x", encoding="utf-8") as handle:
                json.dump(data, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
            print(
                json.dumps(
                    {
                        "output": str(args.output),
                        "workflows": len(data["workflows"]),
                        "notice": "New synthetic dataset; not the frozen published benchmark.",
                    }
                )
            )
            return 0
        settings = Settings(
            model=getattr(args, "model", Settings().model),
            reasoning_effort=getattr(
                args, "reasoning_effort", Settings().reasoning_effort
            ),
        )
        dataset = load_dataset(args.input, settings)
        plan = inspect_dataset(dataset, settings)
        if args.command == "inspect":
            print(json.dumps(plan, indent=2))
            return 0
        # Validate paid parameters before creating a directory or reading a key.
        prices = budget = None
        if not args.prepare_only:
            from .budget import Prices

            required = (
                "budget_usd",
                "input_usd_per_million",
                "cached_input_usd_per_million",
                "cache_write_usd_per_million",
                "output_usd_per_million",
            )
            if any(getattr(args, name) is None for name in required):
                parser.error(
                    "Paid comparison requires --budget-usd and all four --*-usd-per-million prices"
                )
            import math

            if not math.isfinite(args.budget_usd) or args.budget_usd <= 0:
                parser.error("--budget-usd must be positive and finite")
            prices = Prices(
                input=args.input_usd_per_million,
                cached_input=args.cached_input_usd_per_million,
                cache_write=args.cache_write_usd_per_million,
                output=args.output_usd_per_million,
            )
            plan["prices_usd_per_million"] = asdict(prices)
            plan["budget_usd"] = args.budget_usd
        args.output.mkdir(parents=True, exist_ok=False)
        _write_json(args.output / "request-plan.json", plan)
        if args.prepare_only:
            print(
                json.dumps(
                    {
                        "status": "prepared",
                        "output": str(args.output),
                        "network_calls": 0,
                        "planned_queries_per_arm": plan["planned_queries_per_arm"],
                    }
                )
            )
            return 0
        from .budget import BudgetLedger
        from .providers import OpenAIProvider

        key = (
            os.environ.get("OPENAI_API_KEY")
            or getpass("OpenAI API key (hidden, used for this process only): ").strip()
        )
        if not key:
            raise ValueError("An OpenAI API key is required for a paid comparison")
        budget = BudgetLedger(args.output / "budget.json", limit_usd=args.budget_usd)
        provider = OpenAIProvider(
            key,
            model=settings.model,
            reasoning_effort=settings.reasoning_effort,
            prices=prices,
            budget=budget,
        )
        del key

        def save_progress(report: dict[str, Any]) -> None:
            report["budget"] = budget.snapshot()
            report["prices_usd_per_million"] = asdict(prices)
            report["plan"] = plan
            _write_json(args.output / "report.json", report)

        report = run_comparison(
            dataset, provider, settings=settings, on_progress=save_progress
        )
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "output": str(args.output),
                    "summary": report["summary"],
                    "budget": budget.snapshot(),
                },
                indent=2,
            )
        )
        return (
            0
            if all(row["status"] == "completed" for row in report["query_results"])
            else 2
        )
    except (ValueError, OSError, EOFError) as exc:
        print(f"threadmark: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print(
            "threadmark: interrupted; inspect the saved report and budget ledger before starting another run.",
            file=sys.stderr,
        )
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
