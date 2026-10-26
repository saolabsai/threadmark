# Using Threadmark

Threadmark can demonstrate its pipeline offline, prepare an evaluation using cached tokenizer data, or run a paid comparison against OpenAI. Python 3.10 or newer is required.

## Install

```bash
git clone https://github.com/saolabsai/threadmark.git
cd threadmark
python -m venv .venv
```

Activate with `source .venv/bin/activate` on macOS/Linux, or `.venv\Scripts\Activate.ps1` in Windows PowerShell. Then:

```bash
python -m pip install -e .
```

## Start without an API key

```bash
python -m threadmark demo
python -m threadmark setup-tokenizer
python -m threadmark inspect --input examples/workflow.json
python -m threadmark compare --input examples/workflow.json --output runs/prepared --prepare-only
```

`demo` uses a recorded provider, a fictional workflow, and a conservative byte counter. It runs fully offline and demonstrates data flow without making a claim about fresh model quality or benchmark token counts.

Run `setup-tokenizer` once while connected to the internet. It downloads official `o200k_base` tokenizer data into a local cache, without an API key or paid API call. After setup, `inspect`, generation, and `--prepare-only` run offline. The default counter fails with a setup instruction if its cache is missing or corrupt; it does not silently download data. For a computer without internet access, prepare the tokenizer cache on a connected machine and transfer the cache before running these commands.

`inspect` validates the fixture and shows its context plan. `--prepare-only` writes a request plan without reading an API key or making paid requests.

An output directory must be new. Use another path for the next run rather than replacing earlier evidence.

You can also generate a new fictional development dataset after tokenizer setup:

```bash
python -m threadmark generate --split dev --output datasets/development.json --per-cell 1
```

This creates six new workflows across the three domains and two development size bands. It does not recreate the frozen published benchmark. Test-split generation additionally requires a previously locked protocol SHA-256 through `--protocol-sha256`.

## Run a live comparison

A live comparison has two reader arms: full original history, and Threadmark's selected original evidence. It also pays for updater calls. It requires an explicit model, a dollar budget, and four prices per million tokens. The CLI reads `OPENAI_API_KEY` if set; otherwise it asks for the key in a hidden terminal prompt. Do not place the key in a command argument or input file.

The rates in the example below are the **October 27, 2026 experiment's historical rates**. Check the model's current availability and [official pricing](https://developers.openai.com/api/docs/pricing) before a live run, and substitute the rates you intend to use for accounting. A local budget is a guard, not a provider billing limit; charges for requests with unknown usage remain reserved.

For example, this starts a new comparison with a $0.25 local limit and the historical rates above:

```bash
python -m threadmark compare --input examples/workflow.json --output runs/live-example --model gpt-5.6-luna --budget-usd 0.25 --input-usd-per-million 0.20 --cached-input-usd-per-million 0.02 --cache-write-usd-per-million 0.25 --output-usd-per-million 1.20
```

Use `runs/live-example`, not the `runs/prepared` directory from the offline preparation step. Exact options are also listed by:

```bash
python -m threadmark compare --help
```

No command retries paid requests automatically. If an update fails, its completed API calls may already have incurred charges. Inspect the recorded error and budget state before explicitly starting a separate run.

## Use the Python API

This live example tracks one project owner. Its records and people are fictional. As with the CLI, verify model prices before running it.

```python
from getpass import getpass
from pathlib import Path

from threadmark import BudgetLedger, OpenAIProvider, Prices, Settings, Threadmark

settings = Settings(model="gpt-5.6-luna", reasoning_effort="none")
provider = OpenAIProvider(
    api_key=getpass("OpenAI API key: "),
    model=settings.model,
    reasoning_effort=settings.reasoning_effort,
    # Historical study rates, USD per million tokens; verify before use.
    prices=Prices(input=0.20, cached_input=0.02, cache_write=0.25, output=1.20),
    budget=BudgetLedger(Path("runs/api-budget.json"), limit_usd=0.25),
)

catalog = [{"entity_id": "project-a", "fields": ["owner"]}]
tracker = Threadmark(catalog, provider, settings=settings)
tracker.ingest([
    {
        "event_id": "e001",
        "timestamp": "2027-01-01T09:00:00+00:00",
        "sender": "Project lead",
        "text": "Approved: Maya owns project-a.",
    },
    {
        "event_id": "e002",
        "timestamp": "2027-01-02T10:00:00+00:00",
        "sender": "Project lead",
        "text": "Approved: transfer project-a ownership to Sam.",
    },
])

query = {
    "text": "Who currently owns project-a? Cite the approving record.",
    "requested_facts": [{"entity_id": "project-a", "field": "owner"}],
}
print(tracker.evidence(query))  # Original records selected for the reader.
print(tracker.answer(query))   # Paid reader call, using selected evidence.
# tracker.answer(query, mode="full_history") is another paid reader call.
```

The provider reads `OPENAI_API_KEY` when `api_key` is omitted. You can supply another provider implementing the package's provider interface; the offline example in [`examples/use_threadmark.py`](../examples/use_threadmark.py) shows the complete pipeline with recorded responses.

`ingest` processes events in their supplied chronological order. It commits the archive and register only after every chunk validates. If a chunk fails, the tracker keeps its previous state and archive, while completed calls still count toward spend. Use one writer per tracker instance. There is no automatic retry or paid fallback.

`tracker.state` and `tracker.events` return defensive copies. `snapshot()` returns JSON-serializable state and archive, and `Threadmark.from_snapshot(snapshot, provider=provider)` restores them. A snapshot contains original text and predicted facts; protect it like the input records. Provider credentials are not part of the snapshot.

## Input format

The CLI accepts a dataset envelope containing a `manifest` and a `workflows` list. [`examples/workflow.json`](../examples/workflow.json) is the reference fixture. Each workflow contains a public entity catalog, original events, and questions. Expected labels are optional for application use and necessary for scored comparisons.

| Input | Required shape |
| --- | --- |
| Entity catalog | List of objects with `entity_id` and a list of allowed `fields`. |
| Event | `event_id`, timezone-aware ISO 8601 `timestamp`, `sender`, and `text`. IDs must be unique. |
| Query | `text` and `requested_facts`, each identifying an allowed `entity_id` and `field`. CLI fixtures also identify questions with `query_id`. |
| Expected answer | Evaluation-only `facts` entries containing a canonical string or `null`, plus expected source IDs. |

The structured model response is:

```json
{
  "facts": [
    {
      "entity_id": "project-a",
      "field": "owner",
      "value": "Sam",
      "evidence_event_ids": ["e002"]
    }
  ]
}
```

The package validates output fields and citation IDs. It evaluates canonical values exactly; it does not silently repair wrong answers or normalize them after seeing expected labels. The inference boundary discards gold state and other annotations. The active reader sees only selected original events, question, catalog, and instructions.

## Store and interpret results

The CLI writes `request-plan.json` for preparation and live runs. A live run also writes an atomically updated `report.json` and a persistent `budget.json`. The report includes all planned questions from the start, so interrupted work is visible. A failed or unavailable answer remains a failed planned answer when comparing accuracy. An evaluation with failed or skipped questions exits with status 2. Report memory construction and reader costs together, and keep development costs separate from test costs.

`estimated_cost_usd` uses reported token categories and excludes unreported cache writes. `budget_cost_usd` uses the higher applicable input rate when cache-write usage is absent, so the guard can account for more than the displayed estimate. Neither amount is an invoice; missing or invalid usage keeps the request's full reservation held.

The default release runner is sequential for clarity. Its latency and caching conditions differ from the frozen experiment's concurrent runner. Changing the model, prices, prompts, chunk sizes, recent window, or data creates a new experiment; existing benchmark numbers do not automatically apply.

The budget ledger is persistent accounting. Do not delete it to clear a failed request's reservation or reuse a small budget with a separate ledger to imply the original limit was respected. For applications, manage budgets and access controls across all workers that can make paid requests.

## What leaves your machine

Offline commands send no records to a model. Live runs send catalog, workflow text, questions, previous predicted memory for updates, and selected original evidence to OpenAI. The full-history arm sends all original events. Generated artifacts and snapshots may contain the same sensitive text. See [security and data handling](../SECURITY.md) before using private records.
