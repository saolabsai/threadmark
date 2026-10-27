# Local models, current-state lookup, and retrieval

Threadmark supports two ways to use its predicted fact register. The default `active_memory` mode uses stored citations to select original evidence for a reader model. The `state` mode returns the predicted facts themselves. State lookup makes no reader-model call and does not select original source content, but an updater mistake remains an answer mistake.

Both interfaces require a public entity catalog and explicit `requested_facts` keys. State mode uses those keys for routing; it does not interpret the free-form question text. This is a bounded current-fact interface, not a general answer to historical or open-ended document questions. Every mode retains the original archive. Citation validation still visits archive event IDs, so a model-free answer is not a claim of constant-time execution or no archive access.

## Select prompts explicitly

The library retains the original experiment's v1 prompts as defaults. Opt in to the development-selected current-fact prompts with a profile:

```python
from threadmark import Threadmark
from threadmark.prompts import prompt_profile

profile = prompt_profile("current-facts-v2")
tracker = Threadmark(catalog, provider, **profile)
tracker.ingest(events)

predicted_facts = tracker.answer(query, mode="state")
evidence_reader_answer = tracker.answer(query, mode="active_memory")
```

Here `catalog`, `events`, and `query` use the [ordinary Python API schema](usage.md). The profile supplies `updater_instructions` and `reader_instructions` constructor keywords. It emphasizes explicit clears, chronological per-field changes, and the distinction between approved and pending values. A prompt profile alone makes no accuracy guarantee. State lookup returns structurally validated, unverified model predictions; membership of a cited event ID does not prove that the event supports the value.

## Keep a session with its prompts

```python
import json
from pathlib import Path

Path("session.json").write_text(
    json.dumps(tracker.snapshot(), ensure_ascii=False), encoding="utf-8"
)
restored = Threadmark.from_snapshot(
    Path("session.json").read_text(encoding="utf-8"), provider=provider
)
answer = restored.answer(query, mode="state")
```

Default-prompt snapshots retain legacy schema 1. Custom prompts use schema 2 and include both complete instruction strings; restoration rejects missing or invalid prompt fields. Snapshots contain the workflow archive and predicted state, but do not serialize credentials, providers, budget ledgers, or model weights. Supply the intended provider again. For a local session, also pass `count_tokens=provider.count_tokens` when restoring, matching its original tokenizer.

Ingestion commits state and archive only after all new chunks validate. Earlier provider calls can already have incurred costs if a later chunk fails. Nothing retries or switches models automatically.

## Prepare local models explicitly

```sh
python -m pip install -e ".[local]"
python scripts/download_models.py --model all --destination models
```

The command pins model revisions and verifies official weight hashes. SmolLM2 downloads approximately 3.4 GB; MiniLM approximately 91 MB. Model weights retain their upstream licenses. Threadmark's Python code is MIT-licensed. Inference loads existing local safetensors with remote code disabled and never invokes the downloader.

## Experimental sparse-event memory

`EventDeltaMemory` asks the model to extract only explicitly stated changes in each new event. Code then applies those changes chronologically. This removes the need for the model to repeat every unchanged fact correctly at each update.

For example, an earlier record sets Ada as owner and Library as venue. A later record explicitly clears the venue. The reducer retains Ada and sets the venue to `null`, citing the later record. It does not guess a replacement venue or change another field.

Write `k = (entity, field)` and let `delta_t` contain the assignments extracted from event `t`. The reducer is:

```text
state_t[k] = delta_t[k]       if this event explicitly updates k
             state_(t-1)[k]  otherwise
```

An explicit clear is an assignment to `null`. An applied update receives that original event's ID as its citation. The model can still omit or misinterpret a change; the reducer is deterministic, not an oracle that corrects extraction mistakes.

Use the module import below. The `reader`, `updater`, and input objects follow the separately routed hosted example in this guide:

```python
from threadmark import Settings
from threadmark.delta_memory import EventDeltaMemory, DELTA_UPDATER_INSTRUCTIONS
from threadmark.routed_provider import RoleRoutedProvider

delta_provider = RoleRoutedProvider(
    reader, updater, updater_instructions={DELTA_UPDATER_INSTRUCTIONS}
)
delta_settings = Settings(
    model=reader.model, reasoning_effort=reader.reasoning_effort,
    chunk_tokens=2000, updater_max_output_tokens=4096,
)
memory = EventDeltaMemory(
    catalog, delta_provider, delta_settings,
    updater_instructions=DELTA_UPDATER_INSTRUCTIONS,
)
memory.ingest(events)
answer = memory.answer(query, mode="state")
```

Routing matches the exact delta instruction string. If custom instructions are passed to `EventDeltaMemory`, put that same string in the router's set. Otherwise the request routes to the reader. This example makes only updater calls; its configured reader is not called at query time. A direct updater provider also works when `Settings` matches that provider's model and reasoning effort.

Extractor inputs contain only the public catalog and new events, with no current register, gold labels, or prior archive text. Extraction covers explicit assignments and clearings only. It cannot recover an unstated value from an implicit acceptance or a reference to an earlier batch. All original events are still retained. The current state remains unverified, requested keys remain explicit, and all ingestion calls belong in lifecycle cost.

This strategy currently supports only `mode="state"` and has no snapshot/restore interface. It does not silently reuse a register-rewriting snapshot. Ingestion is atomic across its chunks, and sparse per-event extraction can require more output than a final-register summary, so chunk size and output allowance must be chosen and evaluated together.

## Use SmolLM2 locally

```python
from threadmark import LocalTransformersProvider, Settings, Threadmark
from threadmark.prompts import prompt_profile

name = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
provider = LocalTransformersProvider(
    "models/smollm2-1.7b-instruct", model=name, device="cuda"
)
provider.load()  # Separate shared model loading from inference measurements.
settings = Settings(model=name)
tracker = Threadmark(
    catalog, provider, settings,
    count_tokens=provider.count_tokens, **prompt_profile()
)
tracker.ingest(events)
prediction = tracker.answer(query, mode="state")
answer = tracker.answer(query, mode="active_memory")
```

Choose `device="cpu"` explicitly when needed. Generation uses the native chat template, greedy decoding, and frozen weights. Full chat input plus its output reservation must fit the native context; nothing is silently truncated. The pinned SmolLM2 configuration supports 8,192 tokens. Raw-text token counting excludes chat overhead; `complete()` checks the full prepared chat prompt separately.

Replacing Qwen does not make a Qwen adapter compatible with SmolLM2. The [matched backbone evaluation](../research/evaluation-2026-10-28.md#separate-trained-backbone-result) compares separately trained event classifiers; it does not compare unrestricted SmolLM2 reader quality. A separate causal-reader compatibility smoke test passed on one fixture, which is not an accuracy benchmark. Neither this profile nor this adapter implements a learned neural memory layer inside a decoder.

## Route a writer and reader separately

Hosted providers can use different models or reasoning efforts while sharing one persistent project cap and one run cap. Run `python -m threadmark setup-tokenizer` once to prepare its verified local tokenizer cache. The following example assumes `OPENAI_API_KEY`, the two approved cap values, and all four verified USD-per-million token rates are set in the environment:

```python
import os

from threadmark import OpenAIProvider, Settings, Threadmark
from threadmark.budget import BudgetLedger, PairedBudgetLedger, Prices
from threadmark.prompts import UPDATER_INSTRUCTIONS, prompt_profile
from threadmark.routed_provider import RoleRoutedProvider

prices = Prices(
    input=float(os.environ["THREADMARK_INPUT_PRICE"]),
    cached_input=float(os.environ["THREADMARK_CACHED_INPUT_PRICE"]),
    cache_write=float(os.environ["THREADMARK_CACHE_WRITE_PRICE"]),
    output=float(os.environ["THREADMARK_OUTPUT_PRICE"]),
)
project = BudgetLedger(
    "private/project-budget.json", float(os.environ["THREADMARK_PROJECT_CAP_USD"])
)
run = BudgetLedger(
    "private/example-run-budget.json", float(os.environ["THREADMARK_RUN_CAP_USD"])
)
budget = PairedBudgetLedger(project, run)
name = "gpt-5.6-luna"
reader = OpenAIProvider(
    model=name, reasoning_effort="low", prices=prices, budget=budget
)
updater = OpenAIProvider(
    model=name, reasoning_effort="medium", prices=prices, budget=budget
)
profile = prompt_profile()
provider = RoleRoutedProvider(
    reader, updater,
    updater_instructions={UPDATER_INSTRUCTIONS, profile["updater_instructions"]},
)
settings = Settings(
    model=name, reasoning_effort="low", chunk_tokens=8000,
    updater_max_output_tokens=4096, reader_max_output_tokens=1024,
)
tracker = Threadmark(catalog, provider, settings, **profile)
tracker.ingest(events)
answer = tracker.answer(query, mode="state")
```

Set rates for the chosen model before use; this example does not supply current prices. If the writer uses another model, construct it with that model's own `Prices`. `RoleRoutedProvider` matches complete updater instruction strings exactly and forwards every request once. It never chooses a model from input labels or answer quality. Its merged call records identify the actual role, model, usage, and cost.

Keep the same project ledger across runs. An existing cap is immutable; `BudgetLedger.open_existing(path)` opens it without reinitializing it. Unknown outcomes retain their holds. Reusing a run ledger aggregates that run cap across experiments, while distinct run ledgers remain subject to the shared project cap. Paired reservations are conservative local accounting, not a provider invoice or an account-wide spending limit.

## Compare all six strategies

```sh
python scripts/compare_local.py --base models/smollm2-1.7b-instruct --dense-base models/all-MiniLM-L6-v2 --output runs/local-test --bands 2048 4096 --per-cell 2
```

| Strategy | Question-time input | Reader call |
| --- | --- | --- |
| Full history | Complete original archive | Yes |
| Recent only | Recent contiguous whole-event suffix | Yes |
| BM25 | Keyword-ranked events plus recent suffix | Yes |
| Dense | MiniLM-ranked events plus recent suffix | Yes |
| Active memory | Events cited by predicted requested facts plus recent suffix | Yes |
| State | Predicted facts at the supplied requested keys | No |

All five reader strategies share instructions and output limits. The reduced evidence packets share recent-token and source-byte limits; full history receives the complete archive. Direct state has no evidence packet. The comparison charges all updater work to both state-based methods when reporting each method's lifecycle. Those per-method costs reuse the same measured ingestion and must not be summed to infer total experimental spend.

`Threadmark.from_events(...)` loads an archive for `answer(..., mode="full_history")` or `mode="bm25"` without updater calls. Active and state answers remain disabled until explicit ingestion. Dense retrieval is a separate `LocalDenseRetriever` in `threadmark.dense_retrieval`; it returns original events through `select(...)` and is integrated into the comparison script, rather than an engine `answer(mode="dense")` option.

BM25 ranks whole events by keyword overlap. Dense retrieval uses attention-masked mean pooling and L2-normalized vectors. It covers all query/event text with contiguous native-token chunks and ranks an event by its maximum cosine across chunk pairs. MiniLM's intended 256-token window includes special tokens; long endings are chunked, not discarded. The full archive is indexed once per workflow and reused across questions. Neither baseline includes a reranker or every modern RAG technique.

The script locks its protocol and source hashes before generating instances, then saves exact public requests, raw answers, selected sources, failures, and paired results. All planned questions remain in the denominator. Model loading is recorded separately; dense indexing and selection time remain measured work. Direct-state lookup includes updater costs and archive-ID validation time. Local hardware, electricity, and engineering costs are unpriced.

For development, use `--split dev`, `--seed`, `--reader-instructions`, and `--updater-instructions` to disclose the selected configuration. Then freeze the method before generating a fresh final test. `--dataset` reuses exact instances across models; its original generation and split remain authoritative. Reused development data is not a fresh final test.

The hosted backend additionally requires explicit prices, an existing project budget, a run cap, and `OPENAI_API_KEY`. `--run-budget-ledger` can impose an aggregate cap across runs. See `python scripts/compare_local.py --help` for the complete options and the [October 28 evaluation note](../research/evaluation-2026-10-28.md) for this study's protocol and limitations. A result on these synthetic current-fact workflows cannot establish general replacement of RAG.
