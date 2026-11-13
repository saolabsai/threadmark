# Threadmark

**Experimental working memory for evolving workflows, by [SaoLabs AI](https://github.com/saolabsai).**

Threadmark keeps source-linked memory as a workflow changes. `WorkflowMemory` preserves current facts, earlier values, pending and resolved proposals, and recorded reasons. It can answer free-form questions from that structured memory and save or restore its history without another model call. The original archive is retained, and predictions can be wrong.

For example, a project may change owner, deadline, and approval status several times. A proposed change stays pending until an acceptance resolves it; its approved value cites both the proposal and the acceptance. Earlier values remain available for historical questions. Explicit external proposal labels are matched only when they uniquely identify an open proposal for the same entity and field.

The existing `Threadmark` API remains available for its current-fact register and original-evidence reader. `EventDeltaMemory` extracts sparse assignments for direct current-state lookup. These are separate methods with separate measurements. See [workflow memory](docs/workflow-memory.md) for the new history and decision API, its limits, and an offline example.

This is a Python library and comparison harness. Its workflow memory methods use fixed provider weights. A separate research component trains LoRA event classifiers on Qwen and SmolLM2; that narrow component does not establish a new foundation model, a learned decoder memory layer, or general replacement of retrieval-augmented generation.

## Budgeted context for different models

The goal is to process new information once, keep useful state outside the
model, and send a bounded context on each subsequent request. `ActiveContext`
compiles that context from `WorkflowMemory` without another model call. It
preserves selected current facts, pending decisions and their original evidence;
adds history and relevant records when they fit; and reports omissions. Essential
context that cannot fit raises `ContextBudgetExceeded` instead of being cut off.

```python
from threadmark import ActiveContext
from threadmark.context import token_counter

# memory is an existing WorkflowMemory, updated with new events only.
request = ActiveContext(memory).compile(
    "What is approved now, and what is still pending?",
    max_input_tokens=4096,
    count_tokens=token_counter(),  # requires the explicitly prepared local cache
)
# Pass request.instructions and request.content to a compatible reader adapter.
# request.rendered_request is the exact text measured by the supplied counter.
```

The default JSON-envelope count excludes provider-specific chat framing. Supply
the target model's complete request renderer and tokenizer for a native token
budget, including tools and any additional messages. This is portable external
memory; model weights stay unchanged. Compression does not establish extraction
or answer accuracy, and ingesting new text can still require model calls.

See the [active-context guide](docs/active-context.md) for the architecture,
equations, incremental use, token accounting, and limits. Its offline evaluation
uses application-supplied operations and is separate from the failed November 10
extractor experiment; no trained checkpoint was promoted by this change.

## 0.2.0: workflow integrity and local extraction

Version 0.2.0 adds ordered distinct changes, proposals, and notes within one
record; indexed proposal identities; atomic `apply_operations` without an LLM
call; and an optional recent-history window for the updater. New workflow
snapshots use schema 2 and still read schema 1. Existing defaults preserve the
current-facts/open-proposals reference unless history is explicitly enabled.

Local inference can now load an explicitly supplied LoRA adapter alongside a
separately supplied base model. Optional structured workflow decoding constrains
the JSON operation format. It does not determine whether an operation is true.
See [workflow memory](docs/workflow-memory.md) and
[local adapters and decoding](docs/local-and-retrieval.md#use-an-explicit-local-lora-adapter)
for API examples, optional dependencies, and failure behavior.

These changes are **experimental, not a production-readiness claim**. Offline
integrity checks improved from 9/12 to 12/12 cases with recorded operations,
including 1,000 unique updates under an explicit 2 MB packet cap. They do not
measure language-model accuracy. The first local-adapter trial completed 0/4
eight-event workflows in both software arms and is now development evidence.

A separate fresh four-workflow comparison used the same fixed adapter and
improved memory in both arms. Constrained decoding committed 17/32 events and
completed 2/4 workflows, versus 2/32 and 0/4 with ordinary decoding. **Both
produced 0/4 exact final current states.** Committing a valid operation can
still preserve the wrong fact or omit an update. No model was retrained in
these trials, no RAG comparison was run, and local GPU/electricity costs remain
unpriced. Historical benchmarks below do not measure these APIs or decoder
settings. See the [engineering evaluation notes](docs/workflow-memory.md#evaluation).

A follow-up reused those same cases with the unadapted base model. Constraints
committed 18/32 events and completed 2/4 workflows, but both decoding arms again
had 0/4 exact final states. Removing the adapter did not resolve the observed
failures. This is a separate development diagnostic, not fresh validation or
a general conclusion about adapter quality.

## Research and mathematical design

Read the [November 1 long-conversation evaluation](research/evaluation-2026-11-01.md) for the history and proposal improvements, the separate 54-question study and 18-question follow-up, costs, and limitations. The packaged API adds persistence and validation around the tested algorithm; it has not undergone another paid model evaluation.

Read the [October 28 evaluation note](research/evaluation-2026-10-28.md) for the sparse-event and register/retrieval comparisons, plus the separate trained-backbone test. SmolLM2 and Qwen each reached 160/160 on their matched, narrow weekday-state task; this does not measure broader SmolLM2 reader accuracy.

Read the [Threadmark research note](research/threadmark.md) or [download the paper as a PDF](research/threadmark.pdf). The note explains the original active-model idea, the implemented update and evidence-selection equations, lifecycle cost, and the experiment's limits. The [annotated research references](research/references.md) and [BibTeX bibliography](research/references.bib) link to the source papers.

Related work informed the original proposal for a learned latent memory layer. This library implements explicit structured memory and, in its evidence-reader path, retrieval of original records. It is not a reproduction of those papers or an implementation of the proposed latent layer. Its equations describe this implementation and the hypotheses still to test.

## Try it locally

For local model inference, direct predicted-state lookup, BM25, and a dense-vector baseline, see [local models and retrieval comparisons](docs/local-and-retrieval.md). These experimental modes need their own accuracy measurements; direct-state lookup returns unverified model predictions.

Requires Python 3.10 or newer. The demo runs offline. Inspecting or preparing real inputs uses cached tokenizer data; download it once with the setup command below. No API key is needed for these steps.

```bash
git clone https://github.com/saolabsai/threadmark.git
cd threadmark
python -m venv .venv
```

Activate the environment:

```bash
# macOS / Linux
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Then install and run:

```bash
python -m pip install -e .
python examples/use_application_events.py
python examples/use_workflow_memory.py
python -m threadmark demo
python -m threadmark setup-tokenizer
python -m threadmark inspect --input examples/workflow.json
python -m threadmark compare --input examples/workflow.json --output runs/prepared --prepare-only
```

The workflow-memory example and demo use recorded responses and a conservative byte counter to demonstrate the data flow. They are not fresh model evaluations. The example includes an external proposal label, its acceptance, an owner handoff, a free-form answer, and snapshot restoration. `setup-tokenizer` downloads official tokenizer data once; it does not make a paid API call. Subsequent inspection and preparation run offline. Preparation validates the fixture and writes a request plan without calling OpenAI. See [workflow-memory usage](docs/workflow-memory.md) for the new API, and [usage](docs/usage.md) for the existing CLI, paid comparison, input format, and output files.

The [application-events example](examples/use_application_events.py) uses typed
operations supplied by the application. It demonstrates ordered handoffs,
multiple proposals, acceptance/rejection, and snapshot replay without model or
tokenizer calls.

## Latest: long-conversation memory improvement

The November 1 follow-up used **three fresh synthetic histories** in the 32K, 64K and 96K bands, with six questions per history. All methods used the same Luna reader. The largest full-history request had **99,495 input tokens**.

| Method | Correct answers | Correct and source-supported | Known lifecycle API cost |
| --- | ---: | ---: | ---: |
| Full history | 18/18 | 17/18 | $0.30167155 |
| Hybrid RAG | 18/18 | 17/18 | $0.00867095 |
| Current-state memory | 9/18 | 9/18 | $0.07599370 |
| Threadmark history and decisions | 18/18 | 18/18 | $0.11125990 |

The independent blinded AI review agreed with all 72 correctness and supported-correct decisions. The full-history and RAG support failures were missing a proposal citation for the same otherwise correct answer; RAG retrieved that source. This is not evidence of a retrieval failure.

Threadmark used **91.6% fewer reader input tokens than full history**. Including ingestion, its API estimate was **63.1% lower than full history and 12.8 times hybrid RAG's cost**. Building its memories took approximately **193 seconds**; the lower average answer time does not establish an overall speed advantage. Per-method costs include memory building and answering, exclude evaluation judging, and do not price local retrieval compute.

An earlier, separately frozen 54-question study scored 52/54 supported answers for the history method versus 27/54 for current-state memory, 53/54 for full history and 54/54 for hybrid RAG. Secondary blinded AI review found one grader false negative, giving the history method 53/54; the primary score remains unchanged. A proposal-label failure from that study motivated the separately frozen follow-up. **The two studies are not pooled.**

These are small synthetic tests with two entities and 12 current fields per history. Most added length is background conversation, and domain and length are confounded in the follow-up. The results do not establish production readiness, population-level accuracy, or a general RAG replacement. Read the [evaluation and limitations](research/evaluation-2026-11-01.md) and [result summary](benchmarks/long-context-2026-11-01.json).

## Historical October 28 current-state comparison

On 12 fresh synthetic workflows with histories in the 8,192/16,384-token bands, the October 28 test asked 48 questions per method:

| Method | Exact values | Exact values and evidence | Known lifecycle API cost |
| --- | ---: | ---: | ---: |
| Full history | 46/48 | 44/48 | $0.1639632 |
| BM25 plus recent | 34/48 | 34/48 | $0.0376737 |
| Dense plus recent | 37/48 | 36/48 | $0.04033555 |
| Sparse-event state | 46/48 | 46/48 | $0.1854587 |

State lookup used explicit requested keys and made no query-time model call or source-text selection. It matched the full-history aggregate score with **two paired regressions and two improvements**. Its two wrong answers remain wrong under exact canonical-value scoring.

The tradeoff was an approximately **889-second total build across 12 workflows** and **13.1% higher API cost** than full history for four questions per workflow. Fast subsequent lookups do not remove ingestion cost, archive-ID validation, or the need for accurate extraction. This small shared-family test does not establish an accuracy guarantee or general RAG replacement.

See [how to use sparse-event memory](docs/local-and-retrieval.md#experimental-sparse-event-memory), the [full evaluation and limitations](research/evaluation-2026-10-28.md), the [result summary](benchmarks/rag-comparison-2026-10-28.json), and [frozen evidence](benchmarks/rag-comparison-2026-10-28-evidence.zip). The earlier register test remains separate: evidence-guided answers scored 46/48 and direct state 44/48 on different fresh instances, with its negative development history preserved.

## Historical October 27 benchmark

The frozen October 27, 2026 experiment used **90 synthetic workflows** across meeting coordination, customer support, and project management. Each had four questions, giving **360 paired questions**. Actual histories ranged from **1,846 to 15,215 estimated tokens**. Both readers and the updater used `gpt-5.6-luna`. Costs below are totals across all 90 workflows.

| Measure | Full-history reader | Threadmark evidence reader |
| --- | ---: | ---: |
| Exact answer accuracy, all planned questions | 349/360 (96.9%) | 346/360 (96.1%) |
| Available answers | 360/360 | 354/360 |
| Known cost for four questions per workflow | $0.902668 | $0.816890, including memory construction |
| Known cost for the first question per workflow | $0.226120 | $0.661341, including memory construction |

Threadmark used **80.0% fewer reader input tokens**, or **38.9% fewer input tokens including memory construction**. It had 11 regressions and 8 improvements compared with the full-history reader. Three API failures with unknown usage affected six Threadmark answers; those answers remain in the denominator and their charges are not included in the known costs.

**The experiment does not establish equal accuracy or guaranteed savings.** Memory construction makes the first answer more expensive. Synthetic cases share scenario families, and no retrieval-only baseline was evaluated.

Read the [benchmark methodology and limitations](docs/benchmarks.md), inspect the [checked-in result](benchmarks/workflow-2026-10-27.json), or download the [frozen evidence archive](benchmarks/workflow-2026-10-27-evidence.zip). The refactored release has not been rerun to reproduce these numbers. The archive contains the original dataset, protocol, exact requests, per-workflow results, development results, and frozen source snapshots, with private runtime details redacted.

## Evidence-reader path

The diagram below shows `active_memory`. Direct `state` lookup instead returns the predicted values at supplied entity/field keys, with citation-ID validation and no query-time reader. `EventDeltaMemory` builds that state by extracting event assignments and applying them in chronological code.

```text
Original workflow records ───────────────► retained archive
           │                                   │
           ▼                                   │
    updater language model                     │
           │                                   │
           ▼                                   │
 current facts + source IDs                     │
           │                                   │
 question + requested fields                   │
           │                                   │
           └──► select cited originals ◄────────┘
                         + recent records
                                │
                                ▼
                      reader language model
                                │
                                ▼
                        answer + source IDs
```

The reader receives original evidence, not the updater's predicted fact values. An incorrect register can still select the wrong evidence, so memory accuracy and answer accuracy need separate evaluation. Read the [architecture](docs/architecture.md) for the update rules, equations, and failure behavior.

## Repository guide

- [`src/threadmark`](src/threadmark): memory pipeline, provider interface, budget controls, and CLI.
- [`examples`](examples): fictional workflow and offline Python example.
- [`tests`](tests): offline checks for selection, validation, scoring, and budget behavior.
- [`benchmarks`](benchmarks): preserved public experiment summary and provenance.
- [`docs`](docs): usage, architecture, and benchmark interpretation.
- [`research`](research): research note, PDF, equations, and related-paper bibliography.

For development, install `python -m pip install -e ".[dev]"`, run `python -m threadmark setup-tokenizer` once, then run `python -m pytest`. See [contributing](CONTRIBUTING.md) and [security](SECURITY.md).

## License

Threadmark's code is available under the [MIT License](LICENSE). Model access, model weights, and external services remain subject to their respective providers' terms.
