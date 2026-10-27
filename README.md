# Threadmark

**Experimental working memory for evolving workflows, by [SaoLabs AI](https://github.com/saolabsai).**

Threadmark keeps a small register of current facts as a workflow changes. Its evidence-reader mode uses that register to select original records for a language model. Its direct-state mode returns predicted facts at explicitly supplied entity/field keys, without query-time retrieval of source content or another model call. The original archive is retained, and predictions can be wrong.

For example, a project may change owner, deadline, and approval status several times. The sparse-event strategy extracts explicit changes and lets chronological code preserve untouched facts. The register strategy asks an updater to rewrite the current facts. These are separate experimental methods with separate measurements.

This is a Python library and comparison harness. Its workflow memory methods use fixed provider weights. A separate research component trains LoRA event classifiers on Qwen and SmolLM2; that narrow component does not establish a new foundation model, a learned decoder memory layer, or general replacement of retrieval-augmented generation.

## Research and mathematical design

Read the [October 28 evaluation note](research/evaluation-2026-10-28.md) for the sparse-event and register/retrieval comparisons, plus the separate trained-backbone test. SmolLM2 and Qwen each reached 160/160 on their matched, narrow weekday-state task; this does not measure broader SmolLM2 reader accuracy.

Read the [Threadmark research note](research/threadmark.md) or [download the paper as a PDF](research/threadmark.pdf). The note explains the original active-model idea, the implemented update and evidence-selection equations, lifecycle cost, and the experiment's limits. The [annotated research references](research/references.md) and [BibTeX bibliography](research/references.bib) link to the source papers.

Related work informed the original proposal for a learned latent memory layer. This release implements an explicit fact register and retrieval of original evidence; it is not a reproduction of those papers or an implementation of the proposed latent layer. Its equations describe this implementation and the hypotheses still to test.

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
python -m threadmark demo
python -m threadmark setup-tokenizer
python -m threadmark inspect --input examples/workflow.json
python -m threadmark compare --input examples/workflow.json --output runs/prepared --prepare-only
```

The demo uses recorded responses and a conservative byte counter to demonstrate the data flow. It is not a fresh model evaluation. `setup-tokenizer` downloads official tokenizer data once; it does not make a paid API call. Subsequent inspection and preparation run offline. Preparation validates the fixture and writes a request plan without calling OpenAI. See [usage](docs/usage.md) for a paid comparison, the Python API, input format, and output files.

## Latest: sparse-event current-state comparison

On 12 fresh synthetic workflows with histories in the 8,192/16,384-token bands, the latest test asked 48 questions per method:

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
