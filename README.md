# Threadmark

**Memory-guided evidence retrieval for evolving workflows, by [SaoLabs AI](https://github.com/saolabsai).**

Threadmark keeps a small register of current facts as a workflow changes. When you ask a question, that register helps select relevant original records. A language model answers from those records and a recent-history window. The original archive is retained.

For example, a project may change owner, deadline, and approval status several times. Threadmark tracks those changes and sends the reader a smaller evidence packet instead of the entire project history.

This is an experimental Python library and comparison harness. It combines an updater model, structured memory, source selection, and a reader model. It does not train a foundation model or modify a model's weights.

## Research and mathematical design

Read the [Threadmark research note](research/threadmark.md) or [download the paper as a PDF](research/threadmark.pdf). The note explains the original active-model idea, the implemented update and evidence-selection equations, lifecycle cost, and the experiment's limits. The [annotated research references](research/references.md) and [BibTeX bibliography](research/references.bib) link to the source papers.

Related work informed the original proposal for a learned latent memory layer. This release implements an explicit fact register and retrieval of original evidence; it is not a reproduction of those papers or an implementation of the proposed latent layer. Its equations describe this implementation and the hypotheses still to test.

## Try it locally

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

## What the benchmark shows

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

## How it works

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
