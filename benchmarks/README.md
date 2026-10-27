# Benchmark artifacts

## October 28 evaluations

Read the [dated methodology and interpretation note](../research/evaluation-2026-10-28.md). These results are separate experiments and should not be pooled:

| Experiment | Summary | Frozen evidence |
| --- | --- | --- |
| Sparse-event state versus full history, BM25, and dense retrieval | [Sparse-event comparison](rag-comparison-2026-10-28.json) | [Archive](rag-comparison-2026-10-28-evidence.zip) |
| Register rewriting with six context strategies | [Register comparison](rag-register-comparison-2026-10-28.json) | [Archive](rag-register-comparison-2026-10-28-evidence.zip) |
| Matched SmolLM2 versus Qwen weekday classifier | [Backbone comparison](backbone-comparison-2026-10-28.json) | [Archive](backbone-comparison-2026-10-28-evidence.zip) |

The completed sparse-event comparison returned 46/48 exact state answers and 46/48 strict value-and-evidence answers. Full history also returned 46/48 exact values, with two paired gains and two regressions. Sparse-state lifecycle API cost was higher for the measured workload, so the result does not establish free ingestion or guaranteed savings. Register evidence retains negative development attempts and the original completed comparison. The backbone test measures typed weekday state, not unrestricted generated answers.

## Original October 27 workflow study

[`workflow-2026-10-27.json`](workflow-2026-10-27.json) is an unchanged copy of the redacted public `workflow-data.json` export for Threadmark's original 90-workflow experiment. It contains aggregate results, development pilot summaries, source hashes, costs, and limitations. It contains no API credentials or private project accounting.

Export SHA-256: `4e1f3a2c9a42c0de6c51c9c127a4837b13f9ab73615f660440ce637ffee23b7b`.

The result was produced by the original frozen runner, not by a new run of this refactored release. The earlier trained 0.6B weekday classifier is a separate experiment and is not represented by this workflow result.

- [Methodology and interpretation](../docs/benchmarks.md)
- [Research note with mathematical design](../research/threadmark.md)
- [Research paper PDF](../research/threadmark.pdf)
- [Frozen redacted evidence archive](workflow-2026-10-27-evidence.zip)

The archive contains the original dataset, protocol, request plan, exact request payloads, per-workflow and development reports, source snapshots, and an evidence manifest. It is the preserved public export of the frozen experiment, rather than output from the refactored library. Extract it locally to inspect the per-workflow results and their source records.

Archive SHA-256: `006a4e4c64909a6b093ef7dfbc2ee6f7bd18c43508da88c9a0baed474cdb88c7` (3,585,833 bytes).

Known costs exclude three requests whose usage could not be established. Those failures remain in the result; unknown charges must not be treated as zero cost. Public artifacts are redacted and have their own hashes, separate from canonical pre-redaction source hashes. The archive's manifest records both provenance and the hashes to use when verifying the exported files. Canonical source hashes must not be used as byte checks for redacted exports.

Future evaluations should use a new directory or filename. Record data and protocol hashes, model IDs, prices, full lifecycle costs, every failed attempt, planned denominators, and any operational continuation. Do not overwrite this historical result.
