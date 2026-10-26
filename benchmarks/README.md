# Benchmark artifacts

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
