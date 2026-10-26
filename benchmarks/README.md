# Benchmark artifacts

[`workflow-2026-10-27.json`](workflow-2026-10-27.json) is an unchanged copy of the public `workflow-data.json` export for Threadmark's original 90-workflow experiment. It contains aggregate results, development pilot summaries, source hashes, costs, and limitations. It contains no API credentials or private project accounting.

Export SHA-256: `4e1f3a2c9a42c0de6c51c9c127a4837b13f9ab73615f660440ce637ffee23b7b`.

The result was produced by the original frozen runner, not by a new run of this refactored release. The earlier trained 0.6B weekday classifier is a separate experiment and is not represented by this workflow result.

- [Methodology and interpretation](../docs/benchmarks.md)
- [Full public report](https://threadmark-research-20261027.viikys.chatgpt.site/workflow-report)
- [Original public evidence archive](https://threadmark-research-20261027.viikys.chatgpt.site/workflow-evidence.zip)

Known costs exclude three requests whose usage could not be established. Those failures remain in the result; unknown charges must not be treated as zero cost. Public artifacts are redacted and have their own hashes, separate from canonical pre-redaction source hashes.

Future evaluations should use a new directory or filename. Record data and protocol hashes, model IDs, prices, full lifecycle costs, every failed attempt, planned denominators, and any operational continuation. Do not overwrite this historical result.
