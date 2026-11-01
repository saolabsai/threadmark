# Threadmark research

Read the [November 1 long-context evaluation](evaluation-2026-11-01.md) for historical memory, proposal resolution, the separate 54-question study and 18-question follow-up, and the measured cost and speed tradeoffs. Its [sanitized evidence archive](../benchmarks/long-context-2026-11-01-evidence.zip) preserves answer-level results. The public `WorkflowMemory` API packages the tested algorithm with offline-validated persistence and reader checks.

Read the [October 28 evaluation note](evaluation-2026-10-28.md) for the completed sparse-event comparison, preserved register/retrieval comparison, and separately trained SmolLM2-versus-Qwen component test. Each experiment retains its own protocol, results, and limitations, including earlier negative results.

Read the [technical note](threadmark.md) on GitHub or download the [PDF](threadmark.pdf). It describes the implemented method and benchmark, then separately develops the proposed latent VSLM and decoder connection. The [annotated references](references.md) and [BibTeX](references.bib) attribute the four papers that informed the original proposal.

The original PDF remains unchanged. Newer evaluation results are recorded in the dated Markdown note and their evidence archives, rather than presented as though they appeared in the original paper.

The technical note is experimental engineering documentation, not a peer-reviewed publication. The formulas describe this project's implementation and proposal; the repository does not reproduce the cited papers' models or claim their results.

To rebuild the PDF from its Markdown source:

```sh
python -m pip install -e ".[paper]"
python scripts/build_research_pdf.py
```

This uses ReportLab and Matplotlib locally. The build makes no model calls and needs no API key. Temporary equation images are written beneath the ignored `runs/paper-preview/` directory. The final PDF is written beside its source in `research/`.
