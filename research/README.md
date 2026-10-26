# Threadmark research

Read the [technical note](threadmark.md) on GitHub or download the [PDF](threadmark.pdf). It describes the implemented method and benchmark, then separately develops the proposed latent VSLM and decoder connection. The [annotated references](references.md) and [BibTeX](references.bib) attribute the four papers that informed the original proposal.

The technical note is experimental engineering documentation, not a peer-reviewed publication. The formulas describe this project's implementation and proposal; the repository does not reproduce the cited papers' models or claim their results.

To rebuild the PDF from its Markdown source:

```sh
python -m pip install -e ".[paper]"
python scripts/build_research_pdf.py
```

This uses ReportLab and Matplotlib locally. The build makes no model calls and needs no API key. Temporary equation images are written beneath the ignored `runs/paper-preview/` directory. The final PDF is written beside its source in `research/`.
