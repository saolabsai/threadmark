# Contributing

Threadmark is an experimental project by SaoLabs AI. Small, focused contributions with a clear problem statement are welcome.

## Development setup

```bash
python -m venv .venv
# Activate .venv for your shell.
python -m pip install -e ".[dev]"
python -m threadmark setup-tokenizer
python -m pytest
```

Tokenizer setup downloads official encoding data once. Tests run offline after setup; the recorded demo also works without that cache. Do not add live model calls or implicit network downloads to the default test suite. A paid evaluation must be an explicit command with a model, prices, budget, and fresh output directory.

## Pull requests

Explain the problem, resulting behavior, and verification. Add meaningful tests for changes to state updates, source selection, scoring, provider parsing, or budget accounting. Keep expected labels out of model inputs. Preserve original evidence IDs and chronological ordering.

Keep fixtures fictional and small. Never commit API keys, personal workflow records, private run logs, credential stores, local budget ledgers, model weights, or generated environments. Changes to prompts or selection rules need an explanation of their evaluation impact.

Existing benchmark artifacts describe a frozen experiment. Do not edit its results to reflect a new implementation. Add a separately named result with its own data, protocol, provenance, failed attempts, and full lifecycle costs. Report failures in planned denominators; do not present successful requests alone as overall accuracy.

By submitting a contribution, you agree that your contribution may be distributed under the repository's MIT License. For a security concern, follow [SECURITY.md](SECURITY.md).
