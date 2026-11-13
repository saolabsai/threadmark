# Changelog

## Unreleased: budgeted active context

- Add a provider-independent `ActiveContext` compiler for incrementally maintained
  workflow memory, with explicit request rendering and token budgets.
- Preserve current facts, open/active-approved proposals, unresolved references
  and complete source evidence; select optional history, claims and original
  records as whole bundles and expose coverage. Essential overflow raises an
  explicit error without model calls or truncation.
- Export immutable compiled requests, with model-specific renderer/tokenizer
  support. Include offline context-retention/token tests and an integration
  example. These checks do not establish model accuracy or production readiness.

## 0.2.0 — draft (unreleased)

- Preserve distinct ordered assignments, proposals, and notes for the same target within one event; continue rejecting identical duplicate operations.
- Give additional same-event/same-target proposals indexed IDs (`::2`, `::3`, …), retaining existing first-proposal identifiers and scoped alias resolution.
- Add transactional `WorkflowMemory.apply_operations` for externally supplied typed operations without model or tokenizer calls. Validation does not establish semantic correctness.
- Add opt-in `reference_history_entries=0..256` (default zero), bounded by the updater reference limit without pruning the archive or full memory packet.
- Write workflow snapshot schema 2 with the history setting; continue restoring schema 1 with its original no-history behavior.
- Add explicit local LoRA adapter loading, pinned file identities, structural compatibility checks, and separate optional dependencies. The caller must supply the intended local base; adapter metadata cannot choose a remote base.
- Add optional structured workflow decoding for supported local updater calls, with bounded JSON/schema checks and no unconstrained fallback. Format compliance does not verify extracted claims.
- Add a source-snapshotted offline integrity runner: 9/12 cases pass on the prior reducer versus 12/12 on the improved reducer, including 1,000 unique updates with an explicit 2 MB packet cap. These use recorded operations, not live extraction.
- Expose `WorkflowMemory.updater_instructions` and public default workflow instruction constants for exact role routing, including the opt-in history suffix.
- Retain the failed first local trial as development evidence. A separate fresh constrained-decoding trial commits 17/32 events and completes 2/4 workflows versus 2/32 and 0/4, but both arms have 0/4 exact final current states. No model retraining, RAG comparison, or production-readiness claim follows.
- Record a separate unadapted-base diagnostic on the reused development fixture: constrained decoding commits 18/32 events and completes 2/4 workflows, but both arms still have 0/4 exact final states. This is not fresh replication or evidence of a general adapter effect.

- Added `WorkflowMemory` for historical facts, explicit proposal acceptance/rejection, uniquely resolved external proposal labels, and source-linked reasons.
- Added free-form answers from bounded structured memory and portable snapshots restored by validated operation replay without model calls.
- Added an offline workflow-memory example, integration guide, and separately reported November 1 long-context studies with sanitized answer-level evidence and full cost limitations.
- Added optional local Hugging Face generation, BM25 and MiniLM retrieval baselines, and direct predicted-state lookup without a query-time reader call.
- Added explicit v2 prompt profiles, custom-prompt snapshots with backward-compatible restoration, role-specific providers, and paired project/run spending limits.
- Added a six-arm comparison driver that freezes protocols before fresh test generation and preserves public request/response journals and source snapshots.
- Added experimental sparse-event extraction with a deterministic chronological reducer, explicit clears, automatic event citations, and atomic ingestion.
- Preserved separate register, sparse-event, and matched-backbone experiments with independently audited results, negative trials, exact source snapshots, and documented limits.
- Made comparison source snapshots work with ordinary package installations as well as editable checkouts, using the loaded files for both hashes and copies.

- Added a repository research note, downloadable PDF, mathematical design, and annotated bibliography that distinguish the original latent-memory proposal from the implemented explicit register and evidence retrieval.
- Replaced external hosted-page links with repository documentation and the preserved redacted evidence archive. Included research and benchmark evidence in the source distribution.

## 0.1.0 — 2026-10-27

- Extracted the workflow prototype into a reusable Python package with a provider interface, explicit settings, source selection, validation, scoring, and budget controls.
- Added an offline recorded demo, fictional workflow fixture, comparison CLI, and offline tests.
- Added explicit tokenizer setup; regular planning and generation use the local cache and fail closed when it is unavailable.
- Documented installation, data handling, architecture, and evaluation limitations.
- Preserved the redacted public summary of the original 90-workflow experiment with links to its frozen evidence. The refactored package has not been rerun to reproduce that experiment.
- Released SaoLabs AI's code under the MIT License.
