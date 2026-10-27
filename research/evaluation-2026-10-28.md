# Current facts, retrieval, and the cost of maintaining state

**Both phases completed — October 28, 2026.** The sparse-event method returned 46/48 exact current-state answers, matching full history's aggregate score on its fresh final dataset. It had two paired regressions and two improvements, cost 13.1% more in API charges for the measured workload, and required substantial ingestion before its model-free lookups. This small synthetic result does not establish equal population accuracy or general replacement of RAG. The earlier register comparison remains preserved separately below.

## Latest: Phase 2 sparse-event results

The fresh final test used 12 workflows across three domains, two history-size bands, and two scenario variants, with four questions each. All 48 planned answers were available in every arm, and there were no unknown-usage calls.

| Strategy | Exact values / planned | Exact values and evidence | Known lifecycle API cost |
| --- | ---: | ---: | ---: |
| Full history | 46/48 | 44/48 | $0.1639632 |
| BM25 plus recent | 34/48 | 34/48 | $0.0376737 |
| Dense plus recent | 37/48 | 36/48 | $0.04033555 |
| Sparse-event predicted state | 46/48 | 46/48 | $0.1854587 |

State and full history were both correct on 44 questions. State was uniquely correct on two, and full history was uniquely correct on two. Equal aggregate scores therefore retain **two regressions and two improvements**, rather than identical answers or zero quality loss. The two state failures returned `pending_scope` values with literal surrounding single quotes. The required canonical values were unquoted, so both remain incorrect under the frozen exact-value contract. No output normalization or regrading was applied.

Auditing every stored field, including fields outside the questions, found **180/184 exact field values and 10/12 completely correct registers**. Two additional unqueried `approved_scope` values also retained surrounding single quotes. All four field mismatches remain errors; query accuracy alone does not establish that every stored fact is correct.

State lookup made no query-time language-model call or original-source content selection. Its **$0.1854587** includes every extraction call and is **13.1% higher** than full history's **$0.1639632** for four questions per workflow. The run dispatched 228 API calls at a known total of **$0.42743115** across all four arms. Local dense-encoder hardware, energy, and engineering costs are unpriced.

| Measured wall-time total | Full history | Sparse-event state |
| --- | ---: | ---: |
| Ingestion across 12 workflows | None | 889.392 s |
| Answering all 48 queries | 136.888 s | 0.006453 s |

Each state answer became available only after its workflow's ingestion. The roughly 889-second total build cost cannot be omitted when describing those fast lookups. These are sequential experiment totals, including the measured implementation overhead, not production latency guarantees. Lookup still validates citation membership against archive IDs and requires explicit requested keys.

Read the [Phase 2 public summary](../benchmarks/rag-comparison-2026-10-28.json) and [frozen sparse-event evidence](../benchmarks/rag-comparison-2026-10-28-evidence.zip). Phase 1 used different final instances; comparing its direct-state 44/48 with Phase 2's 46/48 does not, by itself, isolate a causal method improvement. Neither result is pooled with the older study or the trained-backbone task.

## Phase 1 results

There were 48 planned answers per strategy. All 48 were available in every arm, with no unknown-usage calls.

| Strategy | Exact values / planned | Exact values and evidence | Known lifecycle API cost |
| --- | ---: | ---: | ---: |
| Full history | 46/48 | 45/48 | $0.163745 |
| Recent only | 36/48 | 36/48 | $0.021391 |
| BM25 plus recent | 34/48 | 33/48 | $0.037559 |
| Dense plus recent | 36/48 | 35/48 | $0.040315 |
| Memory-guided originals | 46/48 | 45/48 | $0.092576 |
| Direct predicted state | 44/48 | 44/48 | $0.070236 |

The evidence-guided method and full history were both correct on 44 questions. Each uniquely answered two of the remaining questions correctly. Equal aggregate scores therefore include **two regressions and two improvements**; they do not show interchangeable answers, zero regressions, or equal population accuracy. Direct-state lookup missed four questions and did not match full history.

The run dispatched 269 API calls at a known total of **$0.3555856**. State lookup made no query-time model calls, but its lifecycle cost includes all updater calls. The evidence-guided cost includes those same updater calls plus its reader calls. These method totals reuse one observed ingestion and must not be added together as actual experimental spend. Dense encoder hardware and energy costs remain unpriced.

The follow-up sparse-event strategy asks a model for per-event explicit assignments and clearings, then applies those deltas with a deterministic chronological reducer. It targets possible forgetting when a model repeatedly rewrites the full register. Extraction can still omit or misinterpret updates. Phase 2 is reported separately and does not replace this completed Phase 1 result.

## Sparse-event method and frozen Phase 2 design

The new `EventDeltaMemory` extractor receives the public catalog and a chunk of new original events. It outputs sparse per-event assignments, not a replacement copy of the whole register. The reducer applies extracted changes in original event order, assigns the event's ID as evidence, and preserves every untouched key. An explicit clear is a real assignment to `null`.

For a partial update map `delta_t` extracted from event `t`, the rule is `state_t[k] = delta_t[k]` when `k` is present in that map; otherwise `state_t[k] = state_(t-1)[k]`. Keeping unchanged values is a code operation. Deciding which explicit changes the event contains remains a fallible model operation.

The disclosed development check returned 12/12 exact state answers and citation sets. That reused development fixture is not the final test. Phase 2 froze the sparse strategy before generating a separate seed's 12 workflows across the same three domains, two history bands, and two variants. It compared full history, BM25, dense retrieval, and direct sparse-event state, with 48 planned answers per arm. Its reader used Luna at low effort with the same v2 reader instructions and 1,024-token allowance. Its extractor used Luna at medium effort, 2,000-token whole-event chunks, and a 4,096-token output allowance.

This extractor has no prior register or old archive in its input. It supports explicit assignments and clearings rather than inferring unstated values from earlier-batch references or implicit approvals. There is no query-time reader or original-source selection for state lookup, but original archives and citation-ID validation remain. The strategy currently has no snapshot/restore interface. Its completed result above retains failures, regressions, and total extraction work.

## What this study tests

The earlier workflow experiment compared complete history against memory-guided original evidence. It did not include retrieval baselines. This study adds keyword retrieval, dense retrieval, recent context alone, and direct predicted-state lookup. The narrow question is whether a maintained register can answer explicitly requested current fields with useful accuracy and lifecycle cost.

The direct-state arm reads predicted register entries rather than retrieving original source content or asking another language model. It still validates citations against archive IDs. The caller already supplies entity/field keys, so this arm does not parse arbitrary questions or answer historical and open-ended document queries. Existing event IDs establish citation membership, not semantic truth.

## Frozen final design

Phase 1 used 12 newly generated synthetic workflows: meeting coordination, support, and project management, each with two history-size bands and two scenario variants. Four questions per workflow gave 48 planned answers per strategy. The six strategies shared the same original archive, catalog, questions, and scoring labels. Labels were used after source selection and inference for scoring and source-coverage diagnostics; they never entered model requests or retrieval ranking.

| Setting | Selected value |
| --- | --- |
| Reader | `gpt-5.6-luna`, reasoning effort `low` |
| Reader output allowance | 1,024 tokens |
| Updater | `gpt-5.6-luna`, reasoning effort `medium` |
| Updater output allowance | 4,096 tokens |
| Whole-event ingestion chunk budget | 8,000 estimated tokens |
| History bands | 8,192 and 16,384 estimated tokens |
| Prompt profile | Current-facts reader v2 and updater v2 |
| Retrieval hits | Up to eight positive-ranked whole events |
| Recent suffix | At most 1,024 estimated tokens, whole events only |
| Reduced original-source packet | At most 20,000 UTF-8 bytes |

History bands describe rendered original events, not complete API payloads. Full reader inputs also contain instructions, catalogs, JSON framing, and questions. The preserved dataset supplies the observed history lengths. Questions occurred after full ingestion, so this was final-state indexing rather than an interleaved streaming-query test.

The protocol and exact implementation were frozen before generating final instances. The evidence records source hashes and frozen source copies. The public library can bind the same v2 profile explicitly through `prompt_profile()`; default v1 instructions remain available. Prompt selection, reasoning effort, and chunk size are configuration changes, not weight training or a new neural decoder architecture.

## Six strategies

1. **Full history:** a reader receives all original events.
2. **Recent only:** a reader receives the newest contiguous suffix of complete events fitting the recent-token limit.
3. **BM25 plus recent:** event-level keyword ranking uses public question text, requested entity labels, and requested field definitions. Positive hits are combined with the recent suffix.
4. **Dense plus recent:** a pinned local `all-MiniLM-L6-v2` encoder applies masked mean pooling and L2 normalization. Contiguous native-token chunks cover entire events and expanded queries; maximum query-chunk/event-chunk cosine ranks each event. Positive hits are combined with the same recent suffix. The per-workflow index is reused across questions.
5. **Memory-guided originals:** the updater's predicted citations for requested facts select original events, combined with the same recent suffix. Predicted values are not supplied to the reader.
6. **Direct predicted state:** the supplied requested keys select facts from the updater's current register, with no reader call or source-text selection.

Selected original events are deduplicated and returned chronologically. Reduced source unions that exceed the byte cap fail without truncating records. Full history is intentionally uncapped by that reduced-packet limit. BM25 and dense retrieval have no reranker; this experiment cannot characterize every RAG design.

## Scoring and accounting

An exact answer requires every requested value to match its label. A separate strict measure also requires the latest direct evidence IDs. Invalid JSON, schema violations, failed ingestion, and unavailable answers remain incorrect in the planned denominator. Paired comparisons identify gains, regressions, both-correct answers, and both-wrong answers on the same questions. Questions within a workflow share history and sometimes facts; they are not independent observations.

All five reader arms use the same selected reader prompt and output limit. Direct-state lookup skips the reader; this saving is measured alongside updater accuracy rather than substituted with a hypothetical perfect answer.

Per-method lifecycle accounting includes every attempted updater call for both state-based methods, including failures. The experiment reuses one measured ingestion, so those two counterfactual method totals must not be added together as actual spend. Actual experimental API spend counts each dispatched call once. Project and aggregate-run ledgers enforce the approved caps; unknown outcomes keep their reservations and must be reported separately from known costs.

Shared model loading is separate from query timing. Retrieval selection and dense indexing are recorded, and direct-state lookup still performs archive-ID validation. Local CPU/GPU, energy, and engineering costs remain unpriced. Reader token reductions alone do not establish lower total cost or a constant-time lookup.

## Development history

The selected configuration followed disclosed development experiments with prompt, reasoning-effort, and ingestion-chunk changes. Those attempts remain in the evidence, including negative results and a denied Sol access attempt. A denied request provides no evidence about Sol answer quality. Development success is a selection signal, not a final accuracy claim; the fresh final test is not used to tune the chosen configuration.

## Separate trained backbone result

A matched, narrow component experiment replaced the Qwen backbone with SmolLM2. Each frozen backbone received its own trained LoRA adapter and nine-way event-effect classifier: seven weekday assignments, `CLEAR`, and `KEEP`. Code replays predicted actions into a weekday-or-unknown state. These are typed state readouts, not generated reader answers or the broad multi-field workflows above.

Both backbones followed the same three-pass plain-data and two-pass augmented-data schedule, with the same data and validation-based selection. SmolLM2 retained plain epoch 1 at augmented epoch 0 (the initializer, before augmentation training); Qwen selected plain epoch 3 followed by augmented epoch 1. Both completed the full requested training schedule before selection. On the fresh final 160 instances arranged in 80 related pairs, both produced **160/160 correct final-state readouts and 80/80 correct pairs**. Both also classified all 1,612 evaluated event occurrences correctly. SmolLM2 matched this Qwen result; the test does not show an accuracy improvement over Qwen.

The selected development checkpoints scored 123/128 for SmolLM2 and 127/128 for Qwen. SmolLM2's development errors included a cancellation wording that it classified as unchanged. The final test's perfect scores do not erase those observed weaknesses or establish that Qwen caused the earlier workflow errors.

State controls gave 20/160 correct when state was erased to unknown, 17/160 when final predicted states were cyclically swapped without looking at labels, and 0/160 with a deliberately wrong donor state. The last donor was selected using different gold labels only after inference; it is a label-informed diagnostic, not a blind retrieval baseline. These controls check dependence on stored state, not general question-answering ability or semantic verification of arbitrary memories.

The final cases use fresh instances from shared finite template families. A perfect small matched test does not establish unseen-family robustness or equal population accuracy. A separate base SmolLM2 causal-reader smoke test passed one JSON compatibility fixture; no broad SmolLM2 reader-accuracy benchmark was established. Neither component comparison implements or validates a learned neural memory layer inside a decoder or online updates to arbitrary language-model parameters.

See the [backbone summary](../benchmarks/backbone-comparison-2026-10-28.json) and [preserved backbone evidence](../benchmarks/backbone-comparison-2026-10-28-evidence.zip). The classifier results are separate from both hosted current-fact phases and must not be pooled with their question scores.

## Limits and evidence

The workflows are synthetic and share finite scenario families with earlier studies. New instances improve separation from development examples but do not establish robustness to unseen tasks, real customer records, unrestricted natural language, or document-scale historical reasoning. The test is small and does not establish equal population accuracy, an accuracy guarantee, or general replacement of retrieval-augmented generation.

The completed [Phase 2 sparse-event summary](../benchmarks/rag-comparison-2026-10-28.json) and [evidence archive](../benchmarks/rag-comparison-2026-10-28-evidence.zip), plus the preserved [Phase 1 register summary](../benchmarks/rag-register-comparison-2026-10-28.json) and [evidence archive](../benchmarks/rag-register-comparison-2026-10-28-evidence.zip), retain their protocols, datasets, exact requests, raw outputs, failure records, and implementation snapshots. The separately linked trained-backbone artifacts retain their own data, selection history, and controls. Negative development attempts remain available; no earlier result is overwritten by a later selected configuration.

The Phase 2 export audit verified all 16 frozen implementation sources and independently replayed all 12 sparse registers from the saved extraction outputs. This checks deterministic reduction and evidence consistency, not the semantic truth of an extracted value. The two wrong state answers remain wrong.

For exact input and scoring reproduction, use the protocol and source copies inside each evidence archive. The current public comparison driver adds two post-run guards: baseline-only selections skip unused ingestion, and `--updater-effort` requires an explicit `--updater-model`. Those guards do not alter either completed configuration; the precise drivers used for the results remain preserved. New live comparisons require a fresh output path, explicit current rates, and their own approved budget while preserving the shared project ledger.

See [library usage and retrieval modes](../docs/local-and-retrieval.md) for the public API and [the earlier benchmark](../docs/benchmarks.md) for the separate October 27 results.
