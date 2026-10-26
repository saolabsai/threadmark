# Benchmark: evolving workflows

The October 27, 2026 study compared a full-history reader with a memory-guided evidence reader on 90 synthetic workflows. The result belongs to the frozen original implementation. This repository refactors that design for reuse; it does not claim a new live reproduction.

## Dataset and protocol

- Three domains: meeting coordination, customer support, and project management.
- Three nominal history bands: 2,048, 8,192, and 16,384 tokens; 10 workflows per domain and band.
- Actual history sizes: 1,846–1,960, 7,407–7,612, and 14,777–15,215 estimated `o200k_base` tokens.
- Two entities and 14 or 16 typed current facts per workflow, with changes, proposals, cancellations, reinstatements, and cross-entity distractors.
- Four final-state questions per workflow: 360 paired questions, 720 planned reader answers.
- The updater and both reader arms used `gpt-5.6-luna`. Questions, catalog, reader instructions, and reader output caps were shared between arms.
- The active reader received originals selected by predicted citations plus a recent window. Gold labels were excluded from inference inputs.

Development pilots informed the architecture before the test instances were generated. Test instances use different seeds but share scenario structures and a finite set of contextual paragraphs with development. This is not an evaluation on unseen task families or real customer data.

## Answer quality

| Result | Full history | Memory-guided evidence |
| --- | ---: | ---: |
| Exact values correct, all planned questions | 349/360 (96.94%) | 346/360 (96.11%) |
| Exact values and expected evidence correct | 347/360 (96.39%) | 343/360 (95.28%) |
| Available answers | 360/360 | 354/360 |

There were **11 regressions**, where full history was correct and Threadmark was wrong or unavailable, and **8 improvements** in the opposite direction. Both were wrong on three questions. The remaining 338 were correct in both arms.

| Domain | Full history | Memory-guided evidence |
| --- | ---: | ---: |
| Meeting coordination | 116/120 (96.67%) | 113/120 (94.17%) |
| Customer support | 116/120 (96.67%) | 118/120 (98.33%) |
| Project management | 117/120 (97.50%) | 115/120 (95.83%) |

The register itself had **1,336/1,380 correct fields (96.81%)**, including failed pipelines in the denominator. A correct reader answer does not imply a perfect register: retrieved recent originals may rescue an earlier memory error.

Three API failures had unknown usage: one updater request and two active-reader requests. They affected six unavailable answers, retained in the denominator. The other eight active-arm failures were semantic answer errors. No failed request was retried. An operational continuation dispatched previously unattempted requests and retained the original failures.

## Cost, tokens, and time

The original experiment made **1,194 API attempts**, including updater calls. Known test spend was **$1.7195583**, excluding charges for the three unknown-usage failures. Development pilots had separate known spend of **$0.18242976**.

| Known spend across all workflows | Full history | Memory-guided evidence |
| --- | ---: | ---: |
| First question per workflow | $0.226120 | $0.661341, including all memory construction |
| Four questions per workflow | $0.902668 | $0.8168903, including all memory construction |

The active arm used **79.99% fewer known reader input tokens**. Including updater input, the reduction was **38.91%**. These counts have different completion coverage, and known costs exclude the failed requests' unknown charges. The small observed cost difference is not a guaranteed saving.

Median recorded reader request time was **2.33 seconds** for full history and **2.20 seconds** for the evidence reader. Median updater ingestion time was **22.89 seconds**. The first-question service-time comparison, including ingestion, was **2.53 seconds** versus **24.48 seconds**, with timing coverage of 90 and 89 workflows respectively. These measurements come from concurrent API work; interruptions changed cache and concurrency conditions.

## Interpretation

The study shows that a citation register can reduce the context supplied to a reader on these tasks. It **does not demonstrate equal accuracy**, general reasoning improvements, or lower cost for a single question. It retains the full archive and shifts work into ingestion and evidence selection.

No retrieval-only, keyword-search, embedding-search, or heuristic selector baseline was evaluated. The experiment therefore does not isolate the benefit of the language-model updater over simpler retrieval. The earlier trained 0.6B weekday classifier was a different, narrower experiment and is not the component evaluated here.

The next useful evaluation would compare source-selection strategies on fresh instances, include more independent workflow families, and measure quality under explicit latency and cost targets. A release reproduction should freeze its model, prices, prompts, settings, data, and dependency versions before running.

## Provenance

The checked-in [public summary](../benchmarks/workflow-2026-10-27.json) is copied from the published site's redacted export. The [full report](https://threadmark-research-20261027.viikys.chatgpt.site/workflow-report) includes per-workflow results; the [public evidence archive](https://threadmark-research-20261027.viikys.chatgpt.site/workflow-evidence.zip) includes the original dataset, protocol, exact requests, reports, and source snapshots.

| Original artifact | SHA-256 |
| --- | --- |
| Dataset | `b88fc69d9f2063d516599c24226fb1add7c5dba3a4513f71b42e36ff8b7cdd9d` |
| Run protocol | `fee8144ce1437fa147a1353aca198fcab905a12bf07125275abab877efe754ca` |
| Request plan | `db4ead6303ff68a93e170e3dd9c03b73b14763fe8895aabecef7a62c2b02af29` |
| Original runner | `32b1ef716a9ba6db6d6ce41ce089e90d8b7d4a617a022ce81a7cfde8d48c1561` |
| Original generator | `4eb3f8585660a6e1eb55fde82f3d3fd4328376e6263d6b7399e85181e0d77379` |

Public exports remove private runtime details. The evidence manifest documents redactions and hashes for the exported artifacts. Do not compare a redacted artifact's bytes with a canonical pre-redaction hash.
