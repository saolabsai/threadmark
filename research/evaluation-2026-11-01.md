# Long-context workflow memory: November 1, 2026

Threadmark's working-memory improvement preserves historical values, proposed and
accepted changes, explicit reasons, and source-event citations. The implementation
keeps model weights fixed. It is an explicit memory layer, not a trained decoder
extension.

The studies below were completed on November 1. Their source code was subsequently
packaged as the public `WorkflowMemory` API. Packaging adds a validated snapshot
journal and a reader wrapper, tested offline; the maintained API has **not** been
rerun to reproduce these paid results. The original experiments remain frozen.

- [Usage and architecture](../docs/workflow-memory.md)
- [Machine-readable results](../benchmarks/long-context-2026-11-01.json)
- [Sanitized answer-level evidence](../benchmarks/long-context-2026-11-01-evidence.zip)

## What changed

The earlier sparse-event store retains current values. Replacing a value loses
the historical information needed to answer an ordinary question about its past.
It also lacks an explicit persistent proposal-to-acceptance relationship.

The improvement records six typed operations: `set`, `clear`, `propose`, `accept`,
`reject`, and `note`. A deterministic reducer maintains current facts, their
observed-event versions, proposal status, explanatory notes and unresolved
references. Accepting a proposal copies its stored value into the current view
and records both the proposal and acceptance sources. Rejection leaves the
approved current value unchanged.

An external proposal label is stored separately from a canonical internal ID.
Acceptance or rejection must uniquely identify an open proposal for the same
entity and field. Unknown or ambiguous references remain unresolved. The code
does not infer an approval merely because a proposal is recent.

Writing the extracted operations as \(o_t\), the source event as \(e_t\), and the
memory as \(M_t\), the implemented path is:

\[
o_t = E_\theta(e_t, \operatorname{reference}(M_{t-1})),\qquad
M_t = \operatorname{reduce}(M_{t-1}, o_t).
\]

\[
a_q = R_\phi(q, \operatorname{packet}(M_T)).
\]

The extractor and reader are fallible language models. Deterministic reduction
and a valid source ID do not prove a model interpreted the source correctly.
The packet includes all bounded structured memory; the question does not select
a subset. Its timestamps represent observed-event order, not general bitemporal
or retroactively effective truth.

## Shared protocol

All arms use the same `gpt-5.6-luna` reader, low reasoning effort, shared reader
instructions, and a 1,024-token output allowance. Both memory updaters use that
model with medium effort, approximately 4,000-token source chunks, and a
4,096-token output allowance. The improved updater additionally receives bounded
current facts and open proposals; that work is charged.

The four methods are full-history prompting, hybrid RAG, the prior sparse-event
current-state store, and versioned workflow memory. The prior-store baseline is
that particular store plus the common reader, not every Threadmark API mode.
The hybrid pipeline combines BM25 and local dense retrieval, then locally
reranks candidates with a cross-encoder and supplies up to eight records.

Full history is a single flattened conversation prompt. Memory updaters ingest
the same source records in chronological chunks, and their readers receive
structured packets without raw-source retrieval fallback. Questions and expected
answers are not supplied to the updater. Queries run after complete ingestion.

Each case has six natural-language questions covering current facts, historical
facts, indirect approval, comparison/arithmetic, reason-bearing summary, and
unanswerable information. All planned answers remain in denominators. All
inference was frozen before anonymous candidate grading. Primary grades come
from a shared-family AI judge; a separate arm-blind AI-agent review was frozen
before method identities were revealed. Neither is human validation.

## Initial long-context study

Nine fresh synthetic histories cover three domains—project management, support,
and meeting coordination—at approximately 8K, 32K and 64K tokens. Each case has
six questions, giving 54 questions per method.

| Method | Correct | Correct and supported | Lifecycle API estimate |
| --- | ---: | ---: | ---: |
| Full history | 53/54 | 53/54 | $0.49140545 |
| Hybrid RAG | 54/54 | 54/54 | $0.02465860 |
| Prior current-state memory | 27/54 | 27/54 | $0.13340075 |
| Versioned workflow memory | 52/54 | 52/54 | $0.21855570 |

The new memory scored 18/18 supported answers in each of the 32K and 64K bands.
The independent stored-state diagnostic found 107/108 exact current values,
compared with 99/108 for the prior store. These auxiliary values are not extra
independent QA samples.

The secondary blinded review judged one historical answer correct and supported,
where the primary judge had incorrectly rejected its boundary citations. That
gives **53/54** in secondary review; the primary **52/54** remains unchanged.

The real remaining error involved an external proposal label. The proposed value
was stored under its canonical ID, but the accepting event referenced its external
label, leaving the proposal unresolved and the old approved value active. That
case became development data for the alias improvement below.

Before this final dataset was generated, two disclosed development trials on
three prior cases scored 17/18 correct and 15/18 supported, followed by 18/18 for
both after a reader clarification. Their combined API estimate was $0.05557230.
No result from those development cases is counted as fresh final accuracy.

## Separate proposal-alias follow-up

After the alias fix passed 43 offline checks and the six questions on the known
failing development case, the method was locked again. Three fresh histories
were then generated: project/32K, support/64K, and meeting/96K, with six questions
each. Reader and judge instructions were unchanged. This targeted 18-question
follow-up is **not pooled with the initial 54-question study**.

| Method | Correct | Correct and supported | Lifecycle API estimate |
| --- | ---: | ---: | ---: |
| Full history | 18/18 | 17/18 | $0.30167155 |
| Hybrid RAG | 18/18 | 17/18 | $0.00867095 |
| Prior current-state memory | 9/18 | 9/18 | $0.07599370 |
| Improved workflow memory | 18/18 | 18/18 | $0.11125990 |

The longest full-history request used 99,495 actual API input tokens. New memory
matched all 36 current-state values; all three external proposal labels resolved
to the correct approved values. Three citation sets used valid older records for
unchanged active values rather than the oracle's latest restatement; these are
not three incorrect facts.

The blind review agreed on correctness and supported-correct status for all 72
answers. Full history and hybrid RAG failed the same citation-support question:
both named the correct venue but cited only its acceptance, omitting the proposal
that named the venue. RAG had retrieved both records. This was a reader citation
omission, not a wrong venue or a retrieval miss.

## Cost and speed

The follow-up's improved memory used $0.08470455 for ingestion and $0.02655535
for answers, totaling $0.11125990. That is approximately 63.1% below full history
and **12.8 times hybrid RAG's API cost**. Its answer-stage input was 100,405 tokens
versus 1,198,897 for full history, a 91.6% reduction. Hybrid RAG used 27,557.

| Follow-up method | Mean query seconds | Memory-ingestion seconds | Ingestion plus 18 queries |
| --- | ---: | ---: | ---: |
| Full history | 3.114 | 0 | 56.057 |
| Hybrid RAG | 3.639 | 0 LLM-memory ingestion | 65.506 |
| Improved workflow memory | 2.624 | 193.116 | 240.357 |

Query timings include the measured retrieval/reader work, while model setup and
loading are excluded. The experiment does not establish an end-to-end speed win.
RAG's local embedding, reranking and infrastructure costs are not priced as API
charges. Memory-building cost is amortized over only six questions per history;
more-query savings have not been established by this test.

Final-run API estimates including evaluation judging were $0.89207855 for the
initial study and $0.50546800 for the follow-up. Development was $0.05557230 plus
$0.01380905, respectively. The complete attached series totals approximately
$1.46692790. Unknown-usage calls in these studies: zero. These estimates are not
provider invoices and are not a claim about lifetime project spend.

## Evidence and limitations

The public archive contains the complete fictional source datasets, questions
and references, answer-level primary grades, input citation membership, structured
memory traces, per-method usage, protocol excerpts, secondary review, stored-state
diagnostics and source/export hashes. The exporter recomputed all planned counts
and citation membership; it did not rejudge semantic correctness. Internal method
identifiers are normalized to descriptive names. Source records, answer text,
scores and denominators are preserved. Private runtime journals, provider IDs,
machine paths and project accounting are excluded. The archive does not contain
the complete network execution harness.

Each history has only two catalog entities, 12 current fields and 14 operative
core records; most added length is background conversation. These tests measure
context length and distraction, not thousands of active facts. The follow-up
confounds domain and length, cases use finite wording families, and questions
within each case are correlated. There is no public LongMemEval/CRAG score,
customer-data validation, interleaved streaming-query evaluation, or guarantee of
equal accuracy. No direct Graphiti, Mem0, or Letta comparison was run.

The evidence supports an improvement over this prior current-state store. It
does not establish a general RAG replacement or competitive production cost.
