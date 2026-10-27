# Architecture

Threadmark reduces the history a reader model must inspect for questions about changing workflow facts. It retains the original records and maintains a compact register that points back to them.

The optional `state` mode returns the register's unverified predicted values directly, without reading original source content or calling another reader model. It requires explicit requested entity/field keys; it does not interpret arbitrary questions. Updater mistakes propagate to its answers. BM25 and local dense retrieval are separate comparison baselines. See [current-state lookup and retrieval](local-and-retrieval.md) for the interfaces, prompt profiles, and lifecycle accounting.

The [research note](../research/threadmark.md) and [PDF](../research/threadmark.pdf) develop the mathematical design and distinguish the original latent-memory proposal from this implemented system. The [annotated bibliography](../research/references.md) identifies related papers and their relevance. This release uses explicit state and source retrieval, and does not reproduce a latent-memory architecture from a paper.

## A plain-language example

A project starts with Maya as owner and a Friday deadline. Later, an approved update transfers ownership to Sam. A draft proposes Tuesday, but no one approves it. The register should say **Sam / Friday**, with source IDs for the two approved decisions.

When asked for the owner and deadline, Threadmark retrieves those original decisions and the most recent records. The reader decides the answer from that evidence. The register's values help organize retrieval; they are not supplied as answer text to the reader.

## Components

| Component | Responsibility |
| --- | --- |
| Original archive | Keep events, IDs, order, and text for evidence retrieval. |
| Public catalog | Define the entities and current fields the application tracks. |
| Updater | Read previous predicted memory and the next chronological batch; produce current values and source IDs. |
| Validator | Check schema, field names, value size, citation IDs, and memory size. |
| Source selector | Retrieve original events cited for requested fields and add a recent-event window. |
| Reader | Answer from the supplied original events and cite their IDs. |
| Budget ledger | Reserve request cost before transport and reconcile known usage afterward. |
| Evaluator | Score answer values and source evidence against separate expected labels. |

The benchmark uses one updater call per history chunk, then asks four questions about final state. It does not interleave questions with incoming events.

## Update and answer equations

Let `A_t` be the original archive through event `t`, `s_t` the predicted register, and `ΔA_t` the next batch of events. With a fixed model and prompt `U`, the update is:

```text
s_t = validate(U(s_(t-1), ΔA_t, catalog))
```

For question `q` and its requested fields `F_q`, the evidence packet is:

```text
E_t(q) = deduplicate_in_original_order(
    cited_originals(A_t, s_t, F_q) ∪ recent_originals(A_t)
)
answer(q) = R(q, E_t(q), catalog)
```

`R` receives original records and their IDs. It does not receive stored predicted fact values. Both models keep fixed weights during a run; changes happen in the external register and archive.

## Frozen experiment settings

The original experiment used approximately 2,000 `o200k_base` tokens per updater chunk, up to 1,024 updater output tokens, and up to 256 reader output tokens. Selection added the newest whole-event suffix that fit within 1,024 tokens. The union of selected original sources was capped at 20,000 UTF-8 bytes. Predicted memory was capped at 12,000 bytes, with at most two evidence IDs per field.

These are implementation limits, not established optimal settings. The [frozen evidence archive](../benchmarks/workflow-2026-10-27-evidence.zip) contains the exact prompts, protocol, and original runner, with private runtime details redacted. The refactored library exposes configuration separately; changing a limit creates a different experiment.

## Rules for changing facts

The updater is instructed to apply explicit approved decisions in chronological order. A proposal or draft should not replace an approved fact. A change affects only the fields it explicitly sets. Clearing a field produces `null` with a source ID; a field never established is `null` without evidence.

The validator checks structure and whether citations refer to allowed original event IDs. It cannot prove that a model understood a record correctly. A valid-looking register may contain a wrong value or cite the wrong entity. This is why the benchmark scores register fields separately from reader answers.

The source selector uses the public requested fields and predicted citations. It does not inspect expected answers, gold memory, or evaluation annotations. The baseline reader receives the complete original archive and the same question and catalog.

## Cost and latency

For `Q` questions about one ingested workflow:

```text
full-history lifecycle cost = Σ full_history_reader_cost(q)
Threadmark lifecycle cost   = Σ updater_cost(batch) + Σ evidence_reader_cost(q)
```

Memory construction is paid before the first Threadmark answer. Reusing it across questions can amortize that cost. The break-even point depends on history length, update frequency, number of questions, model prices, caching, and retrieval quality. Reader-only token savings do not describe total system cost.

The benchmark reports ingestion separately from reader request time. Its concurrent API measurements include interruptions and continuation; they are not a promise of production response time.

## Failure behavior and limits

The release validates structured responses and stops on failed updates instead of inventing a new register. Budget reservations happen before each paid request. There are no automatic paid retries or full-history fallbacks. Unknown usage remains reserved rather than being treated as free.

This design requires retained original records. It saves reader context, not archive storage. It targets questions about explicitly tracked current fields; it does not guarantee arbitrary historical reasoning, complete recall, robust handling of malicious instructions inside records, or production isolation between users. Applications must add authentication, data access controls, retention policies, and operational monitoring appropriate to their own deployment.
