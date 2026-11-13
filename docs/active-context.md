# Portable active context

`ActiveContext` compiles a selected part of `WorkflowMemory` into a bounded,
source-linked reader request. The archive and operation journal stay in the
memory store. Compilation makes no provider call and changes no model weights.
It is experimental context selection, not a trained memory layer or an accuracy
guarantee.

The separation is:

```text
M_t = U(M_(t-1), new_events)
C_t = select(M_t, q, B)
answer = f_theta(q, C_t), with theta fixed
```

The updater `U` applies typed operations to current facts, history, proposals,
claims and unresolved references. An application can supply those operations
through `apply_operations`; text-only records need an extractor through
`ingest`, whose interpretation can be wrong and whose calls may cost money.
Process each new event once. Compile another request for each question or
reader budget without re-ingesting the conversation.

## Compile and reuse

```python
from threadmark import ActiveContext, ContextBudgetExceeded
from threadmark.context import token_counter

# memory is an existing WorkflowMemory with newly ingested application events.
# Loads verified cached o200k data; it never downloads a missing tokenizer.
count_tokens = token_counter()
compiled = ActiveContext(memory).compile(
    "Who owns the review, and which room was approved?",
    max_input_tokens=4096,
    count_tokens=count_tokens,
    requested_facts=[
        {"entity_id": "review", "field": "owner"},
        {"entity_id": "review", "field": "room"},
    ],
    required_event_ids=[],
    history_entries=8,
    claim_entries=8,
    recent_events=2,
    search_events=4,
)
print(compiled.input_tokens, compiled.coverage)

# Optional and potentially billed. Configure the provider and budget yourself.
# completion = reader.complete(
#     instructions=compiled.instructions,
#     content=compiled.content,
#     max_output_tokens=512,
# )
```

The result preserves the exact measured `rendered_request`, instructions,
`input_tokens` and `max_input_tokens`; content and coverage are defensive copies.
It is plain structured content, so it can be passed to compatible cloud or local
reader adapters. Switching readers does not require rebuilding memory. Compile
again when the reader's tokenizer, request frame, question or budget changes.
This is a portable data interface; it does not guarantee equal answers across
models.

The default renderer counts canonical JSON containing `instructions` and
`content`. That is a reproducible estimate, not provider billing. Supply both
the intended model's tokenizer and a `render_request(instructions, content)`
callback for its actual chat template, tools and other input framing. The
compiler measures the complete returned string, including question and
instructions. Reserve output tokens and any provider-specific context overhead
outside `max_input_tokens`; do not append uncounted input after compilation.
The renderer must be deterministic and pure. It must return a string without
modifying its input; mutation raises `ValueError`.

```python
def render_request(instructions, content):
    # Use the same real template as the eventual reader adapter.
    return local_tokenizer.apply_chat_template(
        [
            {"role": "system", "content": instructions},
            {"role": "user", "content": json.dumps(content)},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )

compiled = ActiveContext(memory).compile(
    question,
    max_input_tokens=4096,
    count_tokens=lambda text: len(local_tokenizer.encode(text, add_special_tokens=False)),
    render_request=render_request,
)
```

`json`, `question` and `local_tokenizer` are application-supplied in this second
snippet; no model or tokenizer download is initiated by the compiler.

## Selection, evidence and overflow

Without `requested_facts`, all current facts are mandatory. Explicit scope
requires the caller to name the relevant entity/field keys correctly; the
compiler does not infer a trustworthy semantic scope from the question. The
mandatory bundle includes those current facts, their complete original source
records, open and active approved proposals, unresolved references and
`required_event_ids`. A cleared value remains an explicit null with its clear
event; it is not silently treated as an absent record.

`requested_facts` is a selection scope, not an access-control or tenant boundary.
Whole original source records may contain other fields, and unlinked notes are
selected at entity scope. Enforce authorization before records enter the memory
or before serving a compiled request to a reader.

Optional history and claims are ranked by lexical question relevance and
recency. The compiler then considers recent originals and lexical source search.
Each included record brings its source and applicable proposal or version
dependencies as a whole bundle. It skips an optional bundle that cannot fit;
it never cuts an event's text to make its citations appear complete. The
optional entry settings are candidate limits, not guarantees that every
candidate will fit.

If the mandatory bundle does not fit, `ContextBudgetExceeded` exposes
`required_tokens` and `budget`. The compiler sends nothing, changes no memory,
and starts no automatic model or retrieval fallback. The application can choose
more precise scope, a larger supported input budget, explicit archive retrieval
or an honest inability to answer. Do not remove essential evidence merely to
silence the error.

`coverage` counts selected versus stored current facts, history, proposals,
claims, unresolved references and original sources. It includes the stream
position and `stored_records_complete`. Its
`semantic_completeness_verified` is always false: counts cannot prove that all
evidence needed for a free-form question was selected. Lexical ranking can miss
paraphrases and relationships. Citation presence proves neither correct
extraction nor a supported answer.

## Persistence and one-time ingestion

Use `WorkflowMemory.snapshot()` to persist the catalog, original records,
settings and operation journal. Restore with `WorkflowMemory.from_snapshot`;
restoration validates and replays the journal without a provider call. Then
apply only new chronological events and compile as needed. The compiler is a
view over that memory, not a separate database or a model-specific KV cache.

The [offline example](../examples/use_active_context.py) applies two batches of
known operations, compiles a request, writes a snapshot and confirms that a
restored memory produces the same request. It never invokes its optional
reader adapter. Run from an installed checkout with cached tokenizer data:

```bash
python examples/use_active_context.py
```

Retain source text under the application's normal access and retention rules.
The underlying workflow currently supports at most 32 current fact keys and
has explicit archive, entry and packet caps. Compiled input is bounded, but
stored history and archive size still grow. Neither persistence nor bounded
selection promises lossless answers to arbitrary future history questions.

## Offline evaluation and limits

```bash
python scripts/evaluate_active_context.py --prepare-only
python scripts/evaluate_active_context.py
```

The script writes a frozen scenario file, source snapshots and source hashes
before measuring compilation. It reads verified cached `o200k_base` data and
fails if the cache is unavailable. It does not download assets, call an
extractor/reader or touch cloud resources. Results and a standalone HTML report
are saved under `runs/active-context-20261111/final-with-unpinned/`; existing results are preserved,
and a changed run requires a new `--output` directory.

Cases vary conversation length with fixed state and growing historical state,
increase active fields within the 32-key bound, and exercise changed approvals,
explicit clears, old supporting evidence and oversized mandatory context.
The primary cases explicitly request three review fields and pin known old
source IDs. Their growing historical updates affect only an unrelated field;
this is a favorable scope-selection case, not automatic semantic retrieval.
Two additional cases remove source pins and use the default optional settings.
One has 128 mostly background events; the other has 512 events including 502
operative owner changes inside the requested review scope, while still asking
about the earlier approved room. Both retain explicit field scope.
Operations and expected records are supplied by the fixture. Checks measure
selected-record integrity and token accounting, **not extractor or answer
accuracy**. All scenario outcomes, including failures, remain in the saved
results. Compilation CPU and wall times, ingestion times, raw repetitions,
input-token counts and coverage are reported. There are no billed model calls;
local electricity/hardware cost is unpriced.

Compression is not guaranteed: a short history or a large mandatory source can
produce a compiled request larger than full history. Input-token savings do
not equal end-to-end cost savings. Include extraction,
memory updates, compilation, reader output, provider pricing/caching and any
fallback calls when measuring a real application. A fixed small active state
with long background is especially favorable; increasing retained state or
mandatory evidence can consume the entire budget. Real-reader accuracy and
latency need a separate frozen comparison with full history and appropriate
retrieval baselines.

The November 10 training/repair experiment remains failed on its reused DEV
gate: the repair regressed, the original selected parent was retained only as a
research checkpoint, neither candidate was promoted, and its fresh held test
remained sealed. These offline compiler measurements do not replace that
negative result or establish a new model's accuracy.

The first frozen November 11 run remains in `runs/active-context-20261111/`.
Its intended oversized-source fixture still fit the mandatory budget, so the
compiler omitted optional history and the expected-overflow/history checks
failed. The separate `final/` run freezes a larger essential-source fixture.
The later `final-with-unpinned/` run freezes that correction plus two new
unpinned cases; both earlier results remain intact. CPU samples on Windows can quantize to zero
or 15.625 ms, so the report includes elapsed wall time and does not interpret a
zero CPU sample as instantaneous compilation.
