# Workflow memory

`WorkflowMemory` keeps dated facts and decisions as a workflow changes. An updater extracts typed operations from new records; chronological code applies them without discarding earlier values. A reader can then answer questions about the current state, an earlier state, or an approval chain.

The existing `Threadmark` and `EventDeltaMemory` APIs remain available. Their current-state and evidence-reader methods have different behavior and evaluation results. Installing the library does not enable a background service or change an application's data automatically.

## Run an offline example

From an installed checkout:

```bash
python -m pip install -e .
python examples/use_workflow_memory.py
```

The example uses fictional records and scripted provider responses. It makes no network calls, downloads no model or tokenizer data, and needs no API key. It demonstrates the API, not measured model accuracy.

## Use the Python API

Provide a catalog of entities and the fields the application wants to track. A catalog supports at most **32 current facts in total**, counted as entity/field pairs. A small catalog should describe one bounded workflow rather than an entire organization.

```python
from threadmark import Settings, WorkflowMemory

catalog = [
    {
        "entity_id": "review",
        "label": "Quarterly review",
        "fields": ["owner", "venue"],
    }
]
settings = Settings(
    reasoning_effort="medium",
    chunk_tokens=4000,
    updater_max_output_tokens=4096,
    reader_max_output_tokens=1024,
)

# Configure a provider explicitly; see docs/usage.md for OpenAI budget handling.
# It implements complete(instructions=..., content=..., max_output_tokens=...)
# and returns an object with a text attribute containing JSON.
memory = WorkflowMemory(catalog, provider, settings)
memory.ingest(new_events)
answer = memory.answer("Which venue was approved, and who owns the review now?")
```

If a provider exposes `model` or `reasoning_effort`, those values must match `Settings`. `ingest` and `answer` can incur provider charges. The library does not automatically retry failed requests or start a paid fallback. Use a configured `OpenAIProvider` with explicit prices and a `BudgetLedger` as shown in [usage](usage.md), or supply another compatible provider.

`answer(question, reader_provider=...)` optionally uses a separate reader provider. Otherwise it uses the ingestion provider. The reader receives the catalog, the structured memory packet and the question. It returns exactly:

```json
{
  "answer": "Room B was approved; Bo now owns the review.",
  "evidence_event_ids": ["e2", "e3", "e4"]
}
```

A supported unknown can be represented by a null answer. Non-null answers must cite evidence, and cited IDs must occur in explicit evidence fields in the supplied memory packet. These checks validate structure and citation membership; they do not prove that the answer or the updater's interpretation is correct.

## Supply chronological records

Each event needs a unique `event_id`, a timezone-aware ISO timestamp, a sender and text:

```python
new_events = [
    {
        "event_id": "e1",
        "timestamp": "2026-11-01T09:00:00+00:00",
        "sender": "Coordinator",
        "text": "Ada owns the quarterly review.",
    },
    {
        "event_id": "e2",
        "timestamp": "2026-11-01T09:10:00+00:00",
        "sender": "Ada",
        "text": "Proposal ROOM-7: hold the review in Room B.",
    },
    {
        "event_id": "e3",
        "timestamp": "2026-11-01T09:20:00+00:00",
        "sender": "Coordinator",
        "text": "Accept ROOM-7 because Room B supports the required video call.",
    },
]
```

Call `ingest` with new events only, in chronological order. Duplicate IDs and out-of-order records are rejected. State and archive commit together only after all chunks validate. A failed later chunk can still leave earlier provider charges, but it does not partially commit the new records.

## What the memory retains

| Operation | Effect |
| --- | --- |
| `set` | Assign an explicitly stated current value and preserve the previous version. |
| `clear` | Record an explicit null value without resetting unrelated fields. |
| `propose` | Keep a pending value without treating it as approved. |
| `accept` | Apply a uniquely resolved proposal and cite its proposal and acceptance records. |
| `reject` | Close a uniquely resolved proposal without applying its value. |
| `note` | Preserve a source-linked reason, constraint, dependency or other explanatory claim. |

The model returns operations internally; applications normally supply original records to `ingest`, not operation objects. Operations remain predictions about those records.

Distinct operations within one event retain their stated order. For example,
one event can assign an owner twice, propose two alternatives for the same
field, or record two dependency notes. Earlier assignments remain in history
and the last operative assignment becomes current. An identical duplicate
operation in that event is rejected. Omit an event group with no operations;
an entirely nonoperative input returns `{"events":[]}`.

A proposal has a stable internal identifier derived from its event, entity and
field. The first keeps the existing URL-escaped form
`event_id::entity_id::field`. Additional proposals for that same event and
target append `::2`, `::3`, and so on in proposal order. The public
`threadmark.workflow_memory.proposal_identifier(..., occurrence=2)` helper
constructs that identifier.
When a source supplies an external label such as `ROOM-7`, the memory also
preserves it. Acceptance or rejection matches either identifier only among
open proposals for the same entity and field. If the reference is unknown or
ambiguous, it is recorded as unresolved and the current fact is unchanged.
A reference without an identifier can resolve only when exactly one proposal
is open for that target. Indexed IDs do not resolve an ambiguous external label.

`state` returns the current-fact view. `events` returns the original retained records. `entity_catalog` returns the normalized catalog. These accessors return copies. `diagnostics` reports memory sizes, counts and unresolved references.

`memory_packet()` returns current facts, historical versions, proposals, explanatory claims and unresolved references. It excludes the original record objects, although faithful extracted notes may contain words from a source. **It returns the complete bounded packet.** Its optional `question` argument does not perform semantic selection or narrow the packet. Applications can send the packet to their own reader while retaining a separate search path over the original records.

## Apply operations from another extractor

`apply_operations(events, operations)` accepts the same typed JSON contract as
the updater, as either a dictionary or JSON string. It makes no provider or
tokenizer calls. Use it for separately extracted operations or explicit
application events whose meaning the application already knows:

```python
external_memory = WorkflowMemory(catalog, provider, settings)
record = {
    "event_id": "app-1",
    "timestamp": "2026-11-02T10:00:00+00:00",
    "sender": "Scheduling application",
    "text": "Parker now owns the quarterly review.",
}
external_memory.apply_operations([record], {
    "events": [{
        "event_id": "app-1",
        "operations": [{
            "op": "set", "entity_id": "review", "field": "owner",
            "value": "Parker", "proposal_id": None, "reason": None,
        }],
    }],
})
```

The constructor still requires a provider object for possible later `ingest`
or `answer` calls; this method does not invoke it. It validates chronology,
identifiers, aliases, shapes, provenance, and memory limits, then commits the
archive, state, and journal together. Failure leaves them unchanged. It does
not overwrite existing event IDs or verify that the supplied operations are
true interpretations of the text. Passing an oracle in an offline test is a
reducer check, not evidence of extraction accuracy.

Run the complete [application-events example](../examples/use_application_events.py)
with `python examples/use_application_events.py`. It exercises an ordered
handoff, two proposals, acceptance/rejection, and snapshot replay without a
model or tokenizer call.

## Optional history for extraction

The updater normally sees current facts and open proposals. That may be
insufficient for a record such as “return to the original room.” Opt in to a
bounded recent history window when constructing a new memory:

```python
memory = WorkflowMemory(
    catalog, provider, settings,
    reference_history_entries=16,
)
```

`reference_history_entries` accepts integers from 0 to 256; **0 is the default**.
When enabled, the updater reference can include recent superseded fact
versions and explanatory claims. Entries are ordered by observed recency and
share the existing `max_reference_bytes` cap. Current facts and open proposals
are mandatory and are never removed to fit history. Optional history stops
at the count or byte bound; it is not a query-aware or exhaustive history
search. The reference reports `history_context_complete` when the expanded
envelope fits, so absence from an incomplete window is not evidence that an
older value never existed.

This changes only what is sent to the updater. It never prunes the archive or
the complete memory packet. It can increase input tokens and provider costs,
and it does not guarantee correct resolution of historical references. The
reader still receives the complete bounded packet, subject to its provider's
own context limit.

`memory.updater_instructions` is a read-only property returning the exact
effective updater prompt, including the history suffix when enabled. Use it
when registering a custom prompt with an exact-match role router. The public
`WORKFLOW_UPDATER_INSTRUCTIONS` and `WORKFLOW_HISTORY_UPDATER_INSTRUCTIONS`
constants cover the default profiles. A route registered only for the
no-history prompt will not match a history-enabled updater. The simplest
separate-reader integration remains
`memory.answer(question, reader_provider=reader)`. See
[role routing](local-and-retrieval.md#workflow-updater-routing) for a default
history-enabled example.

## Save and restore

```python
import json
from pathlib import Path

snapshot_path = Path("workflow-memory.json")
snapshot_path.write_text(json.dumps(memory.snapshot()), encoding="utf-8")

restored = WorkflowMemory.from_snapshot(
    json.loads(snapshot_path.read_text(encoding="utf-8")),
    provider=provider,
)
assert restored.memory_packet() == memory.memory_packet()
```

Snapshots include the catalog, settings, prompts, limits, original records and validated operation journal. New workflow snapshots use **schema 2**, including `reference_history_entries`. Restoration also accepts schema 1, treating its absent history setting as zero. This is the `WorkflowMemory` snapshot schema; the older `Threadmark` API has its own schema rules. Restoration validates and replays the journal rather than trusting a supplied derived state. It makes no provider calls. After restoration, new chronological events can be ingested normally.

A structurally valid snapshot can still contain false or malicious source data. Load snapshots only from sources trusted to define the workflow. Snapshots contain the original workflow text and extracted information; apply the same retention and access rules as for the original records. Custom prompts in a snapshot are configuration, not authenticated policy.

## Bounds and integration limits

The defaults allow a 16,000-byte updater reference context, a 256,000-byte structured packet and 10,000 combined historical/proposal/note/unresolved entries. Exceeding a mandatory reference, packet, or entry cap raises an error; the library does not silently discard older stored evidence. Optional history is a bounded reference window as described above. The updater never receives the complete original archive automatically.

Version timestamps follow the order in which events were observed. They do not implement separate retroactive effective dates or a general bitemporal database. Memory extraction and reader answers remain fallible.

Use one writer per instance. Persistence snapshots are available, but a hosted service, tenant isolation, concurrent-writer coordination, event delivery, retention/deletion workflows and application authorization remain the integrating application's responsibility. Keep authoritative application state in its database and approval system; memory provides extracted context.

A typical integration is:

```text
New workflow record → background memory worker → persisted workflow memory
                                                     │
Question ────────────────────────────────────────────┘
                                                     ↓
                                       reader → answer + source IDs
```

An application may choose an explicit original-source retrieval path when ingestion fails, relevant information is missing, a proposal remains unresolved or a memory limit is reached. No such fallback is automatically invoked by `WorkflowMemory`.

## Evaluation

The [November 1 evaluation](../research/evaluation-2026-11-01.md) reports the frozen pre-packaging algorithm. Its fresh 18-question follow-up scored 18/18 correct and source-supported answers across three synthetic histories in the 32K–96K bands. Full history and hybrid RAG also answered all 18 correctly, with one citation-support omission each. The first 54-question study and the follow-up remain separate.

The improvement reduced reader input tokens compared with full history. Including memory building, it remained more expensive than hybrid RAG and took longer overall for six questions per history. These cases contain few active facts despite their long histories. The results do not establish a universal RAG replacement or production accuracy. The packaged persistence and answer interface are covered by offline tests, not a fresh paid benchmark.

The 0.2.0 engineering checks distinguish two questions. The
[offline integrity runner](../scripts/evaluate_workflow_integrity.py) supplies
recorded extractor operations to test ordering, rollback, references, and
snapshot replay. The prior reducer passed **9/12 cases** and the improved
reducer **12/12**. The 1,000-event case contains distinct operative updates,
uses an explicit **2,000,000-byte packet cap**, and passed in both arms.
It does not establish that the default 256,000-byte cap fits that history.
These checks do not run a language model.

A separate local-adapter trial on four fictional eight-event workflows
completed **0/4 workflows in both software arms**. That first 32-event set is
now development evidence. A second comparison froze a fresh 32-event fixture
and held the adapter, improved reducer, prompt, 16-entry history setting,
greedy decoding, and 768-token output budget constant; only structured
decoding changed:

| Fresh local engineering measure | Ordinary decoding | Constrained decoding |
| --- | ---: | ---: |
| Committed events / 32 planned | 2 | 17 |
| Failed events | 4 | 2 |
| Unprocessed events after a failure | 26 | 13 |
| Completed workflows / 4 | 0 | 2 |
| Exact final current states / 4 | 0 | 0 |
| Strict exact operation journals / 32 | 0 | 3 |

Each arm used its own predicted memory. A failure stopped that workflow/arm;
all failed and unprocessed events remain in the denominator. A committed event
means the reducer accepted the output, not that it extracted every fact or
interpreted the record correctly. Complete workflows can end with wrong facts.
Strict operation matching is a diagnostic and can reject faithful note
paraphrases or additional relevant explanations; it is not semantic accuracy.
For example, one saved note merged two stated requirements and retained both,
but did not match the oracle's two-note journal exactly. The constrained
provider produced schema-valid output in 18 of its 19 attempts; the reducer
rejected one duplicate operation, leaving 17 commits. The other attempt did
not reach native EOS. These are separate validation stages.

For example, a completed constrained workflow placed an access requirement in
its room field, and another completed with all three current values null.
Schema constraints did not prevent these semantic failures. The small
fictional sets share operation families, contain three fact slots each, and
are not population or customer-workflow evidence. No weights were trained,
no RAG baseline was measured, and no hosted API/AWS calls were made in these
local trials. Local GPU, electricity, and engineering costs remain unpriced.
The software changes improve integrity and format handling; the tested
extractor is still unsuitable for authoritative workflow updates.

### Unadapted-base development diagnostic

After the first paired run, those exact 32 records were reused with the same
pinned SmolLM2 base and no adapter. Both new arms used the improved memory,
16-entry history, and the same 768-token output and 8,192-token context limits.
This is **reused development evidence**, not another fresh test. The original
paired measurements above remain unchanged and are not pooled with this run.

| Reused-case diagnostic | Ordinary decoding | Constrained decoding |
| --- | ---: | ---: |
| Committed events / 32 | 0 | 18 |
| Failed events | 4 | 2 |
| Unprocessed events after a failure | 28 | 12 |
| Completed workflows / 4 | 0 | 2 |
| Exact final current states / 4 | 0 | 0 |
| Strict exact operation journals / 32 | 0 | 1 |

Removing the adapter did not restore correct final states in these cases.
Cross-run differences are descriptive: the fixture was reused, execution
windows were separate, and source revisions were recorded independently.
They do not establish that adapters generally help or harm extraction. The
same fail-stop rules, strict-journal caveat, local unpriced compute, and absence
of a RAG or reader-quality comparison apply.
