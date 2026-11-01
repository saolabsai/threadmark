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

A proposal has a stable internal identifier derived from its event, entity and field. When a source supplies an external label such as `ROOM-7`, the memory also preserves it. Acceptance or rejection matches either identifier only among open proposals for the same entity and field. If the reference is unknown or ambiguous, it is recorded as unresolved and the current fact is unchanged. A reference without an identifier can resolve only when exactly one proposal is open for that target.

`state` returns the current-fact view. `events` returns the original retained records. `entity_catalog` returns the normalized catalog. These accessors return copies. `diagnostics` reports memory sizes, counts and unresolved references.

`memory_packet()` returns current facts, historical versions, proposals, explanatory claims and unresolved references. It excludes the original record objects, although faithful extracted notes may contain words from a source. **It returns the complete bounded packet.** Its optional `question` argument does not perform semantic selection or narrow the packet. Applications can send the packet to their own reader while retaining a separate search path over the original records.

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

Snapshots include the catalog, settings, prompts, limits, original records and validated operation journal. Restoration validates and replays the journal rather than trusting a supplied derived state. It makes no provider calls. After restoration, new chronological events can be ingested normally.

A structurally valid snapshot can still contain false or malicious source data. Load snapshots only from sources trusted to define the workflow. Snapshots contain the original workflow text and extracted information; apply the same retention and access rules as for the original records. Custom prompts in a snapshot are configuration, not authenticated policy.

## Bounds and integration limits

The defaults allow a 16,000-byte updater reference context, a 256,000-byte structured packet and 10,000 combined historical/proposal/note/unresolved entries. Exceeding a cap raises an error; the library does not silently discard older evidence. The updater reference contains current facts and open proposals, not the complete original archive.

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
