"""Compile portable, budgeted context without an extractor or reader call.

Run from an installed checkout: python examples/use_active_context.py
The application supplies these operations. Cached tokenizer data is required;
this example never downloads it or calls a model.
"""

import argparse
import json
from pathlib import Path

from threadmark import ActiveContext, WorkflowMemory
from threadmark.context import token_counter


class NoModelCalls:
    def complete(self, **kwargs):
        raise RuntimeError("This offline example does not permit model calls")


def operation(kind, field, value=None, *, proposal_id=None):
    return dict(
        op=kind, entity_id="review", field=field, value=value,
        proposal_id=proposal_id, reason=None,
    )


def read_with(reader_provider, compiled):
    """Optional integration point; main() never invokes this function.

    The caller must configure its provider, budget, tokenizer and request frame.
    This call can incur charges. Validate the reader's answer separately.
    """
    return reader_provider.complete(
        instructions=compiled.instructions,
        content=compiled.content,
        max_output_tokens=512,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path,
                        default=Path("runs/active-context-example/memory.json"))
    args = parser.parse_args()
    count_tokens = token_counter()  # Verified local cache only; no download.
    memory = WorkflowMemory(
        [{"entity_id": "review", "fields": ["owner", "room", "deadline"]}],
        NoModelCalls(),
    )
    events = [
        dict(event_id="e1", timestamp="2026-11-11T09:00:00Z", sender="Application",
             text="Ada owns the review. Its deadline is Friday. Propose Room B as ROOM-B."),
        dict(event_id="e2", timestamp="2026-11-11T09:01:00Z", sender="Application",
             text="Approve ROOM-B. Bo now owns the review. Clear the deadline."),
    ]
    groups = [
        dict(event_id="e1", operations=[
            operation("set", "owner", "Ada"),
            operation("set", "deadline", "Friday"),
            operation("propose", "room", "Room B", proposal_id="ROOM-B"),
        ]),
        dict(event_id="e2", operations=[
            operation("accept", "room", proposal_id="ROOM-B"),
            operation("set", "owner", "Bo"),
            operation("clear", "deadline"),
        ]),
    ]
    # Process each new application event once. Text-only inputs instead need a
    # separately validated extractor through memory.ingest(new_events).
    for event, group in zip(events, groups):
        memory.apply_operations([event], {"events": [group]})

    question = "Who owns the review, which room was approved, and is a deadline set?"
    request_options = dict(
        max_input_tokens=4096, count_tokens=count_tokens,
        requested_facts=[dict(entity_id="review", field=field)
                         for field in ("owner", "room", "deadline")],
        history_entries=4, claim_entries=0, recent_events=0, search_events=0,
    )
    compiled = ActiveContext(memory).compile(question, **request_options)

    # Persist the archive and operation journal, not an opaque model cache.
    args.snapshot.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(memory.snapshot(), ensure_ascii=False, indent=2) + "\n"
    if args.snapshot.exists() and args.snapshot.read_text(encoding="utf-8") != serialized:
        raise ValueError("Refusing to overwrite a different snapshot; choose another --snapshot")
    args.snapshot.write_text(serialized, encoding="utf-8")
    restored = WorkflowMemory.from_snapshot(
        json.loads(args.snapshot.read_text(encoding="utf-8")), provider=NoModelCalls(),
    )
    restored_request = ActiveContext(restored).compile(question, **request_options)
    assert restored_request.rendered_request == compiled.rendered_request
    print(json.dumps(dict(
        snapshot=str(args.snapshot.resolve()),
        input_tokens=compiled.input_tokens,
        max_input_tokens=compiled.max_input_tokens,
        tokenizer="cached o200k_base; default JSON frame estimate, not provider billing",
        current_state=compiled.content["context"]["current_state"],
        coverage=compiled.coverage,
        replay_preserves_request=True,
        model_calls=0,
        reader_accuracy="unmeasured",
    ), ensure_ascii=False, indent=2))
    # Reuse this memory with independently configured readers:
    # response_a = read_with(reader_a, compiled)
    # response_b = read_with(reader_b, compiled)
    # For a different tokenizer/template, compile again with that reader's
    # count_tokens and render_request. Ingestion does not need to run again.


if __name__ == "__main__":
    main()
