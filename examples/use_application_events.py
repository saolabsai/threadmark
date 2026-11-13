"""Apply known application operations without a model or network connection.

Run: python examples/use_application_events.py
The application supplies these operations explicitly; this is not an extractor
accuracy benchmark. A text-only conversation still needs a validated extractor.
"""

import json

from threadmark import WorkflowMemory


class NoModelCalls:
    def complete(self, **kwargs):
        raise RuntimeError("This example does not permit model calls")


def operation(kind, field, value=None, *, proposal_id=None):
    return {
        "op": kind,
        "entity_id": "review",
        "field": field,
        "value": value,
        "proposal_id": proposal_id,
        "reason": None,
    }


def main():
    memory = WorkflowMemory(
        [{"entity_id": "review", "fields": ["owner", "room"]}], NoModelCalls()
    )
    events = [
        {
            "event_id": "e1",
            "timestamp": "2026-11-01T10:00:00Z",
            "sender": "Scheduling application",
            "text": "Ada handed ownership to Bo. Propose Room B as B or Room C as C.",
        },
        {
            "event_id": "e2",
            "timestamp": "2026-11-01T10:01:00Z",
            "sender": "Scheduling application",
            "text": "Approve room proposal C and reject B.",
        },
    ]
    memory.apply_operations(
        events,
        {
            "events": [
                {
                    "event_id": "e1",
                    "operations": [
                        operation("set", "owner", "Ada"),
                        operation("set", "owner", "Bo"),
                        operation("propose", "room", "Room B", proposal_id="B"),
                        operation("propose", "room", "Room C", proposal_id="C"),
                    ],
                },
                {
                    "event_id": "e2",
                    "operations": [
                        operation("accept", "room", proposal_id="C"),
                        operation("reject", "room", proposal_id="B"),
                    ],
                },
            ]
        },
    )
    facts = {fact["field"]: fact for fact in memory.state["facts"]}
    assert facts["owner"]["value"] == "Bo"
    assert facts["room"]["value"] == "Room C"
    assert facts["room"]["evidence_event_ids"] == ["e1", "e2"]
    restored = WorkflowMemory.from_snapshot(memory.snapshot(), provider=NoModelCalls())
    assert restored.memory_packet() == memory.memory_packet()
    print("Application-supplied operations; zero model calls.")
    print(json.dumps(memory.memory_packet(), indent=2))


if __name__ == "__main__":
    main()
