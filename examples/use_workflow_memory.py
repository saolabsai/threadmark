"""Offline example of source-linked history, proposals and snapshot restoration.

Run from an installed checkout: ``python examples/use_workflow_memory.py``.
The responses below are scripted fixtures, not fresh model predictions.
"""

from copy import deepcopy
import json

from threadmark import Completion, Settings, WorkflowMemory


def operation(kind, field, value=None, *, proposal_id=None, reason=None):
    return {
        "op": kind,
        "entity_id": "review",
        "field": field,
        "value": value,
        "proposal_id": proposal_id,
        "reason": reason,
    }


class RecordedProvider:
    """Minimal provider with no network access or external dependencies."""

    def __init__(self, responses=()):
        self.responses = list(deepcopy(responses))
        self.calls = 0

    def complete(self, **kwargs):
        self.calls += 1
        if not self.responses:
            raise RuntimeError("No scripted response remains")
        return Completion(text=json.dumps(self.responses.pop(0)))


def main():
    catalog = [
        {
            "entity_id": "review",
            "label": "Quarterly review",
            "fields": ["owner", "venue"],
        }
    ]
    texts = [
        "Ada owns the quarterly review.",
        "Proposal ROOM-7: hold the review in Room B.",
        "Accept ROOM-7 because Room B supports the required video call.",
        "Bo now owns the quarterly review; Ada has handed it over.",
    ]
    events = [
        {
            "event_id": f"e{index + 1}",
            "timestamp": f"2026-11-01T09:{index * 10:02}:00+00:00",
            "sender": "Coordinator",
            "text": text,
        }
        for index, text in enumerate(texts)
    ]
    updates = {
        "events": [
            {"event_id": "e1", "operations": [operation("set", "owner", "Ada")]},
            {
                "event_id": "e2",
                "operations": [
                    operation("propose", "venue", "Room B", proposal_id="ROOM-7")
                ],
            },
            {
                "event_id": "e3",
                "operations": [
                    operation(
                        "accept",
                        "venue",
                        proposal_id="ROOM-7",
                        reason="Room B supports the required video call.",
                    )
                ],
            },
            {"event_id": "e4", "operations": [operation("set", "owner", "Bo")]},
        ]
    }
    scripted_answer = {
        "answer": "Room B was approved; Bo now owns the quarterly review.",
        "evidence_event_ids": ["e2", "e3", "e4"],
    }
    provider = RecordedProvider([updates, scripted_answer])
    settings = Settings(
        reasoning_effort="medium",
        chunk_tokens=4000,
        updater_max_output_tokens=4096,
        reader_max_output_tokens=1024,
    )
    # This conservative byte counter keeps the demonstration offline even when
    # tokenizer data has never been downloaded. Real inference defaults to the
    # configured tokenizer; changing context limits requires its own evaluation.
    memory = WorkflowMemory(
        catalog,
        provider,
        settings,
        count_tokens=lambda text: len(text.encode("utf-8")),
    )
    memory.ingest(events)
    answer = memory.answer("Which venue was approved, and who owns the review now?")

    print("SCRIPTED OFFLINE DEMONSTRATION. This is not measured model accuracy.")
    print(json.dumps(answer, indent=2))
    print("Current facts:", json.dumps(memory.state, indent=2))
    print("Earlier owner remains recorded:")
    owner_history = [
        item
        for item in memory.memory_packet()["fact_history"]
        if item["field"] == "owner"
    ]
    print(json.dumps(owner_history, indent=2))

    # Snapshots contain original workflow text. Protect them like that archive.
    snapshot = json.loads(json.dumps(memory.snapshot()))
    restore_provider = RecordedProvider()
    restored = WorkflowMemory.from_snapshot(
        snapshot,
        provider=restore_provider,
        count_tokens=lambda text: len(text.encode("utf-8")),
    )
    assert restored.memory_packet() == memory.memory_packet()
    assert restore_provider.calls == 0
    print("Snapshot restored without provider calls.")


if __name__ == "__main__":
    main()
