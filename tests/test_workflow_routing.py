"""History-enabled writer prompts must retain exact provider routing."""

from copy import deepcopy

import pytest

from threadmark import (
    Settings,
    WorkflowMemory,
    WORKFLOW_HISTORY_UPDATER_INSTRUCTIONS,
    WORKFLOW_UPDATER_INSTRUCTIONS,
)
from threadmark.providers import Completion
from threadmark.routed_provider import RoleRoutedProvider
from threadmark.workflow_memory import HISTORY_REFERENCE_INSTRUCTIONS


class Provider:
    reasoning_effort = "none"

    def __init__(self, model, response):
        self.model = model
        self.response = response
        self.calls = []

    def complete(self, **kwargs):
        self.calls.append(deepcopy(kwargs))
        return Completion(text=self.response)


@pytest.mark.parametrize("history", [0, 4])
@pytest.mark.parametrize("custom", [False, True])
def test_effective_prompt_matches_ingest_and_survives_snapshot(history, custom):
    prompt = "Custom exact writer prompt" if custom else WORKFLOW_UPDATER_INSTRUCTIONS
    expected = prompt + ("\n" + HISTORY_REFERENCE_INSTRUCTIONS if history else "")
    reader = Provider("reader", '{"answer":null,"evidence_event_ids":[]}')
    updater = Provider("updater", '{"events":[]}')
    routed = RoleRoutedProvider(reader, updater, {expected})
    memory = WorkflowMemory(
        [{"entity_id": "meeting", "fields": ["room"]}],
        routed,
        Settings(model="reader"),
        count_tokens=len,
        updater_instructions=prompt,
        reference_history_entries=history,
    )
    assert memory.updater_instructions == expected
    if not custom:
        assert expected == (
            WORKFLOW_HISTORY_UPDATER_INSTRUCTIONS
            if history
            else WORKFLOW_UPDATER_INSTRUCTIONS
        )
    with pytest.raises(AttributeError):
        memory.updater_instructions = "changed"
    saved = memory.snapshot()
    assert saved["prompts"]["updater"] == prompt
    restored = WorkflowMemory.from_snapshot(saved, provider=routed, count_tokens=len)
    assert restored.updater_instructions == expected
    restored.ingest(
        [
            {
                "event_id": "e1",
                "timestamp": "2026-01-01T00:00:00Z",
                "sender": "Coordinator",
                "text": "Thanks, no changes.",
            }
        ]
    )
    assert len(updater.calls) == 1
    assert updater.calls[0]["instructions"] == expected
    assert reader.calls == []
    assert routed.calls[0]["role"] == "updater"
    # Exact membership remains intentional; a similar prompt is not a writer.
    assert routed.model_for(expected + "\n") == "reader"
