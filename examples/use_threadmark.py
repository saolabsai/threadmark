"""Offline API example using independently scripted demonstration responses.

Run from an installed checkout: ``python examples/use_threadmark.py``.
Replace RecordedDemoProvider with a configured OpenAIProvider for real inference;
see docs/usage.md for explicit prices, budget limits and API key handling.
"""

import json

from threadmark import Threadmark
from threadmark.cli import RecordedDemoProvider
from threadmark.datasets import demo_dataset

workflow = demo_dataset()["workflows"][0]
provider = RecordedDemoProvider()
# The scripted example uses a conservative byte counter to remain offline even
# on first installation. Real inference defaults to the evaluated tokenizer.
memory = Threadmark(
    workflow["entity_catalog"],
    provider,
    count_tokens=lambda text: len(text.encode("utf-8")),
)
memory.ingest(workflow["events"])
query = {key: workflow["queries"][0][key] for key in ("text", "requested_facts")}
answer = memory.answer(query)
print("SCRIPTED OFFLINE DEMONSTRATION. This is not measured model accuracy.")
print(json.dumps(answer, indent=2))
print("Selected source IDs:", [event["event_id"] for event in memory.evidence(query)])

# Snapshots include the original source archive and predicted facts. Store them
# only where those workflow records are allowed to be retained.
snapshot = memory.snapshot()
assert snapshot["events"] == workflow["events"]
