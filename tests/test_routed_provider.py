"""Role routing preserves requests, accounting, and failures without retries."""

from copy import deepcopy

import pytest

from threadmark.prompts import READER_INSTRUCTIONS, UPDATER_INSTRUCTIONS
from threadmark.providers import Completion, ProviderError
from threadmark.routed_provider import RoleRoutedProvider


class Provider:
    reasoning_effort = "none"

    def __init__(self, model, *, failure=None, record_failure=True):
        self.model = model
        self.failure = failure
        self.record_failure = record_failure
        self.received = []
        self._calls = []

    @property
    def calls(self):
        return deepcopy(self._calls)

    def complete(self, **kwargs):
        self.received.append(kwargs)
        if not self.failure or self.record_failure:
            self._calls.append(
                {
                    "status": "failed" if self.failure else "completed",
                    "model": self.model,
                    "usage": {"input_tokens": 10, "output_tokens": 2},
                    "request_started": True,
                    "elapsed_seconds": 0.25,
                    "estimated_cost_usd": 0.001,
                    "budget_cost_usd": 0.001,
                    "content": "PRIVATE_PROVIDER_PAYLOAD",
                    "error": "PRIVATE_ERROR_DETAIL",
                }
            )
        if self.failure:
            raise self.failure
        return Completion(
            '{"facts":[]}',
            usage={"input_tokens": 10, "output_tokens": 2},
            model=self.model,
            elapsed_seconds=0.25,
            estimated_cost_usd=0.001,
            budget_cost_usd=0.001,
        )


def router(reader=None, updater=None, instructions=None):
    reader = reader or Provider("reader-model")
    updater = updater or Provider("updater-model")
    return RoleRoutedProvider(
        reader, updater, instructions or {UPDATER_INSTRUCTIONS, "custom updater"}
    )


def call(provider, instructions, content=None):
    return provider.complete(
        instructions=instructions,
        content=content or {"public": "data"},
        max_output_tokens=64,
    )


def test_exact_instruction_membership_routes_original_and_selected_updater_only():
    provider = router()
    call(provider, UPDATER_INSTRUCTIONS)
    call(provider, "custom updater")
    call(provider, READER_INSTRUCTIONS)
    call(provider, "custom updater\n")
    assert len(provider.updater.received) == 2
    assert len(provider.reader.received) == 2
    assert [record["role"] for record in provider.calls] == [
        "updater",
        "updater",
        "reader",
        "reader",
    ]
    assert provider.model == "reader-model"
    assert provider.reasoning_effort == "none"
    assert provider.model_for("custom updater") == "updater-model"
    assert provider.model_for("custom updater\n") == "reader-model"


def test_content_and_gold_like_labels_cannot_affect_routing_and_arguments_are_unchanged():
    provider = router()
    content = {
        "role": "updater",
        "expected": "custom updater",
        "instructions": UPDATER_INSTRUCTIONS,
    }
    result = call(provider, READER_INSTRUCTIONS, content)
    received = provider.reader.received[0]
    assert received["content"] is content
    assert received == {
        "instructions": READER_INSTRUCTIONS,
        "content": content,
        "max_output_tokens": 64,
    }
    assert result.model == "reader-model"
    assert provider.updater.received == []


def test_mutating_original_instruction_set_cannot_change_routing():
    selected = {UPDATER_INSTRUCTIONS}
    provider = router(instructions=selected)
    selected.add(READER_INSTRUCTIONS)
    call(provider, READER_INSTRUCTIONS)
    assert provider.calls[0]["role"] == "reader"


def test_merged_history_preserves_actual_model_cost_usage_and_dispatch_order():
    provider = router()
    call(provider, READER_INSTRUCTIONS)
    call(provider, UPDATER_INSTRUCTIONS)
    assert [record["model"] for record in provider.calls] == [
        "reader-model",
        "updater-model",
    ]
    assert sum(record["budget_cost_usd"] for record in provider.calls) == 0.002
    assert all(
        record["usage"] == {"input_tokens": 10, "output_tokens": 2}
        for record in provider.calls
    )
    assert "content" not in provider.calls[0]
    assert "error" not in provider.calls[0]
    provider.calls[0]["usage"]["input_tokens"] = 999
    assert provider.calls[0]["usage"]["input_tokens"] == 10


def test_updater_failure_preserves_charged_metadata_without_reader_fallback_or_retry():
    updater = Provider(
        "strong-writer",
        failure=ProviderError("PRIVATE_FAILURE", request_started=True, fatal=False),
    )
    provider = router(updater=updater)
    with pytest.raises(ProviderError):
        call(provider, UPDATER_INSTRUCTIONS)
    assert len(updater.received) == 1
    assert provider.reader.received == []
    assert provider.calls[0]["role"] == "updater"
    assert provider.calls[0]["model"] == "strong-writer"
    assert provider.calls[0]["usage"]["output_tokens"] == 2
    assert provider.calls[0]["budget_cost_usd"] == 0.001
    assert provider.calls[0]["error_type"] == "ProviderError"
    assert "PRIVATE" not in str(provider.calls)


def test_pre_dispatch_budget_failure_still_gets_a_failed_router_record():
    reader = Provider(
        "reader-model", failure=ProviderError("No budget"), record_failure=False
    )
    provider = router(reader=reader)
    with pytest.raises(ProviderError):
        call(provider, READER_INSTRUCTIONS)
    assert len(provider.calls) == 1
    assert provider.calls[0]["request_started"] is False
    assert provider.calls[0]["usage"] is None
    assert provider.calls[0]["model"] == "reader-model"
    assert provider.calls[0]["status"] == "failed"


def test_keyboard_interrupt_propagates_and_actual_attempt_remains_recorded():
    reader = Provider("reader-model", failure=KeyboardInterrupt())
    provider = router(reader=reader)
    with pytest.raises(KeyboardInterrupt):
        call(provider, READER_INSTRUCTIONS)
    assert len(reader.received) == 1
    assert provider.calls[0]["request_started"] is True
    assert provider.calls[0]["error_type"] == "KeyboardInterrupt"


def test_reader_without_native_history_uses_completion_metadata():
    class BareProvider:
        model = "bare-reader"
        reasoning_effort = "none"

        def complete(self, **kwargs):
            return Completion(
                "ok",
                usage={"input_tokens": 3, "output_tokens": 1},
                model=self.model,
                estimated_cost_usd=0.003,
                budget_cost_usd=0.003,
            )

    provider = router(reader=BareProvider())
    call(provider, READER_INSTRUCTIONS)
    assert provider.calls[0]["usage"]["input_tokens"] == 3
    assert provider.calls[0]["estimated_cost_usd"] == 0.003


def test_load_delegates_reader_only_without_generating():
    reader = Provider("local-reader")
    reader.load_count = 0

    def load():
        reader.load_count += 1
        return {"device": "cpu", "model_load_seconds": 1.2}

    reader.load = load
    provider = router(reader=reader)
    metadata = provider.load()
    assert reader.load_count == 1
    assert metadata["device"] == "cpu"
    assert metadata["reader_model"] == "local-reader"
    assert metadata["updater_model"] == "updater-model"
    assert provider.calls == []
    assert reader.received == provider.updater.received == []


def test_hosted_load_metadata_does_not_call_either_provider():
    provider = router()
    assert provider.load()["cold_model_loading"] == "provider_managed_unmeasured"
    assert provider.calls == []


def test_same_provider_object_is_rejected():
    reader = Provider("reader-model")
    with pytest.raises(ValueError, match="separate"):
        RoleRoutedProvider(reader, reader, {UPDATER_INSTRUCTIONS})


@pytest.mark.parametrize("instructions", [set(), [UPDATER_INSTRUCTIONS], {""}, {None}])
def test_invalid_updater_instruction_sets_are_rejected(instructions):
    with pytest.raises(ValueError, match="nonempty set"):
        RoleRoutedProvider(Provider("reader"), Provider("writer"), instructions)
