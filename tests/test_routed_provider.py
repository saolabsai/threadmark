"""Role routing preserves requests, accounting, and failures without retries."""

from copy import deepcopy
from hashlib import sha256
import json

import pytest

from threadmark.prompts import READER_INSTRUCTIONS, UPDATER_INSTRUCTIONS
from threadmark.providers import Completion, ProviderError
from threadmark.routed_provider import RoleRoutedProvider
from threadmark.schema import canonical


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


def adapter_identity():
    config_hash, weights_hash = "1" * 64, "2" * 64
    return {
        "adapter_id": "sha256:"
        + sha256(
            canonical({"config": config_hash, "weights": weights_hash}).encode("utf-8")
        ).hexdigest(),
        "config_sha256": config_hash,
        "weights_sha256": weights_hash,
        "base_config_sha256": "3" * 64,
        "base_model_type": "llama",
        "compatibility": "exact_lora_target_shapes; training_base_revision_unverified",
    }


class MetadataProvider(Provider):
    def __init__(self, metadata):
        super().__init__("local-updater")
        self.metadata = metadata

    def complete(self, **kwargs):
        result = super().complete(**kwargs)
        self._calls[-1].update(self.metadata)
        return result

    def load(self):
        return {"device": "cpu", **self.metadata}


def test_routed_adapter_and_structured_metadata_are_validated_and_defensive():
    identity = adapter_identity()
    native = MetadataProvider(
        {
            "adapter": identity,
            "adapter_requested": True,
            "structured_workflow": True,
            "workflow_schema_sha256": "4" * 64,
            "constraint_engine": "lm-format-enforcer/0.11.3",
            "constraint_setup_seconds": 0.025,
            "max_workflow_value_chars": 256,
        }
    )
    provider = router(updater=native)
    call(provider, UPDATER_INSTRUCTIONS)
    record = provider.calls[0]
    for key, value in native.metadata.items():
        assert record[key] == value
    identity["weights_sha256"] = "changed"
    provider.calls[0]["adapter"]["config_sha256"] = "changed"
    assert provider.calls[0]["adapter"]["weights_sha256"] == "2" * 64
    assert provider.calls[0]["adapter"]["config_sha256"] == "1" * 64


def test_routed_plain_metadata_preserves_explicit_nulls_and_false_flags():
    metadata = {
        "adapter": None,
        "adapter_requested": False,
        "structured_workflow": False,
        "workflow_schema_sha256": None,
        "constraint_engine": None,
        "constraint_setup_seconds": None,
    }
    provider = router(updater=MetadataProvider(metadata))
    call(provider, UPDATER_INSTRUCTIONS)
    assert {key: provider.calls[0][key] for key in metadata} == metadata


def test_unknown_nested_adapter_diagnostics_never_enter_routed_logs_or_load_metadata():
    identity = {
        **adapter_identity(),
        "path": "PRIVATE_LOCAL_PATH",
        "prompt": {"nested": "PRIVATE_WORKFLOW_TEXT"},
    }
    native = MetadataProvider({"adapter": identity, "structured_workflow": True})
    provider = router(reader=native)
    call(provider, READER_INSTRUCTIONS)
    loaded = provider.load()
    assert provider.calls[0]["adapter"] == loaded["adapter"] == adapter_identity()
    assert "PRIVATE" not in json.dumps(provider.calls)
    assert "PRIVATE" not in json.dumps(loaded)
    loaded["adapter"]["weights_sha256"] = "changed"
    assert provider.load()["adapter"]["weights_sha256"] == "2" * 64


@pytest.mark.parametrize(
    "field,value",
    [
        ("adapter_id", "sha256:PRIVATE"),
        ("adapter_id", "sha256:" + "9" * 64),  # Inconsistent identity.
        ("config_sha256", {"private": "PRIVATE_PAYLOAD"}),
        ("weights_sha256", "PRIVATE_PAYLOAD"),
        ("base_config_sha256", ["PRIVATE_PAYLOAD"]),
        ("base_model_type", "PRIVATE_MODEL_TYPE"),
        ("base_model_type", {"private": "PRIVATE_PAYLOAD"}),
        ("compatibility", "PRIVATE_PAYLOAD"),
    ],
)
def test_invalid_adapter_identity_is_omitted_without_leaking_or_becoming_base_only(
    field, value
):
    identity = {**adapter_identity(), field: value}
    native = MetadataProvider({"adapter": identity, "adapter_requested": True})
    provider = router(reader=native)
    call(provider, READER_INSTRUCTIONS)
    assert "adapter" not in provider.calls[0]
    assert provider.calls[0]["adapter_requested"] is True
    assert "adapter" not in provider.load()
    assert "PRIVATE" not in json.dumps(provider.calls)
    assert "PRIVATE" not in json.dumps(provider.load())


@pytest.mark.parametrize(
    "field,value",
    [
        ("adapter", {"prompt": "PRIVATE_PAYLOAD"}),
        ("adapter", "PRIVATE_PAYLOAD"),
        ("adapter_requested", "PRIVATE_PAYLOAD"),
        ("adapter_requested", 1),
        ("structured_workflow", {"private": "PRIVATE_PAYLOAD"}),
        ("structured_workflow", 0),
        ("workflow_schema_sha256", ["PRIVATE_PAYLOAD"]),
        ("workflow_schema_sha256", "PRIVATE_PAYLOAD"),
        ("constraint_engine", {"private": "PRIVATE_PAYLOAD"}),
        ("constraint_engine", "lm-format-enforcer/PRIVATE_PAYLOAD"),
        ("constraint_setup_seconds", "PRIVATE_PAYLOAD"),
        ("constraint_setup_seconds", {"private": "PRIVATE_PAYLOAD"}),
        ("constraint_setup_seconds", float("nan")),
        ("constraint_setup_seconds", float("inf")),
        ("constraint_setup_seconds", -0.1),
        ("constraint_setup_seconds", True),
        ("constraint_setup_seconds", 10**1000),
        ("max_workflow_value_chars", "PRIVATE_PAYLOAD"),
        ("max_workflow_value_chars", True),
        ("max_workflow_value_chars", 0),
    ],
)
def test_new_metadata_rejects_private_payloads_invalid_types_and_nonfinite_times(
    field, value
):
    native = MetadataProvider({field: value})
    provider = router(reader=native)
    call(provider, READER_INSTRUCTIONS)
    assert field not in provider.calls[0]
    assert field not in provider.load()
    assert "PRIVATE" not in json.dumps(provider.calls, allow_nan=False)
    assert "PRIVATE" not in json.dumps(provider.load(), allow_nan=False)
