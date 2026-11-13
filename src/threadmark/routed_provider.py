"""Route writer and reader calls by exact instruction identity only."""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
from math import isfinite
import re
import time
from typing import Any

from .providers import Completion
from .schema import canonical

_METADATA_FIELDS = frozenset(
    {
        "status",
        "model",
        "usage",
        "elapsed_seconds",
        "generation_seconds",
        "estimated_cost_usd",
        "budget_cost_usd",
        "api_charge_usd",
        "hardware_cost_usd",
        "cost_scope",
        "request_started",
        "response_id",
        "input_tokens",
        "context_limit",
        "dtype",
        "error_type",
    }
)

_LOCAL_IDENTITY_FIELDS = frozenset(
    {
        "adapter",
        "adapter_requested",
        "structured_workflow",
        "workflow_schema_sha256",
        "constraint_engine",
        "constraint_setup_seconds",
        "max_workflow_value_chars",
    }
)
_KNOWN_MODEL_TYPES = frozenset(
    {
        "llama",
        "mistral",
        "mixtral",
        "qwen2",
        "qwen2_moe",
        "qwen3",
        "qwen3_moe",
        "gemma",
        "gemma2",
        "gemma3",
        "gemma3_text",
        "phi",
        "phi3",
        "opt",
        "bloom",
        "falcon",
        "gpt2",
        "gpt_neo",
        "gpt_neox",
        "starcoder2",
        "olmo",
        "olmo2",
        "granite",
    }
)
_ADAPTER_COMPATIBILITY = "exact_lora_target_shapes; training_base_revision_unverified"


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _local_identity_metadata(record: dict[str, Any]) -> dict[str, Any]:
    """Keep typed provenance only, never arbitrary nested provider diagnostics.

    An invalid adapter identity is omitted rather than logged or interpreted as
    unadapted. Unknown model types/engine versions need an explicit allowlist
    update. This validates metadata shape and consistency, not model provenance.
    """
    result: dict[str, Any] = {}
    if "adapter" in record:
        adapter = record["adapter"]
        if adapter is None:
            result["adapter"] = None
        elif isinstance(adapter, dict):
            hashes = ("config_sha256", "weights_sha256", "base_config_sha256")
            if (
                all(_is_sha256(adapter.get(key)) for key in hashes)
                and isinstance(adapter.get("adapter_id"), str)
                and isinstance(adapter.get("base_model_type"), str)
                and adapter.get("base_model_type") in _KNOWN_MODEL_TYPES
                and adapter.get("compatibility") == _ADAPTER_COMPATIBILITY
            ):
                expected_id = "sha256:" + sha256(
                    canonical(
                        {
                            "config": adapter["config_sha256"],
                            "weights": adapter["weights_sha256"],
                        }
                    ).encode("utf-8")
                ).hexdigest()
                if adapter["adapter_id"] == expected_id:
                    result["adapter"] = {
                        key: adapter[key]
                        for key in (
                            *hashes,
                            "adapter_id",
                            "base_model_type",
                            "compatibility",
                        )
                    }
    for key in ("adapter_requested", "structured_workflow"):
        if key in record and type(record[key]) is bool:
            result[key] = record[key]
    if "workflow_schema_sha256" in record:
        value = record["workflow_schema_sha256"]
        if value is None or _is_sha256(value):
            result["workflow_schema_sha256"] = value
    if "constraint_engine" in record:
        value = record["constraint_engine"]
        if value is None or value == "lm-format-enforcer/0.11.3":
            result["constraint_engine"] = value
    if "constraint_setup_seconds" in record:
        value = record["constraint_setup_seconds"]
        try:
            valid_duration = (
                type(value) in (int, float) and isfinite(value) and value >= 0
            )
        except OverflowError:
            valid_duration = False
        if value is None or valid_duration:
            result["constraint_setup_seconds"] = value
    value = record.get("max_workflow_value_chars")
    if type(value) is int and 1 <= value <= 1_000_000:
        result["max_workflow_value_chars"] = value
    return result


class RoleRoutedProvider:
    """Use separate providers for state updates and reader answers.

    A call is a writer call only when its instructions exactly match one of the
    supplied updater instruction strings. Content, expected answers, query text,
    and output quality never affect routing. Each request is forwarded once,
    unchanged. There is no automatic fallback or retry.

    Use this instance sequentially and retain exclusive control of its two
    providers while collecting results. Providers can share a budget ledger but
    must be separate instances. ``model`` and ``reasoning_effort`` describe the
    reader for engine compatibility; call records identify the actual model.
    """

    def __init__(
        self, reader: Any, updater: Any, updater_instructions: set[str]
    ) -> None:
        if reader is updater:
            raise ValueError("Reader and updater must be separate provider instances")
        if any(
            not callable(getattr(provider, "complete", None))
            for provider in (reader, updater)
        ):
            raise TypeError("Both providers must implement complete")
        if (
            not isinstance(updater_instructions, (set, frozenset))
            or not updater_instructions
            or any(
                not isinstance(text, str) or not text.strip()
                for text in updater_instructions
            )
        ):
            raise ValueError(
                "updater_instructions must be a nonempty set of exact instruction strings"
            )
        if any(
            not isinstance(getattr(provider, "model", None), str) or not provider.model
            for provider in (reader, updater)
        ):
            raise ValueError("Both providers must declare their model")
        self.reader, self.updater = reader, updater
        self.model = reader.model
        self.reasoning_effort = getattr(reader, "reasoning_effort", "none")
        self._updater_instructions = frozenset(updater_instructions)
        self._calls: list[dict[str, Any]] = []

    @property
    def calls(self) -> list[dict[str, Any]]:
        return deepcopy(self._calls)

    def model_for(self, instructions: str) -> str:
        """Return the configured dispatch model for public request journaling."""
        if not isinstance(instructions, str):
            raise TypeError("instructions must be a string")
        return (
            self.updater.model
            if instructions in self._updater_instructions
            else self.reader.model
        )

    def load(self) -> dict[str, Any]:
        """Initialize the reader when supported; no completion call is made."""
        loader = getattr(self.reader, "load", None)
        if callable(loader):
            metadata = loader()
            if not isinstance(metadata, dict):
                raise TypeError("Reader load metadata must be a dictionary")
            metadata = {
                **{
                    key: deepcopy(value)
                    for key, value in metadata.items()
                    if key not in _LOCAL_IDENTITY_FIELDS
                },
                **_local_identity_metadata(metadata),
            }
        else:
            metadata = {
                "backend": "hosted",
                "cold_model_loading": "provider_managed_unmeasured",
            }
        return {
            **metadata,
            "reader_model": self.reader.model,
            "updater_model": self.updater.model,
            "routing": "exact_updater_instruction_membership",
        }

    def complete(self, **kwargs: Any) -> Completion:
        """Forward one request and preserve actual accounting even on failure."""
        instructions = kwargs.get("instructions")
        if not isinstance(instructions, str):
            raise TypeError("instructions must be a string")
        role = "updater" if instructions in self._updater_instructions else "reader"
        provider = self.updater if role == "updater" else self.reader
        prior = len(getattr(provider, "calls", []))
        started = time.perf_counter()
        response = None
        failure: BaseException | None = None
        try:
            response = provider.complete(**kwargs)
            return response
        except BaseException as exc:
            failure = exc
            raise
        finally:
            native = getattr(provider, "calls", [])[prior:]
            if not native:
                # Some pre-dispatch failures (notably budget reservation
                # rejection) do not create an underlying call record.
                native = [
                    {
                        "status": (
                            "failed"
                            if failure
                            else getattr(response, "status", "completed")
                        ),
                        "request_started": (
                            bool(getattr(failure, "request_started", True))
                            if failure
                            else True
                        ),
                        "usage": deepcopy(getattr(response, "usage", None)),
                        "elapsed_seconds": getattr(
                            response, "elapsed_seconds", time.perf_counter() - started
                        ),
                        "estimated_cost_usd": getattr(
                            response, "estimated_cost_usd", None
                        ),
                        "budget_cost_usd": getattr(response, "budget_cost_usd", None),
                        "response_id": getattr(response, "response_id", None),
                    }
                ]
            for record in native:
                # Keep accounting metadata, never arbitrary provider payloads,
                # prompt text, authorization headers, or raw exception strings.
                metadata = {
                    key: deepcopy(value)
                    for key, value in record.items()
                    if key in _METADATA_FIELDS
                }
                metadata.update(_local_identity_metadata(record))
                metadata.update(
                    role=role,
                    model=record.get("model") or provider.model,
                    routed_model=provider.model,
                )
                if failure is not None:
                    metadata["error_type"] = type(failure).__name__
                self._calls.append(metadata)
