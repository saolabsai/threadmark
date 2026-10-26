"""Explicit token rates and persistent spending reservations.

Rates are supplied by the caller. A budget is a local guard, not a provider
invoice or an account-wide spending limit. Unknown outcomes keep their hold.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any

from . import _ledger

BudgetError = _ledger.BudgetError
BudgetExceeded = _ledger.BudgetExceeded


@dataclass(frozen=True)
class Prices:
    """USD per million tokens; verify rates for the model before use."""

    input: float
    cached_input: float
    cache_write: float
    output: float

    def __post_init__(self) -> None:
        for value in (self.input, self.cached_input, self.cache_write, self.output):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(value)
                or value < 0
            ):
                raise ValueError("Token prices must be finite nonnegative numbers")
        if max(self.input, self.cached_input, self.cache_write, self.output) == 0:
            raise ValueError("At least one token price must be positive")

    def maximum(self, payload: dict[str, Any]) -> float:
        """Conservative UTF-8 byte bound, including prompt framing allowance."""
        input_upper = (
            len((payload["instructions"] + payload["input"]).encode("utf-8")) + 1024
        )
        # Cached tokens could cost more than regular input for custom rates.
        input_rate = max(self.input, self.cached_input, self.cache_write)
        maximum = (
            input_upper * input_rate + payload["max_output_tokens"] * self.output
        ) / 1_000_000
        return max(maximum, 0.000000000001)

    def costs(self, usage: dict[str, Any] | None) -> tuple[float | None, float | None]:
        """Return displayed estimate and conservative accounting cost.

        Missing cache-write counts are excluded from the displayed estimate;
        accounting prices uncached input at the higher applicable input rate.
        Missing required or inconsistent usage leaves the whole hold in place.
        """
        if not isinstance(usage, dict):
            return None, None
        details = usage.get("input_tokens_details", {})
        if not isinstance(details, dict):
            return None, None
        inputs, outputs = usage.get("input_tokens"), usage.get("output_tokens")
        cached = details.get("cached_tokens", 0)
        writes_reported = "cache_write_tokens" in details
        writes = details.get("cache_write_tokens", 0)
        if any(
            type(value) is not int or value < 0
            for value in (inputs, outputs, cached, writes)
        ):
            return None, None
        if cached + writes > inputs:
            return None, None
        estimate = (
            (inputs - cached - writes) * self.input
            + cached * self.cached_input
            + writes * self.cache_write
            + outputs * self.output
        ) / 1_000_000
        accounting = (
            estimate
            if writes_reported
            else (
                (inputs - cached) * max(self.input, self.cache_write)
                + cached * self.cached_input
                + outputs * self.output
            )
            / 1_000_000
        )
        return estimate, accounting


class BudgetLedger:
    """A process-safe file ledger with an explicit, immutable USD cap."""

    def __init__(self, path: str | Path, limit_usd: float) -> None:
        self.path = Path(path)
        _ledger.init_budget(self.path, cap_usd=limit_usd)

    def reserve(self, maximum_usd: float, *, model: str) -> str:
        return _ledger.reserve(
            self.path, maximum_usd, {"model": model, "operation": "responses"}
        )

    def settle(
        self,
        reservation: str,
        actual_usd: float | None,
        *,
        response_id: str | None = None,
    ) -> dict:
        return _ledger.settle(
            self.path, reservation, actual_usd, response_id=response_id
        )

    def snapshot(self) -> dict:
        return _ledger.snapshot(self.path)
