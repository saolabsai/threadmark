"""Explicit token rates and persistent spending reservations.

Rates are supplied by the caller. A budget is a local guard, not a provider
invoice or an account-wide spending limit. Unknown outcomes keep their hold.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from threading import RLock
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

    @staticmethod
    def open_existing(path: str | Path) -> "BudgetLedger":
        """Open and validate existing accounting without reinitializing it.

        The immutable cap, imported historical spend, and all pending/unknown
        holds are preserved. A missing or incompatible ledger is an error.
        Reading uses the same process-safe lock as ordinary snapshots.
        """
        path = Path(path)
        _ledger.snapshot(path)
        ledger = object.__new__(BudgetLedger)
        ledger.path = path
        return ledger

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


class PairedBudgetLedger(BudgetLedger):
    """Require both a persistent project cap and a separate run cap.

    Reservations are not a distributed transaction. The run reserves first;
    project rejection happens before transport and releases only that new run
    hold at a known zero cost. Unexpected reservation failures retain holds.
    Settlement updates the project first, then the run. If either step fails,
    the other outstanding hold is never released as a guessed zero cost.

    The mapping of new run IDs to project IDs lives in this instance. A process
    restart preserves both ledgers' holds but cannot silently reconstruct or
    release their association. Keep this object for the lifetime of a run and
    inspect failed or stale accounting before creating another live run.
    """

    def __init__(self, project: BudgetLedger, run: BudgetLedger) -> None:
        if (
            not isinstance(project, BudgetLedger)
            or not isinstance(run, BudgetLedger)
            or isinstance(project, PairedBudgetLedger)
            or isinstance(run, PairedBudgetLedger)
        ):
            raise TypeError("Project and run must be separate plain budget ledgers")
        project.snapshot()
        run.snapshot()
        if project.path.resolve() == run.path.resolve() or project.path.samefile(
            run.path
        ):
            raise ValueError("Project and run must use different ledger files")
        self.project, self.run = project, run
        self.path = run.path
        self._project_reservations: dict[str, str] = {}
        self._lock = RLock()

    def reserve(self, maximum_usd: float, *, model: str) -> str:
        with self._lock:
            run_id = self.run.reserve(maximum_usd, model=model)
            try:
                project_id = self.project.reserve(maximum_usd, model=model)
            except BudgetError:
                # No reservation ID was returned, so the caller cannot yet
                # start transport. This newly created run request cost zero.
                try:
                    self.run.settle(run_id, 0)
                except Exception:
                    raise _ledger.BudgetStateError(
                        "Project reservation failed and the run hold could not be released"
                    ) from None
                raise
            except Exception:
                # A non-ledger failure is uncertain: keep the run reservation
                # rather than assume what an unexpected implementation did.
                raise _ledger.BudgetStateError(
                    "Project reservation failed unexpectedly; the run hold remains"
                ) from None
            self._project_reservations[run_id] = project_id
            return run_id

    def settle(
        self,
        reservation: str,
        actual_usd: float | None,
        *,
        response_id: str | None = None,
    ) -> dict:
        with self._lock:
            if (
                not isinstance(reservation, str)
                or reservation not in self._project_reservations
            ):
                raise _ledger.BudgetStateError(
                    "Reservation does not belong to this paired budget instance"
                )
            project_id = self._project_reservations[reservation]
            try:
                self.project.settle(project_id, actual_usd, response_id=response_id)
            except BudgetError:
                raise
            except Exception:
                raise _ledger.BudgetStateError(
                    "Project settlement failed; the run hold remains unchanged"
                ) from None
            try:
                self.run.settle(reservation, actual_usd, response_id=response_id)
            except BudgetError:
                raise
            except Exception:
                raise _ledger.BudgetStateError(
                    "Run settlement failed after project accounting; inspect both ledgers"
                ) from None
            # Keep the association for idempotent known settlements and later
            # reconciliation of an unknown outcome with an established cost.
            return self.snapshot()

    @staticmethod
    def _numeric(snapshot: dict) -> dict:
        return {
            key: value
            for key, value in snapshot.items()
            if isinstance(value, (int, float, bool))
        }

    def snapshot(self) -> dict:
        """Return provider-compatible guard fields and private numeric totals.

        Top-level spending totals describe the run. Effective availability is
        the smaller remaining balance, and either ledger's overrun halts the
        provider. Nested project totals are operational accounting, not public
        benchmark output. Snapshotting does not release pre-existing holds.
        """
        with self._lock:
            run = self._numeric(self.run.snapshot())
            project = self._numeric(self.project.snapshot())
            return {
                **run,
                "available_usd": min(run["available_usd"], project["available_usd"]),
                "over_budget": run["over_budget"] or project["over_budget"],
                "reservation_overrun_usd": max(
                    run["reservation_overrun_usd"],
                    project["reservation_overrun_usd"],
                ),
                "run": run,
                "project": project,
            }
