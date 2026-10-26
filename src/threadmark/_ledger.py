"""Persistent USD reservations shared by processes; no network or credentials.

Only explicit ``init_budget`` creates a ledger. Monetary values are stored as
decimal strings. Unknown outcomes keep their maximum reservation until settled
with a known amount. Callers must reserve a conservative maximum BEFORE a paid
request, settle afterward, and stop their run on an uncertain request outcome.

Metadata is restricted to short identifiers, never prompts or credentials.
Existing caps/historical spend cannot be changed through initialization.
"""

from contextlib import contextmanager
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_CEILING
import errno
import json
import os
from pathlib import Path
import re
import tempfile
import time
import uuid


class BudgetError(RuntimeError):
    """Budget operation failed; callers must not proceed with a paid request."""


class BudgetExceeded(BudgetError):
    """The requested reservation exceeds the remaining project budget."""


class BudgetStateError(BudgetError):
    """Missing, incompatible, corrupt or conflicting ledger state."""


_METADATA_KEYS = {"run_id", "case_id", "arm", "model", "operation"}
_IDENTIFIER = re.compile(r"[A-Za-z0-9_.:/-]{1,160}\Z")
_RESERVATION_ID = re.compile(r"rsv_[0-9a-f]{32}\Z")
_ZERO = Decimal("0")


def _money(value, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (str, int, float, Decimal)):
        raise BudgetStateError("USD amount must be a finite nonnegative number")
    try:
        amount = Decimal(str(value))
    except (InvalidOperation, ValueError):
        raise BudgetStateError("Invalid USD amount") from None
    if not amount.is_finite() or amount < 0 or (positive and amount == 0):
        raise BudgetStateError("USD amount is outside the allowed range")
    # Float arithmetic from API usage may contain insignificant trailing digits.
    # Round it conservatively upward to a trillionth of a dollar; never release
    # budget because of binary floating-point noise.
    if amount > Decimal("1000000000"):
        raise BudgetStateError("USD amount exceeds supported magnitude or precision")
    if isinstance(value, float):
        amount = amount.quantize(Decimal("0.000000000001"), rounding=ROUND_CEILING)
    if amount.as_tuple().exponent < -12:
        raise BudgetStateError("USD amount exceeds supported magnitude or precision")
    return amount


def _identifier(value):
    if (
        not isinstance(value, str)
        or not _IDENTIFIER.fullmatch(value)
        or "sk-" in value.lower()
        or "bearer" in value.lower()
    ):
        raise BudgetStateError("Only bounded nonsecret identifiers are accepted")
    return value


def _metadata(value):
    if value is None:
        return {}
    if not isinstance(value, dict) or not set(value).issubset(_METADATA_KEYS):
        raise BudgetStateError(
            "Metadata accepts only run_id, case_id, arm, model and operation identifiers"
        )
    return {key: _identifier(item) for key, item in value.items()}


def _now():
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def _locked(path, *, create=False):
    path = Path(path)
    if create:
        path.parent.mkdir(parents=True, exist_ok=True)
    elif not path.is_file():
        raise BudgetStateError("Budget ledger does not exist")
    lock_path = path.with_name(path.name + ".lock")
    handle = None
    acquired = False
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        handle = os.fdopen(descriptor, "r+b", buffering=0)
        if os.fstat(handle.fileno()).st_size == 0:
            handle.write(b"\0")
        deadline = time.monotonic() + 10.0
        while True:
            try:
                handle.seek(0)
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
                break
            except OSError as exc:
                if (
                    exc.errno not in (errno.EACCES, errno.EAGAIN, errno.EDEADLK)
                    or time.monotonic() >= deadline
                ):
                    raise BudgetStateError(
                        "Unable to acquire budget ledger lock"
                    ) from None
                time.sleep(0.025)
        yield path
    except BudgetError:
        raise
    except OSError:
        raise BudgetStateError("Budget ledger filesystem operation failed") from None
    finally:
        if handle is not None:
            try:
                if acquired:
                    handle.seek(0)
                    if os.name == "nt":
                        import msvcrt

                        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                    else:
                        import fcntl

                        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            finally:
                handle.close()


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise BudgetStateError("Duplicate field in budget ledger")
        result[key] = value
    return result


def _load(path):
    try:
        ledger = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object
        )
        if set(ledger) != {
            "version",
            "cap_usd",
            "historical_cost_usd",
            "created_utc",
            "updated_utc",
            "revision",
            "reservations",
        }:
            raise BudgetStateError("Unsupported budget ledger fields")
        if (
            ledger["version"] != 1
            or type(ledger["revision"]) is not int
            or ledger["revision"] < 0
        ):
            raise BudgetStateError("Unsupported budget ledger version or revision")
        _money(ledger["cap_usd"], positive=True)
        _money(ledger["historical_cost_usd"])
        if not isinstance(ledger["reservations"], dict):
            raise BudgetStateError("Invalid budget reservations")
        responses = set()
        for identifier, row in ledger["reservations"].items():
            if not _RESERVATION_ID.fullmatch(identifier) or set(row) != {
                "maximum_cost_usd",
                "actual_cost_usd",
                "status",
                "metadata",
                "response_id",
                "created_utc",
                "updated_utc",
            }:
                raise BudgetStateError("Invalid reservation fields")
            _money(row["maximum_cost_usd"], positive=True)
            if row["status"] not in ("reserved", "unknown", "settled"):
                raise BudgetStateError("Invalid reservation status")
            if row["status"] == "settled":
                _money(row["actual_cost_usd"])
            elif row["actual_cost_usd"] is not None:
                raise BudgetStateError("Unsettled reservation cannot have a known cost")
            _metadata(row["metadata"])
            response = row["response_id"]
            if response is not None:
                _identifier(response)
                if response in responses:
                    raise BudgetStateError("Duplicate response accounting")
                responses.add(response)
        return ledger
    except BudgetError:
        raise
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        raise BudgetStateError("Cannot read a valid budget ledger") from None


def _write(path, ledger):
    descriptor, name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(ledger, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _totals(ledger):
    known = _money(ledger["historical_cost_usd"])
    reserved, unknown, overrun = _ZERO, _ZERO, _ZERO
    pending_ids, unknown_ids = [], []
    for identifier, row in ledger["reservations"].items():
        maximum = _money(row["maximum_cost_usd"])
        if row["status"] == "settled":
            actual = _money(row["actual_cost_usd"])
            known += actual
            overrun += max(_ZERO, actual - maximum)
        elif row["status"] == "unknown":
            unknown += maximum
            unknown_ids.append(identifier)
        else:
            reserved += maximum
            pending_ids.append(identifier)
    return known, reserved, unknown, overrun, pending_ids, unknown_ids


def _snapshot(ledger):
    known, reserved, unknown, overrun, pending_ids, unknown_ids = _totals(ledger)
    cap = _money(ledger["cap_usd"])
    committed = known + reserved + unknown
    return {
        "cap_usd": float(cap),
        "historical_cost_usd": float(_money(ledger["historical_cost_usd"])),
        "known_spend_usd": float(known),
        "reserved_usd": float(reserved),
        "unknown_hold_usd": float(unknown),
        "committed_usd": float(committed),
        "available_usd": float(max(_ZERO, cap - committed)),
        "over_budget": committed > cap,
        "reservation_overrun_usd": float(overrun),
        "pending_reservations": len(pending_ids),
        "unknown_reservations": len(unknown_ids),
        "pending_reservation_ids": pending_ids,
        "unknown_reservation_ids": unknown_ids,
        "revision": ledger["revision"],
    }


def snapshot(path):
    """Read existing accounting under lock; never create a ledger or expose keys."""
    with _locked(path) as path:
        return _snapshot(_load(path))


def init_budget(path, cap_usd, historical_cost_usd=0):
    """Explicitly create accounting, or require an identical existing budget.

    Historical spend is an initial imported total, not an increment. Repeated
    calls cannot add it twice or change it. Prior spend above the cap is recorded
    truthfully and leaves no available budget.
    """
    cap, historical = _money(cap_usd, positive=True), _money(historical_cost_usd)
    with _locked(path, create=True) as path:
        if path.exists():
            ledger = _load(path)
            if (
                _money(ledger["cap_usd"]) != cap
                or _money(ledger["historical_cost_usd"]) != historical
            ):
                raise BudgetStateError(
                    "Existing cap or historical spend does not match initialization"
                )
        else:
            now = _now()
            ledger = {
                "version": 1,
                "cap_usd": format(cap, "f"),
                "historical_cost_usd": format(historical, "f"),
                "created_utc": now,
                "updated_utc": now,
                "revision": 0,
                "reservations": {},
            }
            _write(path, ledger)
        return _snapshot(ledger)


def reserve(path, maximum_cost_usd, metadata=None):
    """Reserve a positive maximum before sending a request; return its ID."""
    maximum, safe_metadata = _money(maximum_cost_usd, positive=True), _metadata(
        metadata
    )
    with _locked(path) as path:
        ledger = _load(path)
        known, reserved, unknown, _, _, _ = _totals(ledger)
        if known + reserved + unknown + maximum > _money(ledger["cap_usd"]):
            raise BudgetExceeded(
                "Request reservation exceeds the available project budget"
            )
        identifier, now = "rsv_" + uuid.uuid4().hex, _now()
        ledger["reservations"][identifier] = {
            "maximum_cost_usd": format(maximum, "f"),
            "actual_cost_usd": None,
            "status": "reserved",
            "metadata": safe_metadata,
            "response_id": None,
            "created_utc": now,
            "updated_utc": now,
        }
        ledger["revision"] += 1
        ledger["updated_utc"] = now
        _write(path, ledger)
        return identifier


def settle(path, reservation_id, actual_cost_usd=None, response_id=None):
    """Record billed cost, or keep the full maximum on an unknown outcome.

    Unknown reservations can later be reconciled with a known cost (including
    zero when absence of billing is established). Identical settled calls are
    idempotent; conflicting values fail. Actual overruns are recorded truthfully
    and exposed in the snapshot, not dropped from accounting.
    """
    if not isinstance(reservation_id, str) or not _RESERVATION_ID.fullmatch(
        reservation_id
    ):
        raise BudgetStateError("Invalid reservation identifier")
    actual = None if actual_cost_usd is None else _money(actual_cost_usd)
    response = None if response_id is None else _identifier(response_id)
    with _locked(path) as path:
        ledger = _load(path)
        if reservation_id not in ledger["reservations"]:
            raise BudgetStateError("Unknown reservation")
        row = ledger["reservations"][reservation_id]
        if response is not None:
            if row["response_id"] not in (None, response):
                raise BudgetStateError("Reservation response identifier conflict")
            if any(
                key != reservation_id and item["response_id"] == response
                for key, item in ledger["reservations"].items()
            ):
                raise BudgetStateError(
                    "Response is already accounted for by another reservation"
                )
        if row["status"] == "settled":
            if actual is None or _money(row["actual_cost_usd"]) != actual:
                raise BudgetStateError("Cannot change a settled reservation cost")
            if response is None or row["response_id"] == response:
                return _snapshot(ledger)
        row["actual_cost_usd"] = None if actual is None else format(actual, "f")
        row["status"] = "unknown" if actual is None else "settled"
        if response is not None:
            row["response_id"] = response
        row["updated_utc"] = ledger["updated_utc"] = _now()
        ledger["revision"] += 1
        _write(path, ledger)
        return _snapshot(ledger)
