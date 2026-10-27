"""Both spending caps remain authoritative across failures and stale holds."""

import json

import pytest

from threadmark import _ledger
from threadmark.budget import (
    BudgetError,
    BudgetExceeded,
    BudgetLedger,
    PairedBudgetLedger,
    Prices,
)
from threadmark.providers import OpenAIProvider


def pair(tmp_path, *, project_cap=7, run_cap=1, historical=0):
    project_path = tmp_path / "project.json"
    _ledger.init_budget(
        project_path, cap_usd=project_cap, historical_cost_usd=historical
    )
    project = BudgetLedger.open_existing(project_path)
    run = BudgetLedger(tmp_path / "run.json", run_cap)
    return PairedBudgetLedger(project, run)


def test_open_existing_preserves_imported_spend_and_pending_unknown_holds(tmp_path):
    path = tmp_path / "existing.json"
    _ledger.init_budget(path, cap_usd=7, historical_cost_usd="0.8")
    _ledger.reserve(path, "0.5")
    unknown = _ledger.reserve(path, "0.2")
    _ledger.settle(path, unknown, None)
    before = path.read_bytes()

    budget = BudgetLedger.open_existing(path)

    assert path.read_bytes() == before
    snapshot = budget.snapshot()
    assert snapshot["cap_usd"] == 7
    assert snapshot["historical_cost_usd"] == 0.8
    assert snapshot["known_spend_usd"] == 0.8
    assert snapshot["reserved_usd"] == 0.5
    assert snapshot["unknown_hold_usd"] == 0.2
    assert snapshot["available_usd"] == 5.5


def test_open_existing_never_creates_or_repairs_accounting(tmp_path):
    missing = tmp_path / "missing.json"
    with pytest.raises(BudgetError):
        BudgetLedger.open_existing(missing)
    assert not missing.exists()
    malformed = tmp_path / "malformed.json"
    malformed.write_text('{"cap_usd": 7}', encoding="utf-8")
    before = malformed.read_bytes()
    with pytest.raises(BudgetError):
        BudgetLedger.open_existing(malformed)
    assert malformed.read_bytes() == before


def test_run_cap_rejects_before_touching_project(tmp_path):
    budget = pair(tmp_path, run_cap=0.1)
    before = budget.project.path.read_bytes()
    with pytest.raises(BudgetExceeded):
        budget.reserve(0.2, model="fixture-model")
    assert budget.project.path.read_bytes() == before
    assert budget.run.snapshot()["pending_reservations"] == 0


def test_project_rejection_rolls_back_only_the_new_run_hold(tmp_path):
    budget = pair(tmp_path, project_cap=2, historical=1.6)
    first = budget.reserve(0.3, model="fixture-model")
    with pytest.raises(BudgetExceeded):
        budget.reserve(0.2, model="fixture-model")
    snapshot = budget.snapshot()
    assert snapshot["run"]["reserved_usd"] == 0.3
    assert snapshot["project"]["reserved_usd"] == 0.3
    assert snapshot["run"]["known_spend_usd"] == 0
    assert snapshot["run"]["pending_reservations"] == 1
    assert snapshot["available_usd"] == 0.1
    budget.settle(first, 0.05, response_id="response-first")
    assert budget.snapshot()["run"]["known_spend_usd"] == 0.05
    assert budget.snapshot()["project"]["known_spend_usd"] == 1.65


def test_unknown_outcomes_hold_both_caps_and_known_reconciliation_is_idempotent(
    tmp_path,
):
    budget = pair(tmp_path)
    reservation = budget.reserve(0.25, model="fixture-model")
    budget.settle(reservation, None, response_id="response-unknown")
    snapshot = budget.snapshot()
    for ledger in (snapshot["run"], snapshot["project"]):
        assert ledger["unknown_hold_usd"] == 0.25
        assert ledger["unknown_reservations"] == 1
        assert ledger["known_spend_usd"] == 0
    budget.settle(reservation, 0.1, response_id="response-unknown")
    before_run = budget.run.path.read_bytes()
    before_project = budget.project.path.read_bytes()
    budget.settle(reservation, 0.1, response_id="response-unknown")
    assert budget.run.path.read_bytes() == before_run
    assert budget.project.path.read_bytes() == before_project
    assert budget.snapshot()["run"]["available_usd"] == 0.9
    assert budget.snapshot()["project"]["available_usd"] == 6.9


def test_project_settlement_failure_does_not_release_the_run_hold(
    tmp_path, monkeypatch
):
    budget = pair(tmp_path)
    reservation = budget.reserve(0.2, model="fixture-model")
    before_run = budget.run.path.read_bytes()

    def failed_settlement(*args, **kwargs):
        raise BudgetError("simulated project accounting failure")

    with monkeypatch.context() as patch:
        patch.setattr(budget.project, "settle", failed_settlement)
        with pytest.raises(BudgetError):
            budget.settle(reservation, 0.05, response_id="response-partial")
    assert budget.run.path.read_bytes() == before_run
    assert budget.project.snapshot()["reserved_usd"] == 0.2
    budget.settle(reservation, 0.05, response_id="response-partial")
    assert budget.snapshot()["run"]["known_spend_usd"] == 0.05


def test_run_settlement_failure_keeps_run_hold_and_allows_exact_reconciliation(
    tmp_path, monkeypatch
):
    budget = pair(tmp_path)
    reservation = budget.reserve(0.2, model="fixture-model")

    def failed_settlement(*args, **kwargs):
        raise BudgetError("simulated run accounting failure")

    with monkeypatch.context() as patch:
        patch.setattr(budget.run, "settle", failed_settlement)
        with pytest.raises(BudgetError):
            budget.settle(reservation, 0.05, response_id="response-partial")
    assert budget.project.snapshot()["known_spend_usd"] == 0.05
    assert budget.run.snapshot()["reserved_usd"] == 0.2
    budget.settle(reservation, 0.05, response_id="response-partial")
    assert budget.snapshot()["run"]["reserved_usd"] == 0
    assert budget.snapshot()["run"]["known_spend_usd"] == 0.05


def test_failed_rollback_keeps_run_hold_and_unexpected_project_failure_is_conservative(
    tmp_path, monkeypatch
):
    budget = pair(tmp_path, project_cap=1, historical=1)

    def failed_rollback(*args, **kwargs):
        raise BudgetError("simulated failed rollback")

    with monkeypatch.context() as patch:
        patch.setattr(budget.run, "settle", failed_rollback)
        with pytest.raises(BudgetError, match="could not be released"):
            budget.reserve(0.2, model="fixture-model")
    assert budget.run.snapshot()["reserved_usd"] == 0.2
    assert budget.project.snapshot()["reserved_usd"] == 0

    def unexpected_failure(*args, **kwargs):
        raise RuntimeError("simulated unexpected reservation failure")

    with monkeypatch.context() as patch:
        patch.setattr(budget.project, "reserve", unexpected_failure)
        with pytest.raises(BudgetError, match="hold remains"):
            budget.reserve(0.1, model="fixture-model")
    assert budget.run.snapshot()["reserved_usd"] == 0.3


def test_stale_holds_survive_new_pairs_and_snapshot_excludes_identifiers(tmp_path):
    budget = pair(tmp_path)
    stale_project = budget.project.reserve(0.5, model="fixture-model")
    budget.project.settle(stale_project, None)
    stale_run = budget.run.reserve(0.1, model="fixture-model")
    reservation = budget.reserve(0.2, model="fixture-model")
    recreated = PairedBudgetLedger(
        BudgetLedger.open_existing(budget.project.path),
        BudgetLedger.open_existing(budget.run.path),
    )
    with pytest.raises(BudgetError, match="does not belong"):
        recreated.settle(reservation, 0)
    snapshot = recreated.snapshot()
    assert snapshot["run"]["reserved_usd"] == 0.3
    assert snapshot["project"]["unknown_hold_usd"] == 0.5
    assert snapshot["project"]["reserved_usd"] == 0.2
    serialized = json.dumps(snapshot)
    assert stale_project not in serialized
    assert stale_run not in serialized
    assert reservation not in serialized
    assert all(
        isinstance(value, (int, float, bool)) for value in snapshot["run"].values()
    )
    assert all(
        isinstance(value, (int, float, bool)) for value in snapshot["project"].values()
    )


def test_either_overrun_is_visible_to_the_existing_provider_guard(tmp_path):
    budget = pair(tmp_path, run_cap=0.25)
    assert isinstance(budget, BudgetLedger)
    provider = OpenAIProvider(
        api_key="offline-fixture-key",
        model="fixture-model",
        prices=Prices(1, 1, 1, 1),
        budget=budget,
    )
    assert provider.budget is budget
    reservation = budget.reserve(0.2, model="fixture-model")
    budget.settle(reservation, 0.3, response_id="response-overrun")
    snapshot = budget.snapshot()
    assert snapshot["over_budget"] is True
    assert snapshot["run"]["over_budget"] is True
    assert snapshot["project"]["over_budget"] is False
    assert snapshot["reservation_overrun_usd"] == 0.1
    assert snapshot["available_usd"] == 0


def test_pair_rejects_aliases_and_nested_pairs(tmp_path):
    budget = pair(tmp_path)
    alias = BudgetLedger.open_existing(budget.project.path)
    with pytest.raises(ValueError, match="different"):
        PairedBudgetLedger(budget.project, alias)
    with pytest.raises(TypeError, match="plain"):
        PairedBudgetLedger(budget, budget.run)
