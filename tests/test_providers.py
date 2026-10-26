"""Exercise the actual HTTP adapter with mocked transport, never a live key."""

from io import BytesIO
import json
from unittest import mock
from urllib.error import HTTPError

import pytest

from threadmark import BudgetLedger, OpenAIProvider, Prices, ProviderError
from threadmark.budget import BudgetError
from threadmark.providers import ENDPOINT, _NoRedirect

KEY = "unit-test-secret-value"
RATES = Prices(input=0.20, cached_input=0.02, cache_write=0.25, output=1.20)


def usage(**changes):
    value = {
        "input_tokens": 100,
        "output_tokens": 10,
        "input_tokens_details": {"cached_tokens": 20, "cache_write_tokens": 30},
    }
    value.update(changes)
    return value


def body(**changes):
    value = {
        "id": "resp_test_1",
        "model": "resolved-model",
        "status": "completed",
        "usage": usage(),
        "output": [
            {
                "type": "message",
                "content": [{"type": "output_text", "text": '{"facts":[]}'}],
            }
        ],
    }
    value.update(changes)
    return value


def provider(tmp_path, cap=0.01):
    budget = BudgetLedger(tmp_path / "budget.json", limit_usd=cap)
    return OpenAIProvider(KEY, prices=RATES, budget=budget), budget


def fake_transport(response):
    opener = mock.Mock()
    opener.open.return_value = BytesIO(json.dumps(response).encode())
    return (
        mock.patch("threadmark.providers.request.build_opener", return_value=opener),
        opener,
    )


def complete(client):
    return client.complete(
        instructions="Return facts", content={"events": []}, max_output_tokens=256
    )


def test_price_cache_categories_and_unknown_writes():
    assert RATES.costs(usage()) == pytest.approx((0.0000299, 0.0000299))
    estimate, accounting = RATES.costs(
        usage(input_tokens_details={"cached_tokens": 20})
    )
    assert estimate == pytest.approx(0.0000284)
    assert accounting == pytest.approx(0.0000324)
    assert RATES.costs(usage(input_tokens_details={"cached_tokens": True})) == (
        None,
        None,
    )
    assert RATES.costs(usage(input_tokens_details={"cached_tokens": 101})) == (
        None,
        None,
    )
    with pytest.raises(ValueError):
        Prices(input=float("nan"), cached_input=0, cache_write=0, output=1)


def test_explicit_cap_cannot_be_reset(tmp_path):
    budget = BudgetLedger(tmp_path / "budget.json", limit_usd=0.01)
    reservation = budget.reserve(0.005, model="test-model")
    budget.settle(reservation, None)
    with pytest.raises(BudgetError, match="Existing cap"):
        BudgetLedger(tmp_path / "budget.json", limit_usd=1)
    assert (
        BudgetLedger(tmp_path / "budget.json", limit_usd=0.01).snapshot()[
            "unknown_hold_usd"
        ]
        == 0.005
    )


def test_success_settles_once_and_only_posts_to_official_endpoint(tmp_path):
    client, budget = provider(tmp_path)
    transport, opener = fake_transport(body())
    with transport:
        result = complete(client)
    assert result.text == '{"facts":[]}'
    assert result.model == "gpt-5.6-luna"
    assert result.elapsed_seconds > 0
    assert budget.snapshot()["known_spend_usd"] == pytest.approx(0.0000299)
    assert budget.snapshot()["reserved_usd"] == 0
    req = opener.open.call_args.args[0]
    assert req.full_url == ENDPOINT
    assert req.get_header("Authorization") == "Bearer " + KEY
    payload = json.loads(req.data)
    assert payload["store"] is False
    assert payload["model"] == "gpt-5.6-luna"
    assert KEY not in req.data.decode()
    assert KEY not in repr(client.calls)
    assert len(client.calls) == 1
    copy = client.calls
    copy[0]["status"] = "tampered"
    assert client.calls[0]["status"] == "completed"


def test_budget_denial_precedes_any_transport(tmp_path):
    client, budget = provider(tmp_path, cap=0.00000001)
    with mock.patch("threadmark.providers.request.build_opener") as opener:
        with pytest.raises(ProviderError, match="no API call"):
            complete(client)
    opener.assert_not_called()
    assert client.calls == []
    assert budget.snapshot()["known_spend_usd"] == 0


def test_server_error_is_redacted_held_and_never_retried(tmp_path):
    client, budget = provider(tmp_path)
    response = BytesIO(json.dumps({"error": {"message": "Echo " + KEY}}).encode())
    failure = HTTPError(ENDPOINT, 500, "Server error", {}, response)
    with mock.patch("threadmark.providers.request.build_opener") as factory:
        factory.return_value.open.side_effect = failure
        with pytest.raises(ProviderError) as caught:
            complete(client)
        with pytest.raises(ProviderError, match="stopped"):
            complete(client)
        assert factory.return_value.open.call_count == 1
    assert KEY not in str(caught.value)
    assert KEY not in repr(client.calls)
    assert budget.snapshot()["unknown_hold_usd"] > 0
    assert budget.snapshot()["known_spend_usd"] == 0


@pytest.mark.parametrize(
    "response",
    [body(usage=None), body(usage={"input_tokens": KEY}), body(output="invalid")],
)
def test_missing_usage_or_malformed_response_stops_without_retry(tmp_path, response):
    client, budget = provider(tmp_path)
    transport, opener = fake_transport(response)
    with transport, pytest.raises(ProviderError):
        complete(client)
    assert opener.open.call_count == 1
    assert client.halted
    assert KEY not in repr(client.calls)
    snapshot = budget.snapshot()
    assert snapshot["unknown_hold_usd"] > 0 or snapshot["known_spend_usd"] > 0


def test_incomplete_known_usage_is_charged_and_not_retried(tmp_path):
    client, budget = provider(tmp_path)
    transport, opener = fake_transport(body(status="incomplete"))
    with transport, pytest.raises(ProviderError) as caught:
        complete(client)
    assert caught.value.fatal is False
    assert opener.open.call_count == 1
    assert budget.snapshot()["known_spend_usd"] > 0
    assert budget.snapshot()["unknown_hold_usd"] == 0


def test_untrusted_usage_and_model_metadata_cannot_echo_key(tmp_path):
    client, _ = provider(tmp_path)
    counts = usage()
    counts["api_key"] = KEY
    transport, _ = fake_transport(body(usage=counts, model=KEY, id=KEY))
    with transport:
        result = complete(client)
    assert KEY not in repr(result)
    assert KEY not in repr(client.calls)
    assert result.response_id is None


def test_actual_overrun_remains_recorded_and_stops_provider(tmp_path):
    client, budget = provider(tmp_path)
    transport, _ = fake_transport(body(usage=usage(input_tokens=1000000)))
    with transport, pytest.raises(ProviderError, match="exceeded"):
        complete(client)
    assert budget.snapshot()["over_budget"]
    assert budget.snapshot()["known_spend_usd"] > budget.snapshot()["cap_usd"]
    assert client.halted


def test_reservation_overrun_stops_even_below_total_cap(tmp_path):
    client, budget = provider(tmp_path)
    transport, _ = fake_transport(body(usage=usage(input_tokens=10000)))
    with transport, pytest.raises(ProviderError, match="exceeded"):
        complete(client)
    assert not budget.snapshot()["over_budget"]
    assert budget.snapshot()["reservation_overrun_usd"] > 0
    assert client.halted


def test_authorization_redirects_are_disabled():
    assert (
        _NoRedirect().redirect_request(
            None, None, 302, "Found", {}, "https://other.example"
        )
        is None
    )


def test_no_key_and_invalid_input_make_no_request(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        OpenAIProvider(
            prices=RATES, budget=BudgetLedger(tmp_path / "b.json", limit_usd=1)
        )
    client, budget = provider(tmp_path)
    with pytest.raises(ValueError, match="max_output_tokens"):
        client.complete(instructions="Facts", content={}, max_output_tokens=True)
    assert client.calls == []
    assert budget.snapshot()["reserved_usd"] == 0
