"""Small OpenAI Responses adapter with explicit accounting and no retries."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
from math import isfinite
import os
import re
import time
from typing import Any
from urllib import error, request

from .budget import BudgetError, BudgetLedger, Prices

ENDPOINT = "https://api.openai.com/v1/responses"
MAX_RESPONSE_BYTES = 4 * 1024 * 1024


@dataclass(frozen=True)
class Completion:
    text: str
    usage: dict[str, Any] | None = None
    elapsed_seconds: float = 0.0
    model: str = ""
    response_id: str | None = None
    status: str = "completed"
    estimated_cost_usd: float | None = None
    budget_cost_usd: float | None = None


class ProviderError(RuntimeError):
    """A request failed; callers must not automatically retry it."""

    def __init__(
        self, message: str, *, request_started: bool = False, fatal: bool = True
    ) -> None:
        super().__init__(message)
        self.request_started = request_started
        self.fatal = fatal


class _NoRedirect(request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _safe_message(text: Any, api_key: str) -> str:
    """Bound diagnostics and remove credentials even if a server echoes them."""
    if not isinstance(text, str):
        return "The provider returned no readable error message."
    text = text.replace(api_key, "[REDACTED]")
    text = re.sub(r"\bsk-[A-Za-z0-9_-]+", "[REDACTED]", text, flags=re.I)
    text = re.sub(r"\bBearer\s+[^\s\"'<>,;]+", "Bearer [REDACTED]", text, flags=re.I)
    return " ".join(text.split())[:600]


def _http_message(exc: error.HTTPError, api_key: str) -> str:
    message = "The provider returned no readable error message."
    try:
        raw = exc.read(8193)
        if len(raw) <= 8192:
            body = json.loads(raw)
            if isinstance(body, dict) and isinstance(body.get("error"), dict):
                message = _safe_message(body["error"].get("message"), api_key)
    except (OSError, ValueError, UnicodeError):
        pass
    finally:
        exc.close()
    return f"OpenAI returned HTTP {exc.code}. {message} No retry was made."


def _parse_text(response: dict) -> str:
    output = response.get("output")
    if not isinstance(output, list):
        raise ValueError("Response output is not a list")
    fragments = []
    for item in output:
        if not isinstance(item, dict):
            raise ValueError("Invalid response output item")
        if item.get("type") != "message":
            continue
        content = item.get("content")
        if not isinstance(content, list):
            raise ValueError("Invalid response message")
        for part in content:
            if not isinstance(part, dict):
                raise ValueError("Invalid response content")
            if part.get("type") == "output_text":
                if not isinstance(part.get("text"), str):
                    raise ValueError("Invalid response text")
                fragments.append(part["text"])
    return "".join(fragments).strip()


def _usage_metadata(value: Any) -> dict | None:
    """Keep only token counts, never arbitrary provider metadata."""
    if not isinstance(value, dict):
        return None
    count = lambda number: number if type(number) is int and number >= 0 else None
    result = {
        name: count(value.get(name)) for name in ("input_tokens", "output_tokens")
    }
    details = value.get("input_tokens_details", {})
    if not isinstance(details, dict):
        result["input_tokens_details"] = {"cached_tokens": None}
    else:
        result["input_tokens_details"] = {
            "cached_tokens": count(details.get("cached_tokens", 0))
        }
        if "cache_write_tokens" in details:
            result["input_tokens_details"]["cache_write_tokens"] = count(
                details["cache_write_tokens"]
            )
    return result


class OpenAIProvider:
    """Call the official API with an environment key and caller-supplied rates.

    This instance is intended for sequential use. The file budget supports
    multiple processes. Diagnostics omit authorization headers and prompts.
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        model: str = "gpt-5.6-luna",
        reasoning_effort: str | None = "none",
        prices: Prices,
        budget: BudgetLedger,
        timeout: float = 60,
    ) -> None:
        key = os.environ.get("OPENAI_API_KEY") if api_key is None else api_key
        if (
            not isinstance(key, str)
            or not key
            or not key.isascii()
            or any(ord(char) <= 32 or ord(char) == 127 for char in key)
        ):
            raise ValueError(
                "Set a valid OPENAI_API_KEY or provide a key through a secure local prompt"
            )
        if (
            not isinstance(model, str)
            or not re.fullmatch(r"[A-Za-z0-9_.:/-]{1,160}", model)
            or key in model
            or model.lower().startswith("sk-")
        ):
            raise ValueError("Invalid model identifier")
        if reasoning_effort is not None and reasoning_effort not in (
            "none",
            "minimal",
            "low",
            "medium",
            "high",
            "xhigh",
        ):
            raise ValueError("Invalid reasoning effort")
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not isfinite(timeout)
            or not 0 < timeout <= 300
        ):
            raise ValueError("Timeout must be between 0 and 300 seconds")
        if not isinstance(prices, Prices) or not isinstance(budget, BudgetLedger):
            raise TypeError("Explicit Prices and BudgetLedger instances are required")
        self.model, self.reasoning_effort = model, reasoning_effort
        self.prices, self.budget, self.timeout = prices, budget, timeout
        self._key, self._calls, self._halted = key, [], False

    @property
    def calls(self) -> list[dict]:
        return deepcopy(self._calls)

    @property
    def halted(self) -> bool:
        return self._halted

    def complete(
        self, *, instructions: str, content: dict, max_output_tokens: int
    ) -> Completion:
        if self._halted:
            raise ProviderError(
                "This provider stopped after an uncertain request; inspect the budget before a new run"
            )
        if (
            not isinstance(instructions, str)
            or not instructions
            or not isinstance(content, dict)
        ):
            raise ValueError("Instructions and structured content are required")
        if type(max_output_tokens) is not int or not 16 <= max_output_tokens <= 16384:
            raise ValueError("max_output_tokens must be between 16 and 16384")
        payload = {
            "model": self.model,
            "instructions": instructions,
            "input": json.dumps(
                content,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ),
            "store": False,
            "max_output_tokens": max_output_tokens,
        }
        if self.reasoning_effort is not None:
            payload["reasoning"] = {"effort": self.reasoning_effort}
        body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        try:
            reservation = self.budget.reserve(
                self.prices.maximum(payload), model=self.model
            )
        except BudgetError:
            self._halted = True
            raise ProviderError(
                "The budget cannot reserve this request; no API call was made"
            ) from None
        started = time.perf_counter()
        record: dict[str, Any] = {
            "status": "request_failed",
            "model": self.model,
            "usage": None,
            "estimated_cost_usd": None,
            "budget_cost_usd": None,
            "request_started": False,
            "response_id": None,
        }
        response_id = None
        accounting = None
        completion = None
        failure: ProviderError | None = None
        try:
            req = request.Request(
                ENDPOINT,
                data=body,
                method="POST",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": "Bearer " + self._key,
                },
            )
            opener = request.build_opener(_NoRedirect())
            record["request_started"] = True
            with opener.open(req, timeout=self.timeout) as reply:
                raw = reply.read(MAX_RESPONSE_BYTES + 1)
            if len(raw) > MAX_RESPONSE_BYTES:
                raise ValueError("Response is too large")
            response = json.loads(raw)
            if not isinstance(response, dict):
                raise ValueError("Invalid response object")
            usage = _usage_metadata(response.get("usage"))
            estimate, accounting = self.prices.costs(usage)
            response_id = response.get("id")
            if response_id is not None and (
                not isinstance(response_id, str)
                or not re.fullmatch(r"[A-Za-z0-9_.:/-]{1,160}", response_id)
                or self._key in response_id
                or response_id.lower().startswith("sk-")
            ):
                response_id = None
            record.update(
                usage=deepcopy(usage),
                estimated_cost_usd=estimate,
                budget_cost_usd=accounting,
                response_id=response_id,
            )
            if accounting is None:
                failure = ProviderError(
                    "OpenAI returned missing or invalid usage; its full reservation remains held",
                    request_started=True,
                )
            else:
                text = _parse_text(response)
                status = response.get("status", "unknown")
                record["status"] = (
                    status
                    if status
                    in (
                        "completed",
                        "incomplete",
                        "failed",
                        "cancelled",
                        "queued",
                        "in_progress",
                    )
                    else "unknown"
                )
                if status != "completed" or not text:
                    if status == "completed":
                        record["status"] = "empty_output"
                    failure = ProviderError(
                        "OpenAI did not return a complete text answer; no retry was made",
                        request_started=True,
                        fatal=False,
                    )
                else:
                    completion = Completion(
                        text=text,
                        usage=deepcopy(usage),
                        model=self.model,
                        response_id=response_id,
                        estimated_cost_usd=estimate,
                        budget_cost_usd=accounting,
                    )
        except error.HTTPError as exc:
            failure = ProviderError(_http_message(exc, self._key), request_started=True)
        except Exception:
            failure = ProviderError(
                "Transport or response processing failed; no retry was made",
                request_started=record["request_started"],
            )
        finally:
            record["elapsed_seconds"] = time.perf_counter() - started
            try:
                self.budget.settle(
                    reservation,
                    accounting if record["request_started"] else 0,
                    response_id=response_id,
                )
            except BudgetError:
                failure = ProviderError(
                    "Cost accounting failed; inspect the ledger before continuing",
                    request_started=record["request_started"],
                )
            if failure:
                record["error"] = str(failure)
                if failure.fatal:
                    self._halted = True
            try:
                snapshot = self.budget.snapshot()
                if snapshot["over_budget"] or snapshot["reservation_overrun_usd"] > 0:
                    self._halted = True
                    failure = ProviderError(
                        "Actual usage exceeded its reservation; spending was recorded and the provider stopped",
                        request_started=True,
                    )
                    record["error"] = str(failure)
            except BudgetError:
                self._halted = True
                failure = ProviderError(
                    "The budget could not be read after dispatch; inspect the ledger before continuing",
                    request_started=record["request_started"],
                )
                record["error"] = str(failure)
            self._calls.append(record)
        if failure:
            raise failure
        assert completion is not None
        return Completion(
            **{**completion.__dict__, "elapsed_seconds": record["elapsed_seconds"]}
        )
