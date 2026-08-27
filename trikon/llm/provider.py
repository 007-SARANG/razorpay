"""OpenAI-compatible LLM provider with budget guards.

Deliberately provider-agnostic: any endpoint speaking the ``/chat/completions`` shape
works by setting three environment variables. That covers OpenRouter, AgentRouter, a
local llama.cpp server, or anything else, and it means the submission is not hostage to
one service staying up.

Three guards exist because free tiers are genuinely tight -- OpenRouter's free tier is
20 requests/minute and 50/day unless you have ever bought credits:

* **Budget cap** (``llm_max_calls_per_run``) -- the run refuses to exceed it rather than
  dying halfway through a batch with half its exceptions unclassified.
* **Rate limiter** -- paces requests to stay under the per-minute ceiling.
* **Fallback chain** -- if the primary model errors, later models are tried in order
  before giving up.

Failure is never fatal. Every error path returns ``None`` and the caller escalates the
case to human review, which is the correct conservative outcome and keeps a provider
outage from corrupting a reconciliation.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Final

import httpx

from trikon.config import Settings

#: Retryable HTTP statuses. 429 is rate limiting; 5xx are transient upstream faults.
_RETRY_STATUSES: Final[frozenset[int]] = frozenset({408, 409, 429, 500, 502, 503, 504})

_MAX_ATTEMPTS: Final[int] = 3
_BACKOFF_BASE_SECONDS: Final[float] = 1.5


class BudgetExhausted(RuntimeError):
    """Raised when a run has used its allotted LLM calls."""


@dataclass
class CallStats:
    """What a run actually spent, for honest reporting."""

    calls: int = 0
    cached_hits: int = 0
    failures: int = 0
    retries: int = 0
    total_latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    models_used: dict[str, int] = field(default_factory=dict)

    def note_model(self, model: str) -> None:
        self.models_used[model] = self.models_used.get(model, 0) + 1


class _RateLimiter:
    """Minimum-interval pacer.

    A simple sleep between calls rather than a token bucket: adjudication issues a
    handful of requests per run, so the extra precision would buy nothing and the
    behaviour of this version is obvious on inspection.
    """

    def __init__(self, requests_per_minute: int) -> None:
        self._min_interval = 60.0 / max(requests_per_minute, 1)
        self._last_call: float | None = None

    def wait(self) -> None:
        if self._last_call is None:
            self._last_call = time.monotonic()
            return
        elapsed = time.monotonic() - self._last_call
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_call = time.monotonic()


class LLMProvider:
    """Thin, synchronous chat-completions client.

    Synchronous on purpose: adjudication runs on a small batch after the deterministic
    pipeline has finished, so there is no concurrency to exploit, and a sync client keeps
    the call path easy to read and to reason about during a review.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._limiter = _RateLimiter(settings.llm_requests_per_minute)
        self.stats = CallStats()
        self._models = (settings.llm_model, *settings.fallback_models)

    @property
    def budget_remaining(self) -> int:
        return max(0, self._settings.llm_max_calls_per_run - self.stats.calls)

    def complete_json(
        self, *, system: str, user: str, schema_name: str, schema: dict[str, Any]
    ) -> tuple[dict[str, Any] | None, str | None]:
        """Request a JSON object, returning ``(parsed, model_used)``.

        Returns ``(None, None)`` on any failure -- budget exhausted, transport error,
        unparseable output. Callers must treat that as "escalate", never as "no match".

        ``response_format`` is sent as a hint because support is per-endpoint rather than
        per-model on aggregators, and even where accepted ``strict`` is only advisory on
        some providers. So the response is always parsed and validated locally regardless
        of what the endpoint claims to enforce.
        """
        if self._settings.llm_api_key is None:
            return None, None
        if self.budget_remaining <= 0:
            raise BudgetExhausted(
                f"run budget of {self._settings.llm_max_calls_per_run} LLM calls is spent"
            )

        payload_base: dict[str, Any] = {
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.0,  # adjudication should be reproducible, not creative
            "max_tokens": 1600,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": schema_name, "strict": True, "schema": schema},
            },
        }

        for model in self._models:
            parsed = self._try_model(model, payload_base)
            if parsed is not None:
                return parsed, model
        return None, None

    def _try_model(self, model: str, payload_base: dict[str, Any]) -> dict[str, Any] | None:
        url = f"{self._settings.llm_base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._settings.llm_api_key}",
            "Content-Type": "application/json",
        }

        for attempt in range(1, _MAX_ATTEMPTS + 1):
            if self.budget_remaining <= 0:
                return None
            self._limiter.wait()
            started = time.perf_counter()
            try:
                with httpx.Client(timeout=self._settings.llm_timeout_seconds) as client:
                    response = client.post(
                        url, headers=headers, json={**payload_base, "model": model}
                    )
            except httpx.HTTPError:
                self.stats.failures += 1
                self._sleep_backoff(attempt)
                continue
            finally:
                self.stats.calls += 1
                self.stats.total_latency_ms += (time.perf_counter() - started) * 1000.0

            if response.status_code in _RETRY_STATUSES:
                self.stats.retries += 1
                self._sleep_backoff(attempt, response)
                continue
            if response.status_code >= 400:
                # A 4xx that is not rate limiting will not improve on retry -- an
                # unsupported response_format, a bad model id, an invalid key. Move on to
                # the next model in the chain instead of burning budget.
                self.stats.failures += 1
                return None

            parsed = self._extract_json(response)
            if parsed is None:
                self.stats.failures += 1
                self._sleep_backoff(attempt)
                continue

            self.stats.note_model(model)
            usage = (response.json() or {}).get("usage") or {}
            self.stats.prompt_tokens += int(usage.get("prompt_tokens") or 0)
            self.stats.completion_tokens += int(usage.get("completion_tokens") or 0)
            return parsed

        return None

    @staticmethod
    def _sleep_backoff(attempt: int, response: httpx.Response | None = None) -> None:
        """Honour Retry-After when offered, otherwise back off exponentially."""
        if response is not None:
            retry_after = response.headers.get("Retry-After")
            if retry_after:
                try:
                    time.sleep(min(float(retry_after), 30.0))
                    return
                except ValueError:
                    pass
        time.sleep(min(_BACKOFF_BASE_SECONDS**attempt, 20.0))

    @staticmethod
    def _extract_json(response: httpx.Response) -> dict[str, Any] | None:
        """Pull a JSON object out of a chat completion.

        Free-tier models routinely ignore an instruction to emit bare JSON and wrap it in
        a markdown fence or bracket it with prose. Since we cannot rely on the endpoint
        enforcing a schema, the content is salvaged directly: strip a fence if present,
        otherwise take the outermost brace-delimited span. Being liberal here costs
        nothing, because the result is still validated against a Pydantic model before it
        can influence any decision.
        """
        try:
            body = response.json()
            content = body["choices"][0]["message"]["content"]
        except (json.JSONDecodeError, KeyError, IndexError, TypeError):
            return None
        if not isinstance(content, str):
            return None

        text = content.strip()
        if text.startswith("```"):
            text = text.split("```", 2)[1] if text.count("```") >= 2 else text[3:]
            if text.lstrip().lower().startswith("json"):
                text = text.lstrip()[4:]
            text = text.strip()

        try:
            parsed = json.loads(text)
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            pass

        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            return None
        try:
            parsed = json.loads(text[start : end + 1])
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None


def probe_provider(settings: Settings) -> tuple[bool, str]:
    """Cheap reachability check for ``trikon doctor``.

    Uses a one-token request so that confirming configuration costs almost nothing
    against a 50-request daily budget.
    """
    if settings.llm_api_key is None:
        return False, "no API key configured"
    url = f"{settings.llm_base_url.rstrip('/')}/chat/completions"
    try:
        with httpx.Client(timeout=min(settings.llm_timeout_seconds, 30.0)) as client:
            response = client.post(
                url,
                headers={
                    "Authorization": f"Bearer {settings.llm_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": settings.llm_model,
                    "messages": [{"role": "user", "content": "Reply with the single word OK."}],
                    "max_tokens": 5,
                    "temperature": 0.0,
                },
            )
    except httpx.HTTPError as exc:
        return False, f"transport error: {type(exc).__name__}"

    if response.status_code >= 400:
        detail = response.text[:160].replace("\n", " ")
        return False, f"HTTP {response.status_code}: {detail}"
    try:
        content = response.json()["choices"][0]["message"]["content"]
    except (json.JSONDecodeError, KeyError, IndexError, TypeError):
        return False, "unexpected response shape"
    return True, f"model {settings.llm_model} replied {content!r:.40}"
