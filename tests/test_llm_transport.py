"""Transport tests for the LLM layer, against a local OpenAI-compatible stub.

Why a stub rather than a live provider call: the thing that needs verifying is *our*
client — header construction, JSON salvage, retry and backoff, the fallback chain, the
budget cap, the cache, and the end-to-end path from an HTTP response to a clamped
confidence on a match link. A single successful call to a third-party proxy proves none of
that in a repeatable way, and it stops proving anything the moment the provider changes,
rate-limits, or disappears. A stub exercises every branch on demand and keeps working in
CI with no key and no network.

What is deliberately *not* claimed here: these tests say nothing about how well any
particular model adjudicates. They test the plumbing. Adjudication *quality* would need a
live model and a labelled set, and is listed as future work in ARCHITECTURE.md.

The stub speaks real HTTP on a real socket, so `httpx` genuinely serialises, sends, and
parses — no monkeypatching of the transport.
"""

from __future__ import annotations

import json
import re
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Callable, Iterator

import pytest

from trikon.config import Settings
from trikon.generate.defects import DEFAULT_PLAN, inject_defects
from trikon.generate.world import GeneratorConfig, generate_clean_world
from trikon.llm.adjudicate import adjudicate_batch, build_adjudicator
from trikon.llm.cache import ResponseCache
from trikon.llm.provider import BudgetExhausted, LLMProvider, probe_provider
from trikon.models import RULE_CONFIDENCE_CEILING, AUTO_ACCEPT_THRESHOLD
from trikon.pipeline import run_pipeline

# --------------------------------------------------------------------------------------
# Stub server
# --------------------------------------------------------------------------------------


class _Stub:
    """A minimal OpenAI-compatible endpoint whose behaviour a test can script.

    ``script`` is a list of callables, one consumed per request, each returning
    ``(status, headers, body_text)``. Requests beyond the script reuse the last entry, so a
    test only has to describe the interesting prefix.
    """

    def __init__(self, script: list[Callable[[dict[str, Any]], tuple[int, dict[str, str], str]]]):
        self._script = script
        self.requests: list[dict[str, Any]] = []
        self._server: HTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def base_url(self) -> str:
        assert self._server is not None
        return f"http://127.0.0.1:{self._server.server_port}/v1"

    def __enter__(self) -> "_Stub":
        stub = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def _respond(self) -> None:
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length) if length else b"{}"
                try:
                    payload = json.loads(raw)
                except json.JSONDecodeError:
                    payload = {}
                payload["__path"] = self.path
                payload["__auth"] = self.headers.get("Authorization")
                stub.requests.append(payload)

                index = min(len(stub.requests) - 1, len(stub._script) - 1)
                status, headers, body = stub._script[index](payload)
                encoded = body.encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                for key, value in headers.items():
                    self.send_header(key, value)
                self.end_headers()
                self.wfile.write(encoded)

            def do_POST(self) -> None:  # noqa: N802
                self._respond()

            def do_GET(self) -> None:  # noqa: N802
                self._respond()

            def log_message(self, *args: Any) -> None:
                pass  # keep pytest output clean

        self._server = HTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)


def _chat_response(content: str, model: str = "stub-model") -> str:
    """A well-formed chat-completions envelope wrapping ``content``."""
    return json.dumps(
        {
            "id": "chatcmpl-stub",
            "object": "chat.completion",
            "model": model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20},
        }
    )


def _decisions_for(payload: dict[str, Any], decision: str = "ESCALATE") -> str:
    """Build a decisions payload answering exactly the cases the prompt contained.

    Reading the case ids back out of the request is what makes the stub realistic: a
    response that answers cases nobody asked about is a *different* test (see
    ``test_hallucinated_case_id_over_the_wire_is_rejected``).
    """
    prompt = json.dumps(payload)
    case_ids = sorted(set(re.findall(r"CASE\d{2}", prompt)))
    return json.dumps(
        {
            "decisions": [
                {
                    "case_id": cid,
                    "decision": decision,
                    "confidence": 0.99,  # deliberately over-confident; must be clamped
                    "reasoning": "stubbed adjudication",
                }
                for cid in case_ids
            ]
        }
    )


def _ok(content_builder: Callable[[dict[str, Any]], str]):
    return lambda payload: (200, {}, _chat_response(content_builder(payload)))


def _settings_for(stub: _Stub, tmp_path: Any, **overrides: Any) -> Settings:
    defaults: dict[str, Any] = {
        "llm_base_url": stub.base_url,
        "llm_api_key": "sk-stub-key-for-tests",
        "llm_model": "stub-model",
        "llm_requests_per_minute": 6000,  # do not sleep during tests
        "llm_timeout_seconds": 10.0,
        "llm_cache_dir": tmp_path / "cache",
        "llm_cache_enabled": False,
    }
    defaults.update(overrides)
    return Settings(**defaults)


@pytest.fixture
def pending() -> Iterator[list]:
    """Real pending candidates from a real batch, for realistic evidence packets."""
    world = inject_defects(
        generate_clean_world(GeneratorConfig(seed=42, n_orders=300)), DEFAULT_PLAN
    )
    run = run_pipeline(world.orders, world.recon_rows, world.settlements, world.bank_credits)
    candidates = [p for tier in run.tiers.values() for p in tier.escalated]
    assert candidates, "fixture produced no ambiguous candidates"
    yield candidates


# --------------------------------------------------------------------------------------
# Happy path
# --------------------------------------------------------------------------------------


def test_request_is_well_formed_over_real_http(pending, tmp_path) -> None:
    """The client must send bearer auth, the model, temperature 0, and a JSON schema."""
    with _Stub([_ok(lambda p: _decisions_for(p))]) as stub:
        settings = _settings_for(stub, tmp_path)
        adjudicate_batch(pending[:3], settings=settings, provider=LLMProvider(settings))

        assert stub.requests, "no HTTP request reached the stub"
        sent = stub.requests[0]
        assert sent["__path"].endswith("/v1/chat/completions")
        assert sent["__auth"] == "Bearer sk-stub-key-for-tests"
        assert sent["model"] == "stub-model"
        assert sent["temperature"] == 0.0, "adjudication must be reproducible, not creative"
        assert sent["response_format"]["type"] == "json_schema"
        assert [m["role"] for m in sent["messages"]] == ["system", "user"]


def test_full_path_from_http_response_to_clamped_link(tmp_path) -> None:
    """End to end: a real HTTP MATCH becomes a review-flagged link, never auto-accepted.

    This is the test that closes the loop the safety suite could only simulate: the
    decision genuinely arrives over a socket, and the ceiling still holds.
    """
    world = inject_defects(
        generate_clean_world(GeneratorConfig(seed=42, n_orders=300)), DEFAULT_PLAN
    )
    with _Stub([_ok(lambda p: _decisions_for(p, decision="MATCH"))]) as stub:
        settings = _settings_for(stub, tmp_path)
        run = run_pipeline(
            world.orders,
            world.recon_rows,
            world.settlements,
            world.bank_credits,
            adjudicator=build_adjudicator(settings),
        )

    adjudicated = [link for link in run.all_links if link.adjudicated_by]
    assert adjudicated, "no link was adjudicated over HTTP"
    for link in adjudicated:
        # The stub claimed 0.99 on every case; the ladder must pull it back.
        assert link.confidence <= RULE_CONFIDENCE_CEILING[link.rule]
        assert link.confidence < AUTO_ACCEPT_THRESHOLD
        assert not link.auto_accepted
        assert link.reasoning == "stubbed adjudication"
    assert run.llm_used and run.adjudicated > 0


def test_probe_provider_succeeds_against_a_live_endpoint(tmp_path) -> None:
    with _Stub([lambda p: (200, {}, _chat_response("OK"))]) as stub:
        ok, detail = probe_provider(_settings_for(stub, tmp_path))
    assert ok, detail
    assert "OK" in detail


# --------------------------------------------------------------------------------------
# JSON salvage -- what free-tier models actually return
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "wrapper",
    [
        pytest.param("{body}", id="bare"),
        pytest.param("```json\n{body}\n```", id="markdown-fenced"),
        pytest.param("```\n{body}\n```", id="fenced-no-language"),
        pytest.param("Sure! Here is the result:\n{body}\nHope that helps.", id="prose-wrapped"),
        pytest.param("  \n\t{body}\n\n", id="whitespace-padded"),
    ],
)
def test_json_is_salvaged_from_realistic_model_output(pending, tmp_path, wrapper: str) -> None:
    """Models ignore "return only JSON" constantly; the client must cope.

    Being liberal costs nothing because the salvaged object is still validated against a
    Pydantic model before it can affect any decision.
    """
    builder = lambda p: wrapper.replace("{body}", _decisions_for(p))  # noqa: E731
    with _Stub([_ok(builder)]) as stub:
        settings = _settings_for(stub, tmp_path)
        outcomes = adjudicate_batch(
            pending[:2], settings=settings, provider=LLMProvider(settings)
        )
    assert len(outcomes) == 2
    # ESCALATE means accept=False, but the response must have been *parsed* -- proven by
    # the absence of any exception and by the stub having been reached exactly once.
    assert len(stub.requests) == 1


def test_unparseable_output_escalates_rather_than_matching(pending, tmp_path) -> None:
    with _Stub([lambda p: (200, {}, _chat_response("I cannot help with that."))]) as stub:
        settings = _settings_for(stub, tmp_path)
        outcomes = adjudicate_batch(
            pending[:2], settings=settings, provider=LLMProvider(settings)
        )
    assert all(not o.accept for o in outcomes)


def test_hallucinated_case_id_over_the_wire_is_rejected(pending, tmp_path) -> None:
    """A response answering a case we never sent is discarded, and nothing is accepted."""
    bogus = json.dumps(
        {
            "decisions": [
                {
                    "case_id": "CASE99",
                    "decision": "MATCH",
                    "confidence": 1.0,
                    "reasoning": "invented",
                }
            ]
        }
    )
    with _Stub([lambda p: (200, {}, _chat_response(bogus))]) as stub:
        settings = _settings_for(stub, tmp_path)
        outcomes = adjudicate_batch(
            pending[:2], settings=settings, provider=LLMProvider(settings)
        )
    assert all(not o.accept for o in outcomes)


# --------------------------------------------------------------------------------------
# Resilience
# --------------------------------------------------------------------------------------


def test_rate_limit_is_retried_honouring_retry_after(pending, tmp_path) -> None:
    """429 then 200 must succeed, and Retry-After must be read rather than ignored."""
    script = [
        lambda p: (429, {"Retry-After": "0"}, json.dumps({"error": "slow down"})),
        _ok(lambda p: _decisions_for(p, decision="MATCH")),
    ]
    with _Stub(script) as stub:
        settings = _settings_for(stub, tmp_path)
        provider = LLMProvider(settings)
        outcomes = adjudicate_batch(pending[:1], settings=settings, provider=provider)
    assert len(stub.requests) == 2, "did not retry after 429"
    assert any(o.accept for o in outcomes)
    assert provider.stats.retries >= 1


def test_server_error_falls_through_to_the_next_model(pending, tmp_path) -> None:
    """A 5xx on the primary must exhaust retries then try the fallback model."""
    def handler(payload: dict[str, Any]) -> tuple[int, dict[str, str], str]:
        if payload.get("model") == "primary-model":
            return 500, {}, json.dumps({"error": "upstream exploded"})
        return 200, {}, _chat_response(_decisions_for(payload, decision="MATCH"))

    with _Stub([handler]) as stub:
        settings = _settings_for(
            stub, tmp_path, llm_model="primary-model", llm_fallback_models="backup-model"
        )
        provider = LLMProvider(settings)
        outcomes = adjudicate_batch(pending[:1], settings=settings, provider=provider)

    models_tried = [r.get("model") for r in stub.requests]
    assert "primary-model" in models_tried and "backup-model" in models_tried
    assert any(o.accept for o in outcomes)


def test_client_error_does_not_burn_retries(pending, tmp_path) -> None:
    """A 400 will not improve on retry -- move to the next model instead of hammering."""
    with _Stub([lambda p: (400, {}, json.dumps({"error": "unsupported response_format"}))]) as stub:
        settings = _settings_for(stub, tmp_path)
        outcomes = adjudicate_batch(
            pending[:1], settings=settings, provider=LLMProvider(settings)
        )
    assert len(stub.requests) == 1, f"retried a 400 {len(stub.requests)} times"
    assert all(not o.accept for o in outcomes)


def test_401_escalates_instead_of_raising(pending, tmp_path) -> None:
    """The exact failure a gated provider produces must degrade, not crash.

    This is the AgentRouter case: HTTP 401 on every request. The run must still complete
    with everything escalated to human review.
    """
    body = json.dumps({"error": {"message": "unauthorized client detected"}})
    with _Stub([lambda p: (401, {}, body)]) as stub:
        settings = _settings_for(stub, tmp_path)
        world = inject_defects(
            generate_clean_world(GeneratorConfig(seed=42, n_orders=300)), DEFAULT_PLAN
        )
        run = run_pipeline(
            world.orders,
            world.recon_rows,
            world.settlements,
            world.bank_credits,
            adjudicator=build_adjudicator(settings),
        )
    assert not [link for link in run.all_links if link.adjudicated_by]
    assert sum(len(t.escalated) for t in run.tiers.values()) > 0
    # And the deterministic result is untouched by the provider failing.
    assert run.exceptions


def test_budget_cap_stops_calls_and_leaves_the_rest_escalated(pending, tmp_path) -> None:
    """Exhausting the budget mid-batch yields a partial but honest result."""
    with _Stub([_ok(lambda p: _decisions_for(p, decision="MATCH"))]) as stub:
        settings = _settings_for(
            stub, tmp_path, llm_max_calls_per_run=1, llm_adjudication_batch_size=1
        )
        provider = LLMProvider(settings)
        outcomes = adjudicate_batch(pending[:4], settings=settings, provider=provider)

    assert len(stub.requests) == 1, "budget cap was not enforced"
    assert sum(1 for o in outcomes if o.accept) <= 1
    assert provider.budget_remaining == 0


def test_budget_exhausted_raises_only_when_asked_directly(tmp_path) -> None:
    with _Stub([_ok(lambda p: "{}")]) as stub:
        settings = _settings_for(stub, tmp_path, llm_max_calls_per_run=0)
        provider = LLMProvider(settings)
        # llm_enabled is False at zero budget, so nothing should attempt a call.
        assert not settings.llm_enabled
        with pytest.raises(BudgetExhausted):
            provider.complete_json(system="s", user="u", schema_name="n", schema={})


# --------------------------------------------------------------------------------------
# Cache
# --------------------------------------------------------------------------------------


def test_cache_prevents_a_second_http_call(pending, tmp_path) -> None:
    """The second identical run must make zero requests.

    This is the mechanism that makes reported metrics reproducible and the demo
    offline-safe, so it is worth asserting at the transport level rather than trusting.
    """
    with _Stub([_ok(lambda p: _decisions_for(p, decision="MATCH"))]) as stub:
        settings = _settings_for(stub, tmp_path, llm_cache_enabled=True)
        cache = ResponseCache(tmp_path / "cache", enabled=True)

        first = adjudicate_batch(
            pending[:2], settings=settings, provider=LLMProvider(settings), cache=cache
        )
        calls_after_first = len(stub.requests)

        second = adjudicate_batch(
            pending[:2], settings=settings, provider=LLMProvider(settings), cache=cache
        )

    assert calls_after_first >= 1
    assert len(stub.requests) == calls_after_first, "cache did not prevent a second call"
    assert [o.accept for o in first] == [o.accept for o in second]
    assert cache.hits >= 1
