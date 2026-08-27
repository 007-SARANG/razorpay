"""Safety tests for the LLM adjudication layer.

Every test here runs offline. No network call is made, no API key is needed, and none of
these tests will start passing or failing because a provider changed its behaviour -- they
test *our* fences, not the model's judgement.

The scenario throughout is a maximally hostile model: one that accepts every candidate
with near-certain confidence and cites evidence that does not exist. If the invariants
hold against that, they hold against a merely mediocre free-tier model.
"""

from __future__ import annotations

from trikon.config import Settings
from trikon.generate.defects import DEFAULT_PLAN, inject_defects
from trikon.generate.world import GeneratorConfig, generate_clean_world
from trikon.llm.adjudicate import _build_user_prompt, _case_id, _validate
from trikon.llm.cache import ResponseCache, cache_key
from trikon.models import RULE_CONFIDENCE_CEILING, AUTO_ACCEPT_THRESHOLD
from trikon.pipeline import AdjudicationOutcome, run_pipeline


def _dirty_world(seed: int = 42, n_orders: int = 300):
    return inject_defects(
        generate_clean_world(GeneratorConfig(seed=seed, n_orders=n_orders)), DEFAULT_PLAN
    )


def _pending_batch(world, limit: int = 3):
    run = run_pipeline(
        world.orders, world.recon_rows, world.settlements, world.bank_credits
    )
    pending = [p for tier in run.tiers.values() for p in tier.escalated]
    return list(enumerate(pending[:limit]))


# ----------------------------------------------------------------------------------
# The central invariant
# ----------------------------------------------------------------------------------


def test_hostile_model_cannot_create_high_confidence_match() -> None:
    """A model accepting everything at 0.999 must not produce an auto-accepted link.

    This is the property the whole design rests on. If it holds, a hallucination can cost
    a reviewer some attention but can never silently declare money reconciled.
    """
    world = _dirty_world()

    def hostile(pending):  # type: ignore[no-untyped-def]
        return [
            AdjudicationOutcome(
                accept=True, confidence=0.999, reasoning="trust me", model="hostile"
            )
            for _ in pending
        ]

    run = run_pipeline(
        world.orders,
        world.recon_rows,
        world.settlements,
        world.bank_credits,
        adjudicator=hostile,
    )
    adjudicated = [link for link in run.all_links if link.adjudicated_by == "hostile"]
    assert adjudicated, "fixture produced no adjudicable candidates"

    for link in adjudicated:
        assert link.confidence <= RULE_CONFIDENCE_CEILING[link.rule]
        assert link.confidence < AUTO_ACCEPT_THRESHOLD
        assert not link.auto_accepted


def test_hostile_model_cannot_introduce_false_positives_into_auto_accepted_set() -> None:
    """The auto-accepted set must be byte-identical with and without a hostile model."""
    world = _dirty_world()

    def hostile(pending):  # type: ignore[no-untyped-def]
        return [
            AdjudicationOutcome(accept=True, confidence=1.0, reasoning="sure", model="h")
            for _ in pending
        ]

    baseline = run_pipeline(
        world.orders, world.recon_rows, world.settlements, world.bank_credits
    )
    attacked = run_pipeline(
        world.orders,
        world.recon_rows,
        world.settlements,
        world.bank_credits,
        adjudicator=hostile,
    )
    for tier in baseline.tiers:
        assert baseline.auto_accepted_links_for(tier) == attacked.auto_accepted_links_for(
            tier
        ), f"{tier}: a model changed the auto-accepted set"


def test_declining_adjudicator_leaves_candidates_escalated() -> None:
    """A model that declines everything must produce the same result as no model at all."""
    world = _dirty_world()

    def declining(pending):  # type: ignore[no-untyped-def]
        return [
            AdjudicationOutcome(accept=False, confidence=0.0, reasoning="unclear", model="d")
            for _ in pending
        ]

    without = run_pipeline(
        world.orders, world.recon_rows, world.settlements, world.bank_credits
    )
    with_model = run_pipeline(
        world.orders,
        world.recon_rows,
        world.settlements,
        world.bank_credits,
        adjudicator=declining,
    )
    assert sum(len(t.escalated) for t in without.tiers.values()) == sum(
        len(t.escalated) for t in with_model.tiers.values()
    )
    for tier in without.tiers:
        assert without.links_for(tier) == with_model.links_for(tier)


def test_crashing_adjudicator_does_not_corrupt_the_run() -> None:
    """An exception inside adjudication must not be swallowed into a wrong match.

    We assert it propagates rather than silently producing a partial result: a caller that
    cannot adjudicate should learn about it, and the CLI turns this into a clear failure
    instead of quietly reporting inflated numbers.
    """
    import pytest

    world = _dirty_world(seed=13, n_orders=200)

    def exploding(pending):  # type: ignore[no-untyped-def]
        raise RuntimeError("provider melted")

    with pytest.raises(RuntimeError, match="provider melted"):
        run_pipeline(
            world.orders,
            world.recon_rows,
            world.settlements,
            world.bank_credits,
            adjudicator=exploding,
        )


# ----------------------------------------------------------------------------------
# Response validation
# ----------------------------------------------------------------------------------


def test_valid_response_is_accepted() -> None:
    batch = _pending_batch(_dirty_world())
    assert batch, "fixture produced no pending candidates"
    payload = {
        "decisions": [
            {
                "case_id": _case_id(i),
                "decision": "ESCALATE",
                "confidence": 0.1,
                "reasoning": "evidence does not settle it",
                "cited_evidence": [candidate.evidence[0].feature],
            }
            for i, candidate in batch
        ]
    }
    assert _validate(payload, batch) is not None


def test_invented_case_id_is_rejected() -> None:
    """A response referencing a case we never sent is discarded wholesale."""
    batch = _pending_batch(_dirty_world())
    payload = {
        "decisions": [
            {"case_id": "CASE99", "decision": "MATCH", "confidence": 0.9, "reasoning": "x"}
        ]
    }
    assert _validate(payload, batch) is None


def test_invented_evidence_feature_is_rejected() -> None:
    """Citing evidence that does not exist invalidates the whole response.

    Rejecting everything rather than just the offending field is deliberate: a response
    that fabricates one detail has not earned trust in the rest of its output.
    """
    batch = _pending_batch(_dirty_world())
    i, _ = batch[0]
    payload = {
        "decisions": [
            {
                "case_id": _case_id(i),
                "decision": "MATCH",
                "confidence": 0.9,
                "reasoning": "x",
                "cited_evidence": ["telepathy"],
            }
        ]
    }
    assert _validate(payload, batch) is None


def test_out_of_enum_decision_is_rejected() -> None:
    batch = _pending_batch(_dirty_world())
    i, _ = batch[0]
    payload = {
        "decisions": [
            {
                "case_id": _case_id(i),
                "decision": "DEFINITELY",
                "confidence": 0.9,
                "reasoning": "x",
            }
        ]
    }
    assert _validate(payload, batch) is None


def test_malformed_payload_is_rejected() -> None:
    batch = _pending_batch(_dirty_world())
    for payload in ({}, {"decisions": []}, {"decisions": "nope"}, {"wrong_key": []}):
        assert _validate(payload, batch) is None  # type: ignore[arg-type]


# ----------------------------------------------------------------------------------
# Prompt hygiene and configuration
# ----------------------------------------------------------------------------------


def test_prompt_contains_only_derived_evidence() -> None:
    """The packet must carry comparisons, not raw source rows.

    Sending whole records would enlarge a metered prompt and hand the model fields it has
    no business reasoning about (card issuer, FX rate) when the arithmetic has already
    been done in Python.
    """
    batch = _pending_batch(_dirty_world())
    prompt = _build_user_prompt(batch)
    for leaked in ("fx_rate_at_creation", "card_issuer", "card_network", "original_amount"):
        assert leaked not in prompt, f"prompt leaked raw field {leaked}"
    assert "precomputed_evidence" in prompt


def test_prompt_is_stable_for_identical_input() -> None:
    """Cache keys depend on prompt stability, so the same batch must render identically."""
    batch = _pending_batch(_dirty_world())
    assert _build_user_prompt(batch) == _build_user_prompt(batch)


def test_blank_api_key_resolves_to_disabled() -> None:
    """``TRIKON_LLM_API_KEY=`` in a .env must mean absent, not an empty-string key."""
    for blank in ("", "   ", "none", "changeme", "your-key-here"):
        settings = Settings(llm_api_key=blank)
        assert settings.llm_api_key is None
        assert not settings.llm_enabled


def test_settings_never_expose_the_key() -> None:
    """``trikon doctor`` must be safe to run on a screen share or in a recording."""
    settings = Settings(llm_api_key="ak-supersecret-value-1234567890")
    redacted = settings.redacted()
    rendered = repr(redacted)
    assert "supersecret" not in rendered
    assert redacted["llm_api_key"] != settings.llm_api_key


def test_zero_budget_disables_the_llm() -> None:
    settings = Settings(llm_api_key="ak-real-looking-key", llm_max_calls_per_run=0)
    assert not settings.llm_enabled


def test_fallback_chain_excludes_the_primary_model() -> None:
    settings = Settings(
        llm_model="glm-5.2", llm_fallback_models="glm-5.2, other-model , ,third"
    )
    assert settings.fallback_models == ("other-model", "third")


# ----------------------------------------------------------------------------------
# Cache
# ----------------------------------------------------------------------------------


def test_cache_round_trip(tmp_path) -> None:  # type: ignore[no-untyped-def]
    cache = ResponseCache(tmp_path / "cache")
    key = cache_key(model="m", system="s", user="u")
    assert cache.get(key) is None
    cache.put(key, {"decisions": []}, model="m")
    assert cache.get(key) == {"decisions": []}
    assert cache.hits == 1 and cache.misses == 1


def test_cache_key_is_sensitive_to_every_field() -> None:
    base = cache_key(model="m", system="s", user="u")
    assert base != cache_key(model="m2", system="s", user="u")
    assert base != cache_key(model="m", system="s2", user="u")
    assert base != cache_key(model="m", system="s", user="u2")
    # Field boundaries must not collide: ("ab","c") and ("a","bc") are different requests.
    assert cache_key(model="ab", system="c", user="u") != cache_key(
        model="a", system="bc", user="u"
    )


def test_corrupt_cache_entry_is_treated_as_a_miss(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """A damaged cache file must cost one API call, not abort the run."""
    directory = tmp_path / "cache"
    cache = ResponseCache(directory)
    key = cache_key(model="m", system="s", user="u")
    (directory / f"{key}.json").write_text("{not json", encoding="utf-8")
    assert cache.get(key) is None


def test_disabled_cache_never_writes(tmp_path) -> None:  # type: ignore[no-untyped-def]
    cache = ResponseCache(tmp_path / "cache", enabled=False)
    key = cache_key(model="m", system="s", user="u")
    cache.put(key, {"decisions": []}, model="m")
    assert cache.get(key) is None
    assert cache.writes == 0
