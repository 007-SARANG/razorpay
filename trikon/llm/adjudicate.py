"""LLM adjudication of ambiguous candidates.

This is the only place a model influences a reconciliation outcome, and it is fenced in
five ways. Each fence exists because of a specific way an LLM can be wrong.

1. **It only sees the residue.** Deterministic rules resolve the overwhelming majority of
   records; only R6/R7 candidates reach a model. On a 2,000-record batch that is a
   handful of cases, so cost and latency scale with *ambiguity*, not with volume. This is
   what makes the system viable on a 50-request/day free tier at all.
2. **It cannot promote.** A decision is clamped to the deterministic rule's confidence
   ceiling (0.70 for a damaged reference, 0.50 for a genuine tie), both here and again in
   the pipeline. Since auto-accept requires 0.90, **no model output can ever produce an
   auto-accepted match.** Every adjudicated link stays flagged for human review. A
   hallucination therefore cannot silently reconcile money.
3. **It does no arithmetic.** All deltas, fee explanations and working-day gaps are
   computed in Python and handed over as facts. The model is asked to weigh evidence, not
   to add up.
4. **It cannot invent records.** Any cited evidence feature or case id absent from the
   packet invalidates the response, which is then discarded in favour of escalation.
5. **Failure escalates.** Unparseable output, a schema violation, a budget stop or a
   network error all resolve to "leave it for a human". There is no path where a model
   problem becomes a silent match.

What the model genuinely adds: judging whether ``INV-202607-00421`` and
``INV20260700421`` are the same reference given that amount and date agree exactly, and
writing the one-line justification a reviewer reads. That is real linguistic work that
would otherwise need a hand-tuned similarity threshold per merchant.
"""

from __future__ import annotations

import json
from typing import Any, Final, Sequence

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from trikon.config import Settings
from trikon.llm.cache import ResponseCache, cache_key
from trikon.llm.provider import BudgetExhausted, LLMProvider
from trikon.models import Decision, MatchRule
from trikon.money import format_inr
from trikon.pipeline import AdjudicationOutcome, PendingCandidate

_SYSTEM_PROMPT: Final[str] = """\
You are a reconciliation adjudicator for an Indian payments finance team.

A deterministic matching engine has already compared candidate record pairs on reference,
amount and settlement timing. It could not decide these cases. Your job is to judge each
one from the evidence supplied and nothing else.

Rules you must follow:
- Do NOT perform arithmetic. Every amount difference, fee explanation and working-day gap
  has already been computed and is given to you as fact.
- Decide MATCH only when the evidence positively supports the two records being the same
  transaction. A plausible guess is not support.
- Decide NO_MATCH when the evidence indicates they are different transactions.
- Decide ESCALATE whenever the evidence does not settle the question, including when two
  candidates are equally consistent. Escalating is a correct, valued answer -- a human
  reviewer will resolve it. Never guess to appear decisive.
- Cite only evidence feature names that appear in the case you were given.
- Reference strings in Indian finance data often carry transcription damage: case
  changes, separators swapped between - / and space, a zero read as the letter O. Such
  damage is compatible with MATCH when the amount agrees exactly.
- A large unexplained amount difference is never a MATCH, however similar the references.

Return only JSON matching the schema. No prose outside the JSON.
"""

_RESPONSE_SCHEMA: Final[dict[str, Any]] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["decisions"],
    "properties": {
        "decisions": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["case_id", "decision", "confidence", "reasoning"],
                "properties": {
                    "case_id": {"type": "string"},
                    "decision": {"type": "string", "enum": ["MATCH", "NO_MATCH", "ESCALATE"]},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "reasoning": {"type": "string", "maxLength": 400},
                    "cited_evidence": {"type": "array", "items": {"type": "string"}},
                },
            },
        }
    },
}


class _DecisionModel(BaseModel):
    """Validated shape of one adjudication decision."""

    model_config = ConfigDict(extra="ignore")

    case_id: str
    decision: Decision
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = ""
    cited_evidence: tuple[str, ...] = ()


class _ResponseModel(BaseModel):
    model_config = ConfigDict(extra="ignore")

    decisions: list[_DecisionModel]


def _case_id(index: int) -> str:
    return f"CASE{index + 1:02d}"


def _render_case(index: int, candidate: PendingCandidate) -> dict[str, Any]:
    """Build the evidence packet for one case.

    Record identifiers are included because a reviewer needs them in the audit trail, but
    no raw source row is sent: the packet carries only the derived comparison. That keeps
    the prompt small (cheap on a metered account) and gives the model nothing to
    misinterpret beyond the facts already established.
    """
    return {
        "case_id": _case_id(index),
        "tier": candidate.tier.value,
        "rule_reached": candidate.rule.value,
        "why_unresolved": (
            "reference is damaged but amount corroborates"
            if candidate.rule is MatchRule.R6_FUZZY_REF
            else "two or more candidates fit equally well"
        ),
        "left_record": {
            "id": candidate.left.record_id,
            "source": candidate.left.source,
            "reference": candidate.left.ref_strict or "(none)",
            "amount": format_inr(candidate.left.amount),
            "date": str(candidate.left.day),
        },
        "right_record": {
            "id": candidate.right.record_id,
            "source": candidate.right.source,
            "reference": candidate.right.ref_strict or "(none)",
            "amount": format_inr(candidate.right.amount),
            "date": str(candidate.right.day),
        },
        "competing_candidates": list(candidate.competing_right_ids),
        "precomputed_evidence": [
            {
                "feature": ev.feature,
                "observed": ev.observed,
                "supports_match": ev.supports,
                "detail": ev.detail or "",
            }
            for ev in candidate.evidence
        ],
    }


def _build_user_prompt(batch: Sequence[tuple[int, PendingCandidate]]) -> str:
    cases = [_render_case(index, candidate) for index, candidate in batch]
    return (
        "Adjudicate each case below. Return one decision per case_id.\n\n"
        + json.dumps({"cases": cases}, indent=2, sort_keys=True)
    )


def _validate(
    payload: dict[str, Any], batch: Sequence[tuple[int, PendingCandidate]]
) -> dict[str, _DecisionModel] | None:
    """Validate a response against the batch it answers.

    Rejects the whole response if the model returned a case id we did not send, or cited
    an evidence feature that does not exist in that case. Both are hallucination signals,
    and a response that fabricates one field has not earned trust in its others.
    """
    try:
        parsed = _ResponseModel.model_validate(payload)
    except ValidationError:
        return None

    valid_ids = {_case_id(index): candidate for index, candidate in batch}
    out: dict[str, _DecisionModel] = {}

    for decision in parsed.decisions:
        candidate = valid_ids.get(decision.case_id)
        if candidate is None:
            return None  # invented a case id
        allowed_features = {ev.feature for ev in candidate.evidence}
        if any(feature not in allowed_features for feature in decision.cited_evidence):
            return None  # invented evidence
        out[decision.case_id] = decision

    return out or None


def adjudicate_batch(
    candidates: Sequence[PendingCandidate],
    *,
    settings: Settings,
    provider: LLMProvider | None = None,
    cache: ResponseCache | None = None,
) -> list[AdjudicationOutcome]:
    """Adjudicate every pending candidate, batching calls.

    Returns one outcome per input candidate, in order. Any candidate the model does not
    resolve -- or that the budget did not reach -- comes back as a non-acceptance, which
    the pipeline turns into an escalation.
    """
    provider = provider or LLMProvider(settings)
    cache = cache or ResponseCache(
        settings.resolve(settings.llm_cache_dir), enabled=settings.llm_cache_enabled
    )

    outcomes: list[AdjudicationOutcome] = [
        AdjudicationOutcome(accept=False, confidence=0.0, reasoning=None, model=None)
        for _ in candidates
    ]

    batch_size = max(1, settings.llm_adjudication_batch_size)
    indexed = list(enumerate(candidates))

    for start in range(0, len(indexed), batch_size):
        batch = indexed[start : start + batch_size]
        user_prompt = _build_user_prompt(batch)
        model_for_key = settings.llm_model
        key = cache_key(model=model_for_key, system=_SYSTEM_PROMPT, user=user_prompt)

        payload = cache.get(key)
        model_used: str | None = "cache" if payload is not None else None

        if payload is None:
            try:
                payload, model_used = provider.complete_json(
                    system=_SYSTEM_PROMPT,
                    user=user_prompt,
                    schema_name="adjudication",
                    schema=_RESPONSE_SCHEMA,
                )
            except BudgetExhausted:
                # Out of budget: everything remaining stays escalated. Deliberately not an
                # error -- a partially adjudicated run is still a valid, honest result.
                break
            if payload is not None and model_used is not None:
                cache.put(key, payload, model=model_used, note=f"{len(batch)} cases")

        if payload is None:
            continue

        decisions = _validate(payload, batch)
        if decisions is None:
            continue

        for index, candidate in batch:
            decision = decisions.get(_case_id(index))
            if decision is None:
                continue
            accept = decision.decision is Decision.MATCH
            # Clamp to the deterministic ceiling. Enforced here as well as in the pipeline
            # so that neither layer alone is load-bearing for the invariant.
            confidence = min(decision.confidence, candidate.ceiling) if accept else 0.0
            outcomes[index] = AdjudicationOutcome(
                accept=accept,
                confidence=confidence,
                reasoning=decision.reasoning or None,
                model=model_used,
            )

    return outcomes


def build_adjudicator(settings: Settings):
    """Return a callable suitable for ``run_pipeline(adjudicator=...)``.

    The provider and cache are created once and closed over, so a single run reuses one
    budget and one cache rather than resetting them per batch.
    """
    provider = LLMProvider(settings)
    cache = ResponseCache(
        settings.resolve(settings.llm_cache_dir), enabled=settings.llm_cache_enabled
    )

    def _adjudicate(candidates: Sequence[PendingCandidate]) -> list[AdjudicationOutcome]:
        return adjudicate_batch(
            candidates, settings=settings, provider=provider, cache=cache
        )

    _adjudicate.provider = provider  # type: ignore[attr-defined]
    _adjudicate.cache = cache  # type: ignore[attr-defined]
    return _adjudicate
