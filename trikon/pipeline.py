"""The reconciliation pipeline: a fixed, deterministic orchestration.

This is the "agent" in the sense that matters -- a controller that ingests three sources,
decides what it can prove, escalates what it cannot, and produces an auditable account of
both. It is deliberately **not** an LLM deciding what to do next.

Why a fixed DAG rather than a model-driven loop: the stage order here is not a judgement
call, it is forced by data dependencies. Tier-2 arithmetic needs settlement membership;
presence exceptions need to know what matching left over; the cash position needs the
settled flags. A model asked to sequence these would either reproduce this order or get it
wrong, and its choice would vary between runs -- which would make the reported metrics
unreproducible. Reproducibility is the product, so orchestration is code.

The LLM's role is confined to :mod:`trikon.llm.adjudicate`, invoked on the small residue
of genuinely ambiguous candidates, and it can never promote a match above its rule's
confidence ceiling. With no API key configured the residue is escalated to human review
instead and every metric is still produced -- see :class:`RunResult.llm_used`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Sequence

from trikon.assign import (
    AssignmentResult,
    ScoredPair,
    score_candidates,
    solve_assignment,
    solve_subset_sums,
    subset_links,
)
from trikon.block import BlockingStats, generate_candidates
from trikon.classify import (
    CashPosition,
    ReviewCase,
    build_review_cases,
    compute_cash_position,
    detect_duplicates,
    detect_lifecycle_exceptions,
    detect_missing_in_bank,
    detect_missing_in_books,
    detect_missing_in_pg,
    detect_timing_breaches,
    detect_unexplained_bank_credits,
    new_builder,
    pair_residue_by_variance,
    report_ambiguity,
    verify_fees_and_tax,
    verify_fx_conversion,
    verify_settlement_arithmetic,
)
from trikon.models import (
    AUTO_ACCEPT_THRESHOLD,
    BankCredit,
    Evidence,
    ExceptionRecord,
    MatchLink,
    MatchRule,
    Order,
    ReconRow,
    Settlement,
    Tier,
)
from trikon.normalize import (
    NormRecord,
    project_bank_credits,
    project_orders,
    project_pg_payments,
    project_settlements,
)

#: Signature of the optional adjudicator. Takes the pending candidates and returns, for
#: each, a decision plus optional reasoning. Injected rather than imported so the pipeline
#: has no hard dependency on the LLM layer and stays runnable with no provider at all.
Adjudicator = Callable[
    [Sequence["PendingCandidate"]], "Sequence[AdjudicationOutcome]"
]


@dataclass(frozen=True, slots=True)
class PendingCandidate:
    """A candidate that deterministic rules could not settle either way."""

    tier: Tier
    left: NormRecord
    right: NormRecord
    rule: MatchRule
    ceiling: float
    evidence: tuple[Evidence, ...]
    competing_right_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class AdjudicationOutcome:
    """Result of adjudicating one pending candidate."""

    accept: bool
    confidence: float
    reasoning: str | None
    model: str | None


@dataclass
class TierResult:
    """Matching outcome for one tier."""

    tier: Tier
    links: list[MatchLink] = field(default_factory=list)
    escalated: list[PendingCandidate] = field(default_factory=list)
    unmatched_left_ids: list[str] = field(default_factory=list)
    unmatched_right_ids: list[str] = field(default_factory=list)
    blocking: BlockingStats | None = None
    candidates_scored: int = 0
    subset_merges: int = 0
    subset_splits: int = 0
    subset_ambiguous: int = 0

    @property
    def auto_accepted(self) -> int:
        return sum(1 for link in self.links if link.auto_accepted)


@dataclass
class RunResult:
    """Everything one reconciliation run produced."""

    record_count: int
    source_counts: dict[str, int]
    tiers: dict[Tier, TierResult] = field(default_factory=dict)
    exceptions: list[ExceptionRecord] = field(default_factory=list)
    cases: list[ReviewCase] = field(default_factory=list)
    cash: CashPosition | None = None
    timings_ms: dict[str, float] = field(default_factory=dict)
    llm_used: bool = False
    llm_calls: int = 0
    adjudicated: int = 0

    @property
    def all_links(self) -> list[MatchLink]:
        return [link for tier in self.tiers.values() for link in tier.links]

    @property
    def total_ms(self) -> float:
        return sum(self.timings_ms.values())

    @property
    def throughput_per_second(self) -> float:
        seconds = self.total_ms / 1000.0
        return self.record_count / seconds if seconds > 0 else 0.0

    def links_for(self, tier: Tier) -> frozenset[tuple[str, str]]:
        """Produced link pairs for a tier, for comparison against ground truth."""
        result = self.tiers.get(tier)
        if result is None:
            return frozenset()
        return frozenset((link.left_id, link.right_id) for link in result.links)

    def auto_accepted_links_for(self, tier: Tier) -> frozenset[tuple[str, str]]:
        """Only the links accepted without human review."""
        result = self.tiers.get(tier)
        if result is None:
            return frozenset()
        return frozenset(
            (link.left_id, link.right_id) for link in result.links if link.auto_accepted
        )


class _Stopwatch:
    """Records per-stage wall time so throughput can be attributed, not just totalled."""

    def __init__(self) -> None:
        self.timings: dict[str, float] = {}

    def time(self, label: str, fn: Callable[[], object]) -> object:
        start = time.perf_counter()
        try:
            return fn()
        finally:
            self.timings[label] = self.timings.get(label, 0.0) + (
                time.perf_counter() - start
            ) * 1000.0


def _match_tier(
    tier: Tier,
    left: Sequence[NormRecord],
    right: Sequence[NormRecord],
    *,
    auto_accept_threshold: float,
    enable_subset_sum: bool,
) -> tuple[TierResult, AssignmentResult, list[ScoredPair]]:
    """Run blocking, scoring, assignment and (optionally) subset-sum for one tier."""
    result = TierResult(tier=tier)

    pairs, blocking = generate_candidates(left, right)
    result.blocking = blocking

    scored = score_candidates(left, right, pairs, tier=tier)
    result.candidates_scored = len(scored)

    assignment = solve_assignment(
        left, right, scored, tier=tier, auto_accept_threshold=auto_accept_threshold
    )

    for i, j, rule, confidence, evidence in assignment.accepted:
        result.links.append(
            MatchLink(
                tier=tier,
                left_id=left[i].record_id,
                right_id=right[j].record_id,
                rule=rule,
                confidence=confidence,
                auto_accepted=True,
                evidence=evidence,
            )
        )

    competing: dict[int, tuple[str, ...]] = {
        li: tuple(right[j].record_id for j in rivals)
        for li, rivals in assignment.ambiguities
    }
    for i, j, rule, confidence, evidence in assignment.for_adjudication:
        result.escalated.append(
            PendingCandidate(
                tier=tier,
                left=left[i],
                right=right[j],
                rule=rule,
                ceiling=confidence,
                evidence=evidence,
                competing_right_ids=competing.get(i, ()),
            )
        )

    unmatched_left = list(assignment.unmatched_left)
    unmatched_right = list(assignment.unmatched_right)

    if enable_subset_sum:
        solutions, ambiguous = solve_subset_sums(
            left, right, unmatched_left=unmatched_left, unmatched_right=unmatched_right
        )
        result.subset_merges = sum(1 for s in solutions if s.direction == "merge")
        result.subset_splits = sum(1 for s in solutions if s.direction == "split")
        result.subset_ambiguous = len(ambiguous)

        consumed_left: set[int] = set()
        consumed_right: set[int] = set()
        for sol in solutions:
            members = (
                [(i, sol.one_index) for i in sol.many_indices]
                if sol.direction == "merge"
                else [(sol.one_index, j) for j in sol.many_indices]
            )
            group_ids = tuple(
                (left[i].record_id if sol.direction == "merge" else right[j].record_id)
                for i, j in members
            )
            for i, j in members:
                consumed_left.add(i)
                consumed_right.add(j)
                result.links.append(
                    MatchLink(
                        tier=tier,
                        left_id=left[i].record_id,
                        right_id=right[j].record_id,
                        rule=MatchRule.R5_SUBSET_SUM,
                        confidence=0.95,
                        auto_accepted=True,
                        evidence=(
                            Evidence(
                                feature="subset_sum",
                                observed=f"{sol.direction}: {len(members)} records sum "
                                f"exactly to the counterpart",
                                supports=True,
                                detail="Unique exact subset; no alternative subset "
                                "reproduces this total.",
                            ),
                        ),
                        member_ids=group_ids,
                    )
                )
        unmatched_left = [i for i in unmatched_left if i not in consumed_left]
        unmatched_right = [j for j in unmatched_right if j not in consumed_right]

    result.unmatched_left_ids = [left[i].record_id for i in unmatched_left]
    result.unmatched_right_ids = [right[j].record_id for j in unmatched_right]
    return result, assignment, scored


def run_pipeline(
    orders: Sequence[Order],
    rows: Sequence[ReconRow],
    settlements: Sequence[Settlement],
    bank_credits: Sequence[BankCredit],
    *,
    adjudicator: Adjudicator | None = None,
    auto_accept_threshold: float = AUTO_ACCEPT_THRESHOLD,
) -> RunResult:
    """Reconcile three sources and return matches, exceptions, cases and cash position.

    ``adjudicator`` is optional. When absent, ambiguous candidates are escalated to the
    review queue rather than resolved -- which is the correct conservative default and
    keeps the whole pipeline runnable with no provider configured.
    """
    watch = _Stopwatch()

    n_orders = project_orders(orders)
    n_pg = project_pg_payments(rows)
    n_settle = project_settlements(settlements)
    n_bank = project_bank_credits(bank_credits)

    result = RunResult(
        record_count=len(orders) + len(rows) + len(settlements) + len(bank_credits),
        source_counts={
            "orders": len(orders),
            "recon_rows": len(rows),
            "settlements": len(settlements),
            "bank_credits": len(bank_credits),
        },
    )

    # --- Tier 1: order ledger against gateway payments --------------------------------
    tier1, assign1, _ = watch.time(  # type: ignore[assignment]
        "tier1_match",
        lambda: _match_tier(
            Tier.ORDER_PG,
            n_orders,
            n_pg,
            auto_accept_threshold=auto_accept_threshold,
            enable_subset_sum=False,
        ),
    )
    result.tiers[Tier.ORDER_PG] = tier1

    # --- Tier 3: settlements against bank credits -------------------------------------
    tier3, assign3, _ = watch.time(  # type: ignore[assignment]
        "tier3_match",
        lambda: _match_tier(
            Tier.SETTLEMENT_BANK,
            n_settle,
            n_bank,
            auto_accept_threshold=auto_accept_threshold,
            enable_subset_sum=True,
        ),
    )
    result.tiers[Tier.SETTLEMENT_BANK] = tier3

    # --- Adjudication of the ambiguous residue ----------------------------------------
    pending = [*tier1.escalated, *tier3.escalated]
    if adjudicator is not None and pending:

        def _adjudicate() -> None:
            outcomes = adjudicator(pending)
            for candidate, outcome in zip(pending, outcomes):
                result.adjudicated += 1
                if not outcome.accept:
                    continue
                # Safety invariant: adjudication may confirm within the rule's band or
                # decline, but never promote above the deterministic ceiling.
                confidence = min(outcome.confidence, candidate.ceiling)
                tier_result = result.tiers[candidate.tier]
                tier_result.links.append(
                    MatchLink(
                        tier=candidate.tier,
                        left_id=candidate.left.record_id,
                        right_id=candidate.right.record_id,
                        rule=candidate.rule,
                        confidence=confidence,
                        auto_accepted=False,
                        evidence=candidate.evidence,
                        adjudicated_by=outcome.model,
                        reasoning=outcome.reasoning,
                    )
                )
                tier_result.escalated = [
                    p for p in tier_result.escalated if p is not candidate
                ]
                if candidate.left.record_id in tier_result.unmatched_left_ids:
                    tier_result.unmatched_left_ids.remove(candidate.left.record_id)
                if candidate.right.record_id in tier_result.unmatched_right_ids:
                    tier_result.unmatched_right_ids.remove(candidate.right.record_id)

        watch.time("adjudication", _adjudicate)
        result.llm_used = True

    # --- Exception detection -----------------------------------------------------------
    builder = new_builder()
    order_by_id = {o.order_id: o for o in orders}
    row_by_id = {r.entity_id: r for r in rows}
    settlement_by_id = {s.id: s for s in settlements}
    credit_by_id = {c.stmt_id: c for c in bank_credits}

    def _classify() -> None:
        # Tier 2 is pure verification and needs no matching at all.
        verify_settlement_arithmetic(settlements, rows, builder)
        verify_fees_and_tax(rows, builder)
        verify_fx_conversion(rows, builder)

        # A matched-but-late payout is still a finding; matching it must not absolve it.
        detect_timing_breaches(
            result.tiers[Tier.SETTLEMENT_BANK].links,
            settlement_by_id,
            credit_by_id,
            builder,
        )

        # Presence failures, from whatever matching left over. Residue pairing runs first:
        # it claims settlement/credit pairs that are evidently the same payout recorded
        # with different amounts, so they are reported once as a variance rather than
        # twice as "missing" plus "unexplained".
        residue_settlements = [
            settlement_by_id[i] for i in tier3.unmatched_left_ids if i in settlement_by_id
        ]
        residue_credits = [
            credit_by_id[i] for i in tier3.unmatched_right_ids if i in credit_by_id
        ]
        paired_settlements, paired_credits = pair_residue_by_variance(
            residue_settlements, residue_credits, builder
        )

        detect_missing_in_pg(
            [order_by_id[i] for i in tier1.unmatched_left_ids if i in order_by_id], builder
        )
        detect_missing_in_books(
            [row_by_id[i] for i in tier1.unmatched_right_ids if i in row_by_id],
            builder,
            known_order_ids=frozenset(order_by_id),
        )
        detect_missing_in_bank(
            [s for s in residue_settlements if s.id not in paired_settlements], builder
        )
        detect_unexplained_bank_credits(
            [c for c in residue_credits if c.stmt_id not in paired_credits], builder
        )

        detect_duplicates(rows, builder)
        detect_lifecycle_exceptions(settlements, rows, builder)

        # Anything still escalated after adjudication is a genuine refusal to decide.
        for candidate in [*tier1.escalated, *tier3.escalated]:
            report_ambiguity(
                subject_id=candidate.left.record_id,
                candidate_ids=candidate.competing_right_ids
                or (candidate.right.record_id,),
                amount=candidate.left.amount,
                tier=candidate.tier,
                builder=builder,
                detail=f"Rule {candidate.rule.value} reached only "
                f"{candidate.ceiling:.2f} confidence, below the "
                f"{auto_accept_threshold:.2f} auto-accept threshold.",
            )

    watch.time("classify", _classify)
    result.exceptions = builder.collect()
    result.cases = watch.time("review_cases", lambda: build_review_cases(result.exceptions))  # type: ignore[assignment]
    result.cash = watch.time("cash_position", lambda: compute_cash_position(rows, settlements))  # type: ignore[assignment]

    result.timings_ms = watch.timings
    return result
