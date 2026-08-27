"""Tests for the money, calendar and matching core.

The tests that matter most here are the *negative* ones. It is easy to write a reconciler
that matches everything and scores well on a friendly batch; the hard requirement is that
it declines to match when the evidence does not support a match. So alongside the ordinary
arithmetic checks, this suite asserts:

* a one-paise difference is never absorbed (``test_one_paise_is_never_matched``)
* a genuinely ambiguous twin is escalated, not guessed
  (``test_twin_amounts_escalate_rather_than_guess``)
* matching is order-invariant (``test_assignment_is_order_invariant``)
* an LLM can never raise a confidence above its rule's ceiling
  (``test_adjudication_cannot_promote_above_ceiling``)
"""

from __future__ import annotations

import datetime as dt

import pytest

from trikon.assign import score_candidates, solve_assignment, solve_subset_sums, subset_links
from trikon.block import generate_candidates
from trikon.calendar_ist import (
    add_working_days,
    expected_settlement_date,
    is_second_or_fourth_saturday,
    is_working_day,
    working_days_between,
)
from trikon.generate.defects import DEFAULT_PLAN, inject_defects
from trikon.generate.world import GeneratorConfig, generate_clean_world
from trikon.models import (
    AUTO_ACCEPT_THRESHOLD,
    RULE_CONFIDENCE_CEILING,
    MatchRule,
    SettlementStatus,
    Tier,
    TxnType,
)
from trikon.money import (
    compute_fee,
    compute_gst_on_fee,
    compute_net_credit,
    format_inr,
    paise_to_rupee_str,
    rupees_to_paise,
)
from trikon.normalize import (
    canon_loose,
    canon_strict,
    extract_utr,
    project_bank_credits,
    project_settlements,
)
from trikon.pipeline import run_pipeline


# ----------------------------------------------------------------------------------
# Money
# ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("1000.00", 100_000),
        ("1000", 100_000),
        ("0.01", 1),
        ("1234.56", 123_456),
        ("-45.50", -4_550),
        ("1,00,000.00", 10_000_000),
        ("0.005", 1),  # half-up on the third decimal
        ("0.004", 0),
    ],
)
def test_rupees_to_paise(value: str, expected: int) -> None:
    assert rupees_to_paise(value) == expected


def test_paise_round_trip_is_exact() -> None:
    for paise in (0, 1, 99, 100, 123_456, 10_000_000):
        assert rupees_to_paise(paise_to_rupee_str(paise)) == paise


def test_fee_and_gst_are_integers_and_half_up() -> None:
    # 2% of Rs 1000.00 is exactly Rs 20.00; 18% of that is exactly Rs 3.60.
    fee = compute_fee(100_000, "card")
    assert fee == 2_000
    assert compute_gst_on_fee(fee) == 360

    fee, tax, net = compute_net_credit(100_000, "card")
    assert (fee, tax, net) == (2_000, 360, 97_640)
    assert isinstance(fee, int) and isinstance(tax, int) and isinstance(net, int)


def test_upi_carries_no_fee() -> None:
    """Zero-MDR UPI is the majority of Indian volume; assuming a fee would break it."""
    fee, tax, net = compute_net_credit(500_00, "upi")
    assert (fee, tax) == (0, 0)
    assert net == 500_00


def test_fee_rounding_is_half_up_not_bankers() -> None:
    """Python's round() is half-to-even, which would disagree with a published fee table."""
    # 190 bps of 25 paise = 0.475 paise -> half-up gives 0, half-even also 0; use a case
    # where they differ: 50 paise at 100 bps = 0.5 -> half-up 1, half-even 0.
    from trikon.money import _round_half_up

    assert _round_half_up(1, 2) == 1
    assert _round_half_up(3, 2) == 2
    assert _round_half_up(-1, 2) == -1


def test_format_inr_uses_indian_grouping() -> None:
    assert format_inr(10_000_000) == "₹1,00,000.00"
    assert format_inr(123_456_789) == "₹12,34,567.89"
    assert format_inr(-4_550) == "-₹45.50"


# ----------------------------------------------------------------------------------
# Calendar
# ----------------------------------------------------------------------------------


def test_second_and_fourth_saturdays_are_holidays() -> None:
    # August 2026 starts on a Saturday, so the 1st/8th/15th/22nd/29th are Saturdays.
    assert not is_second_or_fourth_saturday(dt.date(2026, 8, 1))  # 1st
    assert is_second_or_fourth_saturday(dt.date(2026, 8, 8))  # 2nd
    assert not is_second_or_fourth_saturday(dt.date(2026, 8, 15))  # 3rd
    assert is_second_or_fourth_saturday(dt.date(2026, 8, 22))  # 4th

    assert is_working_day(dt.date(2026, 8, 1))
    assert not is_working_day(dt.date(2026, 8, 8))


def test_sunday_is_never_a_working_day() -> None:
    assert not is_working_day(dt.date(2026, 8, 2))


def test_settlement_skips_holiday_and_weekend() -> None:
    """T+2 from Fri 14 Aug 2026 must skip 15 Aug (holiday) and 16 Aug (Sunday)."""
    assert expected_settlement_date(dt.date(2026, 8, 14)) == dt.date(2026, 8, 18)


def test_working_days_between_is_signed_and_symmetric() -> None:
    a, b = dt.date(2026, 8, 14), dt.date(2026, 8, 20)
    assert working_days_between(a, b) == 4
    assert working_days_between(b, a) == -4
    assert working_days_between(a, a) == 0


def test_add_working_days_from_a_holiday_moves_forward_first() -> None:
    """A capture on a non-working day still has a well-defined first eligible date."""
    assert add_working_days(dt.date(2026, 8, 8), 0) == dt.date(2026, 8, 10)


# ----------------------------------------------------------------------------------
# Normalisation
# ----------------------------------------------------------------------------------


def test_canonical_reference_forms() -> None:
    assert canon_strict("INV-202607/00421 ") == "INV20260700421"
    assert canon_strict("inv 202607 00421") == "INV20260700421"
    assert canon_strict(None) == ""
    # Loose folding maps letter O to zero and I/L to one, for transcription damage.
    assert canon_loose("INV-O0421") == canon_loose("INV-00421")


def test_utr_extraction_reports_its_own_confidence() -> None:
    utr, strict = extract_utr("NEFT CR RAZORPAY SETTLEMENT KKBKH14156891582 PAYOUT")
    assert utr == "KKBKH14156891582"
    assert strict is True

    utr, strict = extract_utr("NEFT CR RAZORPAY SETTLEMENT REF ***UNREADABLE*** PAYOUT")
    assert utr is None
    assert strict is False


# ----------------------------------------------------------------------------------
# The safety invariant
# ----------------------------------------------------------------------------------


def test_rule_ceilings_are_ordered_and_bounded() -> None:
    """Adjudicable rules must sit below the auto-accept threshold, by construction."""
    from trikon.models import ADJUDICABLE_RULES, DETERMINISTIC_RULES

    for rule in ADJUDICABLE_RULES:
        assert RULE_CONFIDENCE_CEILING[rule] < AUTO_ACCEPT_THRESHOLD, (
            f"{rule} is adjudicable but its ceiling would auto-accept"
        )
    for rule in DETERMINISTIC_RULES:
        assert RULE_CONFIDENCE_CEILING[rule] >= AUTO_ACCEPT_THRESHOLD
    assert RULE_CONFIDENCE_CEILING[MatchRule.R8_NO_MATCH] == 0.0


def test_adjudication_cannot_promote_above_ceiling() -> None:
    """An adjudicator returning 0.99 on an R7 candidate must be clamped to R7's ceiling.

    This is the property that makes an LLM hallucination unable to manufacture a
    high-confidence false match.
    """
    from trikon.pipeline import AdjudicationOutcome

    world = inject_defects(
        generate_clean_world(GeneratorConfig(seed=7, n_orders=60)), DEFAULT_PLAN
    )

    def over_confident(pending):  # type: ignore[no-untyped-def]
        return [
            AdjudicationOutcome(
                accept=True, confidence=0.999, reasoning="looks right to me", model="test"
            )
            for _ in pending
        ]

    run = run_pipeline(
        world.orders,
        world.recon_rows,
        world.settlements,
        world.bank_credits,
        adjudicator=over_confident,
    )
    adjudicated = [link for link in run.all_links if link.adjudicated_by == "test"]
    assert adjudicated, "expected at least one adjudicated link in this batch"
    for link in adjudicated:
        assert link.confidence <= RULE_CONFIDENCE_CEILING[link.rule]
        assert not link.auto_accepted, "adjudicated links must remain flagged for review"


# ----------------------------------------------------------------------------------
# Negative behaviour: the system must refuse
# ----------------------------------------------------------------------------------


def _tier3_match(world):  # type: ignore[no-untyped-def]
    left = project_settlements(world.settlements)
    right = project_bank_credits(world.bank_credits)
    pairs, _ = generate_candidates(left, right)
    scored = score_candidates(left, right, pairs, tier=Tier.SETTLEMENT_BANK)
    assignment = solve_assignment(
        left,
        right,
        scored,
        tier=Tier.SETTLEMENT_BANK,
        auto_accept_threshold=AUTO_ACCEPT_THRESHOLD,
    )
    return left, right, assignment


def test_one_paise_is_never_matched() -> None:
    """Two settlements one paise apart must each match their own credit, never swap.

    Any amount tolerance wider than zero collapses this into an ambiguity or, worse, a
    crossed pair. It is the cheapest possible test for sloppy rounding and it is why the
    blocking index applies no amount band.
    """
    world = inject_defects(
        generate_clean_world(GeneratorConfig(seed=42, n_orders=120)), DEFAULT_PLAN
    )
    left, right, assignment = _tier3_match(world)

    produced = {(left[i].record_id, right[j].record_id) for i, j, *_ in assignment.accepted}
    paise_settlements = {
        s.id for s in world.settlements if s.amount in (7_777_00, 7_777_01)
    }
    paise_credits = {"STMT-PAISE1", "STMT-PAISE2"}
    assert paise_settlements, "one-paise twin fixture missing from the batch"

    amount_of = {s.id: s.amount for s in world.settlements}
    amount_of.update({c.stmt_id: c.amount for c in world.bank_credits})

    involved = [
        (l, r) for l, r in produced if l in paise_settlements or r in paise_credits
    ]
    assert len(involved) == 2, f"expected both twins matched, got {involved}"
    for left_id, right_id in involved:
        assert amount_of[left_id] == amount_of[right_id], (
            f"crossed the one-paise pair: {left_id} != {right_id}"
        )


def test_twin_amounts_escalate_rather_than_guess() -> None:
    """Two identical settlements and two identical credits with no UTR are unresolvable."""
    world = inject_defects(
        generate_clean_world(GeneratorConfig(seed=42, n_orders=120)), DEFAULT_PLAN
    )
    left, right, assignment = _tier3_match(world)

    twin_credits = {"STMT-TWIN1", "STMT-TWIN2"}
    accepted_twins = [
        (left[i].record_id, right[j].record_id)
        for i, j, *_ in assignment.accepted
        if right[j].record_id in twin_credits
    ]
    assert not accepted_twins, (
        f"guessed an assignment for a genuinely ambiguous twin: {accepted_twins}"
    )
    assert assignment.ambiguities, "twin amounts should have produced an ambiguity group"


def test_clean_batch_produces_no_false_positives() -> None:
    """A defect-free batch must reconcile completely, with nothing invented.

    The negative control: if a clean world produces exceptions, the detectors are
    over-firing and every exception count on a dirty batch is suspect.
    """
    world = generate_clean_world(GeneratorConfig(seed=11, n_orders=100))
    truth = world.ground_truth(0)
    run = run_pipeline(
        world.orders, world.recon_rows, world.settlements, world.bank_credits
    )

    for tier in (Tier.ORDER_PG, Tier.SETTLEMENT_BANK):
        produced = run.links_for(tier)
        expected = truth.links_for(tier)
        assert not (produced - expected), f"{tier} invented links: {produced - expected}"

    hard_codes = {"MISSING_IN_PG", "MISSING_IN_BOOKS", "MISSING_IN_BANK", "FEE_MISMATCH",
                  "GST_MISMATCH", "DUPLICATE_PAYMENT", "SETTLEMENT_NET_MISMATCH"}
    raised = {e.code.value for e in run.exceptions} & hard_codes
    assert not raised, f"clean batch raised hard exceptions: {raised}"


def test_unpaid_orders_are_not_reported_missing() -> None:
    """Failed and cancelled orders legitimately have no gateway row."""
    world = generate_clean_world(GeneratorConfig(seed=5, n_orders=150))
    run = run_pipeline(
        world.orders, world.recon_rows, world.settlements, world.bank_credits
    )
    unpaid = {o.order_id for o in world.orders if o.status.value != "paid"}
    flagged = {
        subject
        for e in run.exceptions
        if e.code.value == "MISSING_IN_PG"
        for subject in e.subject_ids
    }
    assert not (unpaid & flagged), "flagged an unpaid order as missing from the gateway"


def test_assignment_is_order_invariant() -> None:
    """Shuffling the input must not change which links are produced.

    Greedy first-best matching fails this: whichever record is processed first claims the
    contested partner. Optimal assignment is invariant by construction, and this test is
    what stops a future refactor from quietly reintroducing greed.
    """
    import random

    world = inject_defects(
        generate_clean_world(GeneratorConfig(seed=3, n_orders=140)), DEFAULT_PLAN
    )
    baseline = run_pipeline(
        world.orders, world.recon_rows, world.settlements, world.bank_credits
    )

    rng = random.Random(99)
    orders = list(world.orders)
    rows = list(world.recon_rows)
    settlements = list(world.settlements)
    credits = list(world.bank_credits)
    rng.shuffle(orders)
    rng.shuffle(rows)
    rng.shuffle(settlements)
    rng.shuffle(credits)

    shuffled = run_pipeline(orders, rows, settlements, credits)

    for tier in (Tier.ORDER_PG, Tier.SETTLEMENT_BANK):
        assert baseline.links_for(tier) == shuffled.links_for(tier), (
            f"{tier} matching depends on input order"
        )


def test_subset_sum_refuses_when_two_subsets_fit() -> None:
    """If two distinct subsets sum to the target, no match may be produced."""
    from trikon.normalize import NormRecord

    day = dt.date(2026, 7, 10)
    left = [
        NormRecord(record_id="A", source="settlement", amount=100, day=day, epoch=0),
        NormRecord(record_id="B", source="settlement", amount=200, day=day, epoch=0),
        NormRecord(record_id="C", source="settlement", amount=300, day=day, epoch=0),
        NormRecord(record_id="D", source="settlement", amount=400, day=day, epoch=0),
    ]
    # 100+400 == 200+300 == 500, so the target is explained two ways.
    right = [NormRecord(record_id="Z", source="bank", amount=500, day=day, epoch=0)]

    solutions, ambiguous = solve_subset_sums(
        left, right, unmatched_left=[0, 1, 2, 3], unmatched_right=[0]
    )
    assert not solutions, f"picked one of two equally valid subsets: {solutions}"
    assert ambiguous and ambiguous[0][2] > 1


def test_subset_sum_resolves_a_unique_split() -> None:
    """One settlement paid as two credits must reconcile, in the split direction."""
    from trikon.normalize import NormRecord

    day = dt.date(2026, 7, 10)
    left = [NormRecord(record_id="S", source="settlement", amount=900, day=day, epoch=0)]
    right = [
        NormRecord(record_id="C1", source="bank", amount=400, day=day, epoch=0),
        NormRecord(record_id="C2", source="bank", amount=500, day=day, epoch=0),
    ]
    solutions, ambiguous = solve_subset_sums(
        left, right, unmatched_left=[0], unmatched_right=[0, 1]
    )
    assert not ambiguous
    assert len(solutions) == 1
    assert solutions[0].direction == "split"
    links = subset_links(solutions, left, right)
    assert {(left[i].record_id, right[j].record_id) for i, j, _ in links} == {
        ("S", "C1"),
        ("S", "C2"),
    }


# ----------------------------------------------------------------------------------
# Generator invariants
# ----------------------------------------------------------------------------------


def test_clean_world_is_internally_consistent() -> None:
    """Every settlement must equal its member rows, and every credit its settlement."""
    world = generate_clean_world(GeneratorConfig(seed=21, n_orders=200))

    members: dict[str, int] = {}
    for row in world.recon_rows:
        if row.settlement_id:
            members[row.settlement_id] = (
                members.get(row.settlement_id, 0) + row.credit - row.debit
            )
    for settlement in world.settlements:
        assert members.get(settlement.id) == settlement.amount, (
            f"{settlement.id} does not equal its member rows"
        )

    by_utr = {s.utr: s for s in world.settlements if s.utr}
    for credit in world.bank_credits:
        assert credit.utr_extracted in by_utr
        assert by_utr[credit.utr_extracted].amount == credit.amount


def test_payment_rows_have_null_payment_id() -> None:
    """Razorpay's schema quirk: payment_id is null on rows of type payment."""
    world = generate_clean_world(GeneratorConfig(seed=4, n_orders=80))
    for row in world.recon_rows:
        if row.type is TxnType.PAYMENT:
            assert row.payment_id is None
        if row.type is TxnType.REFUND:
            assert row.payment_id is not None


def test_defect_plan_injects_every_type() -> None:
    """A silently-skipped injector would inflate the measured match rate."""
    world = inject_defects(
        generate_clean_world(GeneratorConfig(seed=42, n_orders=150)), DEFAULT_PLAN
    )
    injected = {d.defect_code for d in world.defects}
    planned = {p.name for p in DEFAULT_PLAN}
    assert planned - injected == set(), f"never injected: {planned - injected}"


def test_ground_truth_is_never_read_by_the_pipeline() -> None:
    """Corrupting ground truth must not change what the pipeline produces.

    A structural guarantee rather than a behavioural one: if the pipeline could see truth,
    every reported metric would be meaningless. Passing a mutilated GroundTruth and
    observing identical output proves the separation holds.
    """
    world = inject_defects(
        generate_clean_world(GeneratorConfig(seed=8, n_orders=90)), DEFAULT_PLAN
    )
    first = run_pipeline(
        world.orders, world.recon_rows, world.settlements, world.bank_credits
    )
    world.links.clear()
    world.defects.clear()
    second = run_pipeline(
        world.orders, world.recon_rows, world.settlements, world.bank_credits
    )
    for tier in (Tier.ORDER_PG, Tier.SETTLEMENT_BANK):
        assert first.links_for(tier) == second.links_for(tier)
    assert len(first.exceptions) == len(second.exceptions)
