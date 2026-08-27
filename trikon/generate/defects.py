"""Deliberate defect injection, with ground truth recorded for every defect.

This is the module that makes the evaluation trustworthy. Each injector mutates the
clean world and returns an :class:`~trikon.models.InjectedDefect` describing what it
did and what the reconciler is expected to conclude. Nothing here is visible to the
reconciliation pipeline.

**The ground-truth contract**, which the evaluator relies on:

* A **true link is preserved** whenever the two records genuinely correspond, even if
  one of them is now corrupted. Identifying the pair and *flagging the variance* are
  separate skills, and we measure them separately -- a system that escapes a broken
  pair into the review queue should get credit for finding it, not be punished for
  refusing to bless it.
* ``expected_exception is None`` means the defect is **absorbable**: a competent
  reconciler should still match through it (a mutated receipt number, a split bank
  credit) and raise nothing. These are the false-positive traps.
* ``expected_exception is not None`` means the defect **must be reported** under that
  code, and the link must **not** be auto-accepted at high confidence. These are the
  false-negative traps.

Two injectors exist purely as adversarial controls and are expected to be resolved
*correctly*, not escalated: ``MUTATE_RECEIPT`` and ``ONE_PAISE_TWIN``. The latter in
particular will silently pass on any implementation that carries a sloppy amount
tolerance, which is exactly why it is in the suite.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Final

from trikon.calendar_ist import epoch_to_ist_date, ist_date_to_epoch
from trikon.generate.world import World, _Ids as Ids
from trikon.models import (
    BankCredit,
    ExceptionCode,
    InjectedDefect,
    Method,
    ReconRow,
    Settlement,
    SettlementStatus,
    Tier,
    TrueLink,
    TxnType,
)

Injector = Callable[[World, random.Random, Ids], InjectedDefect | None]


@dataclass(frozen=True)
class DefectPlan:
    """How many of each defect to inject."""

    name: str
    count: int


#: Default injection plan, tuned so that a ~120-order batch carries a defect load in the
#: 10-20% range. That is deliberately heavier than a healthy production day: a batch
#: where everything reconciles proves nothing about a reconciler.
DEFAULT_PLAN: Final[tuple[DefectPlan, ...]] = (
    DefectPlan("DROP_PG_ROW", 3),
    DefectPlan("DROP_BANK_CREDIT", 2),
    DefectPlan("ORPHAN_PG_ROW", 2),
    DefectPlan("PERTURB_BANK_AMOUNT", 3),
    DefectPlan("CORRUPT_FEE", 3),
    DefectPlan("CORRUPT_TAX", 2),
    DefectPlan("DUPLICATE_IN_SOURCE", 2),
    DefectPlan("DUPLICATE_PAYMENT", 2),
    DefectPlan("SHIFT_SETTLEMENT_DATE", 2),
    DefectPlan("MUTATE_RECEIPT", 4),
    DefectPlan("SPLIT_BANK_CREDIT", 2),
    DefectPlan("MERGE_BANK_CREDITS", 2),
    DefectPlan("TWIN_AMOUNT_AMBIGUITY", 1),
    DefectPlan("ONE_PAISE_TWIN", 1),
    DefectPlan("FX_DRIFT", 2),
    DefectPlan("UNSETTLED_AGED", 2),
    DefectPlan("SETTLEMENT_FAILED", 1),
    DefectPlan("SCRAMBLE_NARRATION_UTR", 3),
    DefectPlan("DISPUTE_HOLD", 2),
)


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


def _row_ids_touched(world: World) -> set[str]:
    """Ids already mutated by an earlier defect.

    Injectors avoid stacking two defects on one record. Overlapping defects would make
    the expected outcome ambiguous, and an ambiguous expectation cannot be scored.
    """
    touched: set[str] = set()
    for defect in world.defects:
        touched.update(defect.affected_ids)
    return touched


def _drop_link(world: World, tier: Tier, left_id: str | None = None, right_id: str | None = None) -> None:
    """Remove ground-truth links matching the given endpoints."""
    world.links = [
        link
        for link in world.links
        if not (
            link.tier is tier
            and (left_id is None or link.left_id == left_id)
            and (right_id is None or link.right_id == right_id)
        )
    ]


def _members_of(world: World, settlement_id: str) -> list[ReconRow]:
    return [r for r in world.recon_rows if r.settlement_id == settlement_id]


def _replace_row(world: World, entity_id: str, **updates: object) -> ReconRow | None:
    for i, row in enumerate(world.recon_rows):
        if row.entity_id == entity_id:
            new = row.model_copy(update=updates)
            world.recon_rows[i] = new
            return new
    return None


def _replace_credit(world: World, stmt_id: str, **updates: object) -> BankCredit | None:
    for i, credit in enumerate(world.bank_credits):
        if credit.stmt_id == stmt_id:
            new = credit.model_copy(update=updates)
            world.bank_credits[i] = new
            return new
    return None


def _pick(rng: random.Random, items: list, exclude: set[str], key: Callable[[object], str]):
    """Choose a random item whose key is not already touched."""
    pool = [it for it in items if key(it) not in exclude]
    return rng.choice(pool) if pool else None


def _settled_payments(world: World) -> list[ReconRow]:
    return [
        r
        for r in world.recon_rows
        if r.type is TxnType.PAYMENT and r.settled and r.settlement_id is not None
    ]


def _credit_for_settlement(world: World, settlement_id: str) -> BankCredit | None:
    for link in world.links:
        if link.tier is Tier.SETTLEMENT_BANK and link.left_id == settlement_id:
            for credit in world.bank_credits:
                if credit.stmt_id == link.right_id:
                    return credit
    return None


def _settlement_ids_for_credit(world: World, stmt_id: str) -> tuple[str, ...]:
    """Settlement ids linked to a bank credit.

    Used so that a defect targeting a *credit* also claims its *settlement* in
    ``affected_ids``. Without this the two sides are marked independently, and a later
    injector that filters on touched ids can legitimately pick the settlement -- then
    delete the credit an earlier defect had already corrupted. The earlier defect's
    expected exception becomes unsatisfiable because the record no longer exists, and the
    evaluation reports a detection failure that is really a fixture collision. Claiming
    both sides keeps every recorded expectation reachable.
    """
    return tuple(
        link.left_id
        for link in world.links
        if link.tier is Tier.SETTLEMENT_BANK and link.right_id == stmt_id
    )


# --------------------------------------------------------------------------------------
# Presence defects
# --------------------------------------------------------------------------------------


def _inject_drop_pg_row(world: World, rng: random.Random, ids: Ids) -> InjectedDefect | None:
    """Delete a payment row that the merchant's books say was paid.

    Targets an *unsettled* payment so the deletion does not also break the arithmetic
    of a settlement, keeping the defect single-purpose and its expectation unambiguous.
    """
    touched = _row_ids_touched(world)
    candidates = [
        r for r in world.recon_rows if r.type is TxnType.PAYMENT and not r.settled
    ]
    row = _pick(rng, candidates, touched, lambda r: r.entity_id)
    if row is None:
        return None
    world.recon_rows = [r for r in world.recon_rows if r.entity_id != row.entity_id]
    _drop_link(world, Tier.ORDER_PG, right_id=row.entity_id)
    return InjectedDefect(
        defect_code="DROP_PG_ROW",
        affected_ids=(row.order_id or "", row.entity_id),
        expected_exception=ExceptionCode.MISSING_IN_PG,
        amount_at_risk=row.amount,
        note="Order is marked paid in the books but has no gateway row.",
    )


def _inject_drop_bank_credit(world: World, rng: random.Random, ids: Ids) -> InjectedDefect | None:
    """Delete a bank credit for a settlement Razorpay says it processed."""
    touched = _row_ids_touched(world)
    processed = [s for s in world.settlements if s.status is SettlementStatus.PROCESSED]
    settlement = _pick(rng, processed, touched, lambda s: s.id)
    if settlement is None:
        return None
    credit = _credit_for_settlement(world, settlement.id)
    if credit is None:
        return None
    world.bank_credits = [c for c in world.bank_credits if c.stmt_id != credit.stmt_id]
    _drop_link(world, Tier.SETTLEMENT_BANK, left_id=settlement.id)
    return InjectedDefect(
        defect_code="DROP_BANK_CREDIT",
        affected_ids=(settlement.id, credit.stmt_id),
        expected_exception=ExceptionCode.MISSING_IN_BANK,
        amount_at_risk=settlement.amount,
        note="Settlement processed by the gateway never arrived in the bank account.",
    )


def _inject_orphan_pg_row(world: World, rng: random.Random, ids: Ids) -> InjectedDefect | None:
    """Add a gateway payment for an order that does not exist in the books."""
    template = _pick(rng, _settled_payments(world), set(), lambda r: r.entity_id)
    if template is None:
        return None
    orphan = template.model_copy(
        update={
            "entity_id": ids.make("pay_"),
            "order_id": ids.make("order_"),
            "order_receipt": f"UNKNOWN-{rng.randrange(10000, 99999)}",
            "settlement_id": None,
            "settlement_utr": None,
            "settled": False,
            "settled_at": None,
            "description": "Payment captured",
        }
    )
    world.recon_rows.append(orphan)
    return InjectedDefect(
        defect_code="ORPHAN_PG_ROW",
        affected_ids=(orphan.entity_id,),
        expected_exception=ExceptionCode.MISSING_IN_BOOKS,
        amount_at_risk=orphan.amount,
        note="Gateway row references an order absent from the merchant ledger.",
    )


# --------------------------------------------------------------------------------------
# Arithmetic defects
# --------------------------------------------------------------------------------------


def _inject_perturb_bank_amount(world: World, rng: random.Random, ids: Ids) -> InjectedDefect | None:
    """Shift a bank credit by an amount that fee/GST arithmetic cannot explain.

    The link is preserved: the credit still genuinely belongs to that settlement. The
    reconciler should find the pair and refuse to bless it.
    """
    touched = _row_ids_touched(world)
    credit = _pick(rng, world.bank_credits, touched, lambda c: c.stmt_id)
    if credit is None:
        return None
    delta = rng.choice([-1, 1]) * rng.randrange(500, 25_000)
    _replace_credit(world, credit.stmt_id, amount=credit.amount + delta)
    return InjectedDefect(
        defect_code="PERTURB_BANK_AMOUNT",
        affected_ids=(credit.stmt_id, *_settlement_ids_for_credit(world, credit.stmt_id)),
        expected_exception=ExceptionCode.AMOUNT_MISMATCH_UNEXPLAINED,
        amount_at_risk=abs(delta),
        note="Bank credited an amount that is not the settlement net, fee-adjusted or not.",
    )


def _inject_corrupt_fee(world: World, rng: random.Random, ids: Ids) -> InjectedDefect | None:
    """Report a fee that disagrees with an independent recomputation."""
    touched = _row_ids_touched(world)
    candidates = [r for r in _settled_payments(world) if r.method is not Method.UPI and r.fee > 0]
    row = _pick(rng, candidates, touched, lambda r: r.entity_id)
    if row is None:
        return None
    delta = max(100, int(row.fee * rng.uniform(0.15, 0.6)))
    _replace_row(world, row.entity_id, fee=row.fee + delta)
    return InjectedDefect(
        defect_code="CORRUPT_FEE",
        affected_ids=(row.entity_id,),
        expected_exception=ExceptionCode.FEE_MISMATCH,
        amount_at_risk=delta,
        note="Recorded fee does not match the fee recomputed from method and amount.",
    )


def _inject_corrupt_tax(world: World, rng: random.Random, ids: Ids) -> InjectedDefect | None:
    """Report GST that is not 18% of the recorded fee."""
    touched = _row_ids_touched(world)
    candidates = [r for r in _settled_payments(world) if r.tax > 0]
    row = _pick(rng, candidates, touched, lambda r: r.entity_id)
    if row is None:
        return None
    delta = max(50, int(row.tax * rng.uniform(0.2, 0.8)))
    _replace_row(world, row.entity_id, tax=row.tax + delta)
    return InjectedDefect(
        defect_code="CORRUPT_TAX",
        affected_ids=(row.entity_id,),
        expected_exception=ExceptionCode.GST_MISMATCH,
        amount_at_risk=delta,
        note="Recorded tax is not GST at 18% of the recorded fee.",
    )


def _inject_fx_drift(world: World, rng: random.Random, ids: Ids) -> InjectedDefect | None:
    """Break the INR conversion on an international payment."""
    touched = _row_ids_touched(world)
    candidates = [
        r
        for r in world.recon_rows
        if r.original_currency is not None and r.fx_rate_at_creation is not None
    ]
    row = _pick(rng, candidates, touched, lambda r: r.entity_id)
    if row is None:
        return None
    assert row.fx_rate_at_creation is not None and row.original_amount is not None
    drifted = round(row.fx_rate_at_creation * rng.uniform(1.04, 1.12), 4)
    implied = int(round(row.original_amount * drifted))
    _replace_row(world, row.entity_id, fx_rate_at_creation=drifted)
    return InjectedDefect(
        defect_code="FX_DRIFT",
        affected_ids=(row.entity_id,),
        expected_exception=ExceptionCode.FX_VARIANCE,
        amount_at_risk=abs(implied - row.amount),
        note="INR amount does not equal original amount at the recorded creation-time rate.",
    )


# --------------------------------------------------------------------------------------
# Structural defects
# --------------------------------------------------------------------------------------


def _inject_duplicate_in_source(world: World, rng: random.Random, ids: Ids) -> InjectedDefect | None:
    """Emit the same gateway row twice under two ids -- an export artefact."""
    touched = _row_ids_touched(world)
    row = _pick(rng, _settled_payments(world), touched, lambda r: r.entity_id)
    if row is None:
        return None
    clone = row.model_copy(update={"entity_id": ids.make("pay_")})
    world.recon_rows.append(clone)
    # No link for the clone: only one of the two corresponds to the order.
    return InjectedDefect(
        defect_code="DUPLICATE_IN_SOURCE",
        affected_ids=(row.entity_id, clone.entity_id),
        expected_exception=ExceptionCode.DUPLICATE_IN_SOURCE,
        amount_at_risk=clone.amount,
        note="Identical gateway row appears twice under different entity ids.",
    )


def _inject_duplicate_payment(world: World, rng: random.Random, ids: Ids) -> InjectedDefect | None:
    """Charge the same order twice -- a genuine double-charge, not an export artefact.

    Both rows are real payments, so neither can be dismissed. A reconciler must surface
    this rather than quietly matching one and orphaning the other.
    """
    touched = _row_ids_touched(world)
    row = _pick(rng, _settled_payments(world), touched, lambda r: r.entity_id)
    if row is None:
        return None
    second = row.model_copy(
        update={
            "entity_id": ids.make("pay_"),
            "created_at": row.created_at + rng.randrange(60, 900),
            "settled": False,
            "settled_at": None,
            "settlement_id": None,
            "settlement_utr": None,
        }
    )
    world.recon_rows.append(second)
    return InjectedDefect(
        defect_code="DUPLICATE_PAYMENT",
        affected_ids=(row.order_id or "", row.entity_id, second.entity_id),
        expected_exception=ExceptionCode.DUPLICATE_PAYMENT,
        amount_at_risk=second.amount,
        note="One order carries two distinct successful payments minutes apart.",
    )


def _inject_split_bank_credit(world: World, rng: random.Random, ids: Ids) -> InjectedDefect | None:
    """Bank pays one settlement as two credits that sum to it.

    Absorbable: subset-sum over the credit side should recover the pair without raising
    an exception.
    """
    touched = _row_ids_touched(world)
    credit = _pick(
        rng, [c for c in world.bank_credits if c.amount > 20_000], touched, lambda c: c.stmt_id
    )
    if credit is None:
        return None
    settlement_id = next(
        (
            link.left_id
            for link in world.links
            if link.tier is Tier.SETTLEMENT_BANK and link.right_id == credit.stmt_id
        ),
        None,
    )
    if settlement_id is None:
        return None

    first_amount = int(credit.amount * rng.uniform(0.35, 0.65))
    second_amount = credit.amount - first_amount
    world.bank_credits = [c for c in world.bank_credits if c.stmt_id != credit.stmt_id]
    _drop_link(world, Tier.SETTLEMENT_BANK, left_id=settlement_id, right_id=credit.stmt_id)

    new_ids: list[str] = []
    for part, amount in enumerate((first_amount, second_amount), start=1):
        part_id = f"{credit.stmt_id}-P{part}"
        world.bank_credits.append(
            credit.model_copy(
                update={
                    "stmt_id": part_id,
                    "amount": amount,
                    "narration": f"{credit.narration} PART {part}/2",
                }
            )
        )
        world.links.append(
            TrueLink(tier=Tier.SETTLEMENT_BANK, left_id=settlement_id, right_id=part_id)
        )
        new_ids.append(part_id)

    return InjectedDefect(
        defect_code="SPLIT_BANK_CREDIT",
        affected_ids=(settlement_id, *new_ids),
        expected_exception=None,  # absorbable by subset-sum
        amount_at_risk=0,
        note="One settlement arrived as two bank credits; should reconcile by subset sum.",
    )


def _inject_merge_bank_credits(world: World, rng: random.Random, ids: Ids) -> InjectedDefect | None:
    """Bank lumps two settlements into a single credit. Absorbable by subset-sum.

    The two settlements are chosen **adjacent in value date**, because that is the only
    way this happens in reality: a bank consolidates payouts that land in the same
    window, not ones three weeks apart. Picking at random would also inject an
    artificial blocking failure, which would understate the matcher rather than test it.
    """
    touched = _row_ids_touched(world)
    credits = {c.stmt_id: c for c in world.bank_credits}
    pairs = [
        (link.left_id, credits[link.right_id])
        for link in world.links
        if link.tier is Tier.SETTLEMENT_BANK
        and link.left_id not in touched
        and link.right_id not in touched
        and link.right_id in credits
    ]
    if len(pairs) < 2:
        return None
    pairs.sort(key=lambda p: p[1].value_date)
    i = rng.randrange(len(pairs) - 1)
    (s1, first), (s2, second) = pairs[i], pairs[i + 1]

    merged_id = f"{first.stmt_id}-M"
    merged = first.model_copy(
        update={
            "stmt_id": merged_id,
            "amount": first.amount + second.amount,
            "narration": f"NEFT CR RAZORPAY CONSOLIDATED PAYOUT {first.utr_extracted} +1",
            "utr_extracted": None,  # a lumped credit cannot carry one settlement's UTR
            "value_date": max(first.value_date, second.value_date),
        }
    )
    world.bank_credits = [
        c for c in world.bank_credits if c.stmt_id not in {first.stmt_id, second.stmt_id}
    ]
    world.bank_credits.append(merged)
    _drop_link(world, Tier.SETTLEMENT_BANK, left_id=s1, right_id=first.stmt_id)
    _drop_link(world, Tier.SETTLEMENT_BANK, left_id=s2, right_id=second.stmt_id)
    world.links.append(TrueLink(tier=Tier.SETTLEMENT_BANK, left_id=s1, right_id=merged_id))
    world.links.append(TrueLink(tier=Tier.SETTLEMENT_BANK, left_id=s2, right_id=merged_id))
    return InjectedDefect(
        defect_code="MERGE_BANK_CREDITS",
        affected_ids=(s1, s2, merged_id),
        expected_exception=None,  # absorbable by subset-sum
        amount_at_risk=0,
        note="Two consecutive settlements arrived as one bank credit with no usable UTR.",
    )


def _inject_twin_amount_ambiguity(
    world: World, rng: random.Random, ids: Ids
) -> InjectedDefect | None:
    """Create a genuinely unresolvable 2x2: two equal settlements, two equal credits,
    no legible UTR on either side.

    Built from scratch out of ``adjustment`` rows so that tier-2 arithmetic stays valid
    and the only defect present is the ambiguity itself. There is no correct assignment
    here -- a system that picks one is guessing, and we assert that it escalates.
    """
    if not world.bank_credits:
        return None
    day = epoch_to_ist_date(world.bank_credits[-1].value_date)
    when = ist_date_to_epoch(day, hour=9)
    amount = 1_250_00  # Rs 1,250.00 exactly, twice

    affected: list[str] = []
    for n in (1, 2):
        adj = ReconRow(
            entity_id=ids.make("adj_"),
            type=TxnType.ADJUSTMENT,
            debit=0,
            credit=amount,
            amount=amount,
            fee=0,
            tax=0,
            settled=True,
            created_at=when,
            settled_at=when,
            description=f"Promotional credit adjustment {n}",
        )
        settlement = Settlement(
            id=ids.make("setl_"),
            amount=amount,
            status=SettlementStatus.PROCESSED,
            utr=ids.utr(),
            created_at=when,
        )
        adj = adj.model_copy(update={"settlement_id": settlement.id, "settlement_utr": None})
        credit = BankCredit(
            stmt_id=f"STMT-TWIN{n}",
            value_date=when,
            amount=amount,
            narration="NEFT CR RAZORPAY PAYOUT REF UNAVAILABLE",
            utr_extracted=None,
        )
        world.recon_rows.append(adj)
        world.settlements.append(settlement)
        world.bank_credits.append(credit)
        world.settlement_members[settlement.id] = [adj.entity_id]
        affected.extend([settlement.id, credit.stmt_id])
        # No ground-truth link recorded: there is no fact of the matter about which
        # credit belongs to which settlement, so any produced link is a guess.

    return InjectedDefect(
        defect_code="TWIN_AMOUNT_AMBIGUITY",
        affected_ids=tuple(affected),
        expected_exception=ExceptionCode.AMBIGUOUS_MULTI_CANDIDATE,
        amount_at_risk=amount * 2,
        note="Two identical settlements and two identical credits, no legible UTR. "
        "Unresolvable by construction; must escalate rather than guess.",
    )


def _inject_one_paise_twin(world: World, rng: random.Random, ids: Ids) -> InjectedDefect | None:
    """Adversarial control: two settlements one paise apart, UTRs unreadable.

    This is fully resolvable by *exact* amount matching, and only ambiguous to an
    implementation carrying a sloppy rounding tolerance. Expected outcome is therefore
    two correct matches and no exception -- it is a precision trap, not a recall trap.
    """
    if not world.bank_credits:
        return None
    day = epoch_to_ist_date(world.bank_credits[-1].value_date)
    when = ist_date_to_epoch(day, hour=9)
    base = 7_777_00

    affected: list[str] = []
    for n, amount in enumerate((base, base + 1), start=1):
        adj = ReconRow(
            entity_id=ids.make("adj_"),
            type=TxnType.ADJUSTMENT,
            debit=0,
            credit=amount,
            amount=amount,
            fee=0,
            tax=0,
            settled=True,
            created_at=when,
            settled_at=when,
            description=f"Reserve release {n}",
        )
        settlement = Settlement(
            id=ids.make("setl_"),
            amount=amount,
            status=SettlementStatus.PROCESSED,
            utr=ids.utr(),
            created_at=when,
        )
        adj = adj.model_copy(update={"settlement_id": settlement.id, "settlement_utr": None})
        credit = BankCredit(
            stmt_id=f"STMT-PAISE{n}",
            value_date=when,
            amount=amount,
            narration="NEFT CR RAZORPAY PAYOUT REF ILLEGIBLE",
            utr_extracted=None,
        )
        world.recon_rows.append(adj)
        world.settlements.append(settlement)
        world.bank_credits.append(credit)
        world.settlement_members[settlement.id] = [adj.entity_id]
        world.links.append(
            TrueLink(tier=Tier.SETTLEMENT_BANK, left_id=settlement.id, right_id=credit.stmt_id)
        )
        affected.extend([settlement.id, credit.stmt_id])

    return InjectedDefect(
        defect_code="ONE_PAISE_TWIN",
        affected_ids=tuple(affected),
        expected_exception=None,  # resolvable, and must be resolved CORRECTLY
        amount_at_risk=0,
        note="Two settlements differing by exactly 1 paise. Exact-amount matching "
        "resolves both; any amount tolerance wider than 0 makes this ambiguous.",
    )


# --------------------------------------------------------------------------------------
# Timing and lifecycle defects
# --------------------------------------------------------------------------------------


def _inject_shift_settlement_date(
    world: World, rng: random.Random, ids: Ids
) -> InjectedDefect | None:
    """Land a bank credit well outside the expected settlement window."""
    touched = _row_ids_touched(world)
    credit = _pick(rng, world.bank_credits, touched, lambda c: c.stmt_id)
    if credit is None:
        return None
    shift_days = rng.randrange(4, 8)
    _replace_credit(world, credit.stmt_id, value_date=credit.value_date + shift_days * 86_400)
    return InjectedDefect(
        defect_code="SHIFT_SETTLEMENT_DATE",
        affected_ids=(credit.stmt_id, *_settlement_ids_for_credit(world, credit.stmt_id)),
        expected_exception=ExceptionCode.TIMING_BREACH,
        amount_at_risk=0,
        note=f"Credit arrived {shift_days} calendar days after the settlement date.",
    )


def _inject_unsettled_aged(world: World, rng: random.Random, ids: Ids) -> InjectedDefect | None:
    """Age an already-unsettled capture so it is overdue rather than merely in flight."""
    touched = _row_ids_touched(world)
    candidates = [
        r for r in world.recon_rows if not r.settled and r.type is TxnType.PAYMENT and not r.on_hold
    ]
    row = _pick(rng, candidates, touched, lambda r: r.entity_id)
    if row is None:
        return None
    _replace_row(world, row.entity_id, created_at=row.created_at - 12 * 86_400)
    return InjectedDefect(
        defect_code="UNSETTLED_AGED",
        affected_ids=(row.entity_id,),
        expected_exception=ExceptionCode.UNSETTLED_AGED,
        amount_at_risk=row.credit,
        note="Captured well beyond T+2 and still unsettled.",
    )


def _inject_settlement_failed(world: World, rng: random.Random, ids: Ids) -> InjectedDefect | None:
    """Mark a settlement failed and remove its credit.

    A discrimination test: the absent bank credit is *explained* by the failure, so the
    correct output is SETTLEMENT_FAILED alone. Reporting MISSING_IN_BANK as well would
    be double-counting the same rupees.
    """
    touched = _row_ids_touched(world)
    processed = [s for s in world.settlements if s.status is SettlementStatus.PROCESSED]
    settlement = _pick(rng, processed, touched, lambda s: s.id)
    if settlement is None:
        return None
    credit = _credit_for_settlement(world, settlement.id)
    for i, s in enumerate(world.settlements):
        if s.id == settlement.id:
            world.settlements[i] = s.model_copy(
                update={"status": SettlementStatus.FAILED, "utr": None}
            )
    if credit is not None:
        world.bank_credits = [c for c in world.bank_credits if c.stmt_id != credit.stmt_id]
        _drop_link(world, Tier.SETTLEMENT_BANK, left_id=settlement.id)
    return InjectedDefect(
        defect_code="SETTLEMENT_FAILED",
        affected_ids=(settlement.id,),
        expected_exception=ExceptionCode.SETTLEMENT_FAILED,
        amount_at_risk=settlement.amount,
        note="Settlement was rejected by the bank; funds remain with the gateway.",
    )


def _inject_dispute_hold(world: World, rng: random.Random, ids: Ids) -> InjectedDefect | None:
    """Put an unsettled capture on hold behind a dispute."""
    touched = _row_ids_touched(world)
    candidates = [
        r
        for r in world.recon_rows
        if r.type is TxnType.PAYMENT and not r.settled and not r.on_hold and r.dispute_id is None
    ]
    row = _pick(rng, candidates, touched, lambda r: r.entity_id)
    if row is None:
        return None
    _replace_row(world, row.entity_id, on_hold=True, dispute_id=ids.make("disp_"))
    return InjectedDefect(
        defect_code="DISPUTE_HOLD",
        affected_ids=(row.entity_id,),
        expected_exception=ExceptionCode.DISPUTE_HOLD,
        amount_at_risk=row.credit,
        note="Settlement withheld pending dispute resolution.",
    )


# --------------------------------------------------------------------------------------
# Reference-quality defects (absorbable -- these are the false-positive traps)
# --------------------------------------------------------------------------------------

_RECEIPT_MUTATIONS: Final[tuple[Callable[[str], str], ...]] = (
    lambda s: s.lower(),
    lambda s: s.replace("-", "/"),
    lambda s: s.replace("-", " ").replace("/", " "),
    lambda s: f" {s} ",
    lambda s: s.replace("INV-", "INV"),
    lambda s: s.replace("0", "O", 1),  # transcription slip: zero read as letter O
)


def _inject_mutate_receipt(world: World, rng: random.Random, ids: Ids) -> InjectedDefect | None:
    """Corrupt the receipt reference on a gateway row without changing the amount.

    Absorbable: amount and date still agree exactly, so a competent matcher should
    recover the pair via fuzzy reference plus exact amount, and raise nothing.
    """
    touched = _row_ids_touched(world)
    candidates = [
        r for r in world.recon_rows if r.type is TxnType.PAYMENT and r.order_receipt is not None
    ]
    row = _pick(rng, candidates, touched, lambda r: r.entity_id)
    if row is None or row.order_receipt is None:
        return None
    mutated = rng.choice(_RECEIPT_MUTATIONS)(row.order_receipt)
    if mutated == row.order_receipt:
        mutated = row.order_receipt.lower()
    _replace_row(world, row.entity_id, order_receipt=mutated)
    return InjectedDefect(
        defect_code="MUTATE_RECEIPT",
        affected_ids=(row.entity_id,),
        expected_exception=None,  # absorbable via fuzzy reference + exact amount
        amount_at_risk=0,
        note=f"Receipt reference mutated to {mutated!r}; amount and date unchanged.",
    )


def _inject_scramble_narration_utr(
    world: World, rng: random.Random, ids: Ids
) -> InjectedDefect | None:
    """Make the UTR illegible in a bank narration.

    Absorbable: amount and value date are untouched, so the pair should still be
    recovered without the reference. Tests that matching does not depend on one field.
    """
    touched = _row_ids_touched(world)
    candidates = [c for c in world.bank_credits if c.utr_extracted is not None]
    credit = _pick(rng, candidates, touched, lambda c: c.stmt_id)
    if credit is None:
        return None
    _replace_credit(
        world,
        credit.stmt_id,
        narration="NEFT CR RAZORPAY SETTLEMENT REF ***UNREADABLE*** MERCHANT PAYOUT",
        utr_extracted=None,
    )
    return InjectedDefect(
        defect_code="SCRAMBLE_NARRATION_UTR",
        affected_ids=(credit.stmt_id, *_settlement_ids_for_credit(world, credit.stmt_id)),
        expected_exception=None,  # absorbable via amount + date
        amount_at_risk=0,
        note="Bank narration carries no readable UTR; amount and date intact.",
    )


# --------------------------------------------------------------------------------------
# Registry and driver
# --------------------------------------------------------------------------------------

INJECTORS: Final[dict[str, Injector]] = {
    "DROP_PG_ROW": _inject_drop_pg_row,
    "DROP_BANK_CREDIT": _inject_drop_bank_credit,
    "ORPHAN_PG_ROW": _inject_orphan_pg_row,
    "PERTURB_BANK_AMOUNT": _inject_perturb_bank_amount,
    "CORRUPT_FEE": _inject_corrupt_fee,
    "CORRUPT_TAX": _inject_corrupt_tax,
    "FX_DRIFT": _inject_fx_drift,
    "DUPLICATE_IN_SOURCE": _inject_duplicate_in_source,
    "DUPLICATE_PAYMENT": _inject_duplicate_payment,
    "SPLIT_BANK_CREDIT": _inject_split_bank_credit,
    "MERGE_BANK_CREDITS": _inject_merge_bank_credits,
    "TWIN_AMOUNT_AMBIGUITY": _inject_twin_amount_ambiguity,
    "ONE_PAISE_TWIN": _inject_one_paise_twin,
    "SHIFT_SETTLEMENT_DATE": _inject_shift_settlement_date,
    "UNSETTLED_AGED": _inject_unsettled_aged,
    "SETTLEMENT_FAILED": _inject_settlement_failed,
    "DISPUTE_HOLD": _inject_dispute_hold,
    "MUTATE_RECEIPT": _inject_mutate_receipt,
    "SCRAMBLE_NARRATION_UTR": _inject_scramble_narration_utr,
}


def inject_defects(
    world: World,
    plan: tuple[DefectPlan, ...] = DEFAULT_PLAN,
    *,
    seed: int | None = None,
) -> World:
    """Apply a defect plan to ``world`` in place, recording ground truth as it goes.

    Injectors that cannot find a suitable target return ``None`` and are skipped. Skips
    are reported by the caller rather than silently ignored, because a plan that
    quietly failed to inject would inflate the measured match rate.
    """
    rng = random.Random((seed if seed is not None else world.config.seed) ^ 0x5EED)
    ids = Ids(rng)
    for item in plan:
        injector = INJECTORS.get(item.name)
        if injector is None:
            raise KeyError(f"unknown defect {item.name!r}")
        for _ in range(item.count):
            defect = injector(world, rng, ids)
            if defect is not None:
                world.defects.append(defect)
    return world
