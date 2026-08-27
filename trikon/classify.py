"""Deterministic exception detection and the cash position.

Every exception in this module is raised by arithmetic or set logic, never by a model.
That is the point: an exception is an assertion that a specific number is wrong, and it
must be reproducible and arguable. The optional LLM layer may later attach *prose* to an
exception (:attr:`~trikon.models.ExceptionRecord.narrative`), but the code, the subject
records, the rupees at risk and the evidence are all fixed here before any model runs.

Three families of detector live here:

* **Tier 2 -- arithmetic.** Does each settlement equal the sum of its member rows net of
  fee and GST? Is each row's recorded fee what the fee table says it should be? Is tax
  exactly 18% of that fee? Does an international row's INR amount follow from its
  original amount at the recorded creation-time rate? This tier needs no matching at all
  -- rows already carry ``settlement_id`` -- so it is pure verification, and it is where
  the tax-line checking lives.
* **Presence and structure.** Records that matching left over: orders with no gateway
  row, gateway rows with no order, settlements with no bank credit, duplicates.
* **Lifecycle and cash.** Failed settlements, dispute holds, aged unsettled captures,
  and the forward cash position implied by everything not yet settled.

A note on double counting: a settlement that *failed* has no bank credit by definition,
so :func:`detect_missing_in_bank` deliberately skips failed settlements. Reporting both
SETTLEMENT_FAILED and MISSING_IN_BANK for the same rupees would inflate both the
exception count and the amount at risk, which is the sort of accounting a finance
reviewer notices immediately.
"""

from __future__ import annotations

import itertools
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Final, Iterable, Sequence

from trikon.calendar_ist import (
    epoch_to_ist_date,
    expected_settlement_date,
    working_days_between,
)
from trikon.money import compute_fee, compute_gst_on_fee, format_inr
from trikon.models import (
    BankCredit,
    Evidence,
    ExceptionCode,
    ExceptionRecord,
    Method,
    Order,
    OrderStatus,
    ReconRow,
    Settlement,
    SettlementStatus,
    Severity,
    Tier,
    TxnType,
)

#: Rupees-at-risk thresholds (in paise) that lift an exception's severity. Severity is a
#: function of code *and* exposure, because a 12-paise fee discrepancy and a missing
#: 4-lakh settlement are not the same problem even when they share a code.
_HIGH_RISK_PAISE: Final[int] = 50_000_00
_MEDIUM_RISK_PAISE: Final[int] = 5_000_00

#: Codes that are always at least HIGH regardless of amount, because they indicate money
#: that has left one system without arriving in another.
_ALWAYS_HIGH: Final[frozenset[ExceptionCode]] = frozenset(
    {
        ExceptionCode.MISSING_IN_BANK,
        ExceptionCode.SETTLEMENT_FAILED,
        ExceptionCode.DUPLICATE_PAYMENT,
        ExceptionCode.MISSING_IN_PG,
    }
)

#: Codes that are informational rather than errors: the money is accounted for, it is
#: simply not settled yet. Surfacing these as CRITICAL would drown the real breaks.
_INFORMATIONAL: Final[frozenset[ExceptionCode]] = frozenset(
    {ExceptionCode.DISPUTE_HOLD, ExceptionCode.TIMING_BREACH}
)

#: Working days past the expected settlement date before an unsettled capture is aged.
AGED_UNSETTLED_WORKING_DAYS: Final[int] = 3

#: Tolerance for the implied FX reconstruction, in paise. Rounding a rate to four
#: decimals cannot reproduce a paise-exact figure, so a small band is legitimate here --
#: unlike amount matching, where any tolerance would be a bug.
FX_TOLERANCE_PAISE: Final[int] = 100


def severity_for(code: ExceptionCode, amount_at_risk: int) -> Severity:
    """Derive review priority from the exception code and the exposure."""
    if code in _INFORMATIONAL:
        return Severity.LOW
    if amount_at_risk >= _HIGH_RISK_PAISE:
        return Severity.CRITICAL
    if code in _ALWAYS_HIGH:
        return Severity.HIGH
    if amount_at_risk >= _MEDIUM_RISK_PAISE:
        return Severity.MEDIUM
    return Severity.LOW


class _ExceptionBuilder:
    """Accumulates exceptions with stable, deterministic ids.

    Ids are sequential per code (``FEE_MISMATCH-001``) rather than random, so that two
    runs over the same batch produce byte-identical reports. Reproducibility is a feature
    we claim, so it has to hold for identifiers too.
    """

    def __init__(self) -> None:
        self._items: list[ExceptionRecord] = []
        self._counters: Counter[str] = Counter()

    def add(
        self,
        *,
        code: ExceptionCode,
        tier: Tier,
        subject_ids: Sequence[str],
        amount_at_risk: int,
        reason: str,
        evidence: Sequence[Evidence],
        recommended_action: str,
        candidates_considered: Sequence[str] = (),
    ) -> ExceptionRecord:
        self._counters[code.value] += 1
        record = ExceptionRecord(
            exception_id=f"{code.value}-{self._counters[code.value]:03d}",
            code=code,
            tier=tier,
            severity=severity_for(code, amount_at_risk),
            subject_ids=tuple(subject_ids),
            amount_at_risk=amount_at_risk,
            reason=reason,
            evidence=tuple(evidence),
            recommended_action=recommended_action,
            candidates_considered=tuple(candidates_considered),
        )
        self._items.append(record)
        return record

    def collect(self) -> list[ExceptionRecord]:
        """Exceptions ordered by financial impact, then severity.

        A finance controller works a queue top-down by rupees exposed. Ordering by
        insertion or by code would make the most expensive break arbitrarily deep in the
        list, which defeats the purpose of producing the list at all.
        """
        rank = {
            Severity.CRITICAL: 0,
            Severity.HIGH: 1,
            Severity.MEDIUM: 2,
            Severity.LOW: 3,
        }
        return sorted(
            self._items,
            key=lambda e: (-e.amount_at_risk, rank[e.severity], e.exception_id),
        )


# --------------------------------------------------------------------------------------
# Tier 2 -- arithmetic verification (this is the tax-line matching)
# --------------------------------------------------------------------------------------


def verify_settlement_arithmetic(
    settlements: Sequence[Settlement],
    rows: Sequence[ReconRow],
    builder: _ExceptionBuilder,
) -> dict[str, int]:
    """Assert every settlement equals the net movement of its member rows.

    Returns the computed net per settlement id, for reuse by the reporting layer.

    No matching is involved: rows already carry ``settlement_id``. What is being tested
    is whether the gateway's own arithmetic is internally consistent -- the check a
    merchant currently performs by exporting two reports and summing a column by hand.
    """
    members: dict[str, list[ReconRow]] = defaultdict(list)
    for row in rows:
        if row.settlement_id:
            members[row.settlement_id].append(row)

    computed: dict[str, int] = {}
    for settlement in settlements:
        group = members.get(settlement.id, [])
        net = sum(r.credit - r.debit for r in group)
        computed[settlement.id] = net
        if not group:
            # A settlement with no member rows cannot be substantiated at all.
            builder.add(
                code=ExceptionCode.SETTLEMENT_NET_MISMATCH,
                tier=Tier.SETTLEMENT_NET,
                subject_ids=[settlement.id],
                amount_at_risk=settlement.amount,
                reason=f"Settlement {settlement.id} reports {format_inr(settlement.amount)} "
                "but no recon rows reference it.",
                evidence=[
                    Evidence(
                        feature="member_rows",
                        observed="0 rows",
                        supports=False,
                        detail="Payout cannot be substantiated from transaction detail.",
                    )
                ],
                recommended_action="Re-pull the settlement recon report for this settlement id.",
            )
            continue

        if net != settlement.amount:
            delta = settlement.amount - net
            # A duplicated export row inflates its settlement's member sum. That is a
            # symptom of the duplicate, not an independent settlement fault, and
            # detect_duplicates already owns it -- so if removing duplicate-signature rows
            # makes the arithmetic balance, we stay quiet here rather than raise a second
            # exception for one root cause.
            deduped = _sum_unique_signatures(group)
            if deduped == settlement.amount:
                continue
            builder.add(
                code=ExceptionCode.SETTLEMENT_NET_MISMATCH,
                tier=Tier.SETTLEMENT_NET,
                subject_ids=[settlement.id, *(r.entity_id for r in group)],
                amount_at_risk=abs(delta),
                reason=f"Settlement {settlement.id} reports {format_inr(settlement.amount)} "
                f"but its {len(group)} member rows net to {format_inr(net)} "
                f"(delta {format_inr(delta)}).",
                evidence=[
                    Evidence(
                        feature="settlement_amount",
                        observed=format_inr(settlement.amount),
                        supports=False,
                    ),
                    Evidence(
                        feature="sum_of_members",
                        observed=f"{format_inr(net)} across {len(group)} rows",
                        supports=False,
                        detail="Sum of credit minus debit over member rows.",
                    ),
                ],
                recommended_action="Reconcile member rows against the settlement header; "
                "check for rows settled under a different settlement id.",
            )
    return computed


def _sum_unique_signatures(rows: Sequence[ReconRow]) -> int:
    """Net movement over rows, counting economically identical rows only once.

    Two rows sharing order, amount, timestamp, fee and tax are the same transaction
    exported twice; they cannot both represent real money.
    """
    seen: set[tuple[object, ...]] = set()
    total = 0
    for row in rows:
        signature = (row.order_id, row.amount, row.created_at, row.fee, row.tax, row.type)
        if signature in seen:
            continue
        seen.add(signature)
        total += row.credit - row.debit
    return total


def verify_fees_and_tax(rows: Sequence[ReconRow], builder: _ExceptionBuilder) -> None:
    """Independently recompute fee and GST for every row and report disagreements.

    UPI rows are expected to carry zero fee, so a non-zero fee on a UPI row is itself the
    finding. For every other method the fee is recomputed from the amount and the rate
    table.

    **A wrong fee is reported once, not twice.** GST is levied on the fee, so a corrupted
    fee necessarily makes the tax line wrong as well -- but that is one root cause with a
    derived consequence, not two independent problems. Emitting both FEE_MISMATCH and
    GST_MISMATCH for the same row would inflate the exception count and, worse, count the
    same rupees of exposure twice. So when the fee is wrong we report a single
    FEE_MISMATCH whose exposure includes the consequent tax error, and reserve
    GST_MISMATCH for rows whose fee is correct but whose tax still does not follow from
    it. That keeps each exception traceable to exactly one underlying cause.
    """
    for row in rows:
        if row.type is not TxnType.PAYMENT or row.method is None:
            continue

        international = row.original_currency is not None
        try:
            expected_fee = compute_fee(row.amount, row.method.value, international=international)
        except KeyError:  # pragma: no cover - method table is closed
            continue

        expected_tax_on_correct_fee = compute_gst_on_fee(expected_fee)
        expected_tax_on_recorded_fee = compute_gst_on_fee(row.fee)

        if row.fee != expected_fee:
            fee_delta = row.fee - expected_fee
            tax_delta = row.tax - expected_tax_on_correct_fee
            total_delta = fee_delta + tax_delta
            builder.add(
                code=ExceptionCode.FEE_MISMATCH,
                tier=Tier.SETTLEMENT_NET,
                subject_ids=[row.entity_id],
                amount_at_risk=abs(total_delta),
                reason=f"Row {row.entity_id} records fee {format_inr(row.fee)} but "
                f"{format_inr(expected_fee)} is expected for a "
                f"{'international ' if international else ''}{row.method.value} payment of "
                f"{format_inr(row.amount)}. Including the consequent GST error, the total "
                f"discrepancy is {format_inr(total_delta)}.",
                evidence=[
                    Evidence(
                        feature="recorded_fee",
                        observed=format_inr(row.fee),
                        supports=False,
                    ),
                    Evidence(
                        feature="recomputed_fee",
                        observed=format_inr(expected_fee),
                        supports=False,
                        detail=f"Rate table for '{row.method.value}' applied to "
                        f"{format_inr(row.amount)}, rounded half-up.",
                    ),
                    Evidence(
                        feature="consequent_tax",
                        observed=f"{format_inr(row.tax)} recorded vs "
                        f"{format_inr(expected_tax_on_correct_fee)} on the correct fee",
                        supports=False,
                        detail="Derived from the fee error, not an independent tax fault. "
                        "Reported here to avoid double-counting the exposure.",
                    ),
                ],
                recommended_action="Raise a fee discrepancy with the gateway quoting this "
                "entity id; verify the contracted rate for this method. The GST line will "
                "correct itself once the fee is corrected.",
            )
            continue

        if row.tax != expected_tax_on_recorded_fee:
            delta = row.tax - expected_tax_on_recorded_fee
            builder.add(
                code=ExceptionCode.GST_MISMATCH,
                tier=Tier.SETTLEMENT_NET,
                subject_ids=[row.entity_id],
                amount_at_risk=abs(delta),
                reason=f"Row {row.entity_id} records tax {format_inr(row.tax)} but GST at 18% "
                f"of the correctly-recorded fee {format_inr(row.fee)} is "
                f"{format_inr(expected_tax_on_recorded_fee)} (delta {format_inr(delta)}).",
                evidence=[
                    Evidence(
                        feature="recorded_tax", observed=format_inr(row.tax), supports=False
                    ),
                    Evidence(
                        feature="recomputed_tax",
                        observed=format_inr(expected_tax_on_recorded_fee),
                        supports=False,
                        detail="18% of the fee on this row, which is itself correct -- so "
                        "this is an independent tax fault.",
                    ),
                ],
                recommended_action="Check the GST treatment for this row; an incorrect tax "
                "line will not reconcile against the input-credit register.",
            )


def verify_fx_conversion(rows: Sequence[ReconRow], builder: _ExceptionBuilder) -> None:
    """Check that an international row's INR amount follows from its recorded rate.

    Razorpay documents that settlements pay out in INR regardless of the payment
    currency, converted at the rate in force when the payment was created. That makes the
    conversion checkable rather than something a reviewer has to take on trust.
    """
    for row in rows:
        if row.original_amount is None or row.fx_rate_at_creation is None:
            continue
        implied = int(round(row.original_amount * row.fx_rate_at_creation))
        delta = implied - row.amount
        if abs(delta) > FX_TOLERANCE_PAISE:
            builder.add(
                code=ExceptionCode.FX_VARIANCE,
                tier=Tier.SETTLEMENT_NET,
                subject_ids=[row.entity_id],
                amount_at_risk=abs(delta),
                reason=f"Row {row.entity_id}: {row.original_currency} "
                f"{format_inr(row.original_amount)} at recorded rate "
                f"{row.fx_rate_at_creation} implies {format_inr(implied)}, but "
                f"{format_inr(row.amount)} was booked (delta {format_inr(delta)}).",
                evidence=[
                    Evidence(
                        feature="booked_inr", observed=format_inr(row.amount), supports=False
                    ),
                    Evidence(
                        feature="implied_inr",
                        observed=f"{format_inr(implied)} at {row.fx_rate_at_creation}",
                        supports=False,
                        detail=f"Original {row.original_currency} amount times the "
                        "creation-time rate.",
                    ),
                ],
                recommended_action="Confirm the exchange rate applied at payment creation "
                "against the rate booked in the ledger.",
            )


# --------------------------------------------------------------------------------------
# Presence and structural detectors
# --------------------------------------------------------------------------------------


def detect_timing_breaches(
    links: Sequence[object],
    settlement_by_id: dict[str, Settlement],
    credit_by_id: dict[str, BankCredit],
    builder: _ExceptionBuilder,
) -> None:
    """Report settlements that reconciled but arrived outside the expected window.

    A late payout is a genuine finding that a purely match-oriented reconciler loses. Rule
    R3 matches on exact reference and exact amount even when the timing is wrong, and
    auto-accepts it -- correctly, because the pair really is the pair. But "we found it"
    and "it arrived on time" are different claims, and treasury cares about the second
    one. Matching the record must not silently absolve it, so the link is kept *and* the
    breach is raised.

    Severity stays LOW: the money did arrive, so this is a service-level observation
    rather than a reconciliation break, and grading it higher would bury real breaks.
    """
    for link in links:
        rule = getattr(link, "rule", None)
        if rule is None or getattr(rule, "value", "") != "R3_TIMING_SHIFTED":
            continue
        left_id = getattr(link, "left_id", "")
        right_id = getattr(link, "right_id", "")
        settlement = settlement_by_id.get(left_id)
        credit = credit_by_id.get(right_id)
        if settlement is None or credit is None:
            continue

        expected_day = epoch_to_ist_date(settlement.created_at)
        actual_day = epoch_to_ist_date(credit.value_date)
        delta = working_days_between(expected_day, actual_day)
        builder.add(
            code=ExceptionCode.TIMING_BREACH,
            tier=Tier.SETTLEMENT_BANK,
            subject_ids=[settlement.id, credit.stmt_id],
            amount_at_risk=0,
            reason=f"Settlement {settlement.id} matched bank credit {credit.stmt_id} on "
            f"reference and exact amount, but the credit landed {actual_day} against a "
            f"settlement date of {expected_day} ({delta:+d} working days).",
            evidence=[
                Evidence(
                    feature="settlement_date", observed=str(expected_day), supports=True
                ),
                Evidence(
                    feature="bank_value_date",
                    observed=str(actual_day),
                    supports=False,
                    detail=f"{delta:+d} working days from the settlement date, outside the "
                    "expected window.",
                ),
                Evidence(
                    feature="amount",
                    observed=format_inr(credit.amount),
                    supports=True,
                    detail="Amount is exact; only the timing is at issue.",
                ),
            ],
            recommended_action="No cash is missing. Track against the settlement SLA and "
            "raise with the bank if the pattern repeats.",
        )


def pair_residue_by_variance(
    unmatched_settlements: Sequence[Settlement],
    unmatched_credits: Sequence[BankCredit],
    builder: _ExceptionBuilder,
    *,
    relative_tolerance: float = 0.25,
    date_window_days: int = 3,
) -> tuple[set[str], set[str]]:
    """Diagnose leftover settlements and credits that are probably the same payout.

    After matching, a settlement whose bank credit was recorded with the wrong amount ends
    up in one leftover pile and its credit in the other. Reporting them independently
    produces two misleading exceptions -- "settlement never arrived" and "unexplained
    credit" -- for one problem, and double-counts the exposure. Worse, the first claim is
    simply false: the money did arrive, in the wrong amount.

    So before those detectors run, we look for residue pairs that are evidently the same
    payout, preferring a shared UTR and falling back to a near amount on a near date, and
    report a single AMOUNT_MISMATCH_UNEXPLAINED naming both records with the exact delta.

    **This deliberately does not create a match link.** Pairing for diagnosis and matching
    for reconciliation are different acts with different burdens of proof: it is right to
    tell a reviewer "these two are probably related and disagree by 142 rupees", and wrong
    to record that the settlement reconciled. Returns the ids consumed so the
    missing/unexplained detectors skip them.
    """
    consumed_settlements: set[str] = set()
    consumed_credits: set[str] = set()
    available = list(unmatched_credits)

    def _score(settlement: Settlement, credit: BankCredit) -> tuple[int, int] | None:
        """Rank a possible residue pairing. Lower is better; None means implausible."""
        day_gap = abs(
            (epoch_to_ist_date(credit.value_date) - epoch_to_ist_date(settlement.created_at)).days
        )
        if settlement.utr and credit.utr_extracted and settlement.utr == credit.utr_extracted:
            return (0, day_gap)  # shared UTR is conclusive identity
        if day_gap > date_window_days:
            return None
        delta = abs(settlement.amount - credit.amount)
        if settlement.amount == 0 or delta > settlement.amount * relative_tolerance:
            return None
        return (1, delta)

    for settlement in unmatched_settlements:
        if settlement.status is not SettlementStatus.PROCESSED:
            continue
        ranked = sorted(
            (
                (score, credit)
                for credit in available
                if (score := _score(settlement, credit)) is not None
            ),
            key=lambda item: item[0],
        )
        if not ranked:
            continue

        (kind, _), credit = ranked[0]
        delta = credit.amount - settlement.amount
        basis = (
            "a shared UTR"
            if kind == 0
            else f"a near amount within {relative_tolerance:.0%} on an adjacent date"
        )
        builder.add(
            code=ExceptionCode.AMOUNT_MISMATCH_UNEXPLAINED,
            tier=Tier.SETTLEMENT_BANK,
            subject_ids=[settlement.id, credit.stmt_id],
            amount_at_risk=abs(delta),
            reason=f"Settlement {settlement.id} of {format_inr(settlement.amount)} appears "
            f"to correspond to bank credit {credit.stmt_id} of "
            f"{format_inr(credit.amount)} on {basis}, but the amounts differ by "
            f"{format_inr(delta)}, which fee and GST arithmetic does not explain.",
            evidence=[
                Evidence(
                    feature="settlement_amount",
                    observed=format_inr(settlement.amount),
                    supports=False,
                ),
                Evidence(
                    feature="bank_amount",
                    observed=format_inr(credit.amount),
                    supports=False,
                    detail=f"Delta {format_inr(delta)}; not a fee, GST or FX adjustment.",
                ),
                Evidence(
                    feature="pairing_basis",
                    observed=basis,
                    supports=True,
                    detail="Records paired for diagnosis only. No match has been recorded, "
                    "because the amounts do not agree.",
                ),
            ],
            recommended_action="Confirm the payout amount with the bank against the "
            "settlement breakdown; one of the two figures is wrong.",
            candidates_considered=[credit.stmt_id],
        )
        consumed_settlements.add(settlement.id)
        consumed_credits.add(credit.stmt_id)
        available = [c for c in available if c.stmt_id != credit.stmt_id]

    return consumed_settlements, consumed_credits


def detect_missing_in_pg(
    unmatched_orders: Sequence[Order], builder: _ExceptionBuilder
) -> None:
    """Orders the books call paid that no gateway row substantiates."""
    for order in unmatched_orders:
        builder.add(
            code=ExceptionCode.MISSING_IN_PG,
            tier=Tier.ORDER_PG,
            subject_ids=[order.order_id],
            amount_at_risk=order.amount,
            reason=f"Order {order.order_id} ({order.order_receipt}) is marked paid for "
            f"{format_inr(order.amount)} but no gateway payment row was found.",
            evidence=[
                Evidence(
                    feature="order_status",
                    observed=order.status.value,
                    supports=False,
                    detail="Books assert this order was paid.",
                ),
                Evidence(
                    feature="gateway_row",
                    observed="none found",
                    supports=False,
                    detail="No payment row matched on reference, amount or fee-adjusted amount.",
                ),
            ],
            recommended_action="Confirm whether the payment was actually captured; if not, "
            "correct the order status in the books.",
        )


def detect_missing_in_books(
    unmatched_rows: Sequence[ReconRow],
    builder: _ExceptionBuilder,
    *,
    known_order_ids: frozenset[str] = frozenset(),
) -> None:
    """Gateway rows referencing an order the merchant ledger does not contain.

    ``known_order_ids`` guards against a false-orphan trap. Assignment is one-to-one, so
    when two gateway rows both point at one order only one can win the match -- the loser
    lands in the unmatched list even though its order exists and is perfectly well known.
    Reporting that as MISSING_IN_BOOKS would be wrong twice over: the money is not
    unaccounted for, and the row's actual problem (it is a duplicate) is already reported
    by :func:`detect_duplicates`. So a row is only an orphan if its ``order_id`` appears
    nowhere in the ledger.
    """
    for row in unmatched_rows:
        if row.order_id and row.order_id in known_order_ids:
            # The order exists; this row lost a contested assignment rather than being
            # orphaned. Duplicate detection owns this case.
            continue
        builder.add(
            code=ExceptionCode.MISSING_IN_BOOKS,
            tier=Tier.ORDER_PG,
            subject_ids=[row.entity_id],
            amount_at_risk=row.amount,
            reason=f"Gateway row {row.entity_id} for {format_inr(row.amount)} references "
            f"order {row.order_id} / receipt {row.order_receipt}, which is absent from "
            "the merchant ledger.",
            evidence=[
                Evidence(
                    feature="order_reference",
                    observed=str(row.order_receipt),
                    supports=False,
                    detail="No order in the books carries this id, reference or amount.",
                ),
            ],
            recommended_action="Money was collected with no corresponding sale on record. "
            "Investigate before the next close.",
        )


def detect_missing_in_bank(
    unmatched_settlements: Sequence[Settlement], builder: _ExceptionBuilder
) -> None:
    """Settlements the gateway processed that never appeared in the bank.

    Failed settlements are excluded: their absence from the bank is explained by the
    failure, and reporting both codes would double-count the same rupees.
    """
    for settlement in unmatched_settlements:
        if settlement.status is not SettlementStatus.PROCESSED:
            continue
        builder.add(
            code=ExceptionCode.MISSING_IN_BANK,
            tier=Tier.SETTLEMENT_BANK,
            subject_ids=[settlement.id],
            amount_at_risk=settlement.amount,
            reason=f"Settlement {settlement.id} for {format_inr(settlement.amount)} is "
            f"marked processed (UTR {settlement.utr}) but no bank credit matches it, "
            "alone or in combination.",
            evidence=[
                Evidence(
                    feature="settlement_status",
                    observed=settlement.status.value,
                    supports=False,
                ),
                Evidence(
                    feature="bank_credit",
                    observed="none found",
                    supports=False,
                    detail="No credit matched on UTR, exact amount, or any exact subset sum.",
                ),
            ],
            recommended_action="Trace the UTR with the bank; this is unreceived cash.",
        )


def detect_unexplained_bank_credits(
    unmatched_credits: Sequence[BankCredit], builder: _ExceptionBuilder
) -> None:
    """Bank credits that no settlement explains."""
    for credit in unmatched_credits:
        builder.add(
            code=ExceptionCode.AMOUNT_MISMATCH_UNEXPLAINED,
            tier=Tier.SETTLEMENT_BANK,
            subject_ids=[credit.stmt_id],
            amount_at_risk=credit.amount,
            reason=f"Bank credit {credit.stmt_id} of {format_inr(credit.amount)} on "
            f"{epoch_to_ist_date(credit.value_date)} matches no settlement, alone or in "
            "combination.",
            evidence=[
                Evidence(
                    feature="narration",
                    observed=credit.narration[:80],
                    supports=False,
                    detail="No settlement UTR or amount corresponds to this credit.",
                ),
            ],
            recommended_action="Identify the payer; unexplained credits may be a different "
            "settlement cycle, another gateway, or a direct customer transfer.",
        )


def detect_duplicates(rows: Sequence[ReconRow], builder: _ExceptionBuilder) -> None:
    """Find duplicated gateway rows, distinguishing two different problems.

    * **DUPLICATE_IN_SOURCE** -- byte-identical economics under two ids, i.e. the export
      emitted the same row twice. An artefact; no money moved twice.
    * **DUPLICATE_PAYMENT** -- two distinct payments against one order. Real money was
      taken twice and a refund is probably owed.

    Separating these matters because the recommended actions are opposites: one is fixed
    by re-pulling a report, the other by refunding a customer.
    """
    by_signature: dict[tuple[object, ...], list[ReconRow]] = defaultdict(list)
    by_order: dict[str, list[ReconRow]] = defaultdict(list)

    for row in rows:
        if row.type is not TxnType.PAYMENT:
            continue
        by_signature[(row.order_id, row.amount, row.created_at, row.fee, row.tax)].append(row)
        if row.order_id:
            by_order[row.order_id].append(row)

    exact_duplicate_ids: set[str] = set()
    for group in by_signature.values():
        if len(group) < 2:
            continue
        exact_duplicate_ids.update(r.entity_id for r in group)
        primary, *copies = sorted(group, key=lambda r: r.entity_id)
        builder.add(
            code=ExceptionCode.DUPLICATE_IN_SOURCE,
            tier=Tier.ORDER_PG,
            subject_ids=[r.entity_id for r in group],
            amount_at_risk=sum(r.amount for r in copies),
            reason=f"{len(group)} gateway rows share order {primary.order_id}, amount "
            f"{format_inr(primary.amount)} and timestamp -- identical economics under "
            "different entity ids.",
            evidence=[
                Evidence(
                    feature="duplicate_signature",
                    observed=f"{len(group)} rows, identical amount/timestamp/fee",
                    supports=False,
                    detail="Consistent with a duplicated export rather than a double charge.",
                ),
            ],
            recommended_action="De-duplicate the ingest; confirm only one row settled.",
        )

    for order_id, group in by_order.items():
        if len(group) < 2:
            continue
        # Skip pure export artefacts already reported above.
        if all(r.entity_id in exact_duplicate_ids for r in group):
            continue
        ordered = sorted(group, key=lambda r: r.created_at)
        gap_seconds = ordered[-1].created_at - ordered[0].created_at
        builder.add(
            code=ExceptionCode.DUPLICATE_PAYMENT,
            tier=Tier.ORDER_PG,
            subject_ids=[order_id, *(r.entity_id for r in ordered)],
            amount_at_risk=sum(r.amount for r in ordered[1:]),
            reason=f"Order {order_id} carries {len(ordered)} distinct successful payments "
            f"totalling {format_inr(sum(r.amount for r in ordered))}, "
            f"{gap_seconds // 60} minutes apart.",
            evidence=[
                Evidence(
                    feature="payments_per_order",
                    observed=f"{len(ordered)} payments",
                    supports=False,
                    detail="Distinct entity ids with different timestamps: a real double charge.",
                ),
                Evidence(
                    feature="interval",
                    observed=f"{gap_seconds}s between first and last",
                    supports=False,
                ),
            ],
            recommended_action="Likely customer double-charge. Verify and refund the "
            "later payment.",
        )


# --------------------------------------------------------------------------------------
# Lifecycle, timing and cash position
# --------------------------------------------------------------------------------------


def detect_lifecycle_exceptions(
    settlements: Sequence[Settlement],
    rows: Sequence[ReconRow],
    builder: _ExceptionBuilder,
    *,
    as_of: object | None = None,
) -> None:
    """Failed settlements, dispute holds, and aged unsettled captures."""
    for settlement in settlements:
        if settlement.status is SettlementStatus.FAILED:
            builder.add(
                code=ExceptionCode.SETTLEMENT_FAILED,
                tier=Tier.SETTLEMENT_BANK,
                subject_ids=[settlement.id],
                amount_at_risk=settlement.amount,
                reason=f"Settlement {settlement.id} for {format_inr(settlement.amount)} "
                "failed; funds remain with the gateway.",
                evidence=[
                    Evidence(
                        feature="settlement_status", observed="failed", supports=False
                    ),
                ],
                recommended_action="Check bank account validity and request re-initiation. "
                "No bank credit is expected for this settlement.",
            )

    reference_day = as_of
    if reference_day is None:
        settled_days = [epoch_to_ist_date(r.settled_at) for r in rows if r.settled_at]
        created_days = [epoch_to_ist_date(r.created_at) for r in rows]
        reference_day = max(settled_days + created_days) if (settled_days or created_days) else None

    for row in rows:
        if row.on_hold and row.dispute_id:
            builder.add(
                code=ExceptionCode.DISPUTE_HOLD,
                tier=Tier.SETTLEMENT_NET,
                subject_ids=[row.entity_id],
                amount_at_risk=max(row.credit, 0),
                reason=f"Row {row.entity_id} ({format_inr(row.amount)}) is withheld from "
                f"settlement pending dispute {row.dispute_id}.",
                evidence=[
                    Evidence(feature="on_hold", observed="true", supports=False),
                    Evidence(
                        feature="dispute_id",
                        observed=str(row.dispute_id),
                        supports=False,
                        detail="Hold is explained by an open dispute; not a reconciliation break.",
                    ),
                ],
                recommended_action="Track the dispute to resolution; expect settlement or "
                "reversal once closed.",
            )
            continue

        if row.settled or row.type is not TxnType.PAYMENT or reference_day is None:
            continue

        due = expected_settlement_date(epoch_to_ist_date(row.created_at))
        overdue = working_days_between(due, reference_day)  # type: ignore[arg-type]
        if overdue > AGED_UNSETTLED_WORKING_DAYS:
            builder.add(
                code=ExceptionCode.UNSETTLED_AGED,
                tier=Tier.SETTLEMENT_NET,
                subject_ids=[row.entity_id],
                amount_at_risk=max(row.credit, 0),
                reason=f"Row {row.entity_id} captured "
                f"{epoch_to_ist_date(row.created_at)} was due to settle {due} but is "
                f"still unsettled {overdue} working days later.",
                evidence=[
                    Evidence(
                        feature="expected_settlement",
                        observed=str(due),
                        supports=False,
                        detail="T+2 working days from capture over the banking calendar.",
                    ),
                    Evidence(
                        feature="overdue_working_days",
                        observed=f"{overdue}",
                        supports=False,
                    ),
                ],
                recommended_action="Query why this capture has not entered a settlement "
                "batch; check for holds or balance shortfalls.",
            )


@dataclass
class CashPosition:
    """Forward cash view derived from what has not settled yet.

    This is arithmetic over known-unsettled rows, deliberately **not** a forecast model.
    On synthetic data a statistical forecast would be predicting a process we wrote,
    which is circular and unfalsifiable. Summing what is demonstrably still in flight is
    a claim that can be checked line by line.
    """

    expected_by_day: dict[object, int] = field(default_factory=dict)
    on_hold_total: int = 0
    aged_total: int = 0
    failed_total: int = 0
    unsettled_row_count: int = 0

    @property
    def total_in_flight(self) -> int:
        return sum(self.expected_by_day.values())

    def summary_lines(self) -> list[str]:
        lines = [
            f"In flight (expected to settle): {format_inr(self.total_in_flight)} "
            f"across {self.unsettled_row_count} rows"
        ]
        for day in sorted(self.expected_by_day):
            lines.append(f"  {day}: {format_inr(self.expected_by_day[day])}")
        if self.on_hold_total:
            lines.append(f"Withheld under dispute: {format_inr(self.on_hold_total)}")
        if self.aged_total:
            lines.append(f"Overdue past T+2: {format_inr(self.aged_total)}")
        if self.failed_total:
            lines.append(f"Stuck in failed settlements: {format_inr(self.failed_total)}")
        return lines


def compute_cash_position(
    rows: Sequence[ReconRow],
    settlements: Sequence[Settlement],
    *,
    as_of: object | None = None,
) -> CashPosition:
    """Project near-term inflow from rows that have not settled.

    Each unsettled row is attributed to its expected settlement date under the T+2
    working-day rule, so the projection respects weekends and bank holidays rather than
    implying money will arrive on a day the banks are shut.
    """
    position = CashPosition()

    for row in rows:
        if row.settled:
            continue
        net = row.credit - row.debit
        if row.on_hold:
            position.on_hold_total += max(net, 0)
            continue
        if row.type is TxnType.PAYMENT or net != 0:
            due = expected_settlement_date(epoch_to_ist_date(row.created_at))
            position.expected_by_day[due] = position.expected_by_day.get(due, 0) + net
            position.unsettled_row_count += 1
            if as_of is not None:
                overdue = working_days_between(due, as_of)  # type: ignore[arg-type]
                if overdue > AGED_UNSETTLED_WORKING_DAYS:
                    position.aged_total += max(net, 0)

    position.failed_total = sum(
        s.amount for s in settlements if s.status is SettlementStatus.FAILED
    )
    return position


# --------------------------------------------------------------------------------------
# Ambiguity reporting
# --------------------------------------------------------------------------------------


def report_ambiguity(
    *,
    subject_id: str,
    candidate_ids: Sequence[str],
    amount: int,
    tier: Tier,
    builder: _ExceptionBuilder,
    detail: str,
) -> None:
    """Record that the evidence does not determine a single answer.

    Raised when two or more candidates explain a record equally well. This is the
    exception the system is most proud of: the alternative is to pick the first one and
    report a match rate that looks better than the truth.
    """
    builder.add(
        code=ExceptionCode.AMBIGUOUS_MULTI_CANDIDATE,
        tier=tier,
        subject_ids=[subject_id],
        amount_at_risk=amount,
        reason=f"{subject_id} ({format_inr(amount)}) has {len(candidate_ids)} candidates "
        "that fit equally well; no evidence distinguishes them.",
        evidence=[
            Evidence(
                feature="competing_candidates",
                observed=", ".join(candidate_ids[:6]),
                supports=False,
                detail=detail,
            ),
        ],
        recommended_action="Human decision required. Obtain the bank reference or the "
        "gateway settlement breakdown to break the tie.",
        candidates_considered=candidate_ids,
    )


def new_builder() -> _ExceptionBuilder:
    """Public constructor for the exception accumulator."""
    return _ExceptionBuilder()


# --------------------------------------------------------------------------------------
# Review cases -- the human-in-the-loop unit of work
# --------------------------------------------------------------------------------------

#: Codes whose exposure is the *whole* transaction: money is entirely unaccounted for.
#: Several of these on one record still describe the same rupees, so they are combined by
#: taking the maximum rather than by adding.
_PRINCIPAL_EXPOSURE: Final[frozenset[ExceptionCode]] = frozenset(
    {
        ExceptionCode.MISSING_IN_PG,
        ExceptionCode.MISSING_IN_BOOKS,
        ExceptionCode.MISSING_IN_BANK,
        ExceptionCode.DUPLICATE_PAYMENT,
        ExceptionCode.DUPLICATE_IN_SOURCE,
        ExceptionCode.SETTLEMENT_FAILED,
        ExceptionCode.SETTLEMENT_NET_MISMATCH,
        ExceptionCode.AMOUNT_MISMATCH_UNEXPLAINED,
        ExceptionCode.AMBIGUOUS_MULTI_CANDIDATE,
        ExceptionCode.UNSETTLED_AGED,
        ExceptionCode.DISPUTE_HOLD,
    }
)

#: Codes whose exposure is a *difference* -- a fee over-charge, a tax error, FX drift.
#: These are genuinely additive with each other and with a principal exposure.
_DELTA_EXPOSURE: Final[frozenset[ExceptionCode]] = frozenset(
    {
        ExceptionCode.FEE_MISMATCH,
        ExceptionCode.GST_MISMATCH,
        ExceptionCode.FX_VARIANCE,
    }
)


@dataclass(frozen=True)
class ReviewCase:
    """One record needing human attention, with every finding against it attached.

    A reviewer works records, not findings. Presenting eight exceptions that turn out to
    concern three records wastes triage time and, if their amounts are naively summed,
    materially overstates the money at risk. Grouping by record fixes both.

    Exposure is combined by kind rather than by blind addition: several *principal*
    findings on one record describe the same rupees, so the largest is taken, while
    *delta* findings (a fee over-charge, a GST error) are true increments and are added.
    """

    case_id: str
    subject_id: str
    primary_code: ExceptionCode
    severity: Severity
    tier: Tier
    principal_exposure: int
    delta_exposure: int
    findings: tuple[ExceptionRecord, ...]
    recommended_action: str

    @property
    def total_exposure(self) -> int:
        return self.principal_exposure + self.delta_exposure

    @property
    def finding_codes(self) -> tuple[str, ...]:
        return tuple(f.code.value for f in self.findings)


def build_review_cases(exceptions: Sequence[ExceptionRecord]) -> list[ReviewCase]:
    """Group exceptions into per-record review cases, ordered by exposure.

    Grouping is by **connected component over shared record ids**, not by primary subject
    alone. Grouping on the primary id only is not enough: a double-charged order raises
    DUPLICATE_PAYMENT against the *order*, while the unsettled second leg raises
    UNSETTLED_AGED against the *payment row*. Those are the same rupees, and keyed
    separately they would appear as two cases and be added together -- roughly doubling
    the reported exposure for that order. Any two findings that mention a record in common
    therefore land in one case.
    """
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:  # path compression
            parent[x], x = root, parent[x]
        return root

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # Link every id an exception mentions to that exception's primary id.
    for exc in exceptions:
        ids = exc.subject_ids or (exc.exception_id,)
        anchor = ids[0]
        for other in ids:
            union(anchor, other)

    grouped: dict[str, list[ExceptionRecord]] = defaultdict(list)
    for exc in exceptions:
        anchor = (exc.subject_ids or (exc.exception_id,))[0]
        grouped[find(anchor)].append(exc)

    rank = {Severity.CRITICAL: 0, Severity.HIGH: 1, Severity.MEDIUM: 2, Severity.LOW: 3}
    cases: list[ReviewCase] = []

    for findings in grouped.values():
        ordered = sorted(findings, key=lambda e: (rank[e.severity], -e.amount_at_risk))
        primary = ordered[0]
        subject_id = primary.subject_ids[0] if primary.subject_ids else primary.exception_id

        principal = max(
            (f.amount_at_risk for f in findings if f.code in _PRINCIPAL_EXPOSURE), default=0
        )
        delta = sum(f.amount_at_risk for f in findings if f.code in _DELTA_EXPOSURE)

        cases.append(
            ReviewCase(
                case_id=f"CASE-{subject_id}",
                subject_id=subject_id,
                primary_code=primary.code,
                severity=primary.severity,
                tier=primary.tier,
                principal_exposure=principal,
                delta_exposure=delta,
                findings=tuple(ordered),
                recommended_action=primary.recommended_action,
            )
        )

    return sorted(cases, key=lambda c: (-c.total_exposure, rank[c.severity], c.subject_id))


def total_exposure(cases: Sequence[ReviewCase]) -> int:
    """Portfolio exposure across review cases.

    Safe to add because cases are one-per-record and each has already combined its own
    findings by kind, so no rupee is counted twice.
    """
    return sum(c.total_exposure for c in cases)
