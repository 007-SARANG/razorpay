"""Pairwise feature extraction and the deterministic rule ladder.

Scoring is split deliberately into two questions, because they have different answers:

1. **What is true about this pair?** -- :func:`compute_features`, pure and local.
2. **What rule does that justify?** -- :func:`provisional_rule`, which produces a
   *provisional* verdict only.

The provisional verdict is not final because two of the strongest rungs depend on
uniqueness, which is a property of the whole candidate set rather than of one pair. A
bank credit with an illegible narration matching exactly one settlement on amount and
date is nearly certain; the identical pair becomes a coin flip the moment a second
settlement shares that amount and date. :mod:`trikon.assign` resolves that globally and
finalises the rule.

Every rule carries its evidence with it. A confidence number with no attached reasons is
an assertion; the same number with the four comparisons that produced it is a claim a
reviewer can argue with, which is the only kind worth shipping in finance software.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from rapidfuzz import fuzz

from trikon.calendar_ist import working_days_between
from trikon.money import compute_fee, compute_gst_on_fee, format_inr
from trikon.models import Evidence, MatchRule, Tier
from trikon.normalize import NormRecord

#: Reference similarity at or above which two references are considered plausibly the
#: same string with transcription damage. Below this they are treated as unrelated.
#: Chosen conservatively: at 0.85 a mutated receipt still clears, while two genuinely
#: different invoice numbers in the same series do not.
FUZZY_REF_THRESHOLD: Final[float] = 0.85

#: Working-day tolerance for a settlement landing later than expected before it is
#: called a timing breach rather than normal variation.
SETTLEMENT_GRACE_WORKING_DAYS: Final[int] = 1

#: Methods to try when attempting to explain an amount gap as fee plus GST. The
#: counterpart's method is often unknown at comparison time, so we test the table.
_FEE_METHODS: Final[tuple[str, ...]] = (
    "card",
    "netbanking",
    "wallet",
    "emi",
    "international",
    "upi",
)


@dataclass(frozen=True, slots=True)
class PairFeatures:
    """Everything comparable about a candidate pair, computed without judgement."""

    ref_exact: bool
    ref_loose_exact: bool
    ref_similarity: float
    both_refs_present: bool
    either_ref_missing: bool

    amount_delta: int
    amount_exact: bool
    fee_explained_by: str | None
    fee_explained_amount: int

    calendar_day_delta: int
    working_day_delta: int
    within_expected_window: bool

    @property
    def fee_explained(self) -> bool:
        return self.fee_explained_by is not None


def compute_features(
    left: NormRecord,
    right: NormRecord,
    *,
    tier: Tier,
    expected_cycle_days: int = 2,
) -> PairFeatures:
    """Compare two records across reference, amount and time.

    ``amount_delta`` is computed against the counterpart's canonical amount, and the
    fee-explanation search additionally tests the counterpart's *alternative* amount.
    That covers the most common real comparison error -- lining a gross order value up
    against a net settlement figure -- and lets it be reported as "explained by fee and
    GST" rather than as an unexplained variance a human then has to rediscover.
    """
    lref, rref = left.ref_strict, right.ref_strict
    both_present = bool(lref) and bool(rref)
    ref_exact = both_present and lref == rref
    ref_loose_exact = bool(left.ref_loose) and left.ref_loose == right.ref_loose

    similarity = 0.0
    if both_present:
        similarity = fuzz.ratio(lref, rref) / 100.0

    delta = left.amount - right.amount
    amount_exact = delta == 0

    fee_by, fee_amount = _explain_by_fee(left, right)

    cal_delta = (right.day - left.day).days
    wd_delta = working_days_between(left.day, right.day)

    if tier is Tier.SETTLEMENT_BANK:
        # A settlement should appear in the bank on the settlement date, or at worst one
        # working day later.
        within = -1 <= wd_delta <= SETTLEMENT_GRACE_WORKING_DAYS
    else:
        # An order and its payment are near-simultaneous; the settlement cycle is not
        # relevant at tier 1.
        within = abs(cal_delta) <= 1

    return PairFeatures(
        ref_exact=ref_exact,
        ref_loose_exact=ref_loose_exact,
        ref_similarity=similarity,
        both_refs_present=both_present,
        either_ref_missing=not both_present,
        amount_delta=delta,
        amount_exact=amount_exact,
        fee_explained_by=fee_by,
        fee_explained_amount=fee_amount,
        calendar_day_delta=cal_delta,
        working_day_delta=wd_delta,
        within_expected_window=within,
    )


def _explain_by_fee(left: NormRecord, right: NormRecord) -> tuple[str | None, int]:
    """Try to account for an amount gap as exactly fee plus GST.

    Returns ``(method_that_explains_it, fee_plus_tax)`` or ``(None, 0)``. The match must
    be **exact to the paise** -- an approximate explanation is not an explanation, and
    accepting one would reintroduce the tolerance that the one-paise adversarial case
    exists to detect.
    """
    targets = [right.amount]
    if right.alt_amount is not None:
        targets.append(right.alt_amount)
    sources = [left.amount]
    if left.alt_amount is not None:
        sources.append(left.alt_amount)

    for gross in sources:
        for net in targets:
            gap = gross - net
            if gap <= 0:
                continue
            for method in _FEE_METHODS:
                try:
                    fee = compute_fee(gross, method)
                except KeyError:  # pragma: no cover
                    continue
                if fee == 0:
                    continue
                if fee + compute_gst_on_fee(fee) == gap:
                    return method, gap
    return None, 0


def provisional_rule(features: PairFeatures) -> tuple[MatchRule, float]:
    """Map features to a provisional rule and its base score.

    "Provisional" because R4 additionally requires uniqueness, which cannot be known
    here. :mod:`trikon.assign` either confirms R4 or demotes it to R7_AMBIGUOUS.

    The base score doubles as the objective the global assignment maximises, so it is
    monotone in evidence strength rather than being a bare rule label.
    """
    from trikon.models import RULE_CONFIDENCE_CEILING as CEIL

    if features.ref_exact and features.amount_exact and features.within_expected_window:
        return MatchRule.R1_EXACT, CEIL[MatchRule.R1_EXACT]

    if features.ref_exact and features.fee_explained:
        return MatchRule.R2_FEE_EXPLAINED, CEIL[MatchRule.R2_FEE_EXPLAINED]

    if features.ref_exact and features.amount_exact:
        # Reference and amount agree exactly but the timing does not.
        return MatchRule.R3_TIMING_SHIFTED, CEIL[MatchRule.R3_TIMING_SHIFTED]

    if features.either_ref_missing and features.amount_exact and features.within_expected_window:
        # Strong *only if unique*; assign.py decides.
        return MatchRule.R4_UNIQUE_AMOUNT, CEIL[MatchRule.R4_UNIQUE_AMOUNT]

    if (
        features.both_refs_present
        and (features.ref_loose_exact or features.ref_similarity >= FUZZY_REF_THRESHOLD)
        and (features.amount_exact or features.fee_explained)
    ):
        # Reference is damaged but recoverable; amount corroborates. Eligible for
        # adjudication, capped well below auto-accept.
        return MatchRule.R6_FUZZY_REF, CEIL[MatchRule.R6_FUZZY_REF]

    return MatchRule.R8_NO_MATCH, 0.0


def build_evidence(
    left: NormRecord,
    right: NormRecord,
    features: PairFeatures,
    rule: MatchRule,
) -> tuple[Evidence, ...]:
    """Render the features as a reviewable evidence chain.

    Both supporting and opposing items are included. An evidence chain that only lists
    reasons to agree is marketing; the opposing lines are what let a reviewer locate the
    weak point in a decision quickly.
    """
    items: list[Evidence] = []

    if features.ref_exact:
        items.append(
            Evidence(
                feature="reference",
                observed=f"{left.ref_strict} == {right.ref_strict}",
                supports=True,
                detail="Normalised references are identical.",
            )
        )
    elif features.both_refs_present:
        items.append(
            Evidence(
                feature="reference",
                observed=f"{left.ref_strict} ~ {right.ref_strict} ({features.ref_similarity:.2f})",
                supports=features.ref_similarity >= FUZZY_REF_THRESHOLD,
                detail="Similarity above threshold; treated as transcription damage."
                if features.ref_similarity >= FUZZY_REF_THRESHOLD
                else "Similarity below threshold; references treated as unrelated.",
            )
        )
    else:
        missing = "left" if not left.ref_strict else "right"
        items.append(
            Evidence(
                feature="reference",
                observed=f"absent on {missing} record",
                supports=False,
                detail="No usable reference; match must rest on amount and date alone.",
            )
        )

    if features.amount_exact:
        items.append(
            Evidence(
                feature="amount",
                observed=f"{format_inr(left.amount)} exact",
                supports=True,
                detail="Amounts agree to the paise.",
            )
        )
    elif features.fee_explained:
        items.append(
            Evidence(
                feature="amount",
                observed=f"gap {format_inr(features.fee_explained_amount)}",
                supports=True,
                detail=f"Gap equals fee plus 18% GST for method "
                f"'{features.fee_explained_by}', exact to the paise.",
            )
        )
    else:
        items.append(
            Evidence(
                feature="amount",
                observed=f"delta {format_inr(features.amount_delta)}",
                supports=False,
                detail="Gap is not explained by fee or GST arithmetic.",
            )
        )

    items.append(
        Evidence(
            feature="timing",
            observed=f"{features.working_day_delta:+d} working days "
            f"({features.calendar_day_delta:+d} calendar)",
            supports=features.within_expected_window,
            detail="Within the expected settlement window."
            if features.within_expected_window
            else "Outside the expected window; may be a timing breach.",
        )
    )

    if rule is MatchRule.R4_UNIQUE_AMOUNT:
        items.append(
            Evidence(
                feature="uniqueness",
                observed="sole candidate on amount and date",
                supports=True,
                detail="No competing record explains this one equally well.",
            )
        )

    return tuple(items)
