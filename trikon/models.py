"""Domain entities, exception taxonomy, and the rule/confidence contract.

The three source entities deliberately mirror **Razorpay's real published field
names** (verified against their Settlements API and ``settlements/recon/combined``
docs) rather than inventing a tidier schema. A reviewer who works on this system
daily should recognise every field, and any exception the reconciler raises should be
expressible in the vocabulary their own reports already use.

The one place we diverge is money: Razorpay returns ``amount``/``fee``/``tax`` as
integers in paise, and we keep that, but we suffix nothing and document the unit here
once -- **every monetary field in this module is integer paise.**
"""

from __future__ import annotations

from enum import Enum
from typing import Final

from pydantic import BaseModel, ConfigDict, Field


class _Frozen(BaseModel):
    """Immutable base. Source records are evidence; nothing should mutate them."""

    model_config = ConfigDict(frozen=True, extra="forbid")


# --------------------------------------------------------------------------------------
# Enumerations
# --------------------------------------------------------------------------------------


class Method(str, Enum):
    """Payment method, matching Razorpay's ``method`` field values."""

    CARD = "card"
    NETBANKING = "netbanking"
    WALLET = "wallet"
    UPI = "upi"
    EMI = "emi"


class TxnType(str, Enum):
    """Row type in the settlement recon report, matching Razorpay's ``type`` values."""

    PAYMENT = "payment"
    REFUND = "refund"
    TRANSFER = "transfer"
    ADJUSTMENT = "adjustment"


class SettlementStatus(str, Enum):
    """Settlement lifecycle, matching Razorpay's documented ``status`` values."""

    CREATED = "created"
    PROCESSED = "processed"
    FAILED = "failed"


class OrderStatus(str, Enum):
    """Order lifecycle in the merchant's own book of record."""

    CREATED = "created"
    PAID = "paid"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Tier(str, Enum):
    """Which of the three reconciliation questions a link or exception belongs to."""

    ORDER_PG = "tier1_order_pg"
    SETTLEMENT_NET = "tier2_settlement_net"
    SETTLEMENT_BANK = "tier3_settlement_bank"


class MatchRule(str, Enum):
    """The deterministic rule that produced a decision.

    Confidence is a property of *which rule fired*, never a number a model reported
    about itself. See :data:`RULE_CONFIDENCE_CEILING`.

    The ladder is ordered by strength of evidence, and two of the rungs depend on
    *uniqueness* rather than on the pair alone: R4 and R5 only fire when no competing
    candidate explains the same record equally well. That is what separates a
    reference-less match that is genuinely determined (one exact amount on one date)
    from one that is a coin flip (two identical amounts on the same date) -- the same
    pairwise features, opposite correct verdicts.
    """

    R1_EXACT = "R1_EXACT"
    R2_FEE_EXPLAINED = "R2_FEE_EXPLAINED"
    R3_TIMING_SHIFTED = "R3_TIMING_SHIFTED"
    R4_UNIQUE_AMOUNT = "R4_UNIQUE_AMOUNT"
    R5_SUBSET_SUM = "R5_SUBSET_SUM"
    R6_FUZZY_REF = "R6_FUZZY_REF"
    R7_AMBIGUOUS = "R7_AMBIGUOUS"
    R8_NO_MATCH = "R8_NO_MATCH"


#: The maximum confidence each rule may ever yield.
#:
#: This table is the system's core safety invariant. The LLM adjudicator runs only on
#: R6/R7 candidates and may **lower** a confidence or escalate, but it can never raise
#: one above its rule's ceiling. Consequence: no model hallucination can manufacture a
#: high-confidence false match, because the arithmetic-backed rules are the only path
#: to a high-confidence band.
RULE_CONFIDENCE_CEILING: Final[dict[MatchRule, float]] = {
    MatchRule.R1_EXACT: 1.00,
    MatchRule.R2_FEE_EXPLAINED: 0.99,
    MatchRule.R5_SUBSET_SUM: 0.95,
    MatchRule.R3_TIMING_SHIFTED: 0.92,
    MatchRule.R4_UNIQUE_AMOUNT: 0.90,
    MatchRule.R6_FUZZY_REF: 0.70,
    MatchRule.R7_AMBIGUOUS: 0.50,
    MatchRule.R8_NO_MATCH: 0.00,
}

#: Rules whose confidence derives entirely from deterministic arithmetic and exact
#: reference equality. These never consult a model.
DETERMINISTIC_RULES: Final[frozenset[MatchRule]] = frozenset(
    {
        MatchRule.R1_EXACT,
        MatchRule.R2_FEE_EXPLAINED,
        MatchRule.R3_TIMING_SHIFTED,
        MatchRule.R4_UNIQUE_AMOUNT,
        MatchRule.R5_SUBSET_SUM,
    }
)

#: Rules that are eligible for LLM adjudication. Adjudication may confirm within the
#: rule's band or escalate; it may never promote to a higher band.
ADJUDICABLE_RULES: Final[frozenset[MatchRule]] = frozenset(
    {MatchRule.R6_FUZZY_REF, MatchRule.R7_AMBIGUOUS}
)

#: Confidence at or above which a link is auto-accepted with no human review.
#: Deliberately a named constant: the evaluation sweeps it and reports the
#: precision/recall/straight-through tradeoff at each value.
AUTO_ACCEPT_THRESHOLD: Final[float] = 0.90


class ExceptionCode(str, Enum):
    """Every way a record can fail to reconcile.

    Each code has a deterministic detector. The LLM may write the human-readable
    narrative for an exception, but it never chooses the code from free text -- it
    selects from this closed set, and an invalid selection is rejected.
    """

    # Presence failures
    MISSING_IN_PG = "MISSING_IN_PG"
    MISSING_IN_BOOKS = "MISSING_IN_BOOKS"
    MISSING_IN_BANK = "MISSING_IN_BANK"

    # Arithmetic failures
    AMOUNT_MISMATCH_UNEXPLAINED = "AMOUNT_MISMATCH_UNEXPLAINED"
    FEE_MISMATCH = "FEE_MISMATCH"
    GST_MISMATCH = "GST_MISMATCH"
    SETTLEMENT_NET_MISMATCH = "SETTLEMENT_NET_MISMATCH"
    FX_VARIANCE = "FX_VARIANCE"

    # Structural failures
    DUPLICATE_IN_SOURCE = "DUPLICATE_IN_SOURCE"
    DUPLICATE_PAYMENT = "DUPLICATE_PAYMENT"
    SPLIT_SETTLEMENT_INCOMPLETE = "SPLIT_SETTLEMENT_INCOMPLETE"
    AMBIGUOUS_MULTI_CANDIDATE = "AMBIGUOUS_MULTI_CANDIDATE"

    # Timing and lifecycle
    TIMING_BREACH = "TIMING_BREACH"
    UNSETTLED_AGED = "UNSETTLED_AGED"
    SETTLEMENT_FAILED = "SETTLEMENT_FAILED"
    DISPUTE_HOLD = "DISPUTE_HOLD"


class Severity(str, Enum):
    """Review priority. Derived from exception code and rupees at risk, never guessed."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Decision(str, Enum):
    """Outcome of adjudicating an ambiguous candidate."""

    MATCH = "MATCH"
    NO_MATCH = "NO_MATCH"
    ESCALATE = "ESCALATE"


# --------------------------------------------------------------------------------------
# Source A -- the merchant's own book of record (OMS / ERP export)
# --------------------------------------------------------------------------------------


class Order(_Frozen):
    """An order as the merchant's own system believes it happened.

    This is the "should have been paid" side of the triangle. Crucially it carries a
    merchant-controlled ``order_receipt``, which is the field most prone to
    human-entered variation and therefore the main source of fuzzy-reference work.
    """

    order_id: str
    order_receipt: str
    amount: int = Field(description="Gross order value in paise")
    currency: str = "INR"
    status: OrderStatus
    created_at: int = Field(description="Unix epoch seconds")
    customer_ref: str | None = None
    notes: str | None = None


# --------------------------------------------------------------------------------------
# Source B -- Razorpay settlement recon rows and settlement headers
# --------------------------------------------------------------------------------------


class ReconRow(_Frozen):
    """One row of ``GET /v1/settlements/recon/combined``.

    Field names and semantics follow Razorpay's documentation exactly, including the
    quirks: ``payment_id`` is null for rows of type ``payment``; adjustments carry no
    ``settlement_utr`` and no card/method detail; ``debit`` and ``credit`` are
    mutually exclusive and ``amount`` is the gross of whichever side applies.
    """

    entity_id: str = Field(description="pay_/rfnd_/trf_/adj_ prefixed id")
    type: TxnType
    debit: int = 0
    credit: int = 0
    amount: int = Field(description="Gross amount in paise")
    currency: str = "INR"
    fee: int = 0
    tax: int = Field(default=0, description="GST charged on `fee`")
    on_hold: bool = False
    settled: bool = False
    created_at: int
    settled_at: int | None = None
    settlement_id: str | None = None
    description: str | None = None
    payment_id: str | None = Field(default=None, description="Null for type=payment")
    settlement_utr: str | None = None
    order_id: str | None = None
    order_receipt: str | None = None
    method: Method | None = None
    card_network: str | None = None
    card_issuer: str | None = None
    card_type: str | None = None
    dispute_id: str | None = None

    # Non-Razorpay field, used only for international rows so the reconciler can
    # verify the INR conversion instead of treating FX drift as an unexplained break.
    fx_rate_at_creation: float | None = None
    original_currency: str | None = None
    original_amount: int | None = None


class Settlement(_Frozen):
    """A settlement header, per Razorpay's Settlements entity.

    ``amount`` is what Razorpay says it paid out. Tier 2 asserts this equals the sum
    of member rows net of fee and GST; tier 3 asserts it equals what the bank credited.
    """

    id: str = Field(description="setl_ prefixed id")
    amount: int
    status: SettlementStatus
    fees: int = 0
    tax: int = 0
    utr: str | None = None
    created_at: int


# --------------------------------------------------------------------------------------
# Source C -- the bank statement
# --------------------------------------------------------------------------------------


class BankCredit(_Frozen):
    """A credit line on the merchant's bank statement.

    The bank does not know about payments, orders or fees -- it knows a date, an
    amount, and a free-text narration in which a UTR may or may not be legible. That
    asymmetry is the whole difficulty of tier 3, so the narration is kept raw and the
    extracted UTR is stored separately alongside it as derived data.
    """

    stmt_id: str
    value_date: int = Field(description="Unix epoch seconds of the value date")
    amount: int = Field(description="Credited amount in paise; always positive")
    narration: str = Field(description="Raw bank narration, noise included")
    utr_extracted: str | None = Field(
        default=None, description="UTR parsed out of narration, if one was legible"
    )
    bank_ref: str | None = None


# --------------------------------------------------------------------------------------
# Ground truth -- written by the generator, read only by the evaluator
# --------------------------------------------------------------------------------------


class TrueLink(_Frozen):
    """A match that genuinely exists, as recorded by the generator.

    The evaluator compares the reconciler's produced links against these. Nothing in
    the reconciliation pipeline is permitted to read ground truth -- it is loaded by
    ``evaluate.py`` alone, which is what makes the reported metrics meaningful.
    """

    tier: Tier
    left_id: str
    right_id: str


class InjectedDefect(_Frozen):
    """A defect the generator deliberately introduced, with its expected consequence."""

    defect_code: str
    affected_ids: tuple[str, ...]
    expected_exception: ExceptionCode | None = Field(
        default=None,
        description="The code the reconciler should raise. None means the defect is "
        "expected to be absorbed and explained without becoming an exception.",
    )
    amount_at_risk: int = 0
    note: str | None = None


class GroundTruth(_Frozen):
    """Everything the generator knows and the reconciler must not."""

    seed: int
    generated_at: int
    links: tuple[TrueLink, ...]
    defects: tuple[InjectedDefect, ...]

    def links_for(self, tier: Tier) -> frozenset[tuple[str, str]]:
        """Link pairs for one tier, as an order-insensitive set for scoring."""
        return frozenset((link.left_id, link.right_id) for link in self.links if link.tier is tier)


# --------------------------------------------------------------------------------------
# Reconciliation output
# --------------------------------------------------------------------------------------


class Evidence(_Frozen):
    """One checkable statement supporting a decision.

    Evidence is what makes a match defensible rather than asserted. Each item names
    the feature examined, the observed value, and whether it supported or opposed the
    match, so a reviewer can disagree with a specific line rather than the verdict.
    """

    feature: str
    observed: str
    supports: bool
    detail: str | None = None


class MatchLink(_Frozen):
    """A produced match between two records, with its full justification."""

    tier: Tier
    left_id: str
    right_id: str
    rule: MatchRule
    confidence: float = Field(ge=0.0, le=1.0)
    auto_accepted: bool
    evidence: tuple[Evidence, ...]
    adjudicated_by: str | None = Field(
        default=None, description="Model id if an LLM adjudicated; None if purely deterministic"
    )
    reasoning: str | None = None

    # Members are populated for N:M links (a bank credit covering several settlements).
    member_ids: tuple[str, ...] = ()


class ExceptionRecord(_Frozen):
    """A record the reconciler could not resolve, and why.

    This is the honest half of the output and the part the track cares most about.
    ``amount_at_risk`` drives review ordering, because a finance controller triages by
    rupees exposed, not by row order.
    """

    exception_id: str
    code: ExceptionCode
    tier: Tier
    severity: Severity
    subject_ids: tuple[str, ...]
    amount_at_risk: int
    reason: str
    evidence: tuple[Evidence, ...]
    recommended_action: str
    candidates_considered: tuple[str, ...] = ()
    narrative: str | None = Field(
        default=None, description="Optional LLM-written prose; never load-bearing"
    )
