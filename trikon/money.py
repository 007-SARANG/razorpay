"""Money arithmetic for Trikon.

Design rule, enforced everywhere in this codebase: **money is an ``int`` number of
paise.** Never a ``float``, never a ``Decimal`` that escapes this module.

Why this matters enough to be its own module: reconciliation is the act of asserting
that two independently-computed sums are equal. Floating point makes that assertion
unsound -- ``0.1 + 0.2 != 0.3`` -- so a float-based reconciler produces phantom
one-paise breaks that a reviewer then has to triage by hand. Integer paise makes
equality exact and makes every variance we *do* report a real one.

All rounding is explicit half-up on integers. We never rely on Python's banker's
rounding (``round()`` rounds 0.5 to even), because payment processors publish
half-up fee tables and a half-even implementation would disagree with them on
exactly the boundary cases that show up in a large batch.
"""

from __future__ import annotations

from typing import Final, NewType

# A quantity of Indian paise. 100 paise = 1 rupee.
Paise = NewType("Paise", int)

#: GST charged on payment-gateway *fees* (not on the transaction), as a percentage.
#: Razorpay's documented behaviour is that ``tax`` is levied on the collected ``fee``.
GST_PERCENT: Final[int] = 18

#: Basis points of the transaction amount charged as fee, per payment method.
#: These are representative synthetic rates, NOT Razorpay's published price list --
#: the generator and the verifier share this table, so the reconciler's job is to
#: detect rows where the *recorded* fee disagrees with the *recomputed* fee.
FEE_BPS_BY_METHOD: Final[dict[str, int]] = {
    "upi": 0,  # UPI is famously zero-MDR for most merchants in India
    "netbanking": 190,
    "wallet": 200,
    "card": 200,
    "emi": 300,
    "international": 430,
}


def rupees_to_paise(rupees: float | int | str) -> Paise:
    """Convert a rupee quantity to integer paise, rounding half-up.

    Accepts ``str`` so that decimal literals from CSV/JSON can be converted without
    ever passing through a binary float.
    """
    if isinstance(rupees, str):
        text = rupees.strip().replace(",", "")
        if not text:
            raise ValueError("empty rupee string")
        neg = text.startswith("-")
        if neg:
            text = text[1:]
        whole, _, frac = text.partition(".")
        whole_i = int(whole or "0")
        # Pad/truncate the fractional part to exactly 2 digits, rounding half-up on
        # the third digit if present.
        frac_digits = (frac + "000")[:3]
        paise = whole_i * 100 + int(frac_digits[:2])
        if int(frac_digits[2]) >= 5:
            paise += 1
        return Paise(-paise if neg else paise)
    if isinstance(rupees, int):
        return Paise(rupees * 100)
    return Paise(_round_half_up(round(rupees * 1000), 10))


def paise_to_rupee_str(paise: int) -> str:
    """Render paise as a plain ``-1234.56`` style string (no symbol, no grouping)."""
    sign = "-" if paise < 0 else ""
    a = abs(int(paise))
    return f"{sign}{a // 100}.{a % 100:02d}"


def format_inr(paise: int) -> str:
    """Render paise in Indian grouping with a rupee symbol, e.g. ``₹12,34,567.89``.

    Used only for display. Indian grouping puts the last three digits together and
    then groups in pairs, which is why this is not ``f"{n:,}"``.
    """
    sign = "-" if paise < 0 else ""
    a = abs(int(paise))
    whole, frac = divmod(a, 100)
    s = str(whole)
    if len(s) > 3:
        head, tail = s[:-3], s[-3:]
        parts: list[str] = []
        while len(head) > 2:
            parts.insert(0, head[-2:])
            head = head[:-2]
        if head:
            parts.insert(0, head)
        s = ",".join([*parts, tail])
    return f"{sign}₹{s}.{frac:02d}"


def _round_half_up(numerator: int, denominator: int) -> int:
    """Integer division rounding halves away from zero.

    Kept private and used by every fee/tax computation so that rounding policy lives
    in exactly one place and can be pointed at during a review.
    """
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    if numerator >= 0:
        return (numerator * 2 + denominator) // (denominator * 2)
    return -((-numerator * 2 + denominator) // (denominator * 2))


def compute_fee(amount: int, method: str, *, international: bool = False) -> Paise:
    """Recompute the gateway fee for a transaction, in paise.

    This is the *independent* recomputation the reconciler compares against the fee
    reported on a settlement row. Any disagreement is a ``FEE_MISMATCH`` exception.
    """
    key = "international" if international else method
    bps = FEE_BPS_BY_METHOD.get(key)
    if bps is None:
        raise KeyError(f"no fee rate configured for method {method!r}")
    return Paise(_round_half_up(amount * bps, 10_000))


def compute_gst_on_fee(fee: int) -> Paise:
    """Recompute GST on a fee, in paise."""
    return Paise(_round_half_up(fee * GST_PERCENT, 100))


def compute_net_credit(amount: int, method: str, *, international: bool = False) -> tuple[Paise, Paise, Paise]:
    """Return ``(fee, tax, net_credit)`` for a captured payment.

    ``net_credit`` is what should actually reach the merchant's bank for this row:
    gross, minus fee, minus GST on that fee. The three-way reconciler asserts that
    the sum of ``net_credit`` over a settlement's member rows equals the settlement
    amount, which equals the bank credit.
    """
    fee = compute_fee(amount, method, international=international)
    tax = compute_gst_on_fee(fee)
    return fee, tax, Paise(amount - fee - tax)
