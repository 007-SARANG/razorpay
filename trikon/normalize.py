"""Normalisation: turn three dissimilar sources into one comparable substrate.

Matching heterogeneous records is mostly a normalisation problem wearing a matching
problem's clothes. An order has a gross amount and a merchant receipt; a gateway row
has a gross *and* a net and two different reference fields; a bank line has a lump sum
and a free-text narration. Comparing them directly means writing a different comparison
for every pair of sources.

Instead every record is projected onto a single :class:`NormRecord` shape, and all
downstream blocking, scoring and assignment operates on that shape alone. The tiers
then differ only in *which* records are projected and which amount is chosen as
canonical -- not in the matching algorithm.

Two normalised reference forms are produced per record:

* ``ref_strict`` -- uppercased, non-alphanumerics removed. Safe for equality.
* ``ref_loose`` -- additionally folds the character confusions that actually occur in
  hand-transcribed finance references (``O``/``0``, ``I``/``1``). Used for *blocking
  only*, never for asserting a match, because folding is lossy and two genuinely
  different references can collide under it.

That split matters: ``ref_loose`` is allowed to over-generate candidates because the
scorer will reject bad ones, whereas over-generous equality would create false matches.
"""

from __future__ import annotations

import datetime as _dt
import re
from dataclasses import dataclass, field
from typing import Final, Iterable

from trikon.calendar_ist import epoch_to_ist_date
from trikon.models import (
    BankCredit,
    Method,
    Order,
    OrderStatus,
    ReconRow,
    Settlement,
    TxnType,
)

_NON_ALNUM: Final[re.Pattern[str]] = re.compile(r"[^A-Za-z0-9]+")

#: A UTR as Indian banks emit it: a 4-letter bank code then 10-16 digits. Anchored on
#: word boundaries so it does not chop a longer token in half.
_UTR_PATTERN: Final[re.Pattern[str]] = re.compile(r"\b([A-Z]{4}[A-Z0-9]?\d{10,16})\b")

#: Fallback: any long alphanumeric run that plausibly carries a reference. Used only if
#: the strict UTR shape is absent, and flagged as low-confidence by the caller.
_LOOSE_REF_PATTERN: Final[re.Pattern[str]] = re.compile(r"\b([A-Z0-9]{12,22})\b")

_LOOSE_FOLD: Final[dict[int, str]] = str.maketrans({"O": "0", "I": "1", "L": "1"})


def canon_strict(value: str | None) -> str:
    """Uppercase and strip every non-alphanumeric character.

    ``"INV-202607/00421 "`` and ``"inv 202607 00421"`` both become ``"INV20260700421"``.
    Equality on this form is safe to treat as an exact reference match.
    """
    if not value:
        return ""
    return _NON_ALNUM.sub("", value).upper()


def canon_loose(value: str | None) -> str:
    """Strict form plus digit/letter confusion folding. For blocking only."""
    return canon_strict(value).translate(_LOOSE_FOLD)


def extract_utr(narration: str | None) -> tuple[str | None, bool]:
    """Pull a UTR out of a bank narration.

    Returns ``(utr, is_strict)``. ``is_strict`` is False when the value came from the
    permissive fallback pattern, which lets the scorer weight it less. Returning the
    confidence alongside the value -- rather than silently accepting a guess -- is what
    keeps an unreadable narration from being treated as a reliable reference.
    """
    if not narration:
        return None, False
    upper = narration.upper()
    strict = _UTR_PATTERN.search(upper)
    if strict:
        return strict.group(1), True
    loose = _LOOSE_REF_PATTERN.search(upper)
    if loose:
        return loose.group(1), False
    return None, False


@dataclass(slots=True)
class NormRecord:
    """A record projected into the common matching shape.

    ``amount`` is the *canonical amount for this tier* -- gross for tier 1, net for
    tier 3 -- chosen by the projector, so the scorer never has to know which tier it is
    working on.

    ``alt_amount`` carries the other reading where one exists (e.g. an order's gross
    when the canonical figure is net). The scorer uses it to recognise the common
    books-versus-net comparison error and explain it as fee + GST rather than reporting
    an unexplained variance.
    """

    record_id: str
    source: str  # "order" | "pg" | "settlement" | "bank"
    amount: int
    day: _dt.date
    epoch: int
    ref_strict: str = ""
    ref_loose: str = ""
    alt_refs: tuple[str, ...] = ()
    alt_amount: int | None = None
    method: Method | None = None
    ref_is_strict: bool = True
    tags: frozenset[str] = frozenset()
    meta: dict[str, object] = field(default_factory=dict)

    def all_ref_forms(self) -> tuple[str, ...]:
        """Every reference string this record can legitimately be blocked on."""
        forms = {self.ref_strict, self.ref_loose, *self.alt_refs}
        return tuple(f for f in forms if f)


# --------------------------------------------------------------------------------------
# Projectors -- one per source. Each decides the canonical amount for its tier.
# --------------------------------------------------------------------------------------


def project_orders(orders: Iterable[Order], *, paid_only: bool = True) -> list[NormRecord]:
    """Project the merchant's order ledger for tier-1 matching.

    Only ``paid`` orders are projected by default. Unpaid orders legitimately have no
    gateway row, and including them would manufacture MISSING_IN_PG exceptions for
    records that are behaving correctly -- the single easiest way to fake a good
    exception count while actually being wrong.
    """
    out: list[NormRecord] = []
    for order in orders:
        if paid_only and order.status is not OrderStatus.PAID:
            continue
        receipt_strict = canon_strict(order.order_receipt)
        out.append(
            NormRecord(
                record_id=order.order_id,
                source="order",
                amount=order.amount,
                day=epoch_to_ist_date(order.created_at),
                epoch=order.created_at,
                ref_strict=receipt_strict,
                ref_loose=canon_loose(order.order_receipt),
                alt_refs=(canon_strict(order.order_id),),
                tags=frozenset({order.status.value}),
                meta={"currency": order.currency, "receipt_raw": order.order_receipt},
            )
        )
    return out


def project_pg_payments(rows: Iterable[ReconRow]) -> list[NormRecord]:
    """Project gateway payment rows for tier-1 matching.

    Canonical amount is the **gross** ``amount``, because that is what the merchant's
    order carries. The net ``credit`` is kept as ``alt_amount`` so that a comparison
    made against net instead of gross is recognisable as fee-explained rather than
    reported as a variance.
    """
    out: list[NormRecord] = []
    for row in rows:
        if row.type is not TxnType.PAYMENT:
            continue
        tags = {"payment"}
        if row.on_hold:
            tags.add("on_hold")
        if row.settled:
            tags.add("settled")
        if row.dispute_id:
            tags.add("disputed")
        out.append(
            NormRecord(
                record_id=row.entity_id,
                source="pg",
                amount=row.amount,
                alt_amount=row.credit,
                day=epoch_to_ist_date(row.created_at),
                epoch=row.created_at,
                ref_strict=canon_strict(row.order_receipt),
                ref_loose=canon_loose(row.order_receipt),
                alt_refs=(canon_strict(row.order_id),),
                method=row.method,
                tags=frozenset(tags),
                meta={
                    "fee": row.fee,
                    "tax": row.tax,
                    "order_id": row.order_id,
                    "settlement_id": row.settlement_id,
                    "receipt_raw": row.order_receipt,
                },
            )
        )
    return out


def project_settlements(settlements: Iterable[Settlement]) -> list[NormRecord]:
    """Project settlement headers for tier-3 matching.

    Canonical amount is the payout ``amount`` -- what should appear in the bank.
    """
    out: list[NormRecord] = []
    for s in settlements:
        out.append(
            NormRecord(
                record_id=s.id,
                source="settlement",
                amount=s.amount,
                day=epoch_to_ist_date(s.created_at),
                epoch=s.created_at,
                ref_strict=canon_strict(s.utr),
                ref_loose=canon_loose(s.utr),
                tags=frozenset({s.status.value}),
                meta={"utr": s.utr, "status": s.status.value, "fees": s.fees, "tax": s.tax},
            )
        )
    return out


def project_bank_credits(credits: Iterable[BankCredit]) -> list[NormRecord]:
    """Project bank statement credits for tier-3 matching.

    The UTR is re-derived from the raw narration rather than trusting any pre-extracted
    field, and ``ref_is_strict`` records whether the parse was confident. A credit whose
    UTR could not be read is not a broken record -- it simply has to be matched on
    amount and date instead, and the scorer needs to know which situation it is in.
    """
    out: list[NormRecord] = []
    for c in credits:
        utr, strict = extract_utr(c.narration)
        if c.utr_extracted:
            utr, strict = c.utr_extracted, True
        out.append(
            NormRecord(
                record_id=c.stmt_id,
                source="bank",
                amount=c.amount,
                day=epoch_to_ist_date(c.value_date),
                epoch=c.value_date,
                ref_strict=canon_strict(utr),
                ref_loose=canon_loose(utr),
                ref_is_strict=strict,
                tags=frozenset({"credit"} | ({"no_utr"} if not utr else set())),
                meta={"narration": c.narration, "utr": utr, "bank_ref": c.bank_ref},
            )
        )
    return out
