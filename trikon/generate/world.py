"""Generation of a clean, internally-consistent synthetic merchant-month.

This module builds a world in which *everything reconciles perfectly*. Defects are
injected afterwards by :mod:`trikon.generate.defects`, which records each one as
ground truth. Keeping the two stages separate is what makes the evaluation honest: the
reconciler is scored against a defect list that was written down before it ran, and it
never has access to either.

The simulation follows Razorpay's documented behaviour rather than a convenient
approximation, in three places that matter:

* **T+2 working days** from capture, over a calendar that closes on Sundays, on the
  2nd/4th Saturday, and on listed holidays.
* **Live-balance-constrained partial settlement.** Razorpay states that when the amount
  requiring settlement exceeds current live balance, only the subset of transactions
  adding up to live balance is settled and the remainder rolls to the next slot. That
  produces authentic split/rollover breaks that a naive matcher misreads as shortfalls.
* **International payments settle in INR** at the exchange rate captured at payment
  creation, so FX variance is checkable rather than mysterious.
"""

from __future__ import annotations

import datetime as _dt
import random
import string
from dataclasses import dataclass, field
from typing import Final

from trikon.calendar_ist import (
    epoch_to_ist_date,
    expected_settlement_date,
    ist_date_to_epoch,
    is_working_day,
)
from trikon.money import compute_net_credit, rupees_to_paise
from trikon.models import (
    BankCredit,
    GroundTruth,
    InjectedDefect,
    Method,
    Order,
    OrderStatus,
    ReconRow,
    Settlement,
    SettlementStatus,
    Tier,
    TrueLink,
    TxnType,
)

_B62: Final[str] = string.ascii_letters + string.digits

#: Method mix weighted toward UPI, which reflects the actual Indian payments landscape
#: and matters because UPI carries zero MDR -- a reconciler that assumes every row has
#: a fee would break on the majority of a real Indian merchant's volume.
_METHOD_WEIGHTS: Final[dict[Method, float]] = {
    Method.UPI: 0.55,
    Method.CARD: 0.25,
    Method.NETBANKING: 0.12,
    Method.WALLET: 0.05,
    Method.EMI: 0.03,
}

_CARD_NETWORKS: Final[tuple[str, ...]] = ("Visa", "MasterCard", "RuPay", "American Express")
_CARD_ISSUERS: Final[tuple[str, ...]] = ("HDFC", "ICIC", "SBIN", "KARB", "AXIS", "KKBK")
_FX_CURRENCIES: Final[dict[str, float]] = {"USD": 83.45, "EUR": 90.12, "GBP": 105.30}


@dataclass
class GeneratorConfig:
    """Knobs for the synthetic world. Every rate is explicit so runs are auditable."""

    seed: int = 42
    n_orders: int = 120
    start_date: _dt.date = _dt.date(2026, 7, 1)
    days: int = 30

    # Order outcome mix
    paid_rate: float = 0.88
    failed_rate: float = 0.08

    # Post-capture events
    refund_rate: float = 0.09
    partial_refund_share: float = 0.45
    dispute_rate: float = 0.02
    international_rate: float = 0.06

    # Settlement behaviour
    settlement_cycle_days: int = 2
    international_cycle_days: int = 7
    #: Probability that a given settlement day is balance-constrained, forcing the
    #: documented partial-settlement behaviour and a rollover to the next slot.
    partial_settlement_pressure: float = 0.18

    # Amount distribution, in rupees, log-uniform between the bounds
    min_amount_rupees: float = 149.0
    max_amount_rupees: float = 48_000.0


@dataclass
class World:
    """A generated dataset plus the truth about it.

    ``links`` and ``defects`` are the ground truth. The reconciliation pipeline is
    never handed this object -- it receives only ``orders``, ``recon_rows``,
    ``settlements`` and ``bank_credits``.
    """

    config: GeneratorConfig
    orders: list[Order] = field(default_factory=list)
    recon_rows: list[ReconRow] = field(default_factory=list)
    settlements: list[Settlement] = field(default_factory=list)
    bank_credits: list[BankCredit] = field(default_factory=list)
    links: list[TrueLink] = field(default_factory=list)
    defects: list[InjectedDefect] = field(default_factory=list)

    # Bookkeeping the defect stage needs, not part of the delivered dataset.
    settlement_members: dict[str, list[str]] = field(default_factory=dict)
    rollover_count: int = 0

    def ground_truth(self, generated_at: int) -> GroundTruth:
        """Freeze the truth into its serialisable form."""
        return GroundTruth(
            seed=self.config.seed,
            generated_at=generated_at,
            links=tuple(self.links),
            defects=tuple(self.defects),
        )

    def record_count(self) -> int:
        """Total source records delivered to the reconciler.

        This is the number reported as batch size, and it counts every row the system
        must actually consider -- not just orders.
        """
        return (
            len(self.orders) + len(self.recon_rows) + len(self.settlements) + len(self.bank_credits)
        )


class _Ids:
    """Razorpay-shaped identifier generator, deterministic under a seeded RNG."""

    def __init__(self, rng: random.Random) -> None:
        self._rng = rng
        self._seen: set[str] = set()

    def make(self, prefix: str, length: int = 14) -> str:
        while True:
            candidate = f"{prefix}{''.join(self._rng.choices(_B62, k=length))}"
            if candidate not in self._seen:
                self._seen.add(candidate)
                return candidate

    def utr(self) -> str:
        """A bank UTR: 4-letter bank code then 12 digits, as Indian UTRs appear."""
        bank = self._rng.choice(("KKBK", "HDFC", "ICIC", "UTIB", "SBIN"))
        return f"{bank}{''.join(self._rng.choices(string.digits, k=12))}"


def _log_uniform_paise(rng: random.Random, lo_rupees: float, hi_rupees: float) -> int:
    """Draw an amount log-uniformly, which matches real transaction-size skew better
    than a uniform draw: many small payments, a thin tail of large ones."""
    import math

    lo, hi = math.log(lo_rupees), math.log(hi_rupees)
    return rupees_to_paise(round(math.exp(rng.uniform(lo, hi)), 2))


def _pick_method(rng: random.Random) -> Method:
    methods = list(_METHOD_WEIGHTS)
    return rng.choices(methods, weights=[_METHOD_WEIGHTS[m] for m in methods], k=1)[0]


def _receipt(rng: random.Random, index: int, day: _dt.date) -> str:
    """A merchant-controlled receipt number.

    Three shapes are used because merchants are inconsistent, and that inconsistency
    is precisely what makes reference matching non-trivial once mutations are injected.
    """
    style = rng.random()
    if style < 0.5:
        return f"INV-{day.year}{day.month:02d}-{index:05d}"
    if style < 0.8:
        return f"RCPT/{day.strftime('%d%m%y')}/{index:04d}"
    return f"SO{index:06d}"


def generate_clean_world(config: GeneratorConfig) -> World:
    """Build a fully-reconciling synthetic world.

    Returns a :class:`World` whose orders, recon rows, settlements and bank credits are
    mutually consistent to the paise, with ground-truth links recorded for tier 1
    (order to payment) and tier 3 (settlement to bank credit).
    """
    rng = random.Random(config.seed)
    ids = _Ids(rng)
    world = World(config=config)

    # ---- Orders and their payment/refund rows ----------------------------------------
    # `pending` holds rows awaiting settlement, keyed by the working day they first
    # become eligible. The settlement loop below drains it subject to live balance.
    pending: dict[_dt.date, list[ReconRow]] = {}

    for i in range(1, config.n_orders + 1):
        day_offset = rng.randrange(config.days)
        created_day = config.start_date + _dt.timedelta(days=day_offset)
        created_at = ist_date_to_epoch(
            created_day, hour=rng.randrange(6, 23), minute=rng.randrange(60)
        )

        roll = rng.random()
        if roll < config.paid_rate:
            status = OrderStatus.PAID
        elif roll < config.paid_rate + config.failed_rate:
            status = OrderStatus.FAILED
        else:
            status = OrderStatus.CANCELLED

        is_intl = rng.random() < config.international_rate
        if is_intl:
            fx_ccy = rng.choice(list(_FX_CURRENCIES))
            fx_rate = round(_FX_CURRENCIES[fx_ccy] * rng.uniform(0.985, 1.015), 4)
            original_amount = _log_uniform_paise(rng, 5.0, 600.0)
            amount = int(round(original_amount * fx_rate))
        else:
            fx_ccy, fx_rate, original_amount = None, None, None
            amount = _log_uniform_paise(rng, config.min_amount_rupees, config.max_amount_rupees)

        order = Order(
            order_id=ids.make("order_"),
            order_receipt=_receipt(rng, i, created_day),
            amount=amount,
            currency="INR",
            status=status,
            created_at=created_at,
            customer_ref=f"CUST{rng.randrange(1000, 9999)}",
        )
        world.orders.append(order)

        if status is not OrderStatus.PAID:
            # Unpaid orders correctly have no PG row. A reconciler that flags these as
            # MISSING_IN_PG is producing false positives, so the clean world contains
            # plenty of them as a negative control.
            continue

        method = _pick_method(rng)
        fee, tax, net = compute_net_credit(amount, method.value, international=is_intl)
        cycle = config.international_cycle_days if is_intl else config.settlement_cycle_days
        eligible_day = expected_settlement_date(
            epoch_to_ist_date(created_at), cycle_days=cycle
        )
        disputed = rng.random() < config.dispute_rate

        payment = ReconRow(
            entity_id=ids.make("pay_"),
            type=TxnType.PAYMENT,
            debit=0,
            credit=net,
            amount=amount,
            currency="INR",
            fee=fee,
            tax=tax,
            on_hold=disputed,
            settled=False,
            created_at=created_at,
            payment_id=None,  # null for type=payment, per Razorpay's schema
            order_id=order.order_id,
            order_receipt=order.order_receipt,
            method=method,
            card_network=rng.choice(_CARD_NETWORKS) if method is Method.CARD else None,
            card_issuer=rng.choice(_CARD_ISSUERS) if method is Method.CARD else None,
            card_type=rng.choice(("credit", "debit")) if method is Method.CARD else None,
            dispute_id=ids.make("disp_") if disputed else None,
            description="Payment captured",
            fx_rate_at_creation=fx_rate,
            original_currency=fx_ccy,
            original_amount=original_amount,
        )
        world.recon_rows.append(payment)
        world.links.append(
            TrueLink(tier=Tier.ORDER_PG, left_id=order.order_id, right_id=payment.entity_id)
        )
        if not disputed:
            pending.setdefault(eligible_day, []).append(payment)

        # A refund debits the merchant. Refund rows carry payment_id and, per Razorpay's
        # sample, zero fee and tax -- the original fee is not returned.
        if rng.random() < config.refund_rate:
            full = rng.random() >= config.partial_refund_share
            refund_amount = amount if full else int(amount * rng.uniform(0.2, 0.7))
            refund_created = created_at + rng.randrange(3600, 5 * 86400)
            refund = ReconRow(
                entity_id=ids.make("rfnd_"),
                type=TxnType.REFUND,
                debit=refund_amount,
                credit=0,
                amount=refund_amount,
                currency="INR",
                fee=0,
                tax=0,
                created_at=refund_created,
                payment_id=payment.entity_id,
                order_id=order.order_id,
                order_receipt=order.order_receipt,
                method=method,
                description="Refund issued" if full else "Partial refund issued",
            )
            world.recon_rows.append(refund)
            refund_day = expected_settlement_date(
                epoch_to_ist_date(refund_created), cycle_days=config.settlement_cycle_days
            )
            pending.setdefault(refund_day, []).append(refund)

    # ---- Settlement batching, with live-balance-constrained partial settlement -------
    # Index rows by id once. Stamping settlement identity by rescanning the row list per
    # settlement would be O(rows x settlements); at 10k+ records that dominates runtime,
    # and throughput is a headline metric we intend to report honestly.
    row_index: dict[str, int] = {r.entity_id: i for i, r in enumerate(world.recon_rows)}

    all_days = sorted(pending)
    if all_days:
        cursor = all_days[0]
        # The observation cutoff is the end of the window, NOT "whenever everything has
        # settled". A month-end snapshot of a real merchant always has the final days'
        # captures still in flight -- that unsettled tail *is* the cash position, and
        # forcing it to settle would quietly delete the most interesting part of Q4.
        last_day = config.start_date + _dt.timedelta(days=config.days)
        carried: list[ReconRow] = []

        while cursor <= last_day:
            if not is_working_day(cursor):
                cursor += _dt.timedelta(days=1)
                continue

            batch = carried + pending.pop(cursor, [])
            carried = []
            if not batch:
                cursor += _dt.timedelta(days=1)
                continue

            batch.sort(key=lambda r: r.created_at)
            due = sum(r.credit - r.debit for r in batch)

            # Documented behaviour: if the amount requiring settlement exceeds live
            # balance, settle only the subset that adds up to it and roll the rest.
            if due > 0 and rng.random() < config.partial_settlement_pressure and len(batch) > 2:
                cap = int(due * rng.uniform(0.35, 0.75))
                chosen: list[ReconRow] = []
                running = 0
                for row in batch:
                    delta = row.credit - row.debit
                    if running + delta > cap and chosen:
                        break
                    chosen.append(row)
                    running += delta
                chosen_ids = {r.entity_id for r in chosen}
                carried = [r for r in batch if r.entity_id not in chosen_ids]
                batch = chosen
                if carried:
                    world.rollover_count += 1

            net_amount = sum(r.credit - r.debit for r in batch)
            if net_amount <= 0 or not batch:
                # A day whose refunds exceed its credits settles nothing; the debits
                # carry forward. Real and worth modelling.
                carried = carried + batch
                cursor += _dt.timedelta(days=1)
                continue

            settled_at = ist_date_to_epoch(cursor, hour=9)
            utr = ids.utr()
            settlement = Settlement(
                id=ids.make("setl_"),
                amount=net_amount,
                status=SettlementStatus.PROCESSED,
                fees=0,  # normal (non-instant) settlements report 0 here
                tax=0,
                utr=utr,
                created_at=settled_at,
            )
            world.settlements.append(settlement)
            world.settlement_members[settlement.id] = [r.entity_id for r in batch]

            # Stamp settlement identity back onto each member row.
            for row in batch:
                idx = row_index[row.entity_id]
                current = world.recon_rows[idx]
                world.recon_rows[idx] = current.model_copy(
                    update={
                        "settled": True,
                        "settled_at": settled_at,
                        "settlement_id": settlement.id,
                        "settlement_utr": (
                            None if current.type is TxnType.ADJUSTMENT else utr
                        ),
                    }
                )

            credit = BankCredit(
                stmt_id=f"STMT{len(world.bank_credits) + 1:06d}",
                value_date=settled_at,
                amount=net_amount,
                narration=f"NEFT CR RAZORPAY SETTLEMENT {utr} MERCHANT PAYOUT",
                utr_extracted=utr,
                bank_ref=ids.make("bref_", 10),
            )
            world.bank_credits.append(credit)
            world.links.append(
                TrueLink(
                    tier=Tier.SETTLEMENT_BANK, left_id=settlement.id, right_id=credit.stmt_id
                )
            )

            cursor += _dt.timedelta(days=1)

    return world
