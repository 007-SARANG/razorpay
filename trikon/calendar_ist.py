"""Indian banking calendar: settlement timing and working-day arithmetic.

Razorpay's documented domestic settlement cycle is **T+2 working days** from capture,
and "working days do not include the bank holidays" -- if a settlement lands on a
holiday it moves to the next working day. International settlements use a longer
default cycle (their docs example uses T+7).

Two consequences that drive real reconciliation exceptions, both modelled here:

1. A naive ``date + 2 days`` matcher breaks every time a weekend or holiday
   intervenes, producing timing false-positives.
2. Indian banks are additionally closed on the **second and fourth Saturday** of each
   month (RBI convention), while the first, third and fifth Saturdays are working
   days. This is the kind of detail that separates a plausible synthetic dataset from
   an obviously fake one.

The holiday list below is a *representative synthetic* calendar for the generated
data. It is deliberately not presented as an authoritative list of real bank holidays
for any given year -- the reconciler only needs generator and verifier to agree on the
same calendar, and keeping it in one editable place makes that agreement auditable.
"""

from __future__ import annotations

import datetime as _dt
from functools import lru_cache
from typing import Final, Iterable
from zoneinfo import ZoneInfo

IST: Final[ZoneInfo] = ZoneInfo("Asia/Kolkata")

#: Representative synthetic bank-holiday calendar (month, day) pairs, applied to every
#: year the generator produces. Swap or extend freely; both the generator and the
#: reconciler read this same source of truth.
SYNTHETIC_HOLIDAYS_MMDD: Final[frozenset[tuple[int, int]]] = frozenset(
    {
        (1, 26),  # Republic Day
        (3, 25),  # spring festival holiday
        (4, 14),  # Ambedkar Jayanti / regional new year
        (5, 1),  # Maharashtra Day
        (8, 15),  # Independence Day
        (10, 2),  # Gandhi Jayanti
        (10, 21),  # festival of lights
        (11, 5),  # regional holiday
        (12, 25),  # Christmas
    }
)


def is_second_or_fourth_saturday(day: _dt.date) -> bool:
    """True if ``day`` is the 2nd or 4th Saturday of its month.

    Indian banks are closed on those two Saturdays only; the 1st, 3rd and (when it
    exists) 5th Saturday are normal working days.
    """
    if day.weekday() != 5:  # 5 == Saturday
        return False
    occurrence = (day.day - 1) // 7 + 1
    return occurrence in (2, 4)


def is_bank_holiday(day: _dt.date, *, holidays: Iterable[tuple[int, int]] | None = None) -> bool:
    """True if ``day`` is a listed bank holiday (year-agnostic month/day match)."""
    table = SYNTHETIC_HOLIDAYS_MMDD if holidays is None else frozenset(holidays)
    return (day.month, day.day) in table


def is_working_day(day: _dt.date, *, holidays: Iterable[tuple[int, int]] | None = None) -> bool:
    """True if banks settle on ``day``.

    Closed on Sundays, on the 2nd/4th Saturday, and on listed holidays.
    """
    if day.weekday() == 6:  # Sunday
        return False
    if is_second_or_fourth_saturday(day):
        return False
    return not is_bank_holiday(day, holidays=holidays)


def next_working_day(day: _dt.date, *, holidays: Iterable[tuple[int, int]] | None = None) -> _dt.date:
    """The first working day strictly after ``day``."""
    cursor = day + _dt.timedelta(days=1)
    for _ in range(30):
        if is_working_day(cursor, holidays=holidays):
            return cursor
        cursor += _dt.timedelta(days=1)
    raise RuntimeError(f"no working day found within 30 days of {day!r}")


def add_working_days(
    start: _dt.date, n: int, *, holidays: Iterable[tuple[int, int]] | None = None
) -> _dt.date:
    """Advance ``start`` by ``n`` working days.

    ``n == 0`` returns ``start`` itself if it is a working day, otherwise the next
    working day -- which is the behaviour a settlement engine needs, since a capture
    on a holiday still has a well-defined first eligible settlement date.
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    cursor = start
    if not is_working_day(cursor, holidays=holidays):
        cursor = next_working_day(cursor, holidays=holidays)
    for _ in range(n):
        cursor = next_working_day(cursor, holidays=holidays)
    return cursor


def working_days_between(
    start: _dt.date, end: _dt.date, *, holidays: Iterable[tuple[int, int]] | None = None
) -> int:
    """Signed count of working days from ``start`` to ``end``.

    This is the feature the matcher scores on, rather than a raw calendar delta: a
    settlement that lands "3 days late" across a long weekend may be perfectly on
    time in working-day terms, and we do not want to raise an exception for it.
    """
    if end == start:
        return 0
    sign = 1 if end > start else -1
    lo, hi = (start, end) if sign == 1 else (end, start)
    count = 0
    cursor = lo
    while cursor < hi:
        cursor += _dt.timedelta(days=1)
        if is_working_day(cursor, holidays=holidays):
            count += 1
    return sign * count


def expected_settlement_date(
    captured_on: _dt.date, *, cycle_days: int = 2, holidays: Iterable[tuple[int, int]] | None = None
) -> _dt.date:
    """The date a payment captured on ``captured_on`` is expected to settle.

    Defaults to the documented domestic T+2 working-day cycle. Pass ``cycle_days=7``
    for the international default.
    """
    return add_working_days(captured_on, cycle_days, holidays=holidays)


@lru_cache(maxsize=4096)
def epoch_to_ist_date(epoch: int) -> _dt.date:
    """Convert a Unix timestamp to its calendar date in IST.

    Cached because the matcher calls this on every candidate pair, and settlement
    batches reuse the same handful of timestamps heavily.
    """
    return _dt.datetime.fromtimestamp(epoch, tz=IST).date()


def ist_date_to_epoch(day: _dt.date, *, hour: int = 9, minute: int = 0) -> int:
    """Convert an IST calendar date to a Unix timestamp.

    Defaults to 09:00 IST because Razorpay's documented settlement examples land at
    9 a.m., and using a realistic hour keeps generated timestamps plausible.
    """
    return int(_dt.datetime(day.year, day.month, day.day, hour, minute, tzinfo=IST).timestamp())
