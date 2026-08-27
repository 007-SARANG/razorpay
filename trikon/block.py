"""Candidate generation by blocking.

The naive approach to matching two sources is to score every pair, which is O(n*m). At
1,000 records per side that is a million comparisons; at 10,000 it is a hundred million,
and the "throughput" number in a submission quietly becomes a number about a toy batch.

Blocking fixes this the way record-linkage systems have for decades: a pair is only
scored if the two records agree on at least one cheap **blocking key**. Keys are chosen
so that a genuine match is virtually certain to agree on one of them:

* exact normalised reference (an order receipt, a UTR)
* loose normalised reference (folds O/0, I/1 transcription slips)
* exact amount in paise -- a true pair almost always agrees here, and when it does not,
  the fee-explained key below covers the common cause
* amount adjusted for fee + GST, so a books-versus-net comparison still blocks together
* value date, which catches pairs whose reference is unreadable *and* whose amount was
  perturbed

A pair that agrees on none of these is not scored. That is a real recall risk and we
report it honestly: :meth:`BlockingIndex.stats` exposes the reduction ratio, and the
evaluator cross-checks how many ground-truth links were never even generated as
candidates. A recall ceiling you have measured is engineering; one you have not is luck.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Sequence

from trikon.money import compute_gst_on_fee, compute_fee
from trikon.normalize import NormRecord


@dataclass(frozen=True)
class BlockingStats:
    """What blocking cost and what it bought."""

    left_count: int
    right_count: int
    exhaustive_pairs: int
    candidate_pairs: int

    @property
    def reduction_ratio(self) -> float:
        """Fraction of the exhaustive pair space that was never scored."""
        if self.exhaustive_pairs == 0:
            return 0.0
        return 1.0 - (self.candidate_pairs / self.exhaustive_pairs)

    def summary(self) -> str:
        return (
            f"{self.left_count}x{self.right_count} = {self.exhaustive_pairs:,} possible pairs "
            f"reduced to {self.candidate_pairs:,} candidates "
            f"({self.reduction_ratio * 100:.2f}% pruned)"
        )


class BlockingIndex:
    """An inverted index over the right-hand records, queried per left-hand record."""

    #: Amount-window keys, in paise, for date/amount bucketing. Kept at exact-only for
    #: amounts because reconciliation is an equality claim: a tolerance band here is
    #: what lets a one-paise adversary through, so tolerance is applied (visibly) in the
    #: scorer instead, never silently in the index.
    def __init__(self, right: Sequence[NormRecord]) -> None:
        self._right = right
        self._by_ref: dict[str, list[int]] = defaultdict(list)
        self._by_amount: dict[int, list[int]] = defaultdict(list)
        self._by_day: dict[object, list[int]] = defaultdict(list)
        # Right-hand records with no usable reference. Reference buckets can never
        # connect these to anything, so they are always reachable via the date window
        # regardless of how good the left record's reference is. Without this the index
        # would be asymmetric: a perfectly-referenced settlement could never reach a
        # bank credit whose narration was illegible, which is precisely the pairing the
        # matcher most needs to attempt.
        self._no_ref_by_day: dict[object, list[int]] = defaultdict(list)

        for idx, rec in enumerate(right):
            forms = rec.all_ref_forms()
            for form in forms:
                self._by_ref[form].append(idx)
            self._by_amount[rec.amount].append(idx)
            if rec.alt_amount is not None:
                self._by_amount[rec.alt_amount].append(idx)
            self._by_day[rec.day].append(idx)
            if not forms or not rec.ref_is_strict:
                self._no_ref_by_day[rec.day].append(idx)

    def candidates_for(self, left: NormRecord, *, date_window_days: int = 3) -> set[int]:
        """Right-hand indices worth scoring against ``left``.

        The date bucket is a deliberately *narrow, conditional* fallback rather than a
        default key. Blocking on a wide date window would pull in nearly every record
        that shares a month and quietly undo the reduction -- a matcher that scores 47%
        of the pair space is not really blocking, and the throughput figure it produces
        would be meaningless. So dates are only consulted for records that have no
        usable reference at all, which is exactly the population that needs the help
        (an unreadable bank narration) and no one else.
        """
        hits: set[int] = set()

        for form in left.all_ref_forms():
            hits.update(self._by_ref.get(form, ()))

        hits.update(self._by_amount.get(left.amount, ()))
        if left.alt_amount is not None:
            hits.update(self._by_amount.get(left.alt_amount, ()))

        # Fee-explained amount keys: if this record's counterpart recorded the net
        # instead of the gross (or vice versa), it blocks here rather than being missed.
        for probe in self._fee_adjusted_probes(left):
            hits.update(self._by_amount.get(probe, ()))

        # Conditional date fallback: only for records with no reference to block on, and
        # only over a narrow window. These are the cases where amount is the sole signal
        # and it may have been perturbed, so a few extra candidates are worth the cost.
        import datetime as _dt

        needs_date_help = not left.ref_strict or not left.ref_is_strict
        for offset in range(-date_window_days, date_window_days + 1):
            day = left.day + _dt.timedelta(days=offset)
            # Reference-less right records are always reachable; everything else only
            # when this left record has no reference either.
            hits.update(self._no_ref_by_day.get(day, ()))
            if needs_date_help:
                hits.update(self._by_day.get(day, ()))

        return hits

    @staticmethod
    def _fee_adjusted_probes(left: NormRecord) -> tuple[int, ...]:
        """Amounts this record could equal once fee and GST are accounted for.

        Computed for every plausible method rather than the recorded one, because the
        whole point is to catch a counterpart whose method we do not yet know.
        """
        probes: set[int] = set()
        for method in ("card", "netbanking", "wallet", "emi", "international"):
            try:
                fee = compute_fee(left.amount, method)
            except KeyError:  # pragma: no cover - method table is closed
                continue
            probes.add(left.amount - fee - compute_gst_on_fee(fee))
        return tuple(probes)

    def stats(self, left: Sequence[NormRecord], candidate_pairs: int) -> BlockingStats:
        return BlockingStats(
            left_count=len(left),
            right_count=len(self._right),
            exhaustive_pairs=len(left) * len(self._right),
            candidate_pairs=candidate_pairs,
        )


def generate_candidates(
    left: Sequence[NormRecord],
    right: Sequence[NormRecord],
    *,
    date_window_days: int = 9,
) -> tuple[list[tuple[int, int]], BlockingStats]:
    """Produce the candidate pair list and the statistics describing the reduction.

    Returns ``(pairs, stats)`` where each pair is ``(left_index, right_index)``.
    """
    index = BlockingIndex(right)
    pairs: list[tuple[int, int]] = []
    for i, rec in enumerate(left):
        for j in index.candidates_for(rec, date_window_days=date_window_days):
            pairs.append((i, j))
    return pairs, index.stats(left, len(pairs))


def coverage_of(
    pairs: Iterable[tuple[int, int]],
    left: Sequence[NormRecord],
    right: Sequence[NormRecord],
    true_links: frozenset[tuple[str, str]],
) -> tuple[int, int, list[tuple[str, str]]]:
    """How many true links blocking actually made reachable.

    Returns ``(covered, total, missed)``. This is the recall ceiling imposed by
    blocking, independent of how good the scorer is. It belongs in the evaluation
    report: a system cannot match what it never looked at, and hiding that behind a
    scorer's recall number would overstate the scorer.

    Note on N:M links: a settlement that is one of several lumped into a single bank
    credit is *not* reachable by pairwise blocking, and legitimately so -- the credit's
    amount equals no single settlement, so no amount or reference key can connect them.
    Those links are recovered by :func:`trikon.assign.solve_subset_sums`, which searches
    combinations rather than pairs. Measure pairwise coverage over pairwise-resolvable
    links only, via :func:`pairwise_resolvable`, or this figure understates the index.
    """
    generated = {(left[i].record_id, right[j].record_id) for i, j in pairs}
    missed = sorted(link for link in true_links if link not in generated)
    return len(true_links) - len(missed), len(true_links), missed


def pairwise_resolvable(true_links: frozenset[tuple[str, str]]) -> frozenset[tuple[str, str]]:
    """The subset of true links that pairwise matching could possibly resolve.

    A right-hand record claimed by more than one left-hand record (a consolidated bank
    credit) is an N:M relationship. Excluding those here keeps the pairwise recall
    ceiling an honest measure of the *index*, while the N:M links are scored separately
    against the subset-sum solver that is actually responsible for them.
    """
    from collections import Counter

    right_counts = Counter(right_id for _, right_id in true_links)
    return frozenset(link for link in true_links if right_counts[link[1]] == 1)


def nm_links(true_links: frozenset[tuple[str, str]]) -> frozenset[tuple[str, str]]:
    """The complement of :func:`pairwise_resolvable` -- links requiring subset-sum."""
    return true_links - pairwise_resolvable(true_links)
