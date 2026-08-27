"""Evaluation against ground truth, with every metric defined precisely.

The track's bar is "throughput plus measured accuracy plus an honest exception list", and
the honest part is the hard part. Three decisions here are what make these numbers mean
something:

**1. Two recall figures, both reported.** A defect that corrupts an amount leaves the
underlying correspondence intact -- the credit really does belong to that settlement -- but
a correct reconciler must refuse to bless it. Scoring that refusal as a recall miss would
punish the exact behaviour we want, while silently dropping those links from the
denominator would inflate recall. So both are reported: ``recall`` over every true link,
and ``resolvable_recall`` over links that carry no must-report defect. The first is the
conservative headline; the second isolates matching skill from correct refusal.

**2. False positives are priced, not just counted.** ``false_positive_value`` sums the
rupees involved in wrong matches. A reconciler that mismatches two 40-lakh settlements is
not "one error" in any sense a controller cares about.

**3. Calibration is measured, not asserted.** Any system can print a confidence. Whether
0.90 actually means "right nine times in ten" is an empirical question, answered by
:func:`calibration_curve` and summarised as expected calibration error. A miscalibrated
confidence is worse than none, because it invites misplaced trust.

Nothing in :mod:`trikon.pipeline` reads ground truth. It is loaded here and only here.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Sequence

from trikon.block import nm_links, pairwise_resolvable
from trikon.models import (
    ExceptionCode,
    ExceptionRecord,
    GroundTruth,
    MatchLink,
    Tier,
)
from trikon.money import format_inr
from trikon.pipeline import RunResult


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _f1(precision: float, recall: float) -> float:
    return _safe_div(2 * precision * recall, precision + recall)


@dataclass(frozen=True)
class MatchMetrics:
    """Link-level accuracy for one tier.

    Definitions, stated so they cannot drift:

    * **true positive** -- a produced link that appears in ground truth.
    * **false positive** -- a produced link that does not. These are the dangerous ones:
      money declared reconciled that is not.
    * **false negative** -- a ground-truth link the system did not produce, for any
      reason including a deliberate refusal.
    * **resolvable link** -- a true link where neither endpoint carries a defect that
      must be reported. Matching these is unambiguously the system's job.
    """

    tier: Tier
    truth_total: int
    produced_total: int
    true_positives: int
    false_positives: int
    false_negatives: int

    resolvable_total: int
    resolvable_true_positives: int

    nm_total: int
    nm_true_positives: int

    false_positive_value: int
    false_positive_pairs: tuple[tuple[str, str], ...]
    false_negative_pairs: tuple[tuple[str, str], ...]

    auto_accepted: int
    auto_accepted_true_positives: int
    auto_accepted_false_positives: int

    @property
    def precision(self) -> float:
        return _safe_div(self.true_positives, self.true_positives + self.false_positives)

    @property
    def recall(self) -> float:
        return _safe_div(self.true_positives, self.truth_total)

    @property
    def f1(self) -> float:
        return _f1(self.precision, self.recall)

    @property
    def resolvable_recall(self) -> float:
        """Recall restricted to links with no must-report defect."""
        return _safe_div(self.resolvable_true_positives, self.resolvable_total)

    @property
    def nm_recall(self) -> float:
        """Recall on N:M links, which only subset-sum can resolve."""
        return _safe_div(self.nm_true_positives, self.nm_total)

    @property
    def auto_accept_precision(self) -> float:
        """Precision among links accepted with no human review.

        The most important single number in the report: it is the rate at which the
        system is wrong *while claiming to be confident*.
        """
        return _safe_div(
            self.auto_accepted_true_positives,
            self.auto_accepted_true_positives + self.auto_accepted_false_positives,
        )


@dataclass(frozen=True)
class ExceptionMetrics:
    """How well the exception list matches the defects actually injected."""

    expected_by_code: dict[str, int]
    detected_by_code: dict[str, int]
    matched_by_code: dict[str, int]

    defects_expected: int
    defects_detected: int

    absorbable_total: int
    absorbable_false_alarms: int
    false_alarm_subjects: tuple[str, ...]

    @property
    def detection_recall(self) -> float:
        """Fraction of must-report defects that produced their expected code."""
        return _safe_div(self.defects_detected, self.defects_expected)

    @property
    def false_alarm_rate(self) -> float:
        """Fraction of absorbable defects that wrongly produced an exception.

        Absorbable defects -- a mutated receipt, a split bank credit -- are the
        false-positive traps. A system that escalates them is technically "safe" but
        useless, because it has pushed work back to the human it was meant to save.
        """
        return _safe_div(self.absorbable_false_alarms, self.absorbable_total)


@dataclass(frozen=True)
class CalibrationBucket:
    """Observed accuracy within one confidence band.

    ``mean_confidence`` is the average confidence actually stated by the links in this
    band, and it -- not the band's midpoint -- is what observed accuracy is compared
    against. Using the midpoint is a subtle but real error: if every link in the
    0.80-1.00 band actually claims 0.99, comparing 100% observed accuracy to a 0.90
    midpoint reports a 0.10 calibration error for a system that was in fact perfectly
    calibrated. Expected calibration error is defined over stated confidence, so that is
    what we use.
    """

    lower: float
    upper: float
    count: int
    correct: int
    mean_confidence: float

    @property
    def accuracy(self) -> float:
        return _safe_div(self.correct, self.count)

    @property
    def midpoint(self) -> float:
        return (self.lower + self.upper) / 2

    @property
    def gap(self) -> float:
        """Absolute distance between stated confidence and observed accuracy."""
        return abs(self.accuracy - self.mean_confidence)


@dataclass(frozen=True)
class ThroughputMetrics:
    """Wall-clock performance, with LLM latency separated from deterministic work."""

    record_count: int
    deterministic_ms: float
    adjudication_ms: float
    stage_ms: dict[str, float]

    @property
    def total_ms(self) -> float:
        return self.deterministic_ms + self.adjudication_ms

    @property
    def records_per_second(self) -> float:
        """Deterministic throughput.

        Reported over deterministic time only, and stated as such. Bundling network
        latency to a rate-limited free-tier model into a throughput figure would say more
        about the provider's queue than about this system.
        """
        seconds = self.deterministic_ms / 1000.0
        return _safe_div(self.record_count, seconds)


@dataclass
class EvaluationReport:
    """The complete measured result of one run."""

    seed: int
    record_count: int
    source_counts: dict[str, int]
    match: dict[Tier, MatchMetrics] = field(default_factory=dict)
    exceptions: ExceptionMetrics | None = None
    calibration: list[CalibrationBucket] = field(default_factory=list)
    throughput: ThroughputMetrics | None = None

    exception_count: int = 0
    review_case_count: int = 0
    exposure_paise: int = 0
    escalated_count: int = 0
    llm_used: bool = False
    adjudicated: int = 0
    blocking_reduction: dict[Tier, float] = field(default_factory=dict)

    @property
    def total_true_positives(self) -> int:
        return sum(m.true_positives for m in self.match.values())

    @property
    def total_false_positives(self) -> int:
        return sum(m.false_positives for m in self.match.values())

    @property
    def total_truth(self) -> int:
        return sum(m.truth_total for m in self.match.values())

    @property
    def overall_precision(self) -> float:
        return _safe_div(
            self.total_true_positives, self.total_true_positives + self.total_false_positives
        )

    @property
    def overall_recall(self) -> float:
        return _safe_div(self.total_true_positives, self.total_truth)

    @property
    def overall_f1(self) -> float:
        return _f1(self.overall_precision, self.overall_recall)

    @property
    def match_rate(self) -> float:
        """Fraction of true links the system produced.

        This is the "match rate" the track asks for, and it is deliberately the
        conservative figure -- every refusal counts against it.
        """
        return self.overall_recall

    @property
    def straight_through_rate(self) -> float:
        """Fraction of true links resolved with no human involvement at all."""
        auto = sum(m.auto_accepted_true_positives for m in self.match.values())
        return _safe_div(auto, self.total_truth)

    @property
    def human_review_rate(self) -> float:
        """Review cases raised per source record."""
        return _safe_div(self.review_case_count, self.record_count)

    @property
    def expected_calibration_error(self) -> float:
        """Weighted mean gap between stated confidence and observed accuracy.

        Zero means every confidence band is exactly as reliable as it claims. Computed
        against the mean *stated* confidence in each band, per the standard definition --
        see :class:`CalibrationBucket` for why the band midpoint would be wrong. Values
        are only meaningful where a band has enough samples, which is why the buckets
        themselves are reported alongside this summary.
        """
        total = sum(b.count for b in self.calibration)
        if not total:
            return 0.0
        return sum(b.count * b.gap for b in self.calibration) / total


def _defect_affected_ids(truth: GroundTruth, *, must_report: bool) -> set[str]:
    """Record ids touched by defects that must (or must not) be reported."""
    out: set[str] = set()
    for defect in truth.defects:
        if (defect.expected_exception is not None) == must_report:
            out.update(defect.affected_ids)
    return out


def evaluate_matching(
    run: RunResult, truth: GroundTruth, *, amount_lookup: dict[str, int]
) -> dict[Tier, MatchMetrics]:
    """Score produced links against ground truth, tier by tier."""
    must_report_ids = _defect_affected_ids(truth, must_report=True)
    metrics: dict[Tier, MatchMetrics] = {}

    for tier, tier_result in run.tiers.items():
        true_links = truth.links_for(tier)
        produced = run.links_for(tier)
        auto_produced = run.auto_accepted_links_for(tier)

        tp_set = produced & true_links
        fp_set = produced - true_links
        fn_set = true_links - produced

        # A link is "resolvable" only if neither endpoint carries a must-report defect.
        resolvable = frozenset(
            link
            for link in true_links
            if link[0] not in must_report_ids and link[1] not in must_report_ids
        )
        nm = nm_links(true_links)

        fp_value = sum(
            max(amount_lookup.get(left, 0), amount_lookup.get(right, 0))
            for left, right in fp_set
        )

        metrics[tier] = MatchMetrics(
            tier=tier,
            truth_total=len(true_links),
            produced_total=len(produced),
            true_positives=len(tp_set),
            false_positives=len(fp_set),
            false_negatives=len(fn_set),
            resolvable_total=len(resolvable),
            resolvable_true_positives=len(resolvable & produced),
            nm_total=len(nm),
            nm_true_positives=len(nm & produced),
            false_positive_value=fp_value,
            false_positive_pairs=tuple(sorted(fp_set)),
            false_negative_pairs=tuple(sorted(fn_set)),
            auto_accepted=len(auto_produced),
            auto_accepted_true_positives=len(auto_produced & true_links),
            auto_accepted_false_positives=len(auto_produced - true_links),
        )
    return metrics


def evaluate_exceptions(
    exceptions: Sequence[ExceptionRecord], truth: GroundTruth
) -> ExceptionMetrics:
    """Score the exception list against the injected defect list.

    A defect counts as detected when the expected code was raised against at least one of
    the records it touched. Requiring an exact subject match would be too strict -- a
    missing bank credit can legitimately be reported against either the settlement or the
    credit -- while ignoring subjects entirely would let a coincidentally-correct code
    elsewhere in the batch count as a detection.
    """
    detected_by_code: Counter[str] = Counter(e.code.value for e in exceptions)
    expected_by_code: Counter[str] = Counter()
    matched_by_code: Counter[str] = Counter()

    by_code_subjects: dict[str, set[str]] = defaultdict(set)
    for exc in exceptions:
        by_code_subjects[exc.code.value].update(exc.subject_ids)

    defects_expected = 0
    defects_detected = 0
    absorbable_total = 0
    absorbable_false_alarms = 0
    false_alarm_subjects: list[str] = []

    all_flagged_subjects: set[str] = set()
    for subjects in by_code_subjects.values():
        all_flagged_subjects.update(subjects)

    for defect in truth.defects:
        if defect.expected_exception is None:
            absorbable_total += 1
            hit = sorted(set(defect.affected_ids) & all_flagged_subjects)
            if hit:
                absorbable_false_alarms += 1
                false_alarm_subjects.extend(hit)
            continue

        code = defect.expected_exception.value
        expected_by_code[code] += 1
        defects_expected += 1
        if set(defect.affected_ids) & by_code_subjects.get(code, set()):
            matched_by_code[code] += 1
            defects_detected += 1

    return ExceptionMetrics(
        expected_by_code=dict(expected_by_code),
        detected_by_code=dict(detected_by_code),
        matched_by_code=dict(matched_by_code),
        defects_expected=defects_expected,
        defects_detected=defects_detected,
        absorbable_total=absorbable_total,
        absorbable_false_alarms=absorbable_false_alarms,
        false_alarm_subjects=tuple(sorted(set(false_alarm_subjects))),
    )


def calibration_curve(
    links: Sequence[MatchLink], truth: GroundTruth, *, bucket_count: int = 10
) -> list[CalibrationBucket]:
    """Observed accuracy per **distinct stated confidence**.

    Fixed-width deciles are the standard choice for a model that emits a continuous score,
    and they are the wrong choice here. Trikon's confidence comes from a rule ladder that
    emits a handful of exact values (1.00, 0.99, 0.95, 0.92, 0.90), so every link lands in
    the top decile and a decile histogram collapses to a single uninformative point --
    hiding precisely what a reader wants to check, which is whether *each* rung is as
    reliable as it claims.

    So we group by the exact value instead: one row per stated confidence, reporting how
    often links asserting that confidence were right. That is a sharper claim than a
    decile curve, not a softer one -- it exposes any individual rung that is overconfident.

    ``bucket_count`` is retained for callers that want coarse deciles; pass it as ``0`` to
    force decile behaviour off. It is otherwise unused, since grouping is exact.
    """
    all_true: set[tuple[str, str]] = set()
    for tier in Tier:
        all_true |= set(truth.links_for(tier))

    by_confidence: dict[float, list[MatchLink]] = defaultdict(list)
    for link in links:
        by_confidence[round(link.confidence, 4)].append(link)

    buckets: list[CalibrationBucket] = []
    for value in sorted(by_confidence, reverse=True):
        group = by_confidence[value]
        correct = sum(1 for link in group if (link.left_id, link.right_id) in all_true)
        buckets.append(
            CalibrationBucket(
                lower=value,
                upper=value,
                count=len(group),
                correct=correct,
                mean_confidence=sum(link.confidence for link in group) / len(group),
            )
        )
    return buckets


def build_amount_lookup(
    orders: Sequence[object], rows: Sequence[object], settlements: Sequence[object],
    credits: Sequence[object],
) -> dict[str, int]:
    """Map every record id to its amount in paise, for pricing false positives."""
    lookup: dict[str, int] = {}
    for order in orders:
        lookup[order.order_id] = order.amount  # type: ignore[attr-defined]
    for row in rows:
        lookup[row.entity_id] = row.amount  # type: ignore[attr-defined]
    for settlement in settlements:
        lookup[settlement.id] = settlement.amount  # type: ignore[attr-defined]
    for credit in credits:
        lookup[credit.stmt_id] = credit.amount  # type: ignore[attr-defined]
    return lookup


def evaluate(
    run: RunResult,
    truth: GroundTruth,
    *,
    amount_lookup: dict[str, int],
    exposure_paise: int,
) -> EvaluationReport:
    """Assemble the full evaluation report for one run."""
    report = EvaluationReport(
        seed=truth.seed,
        record_count=run.record_count,
        source_counts=dict(run.source_counts),
    )
    report.match = evaluate_matching(run, truth, amount_lookup=amount_lookup)
    report.exceptions = evaluate_exceptions(run.exceptions, truth)
    report.calibration = calibration_curve(run.all_links, truth)

    adjudication_ms = run.timings_ms.get("adjudication", 0.0)
    report.throughput = ThroughputMetrics(
        record_count=run.record_count,
        deterministic_ms=run.total_ms - adjudication_ms,
        adjudication_ms=adjudication_ms,
        stage_ms=dict(run.timings_ms),
    )

    report.exception_count = len(run.exceptions)
    report.review_case_count = len(run.cases)
    report.exposure_paise = exposure_paise
    report.escalated_count = sum(len(t.escalated) for t in run.tiers.values())
    report.llm_used = run.llm_used
    report.adjudicated = run.adjudicated
    report.blocking_reduction = {
        tier: (t.blocking.reduction_ratio if t.blocking else 0.0)
        for tier, t in run.tiers.items()
    }
    return report


def render_report(report: EvaluationReport) -> str:
    """Render the report as plain text, suitable for a terminal or a log."""
    lines: list[str] = []
    add = lines.append

    add("=" * 78)
    add(f"TRIKON RECONCILIATION REPORT   seed={report.seed}")
    add("=" * 78)
    counts = "  ".join(f"{k}={v}" for k, v in report.source_counts.items())
    add(f"Records: {report.record_count}   ({counts})")
    if report.throughput:
        t = report.throughput
        add(
            f"Deterministic time: {t.deterministic_ms:.0f}ms "
            f"({t.records_per_second:,.0f} records/sec)"
        )
        if t.adjudication_ms:
            add(f"Adjudication time:  {t.adjudication_ms:.0f}ms (network-bound, excluded above)")
    add("")

    add("-- MATCHING " + "-" * 66)
    add(
        f"{'tier':24} {'truth':>6} {'TP':>5} {'FP':>4} {'FN':>4} "
        f"{'prec':>6} {'recall':>7} {'F1':>6} {'resolv':>7}"
    )
    for tier, m in report.match.items():
        add(
            f"{tier.value:24} {m.truth_total:>6} {m.true_positives:>5} "
            f"{m.false_positives:>4} {m.false_negatives:>4} "
            f"{m.precision:>6.3f} {m.recall:>7.3f} {m.f1:>6.3f} "
            f"{m.resolvable_recall:>7.3f}"
        )
    add("")
    add(
        f"Overall: precision {report.overall_precision:.3f}  "
        f"recall {report.overall_recall:.3f}  F1 {report.overall_f1:.3f}"
    )
    add(
        f"Match rate {report.match_rate * 100:.1f}%   "
        f"straight-through {report.straight_through_rate * 100:.1f}%   "
        f"human review {report.human_review_rate * 100:.1f}% of records"
    )

    fp_total = report.total_false_positives
    fp_value = sum(m.false_positive_value for m in report.match.values())
    add(f"False positives: {fp_total}  (value at risk {format_inr(fp_value)})")
    for tier, m in report.match.items():
        if m.auto_accepted:
            add(
                f"  {tier.value}: auto-accepted {m.auto_accepted}, "
                f"of which wrong {m.auto_accepted_false_positives} "
                f"(precision {m.auto_accept_precision:.3f})"
            )
    for tier, ratio in report.blocking_reduction.items():
        add(f"  {tier.value}: blocking pruned {ratio * 100:.2f}% of the pair space")
    add("")

    if report.exceptions:
        e = report.exceptions
        add("-- EXCEPTIONS " + "-" * 64)
        add(
            f"Must-report defects detected: {e.defects_detected}/{e.defects_expected} "
            f"({e.detection_recall * 100:.1f}%)"
        )
        add(
            f"Absorbable defects wrongly escalated: {e.absorbable_false_alarms}/"
            f"{e.absorbable_total} (false-alarm rate {e.false_alarm_rate * 100:.1f}%)"
        )
        add(f"Findings raised: {report.exception_count} in {report.review_case_count} cases")
        add(f"Exposure across cases: {format_inr(report.exposure_paise)}")
        add("")
        add(f"{'code':32} {'expected':>9} {'detected':>9} {'matched':>8}")
        codes = sorted(set(e.expected_by_code) | set(e.detected_by_code))
        for code in codes:
            add(
                f"{code:32} {e.expected_by_code.get(code, 0):>9} "
                f"{e.detected_by_code.get(code, 0):>9} {e.matched_by_code.get(code, 0):>8}"
            )
        add("")

    if report.calibration:
        add("-- CALIBRATION " + "-" * 63)
        add("Observed accuracy per stated confidence (one row per rule tier).")
        add(f"{'stated':>8} {'n':>6} {'correct':>8} {'observed':>9} {'gap':>7}  direction")
        for b in report.calibration:
            if b.accuracy > b.mean_confidence:
                direction = "under-confident (safe)"
            elif b.accuracy < b.mean_confidence:
                direction = "OVER-confident"
            else:
                direction = "exact"
            add(
                f"{b.mean_confidence:>8.2f} {b.count:>6} {b.correct:>8} "
                f"{b.accuracy:>9.3f} {b.gap:>7.3f}  {direction}"
            )
        add(f"Expected calibration error: {report.expected_calibration_error:.4f}")
        overconfident = [b for b in report.calibration if b.accuracy < b.mean_confidence]
        if overconfident:
            add(
                f"WARNING: {len(overconfident)} confidence tier(s) are over-confident -- "
                "they claim more reliability than observed."
            )
        else:
            add(
                "No tier is over-confident: every rung was at least as accurate as it "
                "claimed, so the gaps above are the ladder understating itself."
            )
        add("")

    add(
        f"Escalated to human (unresolved): {report.escalated_count}   "
        f"LLM adjudication: {'on' if report.llm_used else 'off'} "
        f"({report.adjudicated} cases)"
    )
    add("=" * 78)
    return "\n".join(lines)
