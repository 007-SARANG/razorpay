"""Serialisation of an evaluated run into a single JSON document.

One serialiser, used by both the CLI (``--json-out``) and the API, so the dashboard and
the terminal can never disagree about what a run produced. Everything the dashboard
renders is in here, and nothing is computed twice.

Monetary values are emitted as integer paise under ``*_paise`` keys **and** as a
pre-formatted display string. The dashboard therefore never does money arithmetic in
JavaScript, where the same floating-point hazard this codebase avoids in Python would
reappear at the last moment.
"""

from __future__ import annotations

from typing import Any

from trikon.classify import total_exposure
from trikon.evaluate import EvaluationReport
from trikon.money import format_inr
from trikon.pipeline import RunResult


def report_to_dict(report: EvaluationReport, run: RunResult) -> dict[str, Any]:
    """Flatten an evaluation report and its run into a JSON-safe dictionary."""
    return {
        "meta": {
            "seed": report.seed,
            "record_count": report.record_count,
            "source_counts": report.source_counts,
            "llm_used": report.llm_used,
            "llm": "enabled" if report.llm_used else "disabled",
            "adjudicated": report.adjudicated,
            "escalated": report.escalated_count,
            "defects_injected": 0,
            "rollovers_simulated": 0,
            "clean": False,
        },
        "overall": {
            "precision": report.overall_precision,
            "recall": report.overall_recall,
            "f1": report.overall_f1,
            "match_rate": report.match_rate,
            "straight_through_rate": report.straight_through_rate,
            "human_review_rate": report.human_review_rate,
            "expected_calibration_error": report.expected_calibration_error,
            "true_positives": report.total_true_positives,
            "false_positives": report.total_false_positives,
            "truth_total": report.total_truth,
        },
        "tiers": [
            {
                "tier": tier.value,
                "truth_total": m.truth_total,
                "produced_total": m.produced_total,
                "true_positives": m.true_positives,
                "false_positives": m.false_positives,
                "false_negatives": m.false_negatives,
                "precision": m.precision,
                "recall": m.recall,
                "f1": m.f1,
                "resolvable_total": m.resolvable_total,
                "resolvable_recall": m.resolvable_recall,
                "nm_total": m.nm_total,
                "nm_recall": m.nm_recall,
                "auto_accepted": m.auto_accepted,
                "auto_accept_precision": m.auto_accept_precision,
                "false_positive_value_paise": m.false_positive_value,
                "false_positive_value_display": format_inr(m.false_positive_value),
                "false_positive_pairs": [list(p) for p in m.false_positive_pairs],
                "false_negative_pairs": [list(p) for p in m.false_negative_pairs],
                "blocking_reduction": report.blocking_reduction.get(tier, 0.0),
                "candidates_scored": run.tiers[tier].candidates_scored,
                "subset_merges": run.tiers[tier].subset_merges,
                "subset_splits": run.tiers[tier].subset_splits,
                "subset_ambiguous": run.tiers[tier].subset_ambiguous,
            }
            for tier, m in report.match.items()
        ],
        "exceptions": _exceptions_block(report),
        "calibration": [
            {
                "lower": b.lower,
                "upper": b.upper,
                "count": b.count,
                "correct": b.correct,
                "accuracy": b.accuracy,
                "mean_confidence": b.mean_confidence,
                "gap": b.gap,
            }
            for b in report.calibration
        ],
        "throughput": {
            "deterministic_ms": report.throughput.deterministic_ms if report.throughput else 0.0,
            "adjudication_ms": report.throughput.adjudication_ms if report.throughput else 0.0,
            "records_per_second": (
                report.throughput.records_per_second if report.throughput else 0.0
            ),
            "stage_ms": report.throughput.stage_ms if report.throughput else {},
        },
        "review_cases": [_case_block(case) for case in run.cases],
        "links": [_link_block(link) for link in run.all_links],
        "cash_position": _cash_block(run),
    }


def _exceptions_block(report: EvaluationReport) -> dict[str, Any]:
    e = report.exceptions
    if e is None:
        return {}
    codes = sorted(set(e.expected_by_code) | set(e.detected_by_code))
    return {
        "detection_recall": e.detection_recall,
        "false_alarm_rate": e.false_alarm_rate,
        "defects_expected": e.defects_expected,
        "defects_detected": e.defects_detected,
        "absorbable_total": e.absorbable_total,
        "absorbable_false_alarms": e.absorbable_false_alarms,
        "finding_count": report.exception_count,
        "case_count": report.review_case_count,
        "exposure_paise": report.exposure_paise,
        "exposure_display": format_inr(report.exposure_paise),
        "by_code": [
            {
                "code": code,
                "expected": e.expected_by_code.get(code, 0),
                "detected": e.detected_by_code.get(code, 0),
                "matched": e.matched_by_code.get(code, 0),
            }
            for code in codes
        ],
    }


def _case_block(case: Any) -> dict[str, Any]:
    return {
        "case_id": case.case_id,
        "subject_id": case.subject_id,
        "primary_code": case.primary_code.value,
        "severity": case.severity.value,
        "tier": case.tier.value,
        "principal_paise": case.principal_exposure,
        "delta_paise": case.delta_exposure,
        "total_paise": case.total_exposure,
        "total_display": format_inr(case.total_exposure),
        "finding_codes": list(case.finding_codes),
        "recommended_action": case.recommended_action,
        "findings": [
            {
                "exception_id": f.exception_id,
                "code": f.code.value,
                "severity": f.severity.value,
                "reason": f.reason,
                "amount_paise": f.amount_at_risk,
                "amount_display": format_inr(f.amount_at_risk),
                "subject_ids": list(f.subject_ids),
                "candidates_considered": list(f.candidates_considered),
                "recommended_action": f.recommended_action,
                "evidence": [
                    {
                        "feature": ev.feature,
                        "observed": ev.observed,
                        "supports": ev.supports,
                        "detail": ev.detail,
                    }
                    for ev in f.evidence
                ],
            }
            for f in case.findings
        ],
    }


def _link_block(link: Any) -> dict[str, Any]:
    return {
        "tier": link.tier.value,
        "left_id": link.left_id,
        "right_id": link.right_id,
        "rule": link.rule.value,
        "confidence": link.confidence,
        "auto_accepted": link.auto_accepted,
        "adjudicated_by": link.adjudicated_by,
        "reasoning": link.reasoning,
        "member_ids": list(link.member_ids),
        "evidence": [
            {
                "feature": ev.feature,
                "observed": ev.observed,
                "supports": ev.supports,
                "detail": ev.detail,
            }
            for ev in link.evidence
        ],
    }


def _cash_block(run: RunResult) -> dict[str, Any]:
    cash = run.cash
    if cash is None:
        return {}
    return {
        "total_in_flight_paise": cash.total_in_flight,
        "total_in_flight_display": format_inr(cash.total_in_flight),
        "on_hold_paise": cash.on_hold_total,
        "on_hold_display": format_inr(cash.on_hold_total),
        "aged_paise": cash.aged_total,
        "failed_paise": cash.failed_total,
        "failed_display": format_inr(cash.failed_total),
        "unsettled_row_count": cash.unsettled_row_count,
        "by_day": [
            {
                "day": str(day),
                "amount_paise": cash.expected_by_day[day],
                "amount_display": format_inr(cash.expected_by_day[day]),
            }
            for day in sorted(cash.expected_by_day)
        ],
    }


def exposure_of(run: RunResult) -> int:
    """Convenience re-export so callers need not import :mod:`trikon.classify`."""
    return total_exposure(run.cases)
