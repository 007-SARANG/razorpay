"""Command-line interface.

Five commands, in the order a reviewer would want them:

* ``doctor``   -- show configuration and whether an LLM provider is reachable
* ``generate`` -- build a synthetic batch with ground truth
* ``run``      -- reconcile a batch and print the report
* ``stress``   -- run several batch sizes and report scaling
* ``sweep``    -- vary the auto-accept threshold and show the precision/recall tradeoff

Every command is deterministic given a seed, and none of them require an API key.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Sequence

from trikon.classify import total_exposure
from trikon.config import load_settings
from trikon.evaluate import build_amount_lookup, evaluate, render_report
from trikon.generate.defects import DEFAULT_PLAN, inject_defects
from trikon.generate.world import GeneratorConfig, World, generate_clean_world
from trikon.models import AUTO_ACCEPT_THRESHOLD, GroundTruth, Tier
from trikon.money import format_inr
from trikon.pipeline import run_pipeline
from trikon.report import report_to_dict


def _build_world(seed: int, n_orders: int, *, clean: bool = False) -> World:
    world = generate_clean_world(GeneratorConfig(seed=seed, n_orders=n_orders))
    if not clean:
        world = inject_defects(world, DEFAULT_PLAN)
    return world


def _serialise_world(world: World, truth: GroundTruth) -> dict[str, object]:
    return {
        "meta": {
            "seed": world.config.seed,
            "n_orders": world.config.n_orders,
            "record_count": world.record_count(),
            "rollover_count": world.rollover_count,
        },
        "orders": [o.model_dump(mode="json") for o in world.orders],
        "recon_rows": [r.model_dump(mode="json") for r in world.recon_rows],
        "settlements": [s.model_dump(mode="json") for s in world.settlements],
        "bank_credits": [c.model_dump(mode="json") for c in world.bank_credits],
        "ground_truth": truth.model_dump(mode="json"),
    }


def cmd_doctor(args: argparse.Namespace) -> int:
    """Report configuration without ever printing a secret."""
    settings = load_settings()
    print("Trikon configuration")
    print("-" * 60)
    for key, value in settings.redacted().items():
        print(f"  {key:26} {value}")
    print()
    if not settings.llm_enabled:
        print("  LLM adjudication: DISABLED (no API key configured).")
        print("  The full pipeline still runs; ambiguous candidates are escalated")
        print("  to the review queue instead of adjudicated. All metrics are produced.")
        return 0

    print("  LLM adjudication: enabled. Probing provider...")
    try:
        from trikon.llm.provider import probe_provider

        ok, detail = probe_provider(settings)
        print(f"  probe: {'OK' if ok else 'FAILED'} -- {detail}")
        return 0 if ok else 1
    except ImportError:
        print("  provider module unavailable; running deterministic-only.")
        return 0


def cmd_generate(args: argparse.Namespace) -> int:
    settings = load_settings()
    world = _build_world(args.seed, args.orders, clean=args.clean)
    truth = world.ground_truth(int(time.time()))

    out_dir = settings.resolve(settings.batch_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"batch_seed{args.seed}_n{args.orders}.json"
    path.write_text(json.dumps(_serialise_world(world, truth), indent=2), encoding="utf-8")

    must_report = sum(1 for d in truth.defects if d.expected_exception is not None)
    absorbable = len(truth.defects) - must_report
    print(f"Wrote {path}")
    print(
        f"  {world.record_count()} records "
        f"(orders={len(world.orders)} rows={len(world.recon_rows)} "
        f"settlements={len(world.settlements)} bank={len(world.bank_credits)})"
    )
    print(f"  ground-truth links: {len(truth.links)}")
    print(
        f"  injected defects: {len(truth.defects)} "
        f"({must_report} must be reported, {absorbable} must be absorbed)"
    )
    print(f"  partial-settlement rollovers simulated: {world.rollover_count}")
    return 0


def _run_once(
    world: World, truth: GroundTruth, *, threshold: float, use_llm: bool
) -> tuple[object, object]:
    adjudicator = None
    if use_llm:
        settings = load_settings()
        if settings.llm_enabled:
            from trikon.llm.adjudicate import build_adjudicator

            adjudicator = build_adjudicator(settings)
        else:
            print(
                "  note: --llm requested but no API key configured; "
                "running deterministic-only.",
                file=sys.stderr,
            )

    run = run_pipeline(
        world.orders,
        world.recon_rows,
        world.settlements,
        world.bank_credits,
        adjudicator=adjudicator,
        auto_accept_threshold=threshold,
    )
    lookup = build_amount_lookup(
        world.orders, world.recon_rows, world.settlements, world.bank_credits
    )
    report = evaluate(
        run, truth, amount_lookup=lookup, exposure_paise=total_exposure(run.cases)
    )
    return run, report


def cmd_run(args: argparse.Namespace) -> int:
    world = _build_world(args.seed, args.orders, clean=args.clean)
    truth = world.ground_truth(int(time.time()))
    run, report = _run_once(world, truth, threshold=args.threshold, use_llm=args.llm)

    print(render_report(report))  # type: ignore[arg-type]

    if args.exceptions:
        print()
        print("-- REVIEW QUEUE (ordered by rupees at risk) " + "-" * 34)
        cases = run.cases[: args.exceptions]  # type: ignore[attr-defined]
        for case in cases:
            print(
                f"{case.severity.value.upper():>8}  {format_inr(case.total_exposure):>15}  "
                f"{case.primary_code.value:28} {case.subject_id}"
            )
            for finding in case.findings:
                print(f"           - {finding.reason}")
                for ev in finding.evidence:
                    mark = "+" if ev.supports else "-"
                    print(f"               {mark} {ev.feature}: {ev.observed}")
            print(f"           => {case.recommended_action}")
            print()
        remaining = len(run.cases) - len(cases)  # type: ignore[attr-defined]
        if remaining > 0:
            print(f"  ... and {remaining} more cases")

    if args.cash and run.cash is not None:  # type: ignore[attr-defined]
        print()
        print("-- CASH POSITION " + "-" * 61)
        for line in run.cash.summary_lines():  # type: ignore[attr-defined]
            print(f"  {line}")

    if args.json_out:
        path = Path(args.json_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report_to_dict(report, run), indent=2), encoding="utf-8")
        print(f"\nWrote machine-readable report to {path}")

    return 0


def cmd_stress(args: argparse.Namespace) -> int:
    """Run several batch sizes and report how accuracy and speed scale."""
    sizes = args.sizes or [120, 500, 1000, 5000]
    print(
        f"{'orders':>7} {'records':>8} {'prec':>6} {'recall':>7} {'resolv':>7} "
        f"{'FP':>4} {'detect':>7} {'alarm':>6} {'ms':>7} {'rec/s':>9} {'ECE':>7}"
    )
    print("-" * 88)
    for n in sizes:
        world = _build_world(args.seed, n)
        truth = world.ground_truth(int(time.time()))
        _, report = _run_once(world, truth, threshold=args.threshold, use_llm=False)
        r = report
        resolvable = [m.resolvable_recall for m in r.match.values()]  # type: ignore[attr-defined]
        print(
            f"{n:>7} {r.record_count:>8} "  # type: ignore[attr-defined]
            f"{r.overall_precision:>6.3f} {r.overall_recall:>7.3f} "  # type: ignore[attr-defined]
            f"{sum(resolvable) / len(resolvable):>7.3f} "
            f"{r.total_false_positives:>4} "  # type: ignore[attr-defined]
            f"{r.exceptions.detection_recall:>7.3f} "  # type: ignore[attr-defined]
            f"{r.exceptions.false_alarm_rate:>6.3f} "  # type: ignore[attr-defined]
            f"{r.throughput.deterministic_ms:>7.0f} "  # type: ignore[attr-defined]
            f"{r.throughput.records_per_second:>9,.0f} "  # type: ignore[attr-defined]
            f"{r.expected_calibration_error:>7.4f}"  # type: ignore[attr-defined]
        )
    return 0


def cmd_sweep(args: argparse.Namespace) -> int:
    """Vary the auto-accept threshold to expose the precision/recall tradeoff.

    This is the table a finance controller actually needs: not "how accurate is it" but
    "what do I give up to reduce manual review, and what does that cost in rupees".
    """
    world = _build_world(args.seed, args.orders)
    truth = world.ground_truth(int(time.time()))
    print(
        f"{'threshold':>10} {'prec':>6} {'recall':>7} {'STP':>7} {'FP':>4} "
        f"{'FP value':>14} {'review':>7}"
    )
    print("-" * 66)
    for threshold in (0.50, 0.70, 0.80, 0.90, 0.95, 1.00):
        _, report = _run_once(world, truth, threshold=threshold, use_llm=False)
        r = report
        fp_value = sum(m.false_positive_value for m in r.match.values())  # type: ignore[attr-defined]
        print(
            f"{threshold:>10.2f} {r.overall_precision:>6.3f} "  # type: ignore[attr-defined]
            f"{r.overall_recall:>7.3f} {r.straight_through_rate:>7.3f} "  # type: ignore[attr-defined]
            f"{r.total_false_positives:>4} {format_inr(fp_value):>14} "  # type: ignore[attr-defined]
            f"{r.human_review_rate:>7.3f}"  # type: ignore[attr-defined]
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="trikon",
        description="Three-way settlement reconciliation controller.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_doctor = sub.add_parser("doctor", help="show configuration and probe the provider")
    p_doctor.set_defaults(func=cmd_doctor)

    p_gen = sub.add_parser("generate", help="write a synthetic batch with ground truth")
    p_gen.add_argument("--seed", type=int, default=42)
    p_gen.add_argument("--orders", type=int, default=120)
    p_gen.add_argument(
        "--clean", action="store_true", help="omit defect injection (negative control)"
    )
    p_gen.set_defaults(func=cmd_generate)

    p_run = sub.add_parser("run", help="reconcile a batch and print the report")
    p_run.add_argument("--seed", type=int, default=42)
    p_run.add_argument("--orders", type=int, default=120)
    p_run.add_argument("--clean", action="store_true")
    p_run.add_argument("--threshold", type=float, default=AUTO_ACCEPT_THRESHOLD)
    p_run.add_argument("--llm", action="store_true", help="enable LLM adjudication")
    p_run.add_argument(
        "--exceptions", type=int, default=0, metavar="N", help="print the top N review cases"
    )
    p_run.add_argument("--cash", action="store_true", help="print the cash position")
    p_run.add_argument("--json-out", metavar="PATH", help="write a machine-readable report")
    p_run.set_defaults(func=cmd_run)

    p_stress = sub.add_parser("stress", help="run several batch sizes")
    p_stress.add_argument("--seed", type=int, default=42)
    p_stress.add_argument("--sizes", type=int, nargs="*")
    p_stress.add_argument("--threshold", type=float, default=AUTO_ACCEPT_THRESHOLD)
    p_stress.set_defaults(func=cmd_stress)

    p_sweep = sub.add_parser("sweep", help="vary the auto-accept threshold")
    p_sweep.add_argument("--seed", type=int, default=42)
    p_sweep.add_argument("--orders", type=int, default=120)
    p_sweep.set_defaults(func=cmd_sweep)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
