"""FastAPI service backing the Trikon dashboard.

Small on purpose. The dashboard is a view over a reconciliation run, so the API's whole
job is to run the pipeline and hand back the report document that
:mod:`trikon.report` already produces. No business logic lives here -- anything the API
computed itself would be a second implementation able to disagree with the CLI.

Notably absent: any endpoint that mutates data or moves money. Trikon is read-and-report
only, which is the correct bounded-autonomy posture for a system that reasons about
settlements. The most it will ever do is tell a human what to look at.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from trikon import __version__
from trikon.config import load_settings
from trikon.evaluate import build_amount_lookup, evaluate
from trikon.generate.defects import DEFAULT_PLAN, inject_defects
from trikon.generate.world import GeneratorConfig, generate_clean_world
from trikon.models import AUTO_ACCEPT_THRESHOLD
from trikon.pipeline import run_pipeline
from trikon.report import exposure_of, report_to_dict

WEB_DIR = Path(__file__).resolve().parent.parent / "web"

app = FastAPI(
    title="Trikon",
    version=__version__,
    description="Three-way settlement reconciliation controller.",
)

# The dashboard is served from the same origin in normal use; CORS is permitted so the
# HTML file can also be opened directly from disk during development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class ReconcileRequest(BaseModel):
    """Parameters for a synthetic batch and the run over it."""

    seed: int = Field(default=42, ge=0, le=2**31 - 1)
    orders: int = Field(default=120, ge=1, le=20_000)
    clean: bool = Field(
        default=False, description="Skip defect injection (negative control)"
    )
    threshold: float = Field(default=AUTO_ACCEPT_THRESHOLD, ge=0.0, le=1.0)
    use_llm: bool = Field(
        default=False, description="Adjudicate ambiguous cases with the configured model"
    )


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "version": __version__}


@app.get("/api/config")
def config() -> dict[str, Any]:
    """Redacted configuration, so the dashboard can show whether an LLM is available.

    Uses :meth:`Settings.redacted`, so the API cannot leak a key even by accident.
    """
    settings = load_settings()
    return settings.redacted()


@app.post("/api/reconcile")
def reconcile(request: ReconcileRequest) -> dict[str, Any]:
    """Generate a batch, reconcile it, evaluate it, and return the full report."""
    started = time.perf_counter()

    world = generate_clean_world(
        GeneratorConfig(seed=request.seed, n_orders=request.orders)
    )
    if not request.clean:
        world = inject_defects(world, DEFAULT_PLAN)
    truth = world.ground_truth(int(time.time()))

    adjudicator = None
    llm_note = "disabled"
    if request.use_llm:
        settings = load_settings()
        if settings.llm_enabled:
            try:
                from trikon.llm.adjudicate import build_adjudicator

                adjudicator = build_adjudicator(settings)
                llm_note = f"enabled ({settings.llm_model})"
            except ImportError as exc:  # pragma: no cover - defensive
                raise HTTPException(500, f"LLM layer unavailable: {exc}") from exc
        else:
            llm_note = "requested but no API key configured"

    run = run_pipeline(
        world.orders,
        world.recon_rows,
        world.settlements,
        world.bank_credits,
        adjudicator=adjudicator,
        auto_accept_threshold=request.threshold,
    )
    lookup = build_amount_lookup(
        world.orders, world.recon_rows, world.settlements, world.bank_credits
    )
    report = evaluate(
        run, truth, amount_lookup=lookup, exposure_paise=exposure_of(run)
    )

    payload = report_to_dict(report, run)
    payload["meta"].update(
        {
            "orders_requested": request.orders,
            "clean": request.clean,
            "threshold": request.threshold,
            "llm": llm_note,
            "defects_injected": len(truth.defects),
            "rollovers_simulated": world.rollover_count,
            "wall_ms": (time.perf_counter() - started) * 1000.0,
        }
    )
    return payload


@app.get("/")
def index() -> FileResponse:
    """Serve the dashboard."""
    target = WEB_DIR / "index.html"
    if not target.is_file():  # pragma: no cover - only if the repo is incomplete
        raise HTTPException(404, "dashboard not found; expected web/index.html")
    return FileResponse(target)


if WEB_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
