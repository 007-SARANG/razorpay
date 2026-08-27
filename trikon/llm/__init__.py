"""LLM adjudication layer.

Optional by design. Nothing in :mod:`trikon.pipeline` imports this package directly --
an adjudicator is injected as a callable -- so the entire reconciliation system runs with
no provider configured, no network access, and no API key.
"""

from __future__ import annotations

__all__ = ["adjudicate", "cache", "provider"]
