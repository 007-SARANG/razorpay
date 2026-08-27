"""On-disk cache for LLM responses.

This cache is not a performance optimisation, it is a reproducibility mechanism, and it
earns its place three times over:

* **Reproducible metrics.** A reviewer re-running the pipeline gets the same adjudication
  decisions, so the reported numbers can be checked rather than taken on faith. Without
  it, every run would sample a model afresh and no figure in the report would be stable.
* **An offline demo.** With the cache committed, the pitch recording cannot be derailed
  by a rate limit, a network drop, or a provider outage mid-take.
* **A survivable submission.** If the free-tier proxy disappears before the deadline, the
  repo still reproduces every claim it makes.

The key is a hash of the *semantic* request -- model, system prompt, and the evidence
packet -- so an unchanged case never costs a second call, while any change to the evidence
or the prompt correctly invalidates the entry.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def cache_key(*, model: str, system: str, user: str) -> str:
    """Stable hash of a request's meaning.

    ``sort_keys`` is applied by callers when serialising the packet, so two logically
    identical requests hash identically regardless of dict ordering.
    """
    digest = hashlib.sha256()
    for part in (model, system, user):
        digest.update(part.encode("utf-8"))
        digest.update(b"\x00")  # separator prevents field-boundary collisions
    return digest.hexdigest()[:32]


class ResponseCache:
    """A flat directory of JSON files, one per cached response.

    Flat files rather than a database: the cache is meant to be committed and read by a
    human during review, and a diff of ``data/cache/*.json`` shows exactly which
    adjudications the reported metrics rest on.
    """

    def __init__(self, directory: Path, *, enabled: bool = True) -> None:
        self._dir = directory
        self._enabled = enabled
        self.hits = 0
        self.misses = 0
        self.writes = 0
        if self._enabled:
            self._dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self._dir / f"{key}.json"

    def get(self, key: str) -> dict[str, Any] | None:
        """Return a cached response, or ``None`` on miss or unreadable entry."""
        if not self._enabled:
            return None
        path = self._path(key)
        if not path.is_file():
            self.misses += 1
            return None
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            # A corrupt entry is treated as a miss rather than an error: the worst case is
            # one extra API call, whereas raising would abort a run over a damaged file.
            self.misses += 1
            return None
        self.hits += 1
        payload = record.get("response")
        return payload if isinstance(payload, dict) else None

    def put(self, key: str, response: dict[str, Any], *, model: str, note: str = "") -> None:
        """Store a response alongside provenance for auditability."""
        if not self._enabled:
            return
        record = {
            "cached_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "model": model,
            "note": note,
            "response": response,
        }
        try:
            self._path(key).write_text(json.dumps(record, indent=2), encoding="utf-8")
            self.writes += 1
        except OSError:
            # Failing to persist a cache entry must never break a reconciliation run.
            pass

    def summary(self) -> str:
        total = self.hits + self.misses
        rate = (self.hits / total * 100) if total else 0.0
        return (
            f"cache: {self.hits} hits, {self.misses} misses "
            f"({rate:.0f}% hit rate), {self.writes} written"
        )
