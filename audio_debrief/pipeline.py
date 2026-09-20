"""Stage graph, analysis_status bookkeeping, config loading, cache helper.

Kept dependency-free (yaml only) so tests and the report module can import it
without pulling in torch.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import yaml

STAGES = ["convert", "transcribe", "diarize", "speakers", "align", "metrics", "prosody", "nouns", "report"]

# stage -> stages it needs to have succeeded
DEPENDS = {
    "convert": [],
    "transcribe": ["convert"],
    "diarize": ["convert", "transcribe"],  # the hybrid diarizer works from Whisper's words
    "speakers": ["diarize"],
    "align": ["transcribe", "diarize"],
    "metrics": ["align"],
    "prosody": ["metrics", "convert"],
    "nouns": ["transcribe"],
    "report": [],
}


class SetupError(Exception):
    """A condition the user must fix before the tool can run (exit code 2)."""


def app_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_config(path: str | Path | None = None) -> dict:
    p = Path(path) if path else app_root() / "config.yaml"
    with open(p, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def mmss(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


# --- analysis_status -------------------------------------------------------

def new_status() -> dict:
    return {s: {"status": "pending", "detail": ""} for s in STAGES}


def mark(status: dict, stage: str, state: str, detail: str = "") -> None:
    status[stage] = {"status": state, "detail": detail}


def mark_ok(status: dict, stage: str, detail: str = "") -> None:
    mark(status, stage, "ok", detail)


def dependents(stage: str) -> list[str]:
    out: list[str] = []
    frontier = [stage]
    while frontier:
        cur = frontier.pop()
        for s, deps in DEPENDS.items():
            if cur in deps and s not in out:
                out.append(s)
                frontier.append(s)
    return [s for s in STAGES if s in out]


def mark_failed(status: dict, stage: str, detail: str) -> None:
    mark(status, stage, "failed", detail)
    for dep in dependents(stage):
        if status[dep]["status"] in ("pending", "skipped"):
            mark(status, dep, "skipped", f"depends on {stage}, which {status[stage]['status']}")


def run_stage(status: dict, stage: str, fn: Callable[[], Any], log: Callable[[str], None] = print) -> Any:
    """Run one stage; record ok/failed and cascade skips. Never raises except SetupError/KeyboardInterrupt."""
    if status[stage]["status"] == "skipped":
        log(f"[{stage}] skipped: {status[stage]['detail']}")
        return None
    log(f"[{stage}] running...")
    try:
        result = fn()
    except (SetupError, KeyboardInterrupt):
        raise
    except Exception as exc:  # noqa: BLE001 - any stage failure must land in the report
        mark_failed(status, stage, f"{type(exc).__name__}: {exc}")
        log(f"[{stage}] FAILED: {status[stage]['detail']}")
        return None
    if status[stage]["status"] == "pending":
        mark_ok(status, stage)
    log(f"[{stage}] {status[stage]['status']}")
    return result


# --- cache ------------------------------------------------------------------

def cached_json(path: Path, compute: Callable[[], Any], use_cache: bool = True) -> Any:
    if use_cache and path.exists():
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    result = compute()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False)
    return result
