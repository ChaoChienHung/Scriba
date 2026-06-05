from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .paths import repo_root, safe_child_dir


_RUN_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")


def runs_root() -> Path:
    return repo_root() / "runs"


def list_runs() -> list[Path]:
    root = runs_root()
    if not root.exists():
        return []
    out: list[Path] = []
    for p in sorted(root.iterdir(), key=lambda x: x.name, reverse=True):
        if p.is_dir():
            out.append(p)
    return out


def resolve_run_dir(run_id: str) -> Path:
    run_id = run_id.strip()
    if not _RUN_ID_RE.match(run_id):
        raise ValueError("invalid run_id")
    return safe_child_dir(root=runs_root(), child=run_id)


def read_trainer_state(run_dir: Path) -> dict[str, Any]:
    p = run_dir / "trainer_state.json"
    if not p.exists():
        raise FileNotFoundError(p)
    return json.loads(p.read_text(encoding="utf-8"))


def extract_metric_series(trainer_state: dict[str, Any]) -> tuple[list[str], list[dict[str, Any]]]:
    log_history = trainer_state.get("log_history", [])
    if not isinstance(log_history, list):
        return [], []

    keys: set[str] = set()
    series: list[dict[str, Any]] = []
    for row in log_history:
        if not isinstance(row, dict):
            continue
        point: dict[str, Any] = {
            "step": row.get("step"),
            "epoch": row.get("epoch"),
            "timestamp": row.get("timestamp"),
            "values": {},
        }
        for k, v in row.items():
            if k in {"step", "epoch", "timestamp"}:
                continue
            if isinstance(v, (int, float, str, bool)) or v is None:
                point["values"][k] = v
                keys.add(k)
        if point["values"]:
            series.append(point)

    return sorted(keys), series

