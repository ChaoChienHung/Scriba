from __future__ import annotations

from pathlib import Path

from scriba.checkpoints import project_root


def repo_root() -> Path:
    return project_root()


def safe_child_dir(*, root: Path, child: str) -> Path:
    p = (root / child).resolve()
    root_resolved = root.resolve()
    if p == root_resolved:
        return p
    if root_resolved not in p.parents:
        raise ValueError("path escapes root")
    return p


def safe_resolve_dir(*, root: Path, p: str | Path) -> Path:
    pp = Path(p).expanduser()
    rp = (root / pp).resolve() if not pp.is_absolute() else pp.resolve()
    root_resolved = root.resolve()
    if rp == root_resolved:
        return rp
    if root_resolved not in rp.parents:
        raise ValueError("path escapes root")
    if not rp.exists():
        raise FileNotFoundError(rp)
    if not rp.is_dir():
        raise ValueError("path is not a directory")
    return rp
