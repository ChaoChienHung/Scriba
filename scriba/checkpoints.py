from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def checkpoints_root() -> Path:
    return project_root() / "models"


def latest_checkpoint_dir(arch: str) -> Path:
    return checkpoints_root() / arch.strip().lower() / "latest"


def resolve_checkpoint_dir(
    *,
    arch: str,
    checkpoint_dir: Optional[str | Path] = None,
) -> Optional[Path]:
    if checkpoint_dir is not None:
        p = Path(checkpoint_dir)
        return p
    p = latest_checkpoint_dir(arch)
    return p if p.exists() else None


def publish_latest(*, arch: str, run_dir: str | Path) -> Path:
    src = Path(run_dir)
    if not src.exists():
        raise FileNotFoundError(src)

    dst = latest_checkpoint_dir(arch)
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.is_symlink() or dst.exists():
        if dst.is_symlink() or dst.is_file():
            dst.unlink(missing_ok=True)
        else:
            shutil.rmtree(dst)

    dst.symlink_to(src.resolve(), target_is_directory=True)
    return dst
