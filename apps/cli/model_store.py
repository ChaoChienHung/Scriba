from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from transformers import VisionEncoderDecoderModel

from scriba.checkpoints import latest_checkpoint_dir, project_root, publish_latest
from scriba.registry import get_model_factory


@dataclass(frozen=True)
class StoredModel:
    name: str
    path: Path


def models_root() -> Path:
    return project_root() / "models"


def arch_root(arch: str) -> Path:
    return models_root() / arch.strip().lower()


def list_stored_models(arch: str) -> list[StoredModel]:
    root = arch_root(arch)
    if not root.exists():
        return []
    out: list[StoredModel] = []
    for p in sorted(root.iterdir(), key=lambda x: x.name):
        if not p.is_dir():
            continue
        if p.name == "latest":
            continue
        if (p / "model").exists() or (p / "processor").exists():
            out.append(StoredModel(name=p.name, path=p))
    return out


def sanitize_model_name(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9._-]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s or "model"


def ensure_pretrained_saved(
    *,
    arch: str,
    pretrained_id: str,
    name: Optional[str] = None,
    set_latest: bool = True,
) -> Path:
    arch = arch.strip().lower()
    pretrained_id = pretrained_id.strip()
    if not pretrained_id:
        raise ValueError("pretrained_id is empty")

    model_name = sanitize_model_name(name or pretrained_id)
    dst = arch_root(arch) / model_name
    model_dir = dst / "model"
    processor_dir = dst / "processor"

    if model_dir.exists() and processor_dir.exists():
        if set_latest:
            publish_latest(arch=arch, run_dir=dst)
        return dst

    model_dir.mkdir(parents=True, exist_ok=True)
    processor_dir.mkdir(parents=True, exist_ok=True)

    factory = get_model_factory(arch)
    _, processor = factory.build(pretrained_id)

    model = VisionEncoderDecoderModel.from_pretrained(pretrained_id)
    model.save_pretrained(model_dir)
    processor.save_pretrained(processor_dir)

    if set_latest:
        publish_latest(arch=arch, run_dir=dst)

    return dst


def resolve_model_dir(
    *,
    arch: str,
    selection: str,
) -> Optional[Path]:
    selection = selection.strip()
    if not selection:
        return None

    if selection == "latest":
        p = latest_checkpoint_dir(arch)
        return p if p.exists() else None

    p = arch_root(arch) / selection
    return p if p.exists() else None
