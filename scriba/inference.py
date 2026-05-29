from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
from PIL import Image
from transformers import VisionEncoderDecoderModel

from .checkpoints import resolve_checkpoint_dir
from .registry import get_model_factory


@dataclass(frozen=True)
class InferenceResult:
    arch: str
    output: dict[str, Any]
    latency_ms: float


def _default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_from_checkpoint_dir(arch: str, checkpoint_dir: Path) -> tuple[VisionEncoderDecoderModel, Any]:
    model_dir = checkpoint_dir / "model"
    processor_dir = checkpoint_dir / "processor"

    if not model_dir.exists():
        model_dir = checkpoint_dir
    if not processor_dir.exists():
        processor_dir = checkpoint_dir

    model = VisionEncoderDecoderModel.from_pretrained(model_dir)

    factory = get_model_factory(arch)
    _, base_processor = factory.build(None)
    processor_cls = type(base_processor)
    processor = processor_cls.from_pretrained(processor_dir)
    return model, processor


def load_model_and_processor(
    *,
    arch: str,
    pretrained: Optional[str] = None,
    checkpoint_dir: Optional[str | Path] = None,
) -> tuple[VisionEncoderDecoderModel, Any, Optional[Path]]:
    resolved = resolve_checkpoint_dir(arch=arch, checkpoint_dir=checkpoint_dir)
    if resolved is not None:
        model, processor = _load_from_checkpoint_dir(arch, resolved)
        return model, processor, resolved

    factory = get_model_factory(arch)
    model, processor = factory.build(pretrained)
    return model, processor, None


def run_inference(
    *,
    arch: str,
    image: Image.Image,
    pretrained: Optional[str] = None,
    checkpoint_dir: Optional[str | Path] = None,
    device: Optional[torch.device] = None,
    max_new_tokens: int = 128,
    num_beams: int = 1,
) -> InferenceResult:
    model, processor, resolved = load_model_and_processor(
        arch=arch,
        pretrained=pretrained,
        checkpoint_dir=checkpoint_dir,
    )
    device = device or _default_device()
    model.to(device)
    model.eval()

    inputs = processor(images=image.convert("RGB"), return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)

    started = time.perf_counter()
    generated_ids = model.generate(pixel_values, max_new_tokens=max_new_tokens, num_beams=num_beams)
    latency_ms = (time.perf_counter() - started) * 1000.0

    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    output: dict[str, Any] = {
        "source": str(resolved) if resolved is not None else (pretrained or "pretrained_default"),
    }

    if arch.strip().lower() == "donut":
        parsed = None
        if hasattr(processor, "token2json"):
            try:
                parsed = processor.token2json(text)
            except Exception:
                parsed = None
        output.update({"raw": text, "json": parsed})
    else:
        output.update({"text": text})

    return InferenceResult(arch=arch, output=output, latency_ms=latency_ms)

