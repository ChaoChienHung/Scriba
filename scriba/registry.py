from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Optional


@dataclass(frozen=True)
class ModelFactory:
    arch: str
    default_pretrained: str
    build: Callable[[Optional[str]], tuple[object, object]]


def _build_trocr(pretrained: Optional[str]) -> tuple[object, object]:
    from .models.trocr.model import build_model_and_processor

    return build_model_and_processor(pretrained=pretrained)


def _build_donut(pretrained: Optional[str]) -> tuple[object, object]:
    from .models.donut.model import build_model_and_processor

    return build_model_and_processor(pretrained=pretrained)


_REGISTRY: Mapping[str, ModelFactory] = {
    "trocr": ModelFactory(
        arch="trocr",
        default_pretrained="microsoft/trocr-base-handwritten",
        build=_build_trocr,
    ),
    "donut": ModelFactory(
        arch="donut",
        default_pretrained="naver-clova-ix/donut-base",
        build=_build_donut,
    ),
}


def get_model_factory(arch: str) -> ModelFactory:
    key = arch.strip().lower()
    if key not in _REGISTRY:
        raise ValueError(f"Unknown arch: {arch}. Available: {', '.join(sorted(_REGISTRY))}")
    return _REGISTRY[key]

