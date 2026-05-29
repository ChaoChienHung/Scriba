from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrOCRConfig:
    pretrained: str = "microsoft/trocr-base-handwritten"
    max_target_length: int = 128

