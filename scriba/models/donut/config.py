from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DonutConfig:
    pretrained: str = "naver-clova-ix/donut-base"
    max_target_length: int = 256

