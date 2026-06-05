from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import torch
from transformers import VisionEncoderDecoderModel


@dataclass
class CachedBundle:
    model: VisionEncoderDecoderModel
    processor: Any
    resolved_checkpoint: Optional[Path]


class ModelCache:
    def __init__(self, *, max_items: int = 2) -> None:
        self._max_items = max_items
        self._lock = threading.RLock()
        self._items: OrderedDict[tuple[str, str, str], CachedBundle] = OrderedDict()

    def get_or_load(
        self,
        *,
        key: tuple[str, str, str],
        loader: Callable[[], tuple[VisionEncoderDecoderModel, Any, Optional[Path]]],
        device: torch.device,
    ) -> CachedBundle:
        with self._lock:
            hit = self._items.get(key)
            if hit is not None:
                self._items.move_to_end(key)
                return hit

            model, processor, resolved = loader()
            model.to(device)
            model.eval()
            bundle = CachedBundle(model=model, processor=processor, resolved_checkpoint=resolved)
            self._items[key] = bundle
            self._items.move_to_end(key)
            while len(self._items) > self._max_items:
                self._items.popitem(last=False)
            return bundle
