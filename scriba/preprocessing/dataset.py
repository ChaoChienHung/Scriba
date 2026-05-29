from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset


def _pick_column(fieldnames: Iterable[str], candidates: list[str]) -> Optional[str]:
    lowered = {name.lower(): name for name in fieldnames}
    for c in candidates:
        if c.lower() in lowered:
            return lowered[c.lower()]
    return None


@dataclass(frozen=True)
class CsvSchema:
    image_column: str
    text_column: str


def infer_csv_schema(csv_path: Path) -> CsvSchema:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {csv_path}")
        fieldnames = list(reader.fieldnames)

    image_column = _pick_column(fieldnames, ["image", "img", "path", "filepath", "filename", "file"])
    text_column = _pick_column(fieldnames, ["text", "label", "transcription", "gt", "ground_truth"])

    if image_column and text_column:
        return CsvSchema(image_column=image_column, text_column=text_column)

    if len(fieldnames) == 2:
        return CsvSchema(image_column=fieldnames[0], text_column=fieldnames[1])

    raise ValueError(
        f"Cannot infer CSV schema for {csv_path}. "
        f"Please provide columns for image/text. Found: {fieldnames}"
    )


class OCRCsvDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        csv_path: str | Path,
        images_dir: str | Path,
        processor: Any,
        *,
        max_target_length: int,
        image_column: Optional[str] = None,
        text_column: Optional[str] = None,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.images_dir = Path(images_dir)
        self.processor = processor
        self.max_target_length = max_target_length

        schema = infer_csv_schema(self.csv_path)
        self.image_column = image_column or schema.image_column
        self.text_column = text_column or schema.text_column

        self._rows: list[dict[str, str]] = []
        with self.csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get(self.image_column) is None or row.get(self.text_column) is None:
                    continue
                self._rows.append(row)

        if len(self._rows) == 0:
            raise ValueError(f"No rows loaded from {self.csv_path}")

    def __len__(self) -> int:
        return len(self._rows)

    def _resolve_image_path(self, raw: str) -> Path:
        p = Path(raw)
        if not p.is_absolute():
            p = self.images_dir / p
        return p

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self._rows[idx]
        image_path = self._resolve_image_path(row[self.image_column])
        text = row[self.text_column]

        image = Image.open(image_path).convert("RGB")

        processed = self.processor(
            images=image,
            text=text,
            padding="max_length",
            truncation=True,
            max_length=self.max_target_length,
            return_tensors="pt",
        )

        pixel_values = processed["pixel_values"][0]
        labels_key = "labels" if "labels" in processed else "input_ids"
        labels = processed[labels_key][0].clone()

        pad_id = getattr(self.processor, "tokenizer", None)
        if pad_id is not None and getattr(self.processor.tokenizer, "pad_token_id", None) is not None:
            labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {
            "pixel_values": pixel_values,
            "labels": labels,
        }


@dataclass(frozen=True)
class OCRDataCollator:
    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        pixel_values = torch.stack([f["pixel_values"] for f in features])
        labels = torch.stack([f["labels"] for f in features])
        return {
            "pixel_values": pixel_values,
            "labels": labels,
        }
