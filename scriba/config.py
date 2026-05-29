from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataPaths:
    root: Path

    @property
    def raw_landlord_root(self) -> Path:
        return self.root / "raw" / "landlord"

    def split_images_dir(self, split: str) -> Path:
        return self.raw_landlord_root / split / "images"

    def split_csv_path(self, split: str) -> Path:
        return self.raw_landlord_root / split / "label.csv"


def default_data_paths(project_root: Path) -> DataPaths:
    return DataPaths(root=project_root / "data")

