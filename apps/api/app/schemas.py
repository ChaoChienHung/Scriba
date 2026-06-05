from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


ModelSource = Literal["latest", "stored", "pretrained", "custom"]


class CapabilitiesResponse(BaseModel):
    arches: list[str] = Field(default_factory=lambda: ["trocr", "donut"])
    compare_min: int = 2
    compare_max: int = 4
    decode_defaults: dict[str, Any] = Field(default_factory=lambda: {"max_new_tokens": 128, "num_beams": 1})


class StoredModelItem(BaseModel):
    name: str
    path: str


class ModelsResponse(BaseModel):
    arch: str
    latest_exists: bool
    stored: list[StoredModelItem]


class PretrainedCacheRequest(BaseModel):
    arch: str
    pretrained_id: str
    name: Optional[str] = None
    set_latest: bool = True


class PretrainedCacheResponse(BaseModel):
    saved_path: str
    latest_updated: bool


class InferSourceResolved(BaseModel):
    kind: Literal["checkpoint_dir", "pretrained"]
    value: str


class InferResponse(BaseModel):
    request_id: str
    arch: str
    source_resolved: InferSourceResolved
    latency_ms: float
    output: dict[str, Any]


class CompareResponse(BaseModel):
    request_id: str
    results: list[InferResponse]


class RunSummary(BaseModel):
    run_id: str
    path: str
    has_trainer_state: bool


class MetricPoint(BaseModel):
    step: Optional[int] = None
    epoch: Optional[float] = None
    timestamp: Optional[float] = None
    values: dict[str, Any]


class MetricSeriesResponse(BaseModel):
    run_id: str
    keys: list[str]
    series: list[MetricPoint]

