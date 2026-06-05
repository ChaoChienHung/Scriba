from __future__ import annotations

import json
import time
import uuid
from io import BytesIO
from pathlib import Path
from typing import Any, Optional

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from scriba.checkpoints import latest_checkpoint_dir
from scriba.inference import InferenceResult, load_model_and_processor
from apps.cli.model_store import ensure_pretrained_saved, list_stored_models, resolve_model_dir

from .metrics_reader import extract_metric_series, list_runs, read_trainer_state, resolve_run_dir
from .model_cache import ModelCache
from .paths import repo_root, safe_resolve_dir
from .schemas import (
    CapabilitiesResponse,
    CompareResponse,
    InferResponse,
    InferSourceResolved,
    MetricPoint,
    MetricSeriesResponse,
    ModelsResponse,
    PretrainedCacheRequest,
    PretrainedCacheResponse,
    RunSummary,
    StoredModelItem,
)


def _default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_cache = ModelCache(max_items=2)


@app.get("/api/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/capabilities", response_model=CapabilitiesResponse)
def capabilities() -> CapabilitiesResponse:
    return CapabilitiesResponse()


@app.get("/api/models", response_model=ModelsResponse)
def models(arch: str) -> ModelsResponse:
    arch = arch.strip().lower()
    if arch not in {"trocr", "donut"}:
        raise HTTPException(status_code=400, detail="invalid arch")

    latest = (repo_root() / "models" / arch / "latest").exists()
    stored = [StoredModelItem(name=m.name, path=str(m.path)) for m in list_stored_models(arch)]
    return ModelsResponse(arch=arch, latest_exists=latest, stored=stored)


@app.post("/api/models/pretrained-cache", response_model=PretrainedCacheResponse)
def pretrained_cache(req: PretrainedCacheRequest) -> PretrainedCacheResponse:
    arch = req.arch.strip().lower()
    if arch not in {"trocr", "donut"}:
        raise HTTPException(status_code=400, detail="invalid arch")

    dst = ensure_pretrained_saved(
        arch=arch,
        pretrained_id=req.pretrained_id,
        name=req.name,
        set_latest=req.set_latest,
    )
    return PretrainedCacheResponse(saved_path=str(dst), latest_updated=bool(req.set_latest))


def _read_image(upload: UploadFile) -> Image.Image:
    try:
        data = upload.file.read()
        img = Image.open(BytesIO(data))
        img.load()
        return img.convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="invalid image")


def _resolve_source(
    *,
    arch: str,
    source: str,
    stored_name: Optional[str],
    pretrained_id: Optional[str],
    custom_checkpoint_dir: Optional[str],
) -> tuple[Optional[str], Optional[Path], tuple[str, str, str]]:
    source = source.strip().lower()
    if source not in {"latest", "stored", "pretrained", "custom"}:
        raise HTTPException(status_code=400, detail="invalid source")

    if source == "latest":
        p = latest_checkpoint_dir(arch)
        if p.exists():
            rp = p.resolve()
            return None, rp, ("checkpoint_dir", arch, str(rp))
        pid = pretrained_id.strip() if pretrained_id else None
        return pid, None, ("pretrained", arch, (pid or "__default__"))

    if source == "stored":
        if not stored_name:
            raise HTTPException(status_code=400, detail="stored_name is required")
        resolved = resolve_model_dir(arch=arch, selection=stored_name)
        if resolved is None:
            raise HTTPException(status_code=404, detail="stored model not found")
        ckpt = resolved.resolve()
        return None, ckpt, ("checkpoint_dir", arch, str(ckpt))

    if source == "custom":
        if not custom_checkpoint_dir:
            raise HTTPException(status_code=400, detail="custom_checkpoint_dir is required")
        ckpt = safe_resolve_dir(root=repo_root(), p=custom_checkpoint_dir)
        return None, ckpt, ("checkpoint_dir", arch, str(ckpt))

    if source == "pretrained":
        if not pretrained_id:
            raise HTTPException(status_code=400, detail="pretrained_id is required")
        pid = pretrained_id.strip()
        return pid, None, ("pretrained", arch, pid)

    pid = pretrained_id.strip() if pretrained_id else None
    return pid, None, ("pretrained", arch, (pid or "__default__"))


def _infer_with_cache(
    *,
    arch: str,
    image: Image.Image,
    pretrained: Optional[str],
    checkpoint_dir: Optional[Path],
    cache_key: tuple[str, str, str],
    max_new_tokens: int,
    num_beams: int,
) -> InferenceResult:
    device = _default_device()

    def _loader() -> tuple[Any, Any, Optional[Path]]:
        model, processor, resolved = load_model_and_processor(
            arch=arch,
            pretrained=pretrained,
            checkpoint_dir=checkpoint_dir,
        )
        return model, processor, resolved

    bundle = _cache.get_or_load(key=cache_key, loader=_loader, device=device)

    inputs = bundle.processor(images=image.convert("RGB"), return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)

    started = time.perf_counter()
    generated_ids = bundle.model.generate(pixel_values, max_new_tokens=max_new_tokens, num_beams=num_beams)
    latency_ms = (time.perf_counter() - started) * 1000.0

    text = bundle.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    output: dict[str, Any] = {
        "source": str(bundle.resolved_checkpoint) if bundle.resolved_checkpoint is not None else (pretrained or "pretrained_default"),
    }
    if arch.strip().lower() == "donut":
        parsed = None
        if hasattr(bundle.processor, "token2json"):
            try:
                parsed = bundle.processor.token2json(text)
            except Exception:
                parsed = None
        output.update({"raw": text, "json": parsed})
    else:
        output.update({"text": text})

    return InferenceResult(arch=arch, output=output, latency_ms=latency_ms)


def _to_infer_response(*, request_id: str, result: InferenceResult) -> InferResponse:
    src = result.output.get("source", "unknown")
    p = Path(str(src))
    if not p.is_absolute():
        p = (repo_root() / p).resolve()
    kind = "checkpoint_dir" if p.exists() else "pretrained"
    return InferResponse(
        request_id=request_id,
        arch=result.arch,
        source_resolved=InferSourceResolved(kind=kind, value=str(p) if kind == "checkpoint_dir" else str(src)),
        latency_ms=result.latency_ms,
        output=result.output,
    )


@app.post("/api/infer", response_model=InferResponse)
def infer(
    arch: str = Form(...),
    source: str = Form("latest"),
    stored_name: Optional[str] = Form(None),
    pretrained_id: Optional[str] = Form(None),
    custom_checkpoint_dir: Optional[str] = Form(None),
    max_new_tokens: int = Form(128),
    num_beams: int = Form(1),
    image: UploadFile = File(...),
) -> InferResponse:
    arch = arch.strip().lower()
    if arch not in {"trocr", "donut"}:
        raise HTTPException(status_code=400, detail="invalid arch")

    img = _read_image(image)
    pretrained, checkpoint_dir, cache_key = _resolve_source(
        arch=arch,
        source=source,
        stored_name=stored_name,
        pretrained_id=pretrained_id,
        custom_checkpoint_dir=custom_checkpoint_dir,
    )

    request_id = str(uuid.uuid4())
    result = _infer_with_cache(
        arch=arch,
        image=img,
        pretrained=pretrained,
        checkpoint_dir=checkpoint_dir,
        cache_key=cache_key,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
    )
    return _to_infer_response(request_id=request_id, result=result)


@app.post("/api/compare", response_model=CompareResponse)
def compare(
    specs_json: str = Form(...),
    max_new_tokens: int = Form(128),
    num_beams: int = Form(1),
    image: UploadFile = File(...),
) -> CompareResponse:
    try:
        specs = json.loads(specs_json)
    except Exception:
        raise HTTPException(status_code=400, detail="invalid specs_json")

    if not isinstance(specs, list) or not (2 <= len(specs) <= 4):
        raise HTTPException(status_code=400, detail="specs must be a list with length 2~4")

    img = _read_image(image)
    request_id = str(uuid.uuid4())

    results: list[InferResponse] = []
    for s in specs:
        if not isinstance(s, dict):
            raise HTTPException(status_code=400, detail="invalid spec item")
        arch = str(s.get("arch", "")).strip().lower()
        if arch not in {"trocr", "donut"}:
            raise HTTPException(status_code=400, detail="invalid arch")
        pretrained, checkpoint_dir, cache_key = _resolve_source(
            arch=arch,
            source=str(s.get("source", "latest")),
            stored_name=s.get("stored_name"),
            pretrained_id=s.get("pretrained_id"),
            custom_checkpoint_dir=s.get("custom_checkpoint_dir"),
        )
        r = _infer_with_cache(
            arch=arch,
            image=img,
            pretrained=pretrained,
            checkpoint_dir=checkpoint_dir,
            cache_key=cache_key,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
        )
        results.append(_to_infer_response(request_id=request_id, result=r))

    return CompareResponse(request_id=request_id, results=results)


@app.get("/api/runs", response_model=list[RunSummary])
def runs() -> list[RunSummary]:
    out: list[RunSummary] = []
    for p in list_runs():
        out.append(
            RunSummary(
                run_id=p.name,
                path=str(p),
                has_trainer_state=(p / "trainer_state.json").exists(),
            )
        )
    return out


@app.get("/api/runs/{run_id}/metrics", response_model=MetricSeriesResponse)
def run_metrics(run_id: str) -> MetricSeriesResponse:
    try:
        run_dir = resolve_run_dir(run_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        trainer_state = read_trainer_state(run_dir)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="trainer_state.json not found")

    keys, series_raw = extract_metric_series(trainer_state)
    series = [MetricPoint(**p) for p in series_raw]
    return MetricSeriesResponse(run_id=run_id, keys=keys, series=series)
