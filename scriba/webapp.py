from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import streamlit as st
from PIL import Image

from .checkpoints import latest_checkpoint_dir, project_root
from .inference import InferenceResult, run_inference


@dataclass(frozen=True)
class ModelChoice:
    arch: str
    checkpoint_dir: Optional[str]
    pretrained: Optional[str]


def _runs_root() -> Path:
    return project_root() / "runs"


def _list_runs() -> list[Path]:
    root = _runs_root()
    if not root.exists():
        return []
    return sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.name, reverse=True)


def _maybe_read_json(path: Path) -> Optional[dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _run_predict(choice: ModelChoice, image: Image.Image) -> InferenceResult:
    return run_inference(
        arch=choice.arch,
        image=image,
        checkpoint_dir=choice.checkpoint_dir,
        pretrained=choice.pretrained,
        max_new_tokens=256 if choice.arch == "donut" else 128,
        num_beams=1,
    )


def _render_output(res: InferenceResult) -> None:
    st.metric("Latency (ms)", f"{res.latency_ms:.1f}")
    if res.arch == "donut":
        st.subheader("Structured JSON")
        st.json(res.output.get("json"))
        st.subheader("Raw")
        st.code(res.output.get("raw") or "", language="text")
    else:
        st.subheader("Text")
        st.code(res.output.get("text") or "", language="text")


def _choice_ui(prefix: str, default_arch: str) -> ModelChoice:
    arch = st.selectbox(f"{prefix} Model", ["trocr", "donut"], index=0 if default_arch == "trocr" else 1)

    latest = latest_checkpoint_dir(arch)
    use_latest = st.checkbox(f"{prefix} Use models/{arch}/latest", value=latest.exists())
    checkpoint_dir = str(latest) if use_latest and latest.exists() else None

    custom = st.text_input(f"{prefix} Custom checkpoint dir (optional)", value="")
    if custom.strip():
        checkpoint_dir = custom.strip()

    pretrained = st.text_input(f"{prefix} Pretrained id (fallback)", value="")
    pretrained = pretrained.strip() or None
    return ModelChoice(arch=arch, checkpoint_dir=checkpoint_dir, pretrained=pretrained)


def _runs_tab() -> None:
    runs = _list_runs()
    if not runs:
        st.info("No runs found under runs/. Train first or copy a run folder here.")
        return

    selected = st.selectbox("Select run", runs, format_func=lambda p: p.name)
    trainer_state = _maybe_read_json(selected / "trainer_state.json")
    if trainer_state is None:
        st.warning("trainer_state.json not found in this run folder.")
        return

    log_history = trainer_state.get("log_history") or []
    df = pd.DataFrame(log_history)
    if "step" in df.columns:
        df = df.sort_values("step")
        df = df.set_index("step")

    st.subheader("Raw trainer_state.json")
    st.json(trainer_state)

    st.subheader("log_history (table)")
    st.dataframe(df)

    metric_cols = [c for c in df.columns if c.startswith("eval_") or c in {"loss", "eval_loss"}]
    if metric_cols:
        st.subheader("Metrics")
        st.line_chart(df[metric_cols])


def main() -> None:
    st.set_page_config(page_title="Scriba", layout="wide")
    st.title("Scriba")

    tab_infer, tab_compare, tab_runs = st.tabs(["Infer", "Compare", "Runs"])

    with tab_infer:
        choice = _choice_ui("A", "trocr")
        upload = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "webp"])
        if upload is not None:
            image = Image.open(upload).convert("RGB")
            st.image(image, caption="Input", use_container_width=True)
            if st.button("Run inference", type="primary"):
                res = _run_predict(choice, image)
                _render_output(res)

    with tab_compare:
        col_left, col_right = st.columns(2)
        with col_left:
            choice_a = _choice_ui("A", "trocr")
        with col_right:
            choice_b = _choice_ui("B", "donut")

        upload = st.file_uploader("Upload an image for comparison", type=["png", "jpg", "jpeg", "webp"], key="cmp")
        if upload is not None:
            image = Image.open(upload).convert("RGB")
            st.image(image, caption="Input", use_container_width=True)
            if st.button("Run compare", type="primary"):
                a, b = st.columns(2)
                with a:
                    st.header(f"A: {choice_a.arch}")
                    res_a = _run_predict(choice_a, image)
                    _render_output(res_a)
                with b:
                    st.header(f"B: {choice_b.arch}")
                    res_b = _run_predict(choice_b, image)
                    _render_output(res_b)

    with tab_runs:
        _runs_tab()


if __name__ == "__main__":
    main()
