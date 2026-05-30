from __future__ import annotations

import sys
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st
from PIL import Image

from scriba.checkpoints import latest_checkpoint_dir, project_root
from scriba.inference import InferenceResult, run_inference
from scriba.model_store import ensure_pretrained_saved, list_stored_models, resolve_model_dir, sanitize_model_name


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
    if isinstance(res.output, dict) and "source" in res.output:
        st.caption(f"source: {res.output.get('source')}")
    if res.arch == "donut":
        st.subheader("Structured JSON")
        st.json(res.output.get("json"))
        st.subheader("Raw")
        st.code(res.output.get("raw") or "", language="text")
    else:
        st.subheader("Text")
        st.code(res.output.get("text") or "", language="text")


def _run_predict_with_params(
    choice: ModelChoice,
    image: Image.Image,
    *,
    max_new_tokens: int,
    num_beams: int,
) -> InferenceResult:
    return run_inference(
        arch=choice.arch,
        image=image,
        checkpoint_dir=choice.checkpoint_dir,
        pretrained=choice.pretrained,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
    )


def _choice_ui(*, slot: str, default_arch: str, key_prefix: str) -> ModelChoice:
    arch = st.selectbox(
        "Model",
        ["trocr", "donut"],
        index=0 if default_arch == "trocr" else 1,
        key=f"{key_prefix}-{slot}-arch",
    )

    stored = list_stored_models(arch)
    stored_names = [m.name for m in stored]
    source = st.selectbox(
        "Source",
        ["latest", "stored", "pretrained", "custom"],
        index=0,
        key=f"{key_prefix}-{slot}-source",
    )

    checkpoint_dir: Optional[str] = None
    pretrained: Optional[str] = None

    if source == "latest":
        latest = latest_checkpoint_dir(arch)
        checkpoint_dir = str(latest) if latest.exists() else None

    elif source == "stored":
        if stored_names:
            selected = st.selectbox(
                "Stored model",
                stored_names,
                index=0,
                key=f"{key_prefix}-{slot}-stored",
            )
            resolved = resolve_model_dir(arch=arch, selection=selected)
            checkpoint_dir = str(resolved) if resolved is not None else None
        else:
            st.info(f"No stored models under models/{arch}/ yet.")

    elif source == "pretrained":
        pretrained = st.text_input("Pretrained id", value="", key=f"{key_prefix}-{slot}-pretrained")
        name = st.text_input("Save as (optional)", value="", key=f"{key_prefix}-{slot}-pretrained-name")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Download to ./models", type="secondary", key=f"{key_prefix}-{slot}-download"):
                if pretrained.strip():
                    dst = ensure_pretrained_saved(
                        arch=arch,
                        pretrained_id=pretrained.strip(),
                        name=name.strip() or sanitize_model_name(pretrained),
                        set_latest=True,
                    )
                    st.success(f"Saved to {dst} and updated models/{arch}/latest")
        with col2:
            latest = latest_checkpoint_dir(arch)
            if latest.exists():
                st.caption(f"latest → {latest.resolve()}")

    else:
        custom = st.text_input("Custom checkpoint dir", value="", key=f"{key_prefix}-{slot}-custom")
        checkpoint_dir = custom.strip() or None

    return ModelChoice(arch=arch, checkpoint_dir=checkpoint_dir, pretrained=pretrained)


def _runs_tab() -> None:
    runs = _list_runs()
    if not runs:
        st.info("No runs found under runs/. Train first or copy a run folder here.")
        return

    selected = st.selectbox("Select run", runs, format_func=lambda p: p.name, key="runs-select")
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

    tab_infer, tab_compare, tab_runs = st.tabs(["Inference", "Comparison", "Training Metrics"])

    with tab_infer:
        st.subheader("Single Model Inference")
        left, right = st.columns([1, 2])
        with left:
            st.markdown("**Model**")
            choice = _choice_ui(slot="single", default_arch="trocr", key_prefix="single")
            st.markdown("**Decoding**")
            max_new_tokens = st.number_input(
                "max_new_tokens",
                min_value=1,
                max_value=2048,
                value=256 if choice.arch == "donut" else 128,
                step=1,
                key="single-max-new-tokens",
            )
            num_beams = st.number_input(
                "num_beams",
                min_value=1,
                max_value=10,
                value=1,
                step=1,
                key="single-num-beams",
            )
            upload = st.file_uploader(
                "Upload image",
                type=["png", "jpg", "jpeg", "webp"],
                key="single-upload",
            )
            run_btn = st.button("Run", type="primary", key="single-run")

        with right:
            if upload is not None:
                image = Image.open(upload).convert("RGB")
                st.image(image, caption="Input", use_container_width=True)
                if run_btn:
                    res = _run_predict_with_params(
                        choice,
                        image,
                        max_new_tokens=int(max_new_tokens),
                        num_beams=int(num_beams),
                    )
                    _render_output(res)

    with tab_compare:
        st.subheader("Model Comparison (up to 4)")
        if "compare_n" not in st.session_state:
            st.session_state.compare_n = 2

        controls = st.columns([1, 1, 2])
        with controls[0]:
            if st.button("+ Add model", key="cmp-add"):
                st.session_state.compare_n = min(4, int(st.session_state.compare_n) + 1)
        with controls[1]:
            if st.button("- Remove model", key="cmp-remove"):
                st.session_state.compare_n = max(2, int(st.session_state.compare_n) - 1)

        upload = st.file_uploader(
            "Upload image",
            type=["png", "jpg", "jpeg", "webp"],
            key="cmp-upload",
        )

        st.markdown("**Models**")
        n = int(st.session_state.compare_n)
        choice_list: list[ModelChoice] = []
        cols = st.columns(n)
        for i in range(n):
            with cols[i]:
                choice = _choice_ui(
                    slot=f"m{i}",
                    default_arch="trocr" if i == 0 else ("donut" if i == 1 else "trocr"),
                    key_prefix="cmp",
                )
                choice_list.append(choice)

        st.markdown("**Decoding**")
        decode_cols = st.columns(2)
        with decode_cols[0]:
            max_new_tokens = st.number_input(
                "max_new_tokens",
                min_value=1,
                max_value=2048,
                value=128,
                step=1,
                key="cmp-max-new-tokens",
            )
        with decode_cols[1]:
            num_beams = st.number_input(
                "num_beams",
                min_value=1,
                max_value=10,
                value=1,
                step=1,
                key="cmp-num-beams",
            )

        run_btn = st.button("Run compare", type="primary", key="cmp-run")
        if upload is not None:
            image = Image.open(upload).convert("RGB")
            st.image(image, caption="Input", use_container_width=True)
            if run_btn:
                out_cols = st.columns(len(choice_list))
                for i, choice in enumerate(choice_list):
                    with out_cols[i]:
                        st.header(f"Model {i + 1}: {choice.arch}")
                        res = _run_predict_with_params(
                            choice,
                            image,
                            max_new_tokens=int(max_new_tokens),
                            num_beams=int(num_beams),
                        )
                        _render_output(res)

    with tab_runs:
        _runs_tab()


if __name__ == "__main__":
    main()
