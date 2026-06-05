from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, VisionEncoderDecoderModel

from ..config import default_data_paths
from ..eval.metrics import cer, compute_ocr_metrics, wer
from ..logging import setup_logging
from ..preprocessing.dataset import OCRCsvDataset, OCRDataCollator
from ..registry import get_model_factory


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m scriba.eval")
    p.add_argument("--arch", required=True, choices=["trocr", "donut"])
    p.add_argument("--pretrained", default=None)
    p.add_argument("--model-dir", default=None)
    p.add_argument("--processor-dir", default=None)
    p.add_argument("--run-dir", default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--write-predictions", action="store_true")

    p.add_argument("--split", default="test", choices=["train", "validation", "test"])
    p.add_argument("--csv", default=None)
    p.add_argument("--images-dir", default=None)
    p.add_argument("--max-target-length", type=int, default=None)

    p.add_argument("--batch-size", type=int, default=8)
    return p


def main(argv: Optional[list[str]] = None) -> None:
    args = build_argparser().parse_args(argv)
    setup_logging()

    run_dir = Path(args.run_dir) if args.run_dir else None
    model_dir = Path(args.model_dir) if args.model_dir else None
    processor_dir = Path(args.processor_dir) if args.processor_dir else None

    if run_dir is not None and model_dir is None:
        model_dir = run_dir / "model"
        processor_dir = run_dir / "processor"

    if model_dir is not None:
        model = VisionEncoderDecoderModel.from_pretrained(model_dir)
        if processor_dir is None:
            candidate = model_dir.parent / "processor"
            processor_dir = candidate if candidate.exists() else model_dir
        factory = get_model_factory(args.arch)
        _, base_processor = factory.build(args.pretrained)
        processor_cls = type(base_processor)
        processor = processor_cls.from_pretrained(processor_dir)
    else:
        factory = get_model_factory(args.arch)
        model, processor = factory.build(args.pretrained)

    project_root = _project_root()
    data_paths = default_data_paths(project_root)

    csv_path = Path(args.csv) if args.csv else data_paths.split_csv_path(args.split)
    images_dir = Path(args.images_dir) if args.images_dir else data_paths.split_images_dir(args.split)

    max_target_length = args.max_target_length or getattr(model.config, "max_length", 128)
    dataset = OCRCsvDataset(
        csv_path,
        images_dir,
        processor,
        max_target_length=max_target_length,
        return_metadata=bool(args.write_predictions),
    )

    eval_output_dir: Optional[Path] = None
    if args.output_dir:
        eval_output_dir = Path(args.output_dir)
    elif run_dir is not None:
        eval_output_dir = run_dir / "eval" / args.split

    if eval_output_dir is not None:
        eval_output_dir.mkdir(parents=True, exist_ok=True)
        hf_output_dir = eval_output_dir / "_hf"
        hf_output_dir.mkdir(parents=True, exist_ok=True)
    else:
        hf_output_dir = project_root / "runs" / "_eval_tmp"

    eval_args = Seq2SeqTrainingArguments(
        output_dir=str(hf_output_dir),
        per_device_eval_batch_size=args.batch_size,
        predict_with_generate=True,
        report_to=[],
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=eval_args,
        tokenizer=getattr(processor, "tokenizer", None),
        data_collator=OCRDataCollator(),
        compute_metrics=lambda p: compute_ocr_metrics(processor, p),
    )

    out = trainer.predict(dataset)
    metrics = dict(out.metrics or {})
    print(metrics)

    if eval_output_dir is not None:
        (eval_output_dir / "metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    if eval_output_dir is not None and args.write_predictions:
        preds = out.predictions
        labels = out.label_ids
        if isinstance(preds, tuple):
            preds = preds[0]
        if preds is None or labels is None:
            return

        label_ids = np.where(labels == -100, processor.tokenizer.pad_token_id, labels)
        pred_str = processor.batch_decode(preds, skip_special_tokens=True)
        label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

        rows: list[dict[str, str]] = getattr(dataset, "_rows", [])
        image_column = getattr(dataset, "image_column", None)

        pred_path = eval_output_dir / "predictions.jsonl"
        with pred_path.open("w", encoding="utf-8") as f:
            for i, (p, r) in enumerate(zip(pred_str, label_str)):
                img_path = None
                if i < len(rows) and image_column in rows[i]:
                    raw = rows[i][image_column]
                    img_path = str((images_dir / raw).resolve()) if not Path(raw).is_absolute() else str(Path(raw))

                item = {
                    "idx": i,
                    "image_path": img_path,
                    "pred": p,
                    "label": r,
                    "cer": cer(p.strip(), r.strip()),
                    "wer": wer(p.strip(), r.strip()),
                }
                f.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
