from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, VisionEncoderDecoderModel

from ..config import default_data_paths
from ..eval.metrics import compute_ocr_metrics
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

    p.add_argument("--split", default="test", choices=["train", "validation", "test"])
    p.add_argument("--csv", default=None)
    p.add_argument("--images-dir", default=None)
    p.add_argument("--max-target-length", type=int, default=None)

    p.add_argument("--batch-size", type=int, default=8)
    return p


def main(argv: Optional[list[str]] = None) -> None:
    args = build_argparser().parse_args(argv)
    setup_logging()

    model_dir = Path(args.model_dir) if args.model_dir else None
    processor_dir = Path(args.processor_dir) if args.processor_dir else None

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
    )

    eval_args = Seq2SeqTrainingArguments(
        output_dir=str(project_root / "runs" / "_eval_tmp"),
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
    print(out.metrics)


if __name__ == "__main__":
    main()
