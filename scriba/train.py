from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from .checkpoints import publish_latest
from .config import default_data_paths
from .distributed import is_main_process
from .engines.trainer import TrainerConfig, build_trainer
from .logging import setup_logging
from .preprocessing.dataset import OCRCsvDataset, OCRDataCollator
from .registry import get_model_factory


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m scriba.train")
    p.add_argument("--arch", required=True, choices=["trocr", "donut"])
    p.add_argument("--pretrained", default=None)
    p.add_argument("--run-name", default=None)
    p.add_argument("--output-dir", default=None)

    p.add_argument("--train-csv", default=None)
    p.add_argument("--train-images-dir", default=None)
    p.add_argument("--val-csv", default=None)
    p.add_argument("--val-images-dir", default=None)

    p.add_argument("--max-target-length", type=int, default=None)

    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--train-bs", type=int, default=8)
    p.add_argument("--eval-bs", type=int, default=8)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--logging-steps", type=int, default=50)
    p.add_argument("--publish-latest", action="store_true")
    p.add_argument(
        "--report-to",
        default="",
        help="Comma-separated list for HF Trainer report_to (e.g. wandb,tensorboard). Empty disables.",
    )
    p.add_argument("--run-name-hf", default=None, help="Override HF Trainer run_name (defaults to --run-name).")
    return p


def main(argv: Optional[list[str]] = None) -> None:
    args = build_argparser().parse_args(argv)
    setup_logging()

    factory = get_model_factory(args.arch)
    model, processor = factory.build(args.pretrained)

    project_root = _project_root()
    data_paths = default_data_paths(project_root)

    train_csv = Path(args.train_csv) if args.train_csv else data_paths.split_csv_path("train")
    train_images_dir = Path(args.train_images_dir) if args.train_images_dir else data_paths.split_images_dir("train")
    val_csv = Path(args.val_csv) if args.val_csv else data_paths.split_csv_path("validation")
    val_images_dir = Path(args.val_images_dir) if args.val_images_dir else data_paths.split_images_dir("validation")

    max_target_length = args.max_target_length or getattr(model.config, "max_length", 128)

    train_dataset = OCRCsvDataset(
        train_csv,
        train_images_dir,
        processor,
        max_target_length=max_target_length,
    )
    val_dataset = OCRCsvDataset(
        val_csv,
        val_images_dir,
        processor,
        max_target_length=max_target_length,
    )

    run_name = args.run_name or f"{args.arch}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    output_dir = Path(args.output_dir) if args.output_dir else (project_root / "runs" / run_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    report_to = [x.strip() for x in str(args.report_to).split(",") if x.strip()]
    hf_run_name = args.run_name_hf or run_name

    if is_main_process():
        cfg = {
            "arch": args.arch,
            "pretrained": args.pretrained,
            "run_name": run_name,
            "output_dir": str(output_dir),
            "train_csv": str(train_csv),
            "train_images_dir": str(train_images_dir),
            "val_csv": str(val_csv),
            "val_images_dir": str(val_images_dir),
            "max_target_length": max_target_length,
            "epochs": args.epochs,
            "lr": args.lr,
            "train_bs": args.train_bs,
            "eval_bs": args.eval_bs,
            "fp16": bool(args.fp16),
            "bf16": bool(args.bf16),
            "logging_steps": args.logging_steps,
            "publish_latest": bool(args.publish_latest),
            "report_to": report_to,
            "hf_run_name": hf_run_name,
            "argv": sys.argv,
            "env": {
                "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            },
        }
        (output_dir / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    trainer_cfg = TrainerConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.train_bs,
        per_device_eval_batch_size=args.eval_bs,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        fp16=args.fp16,
        bf16=args.bf16,
        logging_steps=args.logging_steps,
        report_to=report_to,
        run_name=hf_run_name,
    )

    trainer = build_trainer(
        model=model,
        processor=processor,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=OCRDataCollator(),
        cfg=trainer_cfg,
        generation_max_length=max_target_length,
    )

    trainer.train()
    if is_main_process():
        trainer.save_model(str(output_dir / "model"))
        processor.save_pretrained(str(output_dir / "processor"))
        if args.publish_latest:
            publish_latest(arch=args.arch, run_dir=output_dir)


if __name__ == "__main__":
    main()
