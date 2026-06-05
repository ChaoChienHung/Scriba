from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

from ..eval.metrics import compute_ocr_metrics


@dataclass(frozen=True)
class TrainerConfig:
    output_dir: str
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    num_train_epochs: float = 3.0
    learning_rate: float = 5e-5
    warmup_ratio: float = 0.0
    weight_decay: float = 0.0
    logging_steps: int = 50
    save_strategy: str = "epoch"
    evaluation_strategy: str = "epoch"
    save_total_limit: int = 2
    predict_with_generate: bool = True
    fp16: bool = False
    bf16: bool = False
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "cer"
    greater_is_better: bool = False
    report_to: list[str] = field(default_factory=list)
    run_name: Optional[str] = None


def build_trainer(
    *,
    model: Any,
    processor: Any,
    train_dataset: Any,
    eval_dataset: Any,
    data_collator: Any,
    cfg: TrainerConfig,
    generation_max_length: Optional[int] = None,
) -> Seq2SeqTrainer:
    training_args = Seq2SeqTrainingArguments(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        num_train_epochs=cfg.num_train_epochs,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        save_strategy=cfg.save_strategy,
        evaluation_strategy=cfg.evaluation_strategy,
        logging_steps=cfg.logging_steps,
        save_total_limit=cfg.save_total_limit,
        predict_with_generate=cfg.predict_with_generate,
        fp16=cfg.fp16,
        bf16=cfg.bf16,
        load_best_model_at_end=cfg.load_best_model_at_end,
        metric_for_best_model=cfg.metric_for_best_model,
        greater_is_better=cfg.greater_is_better,
        report_to=cfg.report_to,
        run_name=cfg.run_name,
    )

    if generation_max_length is not None:
        training_args.generation_max_length = generation_max_length

    compute_metrics = lambda p: compute_ocr_metrics(processor, p)
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=getattr(processor, "tokenizer", None),
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    return trainer
