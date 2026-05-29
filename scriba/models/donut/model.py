from __future__ import annotations

from typing import Optional

from transformers import DonutProcessor, VisionEncoderDecoderModel

from .config import DonutConfig


def build_model_and_processor(pretrained: Optional[str] = None) -> tuple[VisionEncoderDecoderModel, DonutProcessor]:
    cfg = DonutConfig(pretrained=pretrained or DonutConfig.pretrained)
    processor = DonutProcessor.from_pretrained(cfg.pretrained)
    model = VisionEncoderDecoderModel.from_pretrained(cfg.pretrained)
    model.config.decoder_start_token_id = processor.tokenizer.bos_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.eos_token_id = processor.tokenizer.eos_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    model.config.max_length = cfg.max_target_length
    return model, processor

