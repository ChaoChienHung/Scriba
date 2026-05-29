from __future__ import annotations

from typing import Optional

from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from .config import TrOCRConfig


def build_model_and_processor(pretrained: Optional[str] = None) -> tuple[VisionEncoderDecoderModel, TrOCRProcessor]:
    cfg = TrOCRConfig(pretrained=pretrained or TrOCRConfig.pretrained)
    processor = TrOCRProcessor.from_pretrained(cfg.pretrained)
    model = VisionEncoderDecoderModel.from_pretrained(cfg.pretrained)
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    model.config.max_length = cfg.max_target_length
    return model, processor

