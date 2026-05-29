
# Dev Notes

## Output Convention

- Training outputs go to `runs/<run_name>/`
  - `model/` HuggingFace `VisionEncoderDecoderModel`
  - `processor/` corresponding processor (TrOCRProcessor / DonutProcessor)

## Evaluation Tiers

- Tier 0 (Smoke)
  - One batch forward + generate
  - Verify decoding works and no obvious NaNs
- Tier 1 (Offline metrics)
  - Run `python -m scriba.eval` on validation/test split
  - Track `cer` and `wer`
- Tier 2 (Qualitative)
  - Save a small set of predictions with image paths for manual inspection
