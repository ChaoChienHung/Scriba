#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

ZIP_PATH="$ROOT/scripts/handwriting-data.zip"
RAW_ROOT="$ROOT/data/raw/landlord"

mkdir -p "$RAW_ROOT"

curl -L -o "$ZIP_PATH" "https://www.kaggle.com/api/v1/datasets/download/landlord/handwriting-recognition"
unzip -o "$ZIP_PATH" -d "$RAW_ROOT"
rm -f "$ZIP_PATH"

mkdir -p "$RAW_ROOT/test/images" "$RAW_ROOT/train/images" "$RAW_ROOT/validation/images"

if [ -d "$RAW_ROOT/test_v2/test" ]; then
  mv "$RAW_ROOT/test_v2/test" "$RAW_ROOT/test/images/"
fi
if [ -d "$RAW_ROOT/train_v2/train" ]; then
  mv "$RAW_ROOT/train_v2/train" "$RAW_ROOT/train/images/"
fi
if [ -d "$RAW_ROOT/validation_v2/validation" ]; then
  mv "$RAW_ROOT/validation_v2/validation" "$RAW_ROOT/validation/images/"
fi

rm -rf "$RAW_ROOT/test_v2" "$RAW_ROOT/train_v2" "$RAW_ROOT/validation_v2"

if [ -f "$RAW_ROOT/written_name_test_v2.csv" ]; then
  mv "$RAW_ROOT/written_name_test_v2.csv" "$RAW_ROOT/test/label.csv"
fi
if [ -f "$RAW_ROOT/written_name_train_v2.csv" ]; then
  mv "$RAW_ROOT/written_name_train_v2.csv" "$RAW_ROOT/train/label.csv"
fi
if [ -f "$RAW_ROOT/written_name_validation_v2.csv" ]; then
  mv "$RAW_ROOT/written_name_validation_v2.csv" "$RAW_ROOT/validation/label.csv"
fi

