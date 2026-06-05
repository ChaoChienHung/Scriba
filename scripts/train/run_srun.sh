#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.." || exit 1

if [ -d "venv" ]; then
  source venv/bin/activate
fi

srun -p gpu --gres=gpu:1 -c 4 --mem=16G -t 02:00:00 -u \
  python3 -m scriba.train \
    --arch trocr \
    --run-name trocr_srun_run \
    --publish-latest

