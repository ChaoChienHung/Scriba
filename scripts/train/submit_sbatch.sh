#!/usr/bin/env bash
#SBATCH --job-name=scriba-train
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=runs/slurm-train-%j.out
#SBATCH --error=runs/slurm-train-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p runs

if [ -d "venv" ]; then
  source venv/bin/activate
fi

python3 -m scriba.train \
  --arch trocr \
  --run-name trocr_sbatch_run \
  --publish-latest

