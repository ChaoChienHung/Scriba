# Scriba

## Introduction

This repository serves as a **sandbox** for **building**, **training**, and **evaluating handwritten text** parsing models, while **also supporting the development of real-world applications**. It contains experiments with **different model architectures**, **preprocessing pipelines**, and **decoding strategies** for **recognizing** and **structuring handwritten text**.

## Project Rules

- Immutable constraints / guardrails: see [AGENTS.md](file:///Users/bytedance/Desktop/Ludwig/Scriba/AGENTS.md)
- Design decisions / experiment notes: see [DevNotes.md](file:///Users/bytedance/Desktop/Ludwig/Scriba/DevNotes.md)
- Roadmap / backlog: see [TODO.md](file:///Users/bytedance/Desktop/Ludwig/Scriba/TODO.md)

## Model Comparison

| Model | Arch | Size | Output | Brief |
|---|---|---:|---|---|
| microsoft/trocr-small-handwritten | trocr | small | text | Small TrOCR handwritten OCR baseline（較快，適合先跑通流程） |
| microsoft/trocr-base-handwritten | trocr | base | text | Default handwritten OCR baseline（速度/效果折衷） |
| microsoft/trocr-large-handwritten | trocr | large | text | Larger TrOCR baseline（通常更準但更慢） |
| naver-clova-ix/donut-base | donut | base | raw + json | Donut baseline（可輸出 structured JSON，適合做 parsing/結構化對照） |
| naver-clova-ix/donut-base-finetuned-cord-v2 | donut | base-ft | raw + json | 偏收據/表單結構化（適合當 structured JSON 的對照樣例） |

## Folder Structure (Karen-style)

```bash
Scriba/
│
├── data/                      # Local datasets (git ignored), .gitkeep keeps folder
│   └── raw/landlord/{train,validation,test}/...
├── runs/                      # Training outputs (git ignored), .gitkeep keeps folder
├── models/                    # Default model pointers (models/<arch>/latest)
├── scripts/                   # One-off helpers (download/train templates)
├── apps/                      # Applications (API/Web/CLI) built on top of scriba core
├── utils/                     # Small reusable helpers (optional)
└── scriba/                    # Main Python package
    ├── train.py               # Entry: python -m scriba.train
    ├── eval/                  # Entry: python -m scriba.eval
    ├── engines/               # Trainer/loop abstraction
    ├── models/                # Each model in its own folder
    │   ├── trocr/
    │   └── donut/
    └── preprocessing/         # Dataset loading & preprocessing

```
## Quickstart

### 1. Install dependencies

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Download the dataset (optional)

```bash
chmod 600 ./scripts/download/download_landlord.sh
bash ./scripts/download/download_landlord.sh
```

### 3. Train (and publish a default checkpoint)

This will create `runs/<run_name>/` and also publish a symlink to `models/<arch>/latest`.

```bash
python3 -m scriba.train --arch trocr --publish-latest
python3 -m scriba.train --arch donut --max-target-length 256 --publish-latest
```

### 4. Download pretrained into ./models (optional)

This will save a local copy under `models/<arch>/<name>/{model,processor}`.

```bash
python3 -m apps.cli.download --arch trocr --pretrained microsoft/trocr-base-handwritten --set-latest
python3 -m apps.cli.download --arch donut --pretrained naver-clova-ix/donut-base --set-latest
```

### 5. Inference (CLI)

If `models/<arch>/latest` exists, it will be used automatically.

```bash
python3 -m apps.cli.infer --arch trocr --image path/to/image.jpg
python3 -m apps.cli.infer --arch donut --image path/to/image.jpg
```

### 6. Web UI (Next.js)

```bash
cd apps/web
npm run dev -- --port 3000
```

### 7. API (FastAPI)

```bash
python3 -m uvicorn apps.api.app.main:app --reload --port 8000
```

### 8. Evaluate

```bash
python3 -m scriba.eval --arch trocr --split test
python3 -m scriba.eval --arch donut --split test
```

If you trained a run under `runs/<run_name>/model`:

```bash
python3 -m scriba.eval --arch trocr --model-dir runs/<run_name>/model --split test
```

## References
