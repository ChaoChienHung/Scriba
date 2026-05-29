# Scriba

## Introduction

This repository serves as a **sandbox** for **building**, **training**, and **evaluating handwritten text** parsing models, while **also supporting the development of real-world applications**. It contains experiments with **different model architectures**, **preprocessing pipelines**, and **decoding strategies** for **recognizing** and **structuring handwritten text**.

## Folder Structure (Karen-style)

```bash
Scriba/
│
├── data/                      # Local datasets (git ignored), .gitkeep keeps folder
│   └── raw/landlord/{train,validation,test}/...
├── runs/                      # Training outputs (git ignored), .gitkeep keeps folder
├── models/                    # Default model pointers (models/<arch>/latest)
├── scripts/                   # One-off helpers (e.g. download dataset)
├── utils/                     # Small reusable helpers (optional)
└── scriba/                    # Main Python package
    ├── train.py               # Entry: python -m scriba.train
    ├── infer.py               # Entry: python -m scriba.infer
    ├── webapp.py              # Entry: streamlit run scriba/webapp.py
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
chmod 600 ./scripts/download_dataset.sh
bash ./scripts/download_dataset.sh
```

### 3. Train (and publish a default checkpoint)

This will create `runs/<run_name>/` and also publish a symlink to `models/<arch>/latest`.

```bash
python3 -m scriba.train --arch trocr --publish-latest
python3 -m scriba.train --arch donut --max-target-length 256 --publish-latest
```

### 4. Inference (CLI)

If `models/<arch>/latest` exists, it will be used automatically.

```bash
python3 -m scriba.infer --arch trocr --image path/to/image.jpg
python3 -m scriba.infer --arch donut --image path/to/image.jpg
```

### 5. Web UI (comparison + runs dashboard)

```bash
streamlit run scriba/webapp.py
```

### 6. Evaluate

```bash
python3 -m scriba.eval --arch trocr --split test
python3 -m scriba.eval --arch donut --split test
```

If you trained a run under `runs/<run_name>/model`:

```bash
python3 -m scriba.eval --arch trocr --model-dir runs/<run_name>/model --split test
```

## References
