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
├── scripts/                   # One-off helpers (e.g. download dataset)
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
## Instruction (Train)
### 1. Install dependencies
### 1. Download the training data
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Download the dataset (optional)

You can download the Kaggle dataset using:
You can download the provided training dataset using the script:

chmod 600 ./scripts/download_dataset.sh
bash ./scripts/download_dataset.sh
chmod 600 ./scripts/download_dataset.sh
**Optional**: If you prefer, you can also prepare your own dataset and place it inside the data/ folder.
### 3. Train

TrOCR:

```bash
python -m scriba.train --arch trocr
```

Donut:

```bash
python -m scriba.train --arch donut --max-target-length 256
```

### 4. Evaluate

```bash
python -m scriba.eval --arch trocr --split test
python -m scriba.eval --arch donut --split test
```

If you trained a run under `runs/<run_name>/model`:

```bash
python -m scriba.eval --arch trocr --model-dir runs/<run_name>/model --split test
```
## Instruction (Inference)

## References
