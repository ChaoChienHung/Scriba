# Scriba

## Introduction

This repository serves as a **sandbox** for **building**, **training**, and **evaluating handwritten text** parsing models, while **also supporting the development of real-world applications**. It contains experiments with **different model architectures**, **preprocessing pipelines**, and **decoding strategies** for **recognizing** and **structuring handwritten text**.

## Folder Structure

```bash
repo/
│
├── cache/                                  # Tokenizers, Weights, etc.
│
├── data/ 
│   ├── processed/
│   │   ├── v1/
│   │   │   ├── train/
│   │   │   │   ├── images/
│   │   │   │   └── labels.csv
│   │   │   │
│   │   │   ├── val/
│   │   │   │   ├── images/
│   │   │   │   └── labels.csv
│   │   │   │
│   │   │   └── train/
│   │   │       ├── images/
│   │   │       └── labels.csv
│   │   ├── v2/
│   │   │   ├── train/
│   │   │   │   ├── images/
│   │   │   │   └── labels.csv
│   │   │   │
│   │   │   ├── val/
│   │   │   │   ├── images/
│   │   │   │   └── labels.csv
│   │   │   │
│   │   │   └── train/
│   │   │       ├── images/
│   │   │       └── labels.csv
│   │   │   
│   │   └── README.md
│   │
│   └── raw/
│       └── landlord/
│           ├── test_v2/test                    # Testing Data
│           ├── train_v2/train                  # Training Data
│           ├── validation_v2/validation        # Validation Data
│           ├── written_name_test_v2.csv
│           ├── written_name_train_v2.csv
│           └── written_name_validation_v2.csv
│
├── preprocessing/                          # Data Transformations
│   ├── image.py
│   ├── text.py
│   ├── tokenizer.py
│   └── __init__.py
│
├── models/                                 # Model Architectures
│   ├── donut/
│   │   ├── model.py
│   │   └── config.py
│   ├── trocr/
│   │   ├── model.py
│   │   └── config.py
│   └── __init__.py
│
├── training/                               # Training and Evaluation Logic
│   ├── train_donut.py
│   ├── train_trocr.py
│   ├── evaluate.py
│   └── common.py
│
├── experiments/                            # Experiment Result
│   ├── donut_v1/
│   │   ├── checkpoints/
│   │   ├── metrics.json
│   │   └── config.yaml
│   └── trocr_v1/
│
├── models/
│   ├── Donut/
│   └── TrOCR
│
├── scripts/
│   └── download_dataset.sh
│
├── utils/
│
├── .gitignore
├── DevNotes.md
├── README.md
└── requirements.txt

```

## Instruction (Train)

### 1. Download the training data

You can download the provided training dataset using the script:

```bash
chmod 600 ./scripts/download_dataset.sh
```
**Optional**: If you prefer, you can also prepare your own dataset and place it inside the data/ folder.

## Instruction (Inference)

## References
1. [Handwriting Recognition Data](https://www.kaggle.com/datasets/landlord/handwriting-recognition/data)