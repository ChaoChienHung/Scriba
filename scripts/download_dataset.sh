#!/bin/bash

# --------------------------------------------------------
# Script to download the "handwriting-recognition" dataset
# from Kaggle, unzip it, and remove it. 
# --------------------------------------------------------

# Download Dataset
# ----------------
# -L : follow redirects (needed for Kaggle download links)
# -o : specify output file path and name

curl -L -o ./handwriting-data.zip https://www.kaggle.com/api/v1/datasets/download/landlord/handwriting-recognition

# Unzip Dataset
# -------------
unzip -o ./handwriting-data.zip -d ../data/raw/landlord/

# Remove the Zip File
# -------------------
rm ./handwriting-data.zip


# Organize Folders and Files
# --------------------------
mkdir ../data/raw/landlord/images
mv ../data/raw/landlord/test_v2/test ../data/raw/landlord/images/test/
mv ../data/raw/landlord/train_v2/train ../data/raw/landlord/images/train/
mv ../data/raw/landlord/validation_v2/validation ../data/raw/landlord/images/validation


mkdir ../data/raw/landlord/labels
mv ../data/raw/landlord/written_name_test_v2.csv ../data/raw/landlord/labels/test.csv
mv ../data/raw/landlord/written_name_train_v2.csv ../data/raw/landlord/labels/train.csv
mv ../data/raw/landlord/written_name_validation_v2.csv ../data/raw/landlord/labels/validation.csv

# ----------------------------------------------------------------------
# Note:
# 1. This will only work if you have Kaggle API credentials set up.
#    Place your kaggle.json file in ~/.kaggle/kaggle.json
#    and set permissions: chmod 600 ~/.kaggle/kaggle.json
# 2. If you face authentication errors, consider using the
#    Kaggle CLI instead: 
#       kaggle datasets download landlord/handwriting-recognition -p ../
# ----------------------------------------------------------------------
