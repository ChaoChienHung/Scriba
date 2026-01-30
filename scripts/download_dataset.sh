#!/bin/bash

# --------------------------------------------------------
# Script to download the "handwriting-recognition" dataset
# from Kaggle, unzip it, and remove it.
# 
# -L : follow redirects (needed for Kaggle download links)
# -o : specify output file path and name
# --------------------------------------------------------

curl -L -o ../data/handwriting-data.zip https://www.kaggle.com/api/v1/datasets/download/landlord/handwriting-recognition && \
unzip -o ../data/handwriting-data.zip -d ../data/landlord/ && \
rm ../data/handwriting-data.zip

# ----------------------------------------------------------------------
# Note:
# 1. This will only work if you have Kaggle API credentials set up.
#    Place your kaggle.json file in ~/.kaggle/kaggle.json
#    and set permissions: chmod 600 ~/.kaggle/kaggle.json
# 2. If you face authentication errors, consider using the
#    Kaggle CLI instead: 
#       kaggle datasets download landlord/handwriting-recognition -p ../
# ----------------------------------------------------------------------
