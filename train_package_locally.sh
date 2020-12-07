#!/bin/bash

# -e: Exit immediately if a command exits with a non-zero status, -v: Print shell input lines as they are read.
set -evx

# Defining other variables
PACKAGE_PATH=$(pwd)/package/bert_faqclass
MODULE_NAME=bert_faqclass

gcloud config set ml_engine/local_python $(pwd)/venv/bin/python3

# This is similar to `python -m trainer.task --job-dir local-training-output`
# but it better replicates the AI Platform environment, especially
# for distributed training (not applicable here).
gcloud ai-platform local train \
  --package-path "$PACKAGE_PATH" \
  --module-name "$MODULE_NAME"