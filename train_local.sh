#!/bin/bash

export PYTHONPATH=/home/nicolamassarenti/Documents/Progetti/bert-faqclass/model-bert-faqclass
# -e: Exit immediately if a command exits with a non-zero status, -v: Print shell input lines as they are read.
set -ev

# Reading and loading env variables defined in .env file
DOTENV_FILENAME=".env"
source $DOTENV_FILENAME
set +a

# Activating the virtualenv
source "./venv/bin/activate"

# Defining other variables
DATE=$(date '+%Y%m%d_%H%M%S')
PACKAGE_PATH=./src
MODULE_NAME=main

# This is similar to `python -m trainer.task --job-dir local-training-output`
# but it better replicates the AI Platform environment, especially
# for distributed training (not applicable here).
gcloud ai-platform local train \
  --package-path $PACKAGE_PATH \
  --module-name $MODULE_NAME \
  --job-dir local-training-output