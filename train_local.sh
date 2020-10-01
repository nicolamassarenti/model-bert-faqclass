#!/bin/bash

# -e: Exit immediately if a command exits with a non-zero status, -v: Print shell input lines as they are read.
set -ev

# Reading and loading env variables defined in .env file
DOTENV_FILENAME="$(pwd)/.env"
source "$DOTENV_FILENAME"
set +a

# Activating the virtualenv
source "./venv/bin/activate"

# Defining other variables
PACKAGE_PATH=trainer
MODULE_NAME="$PACKAGE_PATH".task

# This is similar to `python -m trainer.task --job-dir local-training-output`
# but it better replicates the AI Platform environment, especially
# for distributed training (not applicable here).
gcloud ai-platform local train \
  --package-path "$PACKAGE_PATH" \
  --module-name $MODULE_NAME \
  --job-dir local-training-output