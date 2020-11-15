#!/bin/bash

# -e: Exit immediately if a command exits with a non-zero status, -v: Print shell input lines as they are read.
set -evx

# Defining other variables
PACKAGE_PATH=./src
MODULE_NAME=src.main

# This is similar to `python -m trainer.task --job-dir local-training-output`
# but it better replicates the AI Platform environment, especially
# for distributed training (not applicable here).
gcloud ai-platform local train \
  --package-path "$PACKAGE_PATH" \
  --module-name "$MODULE_NAME" \
  -- \
  --google_application_credential="/home/nicolamassarenti/Documents/Progetti/bert-faqclass/model-bert-faqclass/auth/bert-faqclass-a96dec925432.json" \
  --bert_url="https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/2" \
  --model_checkpoint_path="/home/nicolamassarenti/Documents/Progetti/bert-faqclass/model-bert-faqclass/model/checkpoint"\
  --logs_config_path="/home/nicolamassarenti/Documents/Progetti/bert-faqclass/model-bert-faqclass/config/log_config.ini"