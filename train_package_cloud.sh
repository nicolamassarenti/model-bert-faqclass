#!/bin/bash

# -e: Exit immediately if a command exits with a non-zero status, -v: Print shell input lines as they are read.
set -evx
TIMESTAMP=$(date +"%Y%m%dT%H%M%S")

# VARIABLES
MODEL_NAME="model_bert_faqclass"
PACKAGE_PATH=$(pwd)/package/bert_faqclass
MODULE_NAME=bert_faqclass

STAGING_BUCKET="gs://$MODEL_NAME/"
REGION="europe-west1"

JOB_NAME="$MODEL_NAME"_"$TIMESTAMP"_training
JOB_DIR="$STAGING_BUCKET$JOB_NAME"

# Creating the bucket in the region if it doesn't already exists
gsutil ls -b $STAGING_BUCKET || gsutil mb -l $REGION $STAGING_BUCKET

# Running the training on cloud
gcloud ai-platform jobs submit training "$JOB_NAME" \
  --staging-bucket=$STAGING_BUCKET \
  --package-path "$PACKAGE_PATH" \
  --module-name "$MODULE_NAME" \
  --region "$REGION" \
  --python-version 3.7 \
  --runtime-version 2.2 \
  --job-dir "$JOB_DIR"