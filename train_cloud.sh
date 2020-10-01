#!/bin/bash

# -e: Exit immediately if a command exits with a non-zero status, -v: Print shell input lines as they are read.
set -ev

JOB_NAME='JOB_NAME'
JOB_DIR='JOB_OUTPUT_PATH'

# VARIABLES
MODEL_NAME="model_bert_faqclass"
PACKAGE_PATH="$(pwd)/trainer"
MODULE_NAME="$PACKAGE_PATH"/task
STAGING_BUCKET=$MODEL_NAME
BUCKET_PATH="gs://$STAGING_BUCKET/"
REGION="europe-west1"
JOB_DIR="$BUCKET_PATH$JOB_NAME"

# Creating the bucket in the region if it doesn't already exists
gsutil ls -b $BUCKET_PATH || gsutil mb -l $REGION $BUCKET_PATH

# Setting job name
DATE=$(date '+%Y%m%d_%H%M%S')
JOB_NAME="$STAGING_BUCKET"_"$DATE"_training

# Running the training on cloud
gcloud ai-platform jobs submit training "$JOB_NAME" \
  --staging-bucket=$BUCKET_PATH \
  --package-path "$PACKAGE_PATH" \
  --module-name "$MODULE_NAME" \
  --region "$REGION" \
  --python-version 3.7 \
  --runtime-version 1.15 \
  --job-dir "$JOB_DIR" \
  --stream-logs
