#!/bin/bash

# -e: Exit immediately if a command exits with a non-zero status, -v: Print shell input lines as they are read.
set -ev

# Name of the model
MODEL_NAME="model_bert_faqclass"

# Name of the bucket where the code will be uploaded
BUCKET_NAME=$MODEL_NAME
BUCKET_PATH="gs://$BUCKET_NAME/"

# Region where the model will be deployed
REGION="europe-west1"

# Creating the bucket in the region if it doesn't already exists
gsutil ls -b $BUCKET_PATH || gsutil mb -l $REGION $BUCKET_PATH

# Setting job name
DATE=$(date '+%Y%m%d_%H%M%S')
JOB_NAME="$BUCKET_NAME"_"$DATE"_training

echo "$JOB_NAME"

# Setting package configuration
PACKAGE_PATH=$(pwd)/src
MODULE_NAME=main

# Setting job dir
JOB_DIR="$BUCKET_PATH$JOB_NAME"

# Running the training on cloud
gcloud ai-platform jobs submit training "$JOB_NAME" \
  --package-path "$PACKAGE_PATH" \
  --module-name "$MODULE_NAME" \
  --region "$REGION" \
  --python-version 3.7 \
  --runtime-version 1.15 \
  --job-dir "$JOB_DIR" \
  --stream-logs
