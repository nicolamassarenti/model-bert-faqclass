#!/bin/bash

# -e: Exit immediately if a command exits with a non-zero status, -v: Print shell input lines as they are read.
set -evx

TIMESTAMP=$(date +"%Y%m%dT%H%M%S")

# VARIABLES
MODEL_NAME="model_bert_faqclass"
STAGING_BUCKET="gs://$MODEL_NAME/"
REGION="europe-west1"

JOB_NAME="$MODEL_NAME"_"$TIMESTAMP"_training
JOB_DIR="$STAGING_BUCKET$JOB_NAME"

PROJECT_ID="bert-faqclass"

SOURCE_IMAGE="bert_faqclass_model"
REMOTE_IMAGE=${SOURCE_IMAGE}

HOSTNAME=eu.gcr.io

IMAGE_URI=${HOSTNAME}/${PROJECT_ID}/${REMOTE_IMAGE}

# Running model training on AI Platform
gcloud ai-platform jobs submit training "$JOB_NAME" \
  --staging-bucket=$STAGING_BUCKET \
  --master-image-uri $IMAGE_URI \
  --region "$REGION" \
  --job-dir "$JOB_DIR"