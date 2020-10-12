#!/bin/bash

set -avx
TIMESTAMP=$(date +"%Y%m%dT%H%M%S")

# Constants for docker image and Google Container Registry
HOSTNAME="eu.gcr.io"
IMAGE_NAME="trainer-bert-faqclass"
PROJECT_ID="bert-faqclass"
TAG="latest"

IMAGE_URI=$HOSTNAME/$PROJECT_ID/$IMAGE_NAME:$TAG

# Configure docker to use the gcloud command-line tool as a credential helper
gcloud auth configure-docker

# Building the docker image
docker build -t $IMAGE_NAME .

# Tag the image with a registry name
docker tag $IMAGE_NAME $HOSTNAME/$PROJECT_ID/$IMAGE_NAME:$TAG

# Pushing the image on Google Container Registry
docker push $IMAGE_URI

# Constants for training job
JOB_NAME="$IMAGE_NAME"_"$TIMESTAMP"
REGION="europe-west1"
STAGING_BUCKET="gs://$IMAGE_NAME/"

# Creating the bucket in the region if it doesn't already exists
gsutil ls $STAGING_BUCKET || gsutil mb -l $REGION $STAGING_BUCKET


# Starting the training job
#gcloud ai-platform jobs submit training "$JOB_NAME" \
#                                        --staging-bucket "$STAGING_BUCKET" \
#                                        --master-image-uri=$IMAGE_URI