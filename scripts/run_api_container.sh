#!/bin/bash

set -euo pipefail

export containerName=distilbert_api_$USER

docker run \
  -d \
  --rm \
  -p 8000:8000 \
  --shm-size=1g \
  --memory=4g \
  --name "$containerName" \
  distilbert-api:latest
