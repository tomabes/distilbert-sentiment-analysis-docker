#!/bin/bash

set -euo pipefail

curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I really enjoyed this assignment."}'
