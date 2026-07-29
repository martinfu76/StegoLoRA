#!/usr/bin/env bash
set -euo pipefail

# Native Ubuntu + Conda + NVIDIA L40 launcher.
# Activate the vLLM Conda environment before running this script.
# Required: MODEL_PATH, ADAPTER_PATH, VLLM_API_KEY
# Optional: HOST, PORT, MAX_MODEL_LEN, GPU_ID, MAX_LORA_RANK

: "${MODEL_PATH:?Set MODEL_PATH to the Llama base-model directory}"
: "${ADAPTER_PATH:?Set ADAPTER_PATH to the PEFT adapter directory}"
: "${VLLM_API_KEY:?Set VLLM_API_KEY to a non-empty secret}"

if ! command -v vllm >/dev/null 2>&1; then
  echo "vllm is not available in the active environment." >&2
  echo "Run: conda activate stegolora-vllm" >&2
  exit 1
fi

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
GPU_ID="${GPU_ID:-0}"
MAX_LORA_RANK="${MAX_LORA_RANK:-8}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"

exec vllm serve "${MODEL_PATH}" \
  --served-model-name stegolora-base \
  --enable-lora \
  --max-lora-rank "${MAX_LORA_RANK}" \
  --lora-modules "stegolora=${ADAPTER_PATH}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --api-key "${VLLM_API_KEY}" \
  --dtype float16 \
  --generation-config vllm \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization 0.90
