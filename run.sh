#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODELS_DIR="${MODELS_DIR:-$SCRIPT_DIR/models}"
LLAMA_DIR="${LLAMA_DIR:-$SCRIPT_DIR/llama.cpp}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8080}"
CTX_SIZE="${CTX_SIZE:-8192}"
GPU_LAYERS="${GPU_LAYERS:-99}"
MODEL="${MODEL:-}"

if [ -z "$MODEL" ]; then
  candidates=("$MODELS_DIR"/*.gguf)
  if [ "${#candidates[@]}" -eq 0 ]; then
    echo "No .gguf files found in $MODELS_DIR" >&2
    exit 1
  fi
  echo "Available models:"
  for i in "${!candidates[@]}"; do
    echo "  [$((i + 1))] $(basename "${candidates[$i]}")"
  done
  read -r -p "Pick a model number: " choice
  if ! [[ "$choice" =~ ^[0-9]+$ ]] || [ "$choice" -lt 1 ] || [ "$choice" -gt "${#candidates[@]}" ]; then
    echo "Invalid choice." >&2
    exit 1
  fi
  MODEL="${candidates[$((choice - 1))]}"
fi

echo "Starting server with: $(basename "$MODEL")"
echo "  Host: $HOST  Port: $PORT  Context: $CTX_SIZE  GPU layers: $GPU_LAYERS"

exec "$LLAMA_DIR/build/bin/llama-server" \
  -m "$MODEL" \
  --host "$HOST" \
  --port "$PORT" \
  -c "$CTX_SIZE" \
  -ngl "$GPU_LAYERS" \
  "$@"
