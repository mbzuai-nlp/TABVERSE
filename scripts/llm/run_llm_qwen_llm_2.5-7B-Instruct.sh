#!/bin/bash
set -e
set -o pipefail

MODEL_NAME="Qwen2.5-7B-Instruct"
MODEL_PATH="Qwen/Qwen2.5-7B-Instruct"
PORT=8036
BATCH_SIZE="${2:-8}"
MAX_SAMPLES="${1:-$MAX_SAMPLES}"

export PYTORCH_ALLOC_CONF=expandable_segments:True
export PATH=$HOME/.local/bin:$PATH

LOG_DIR="logs"
mkdir -p "$LOG_DIR"

# Load HF token
if [ -f ".env" ]; then
    export $(grep -v '^#' .env | xargs)
else
    echo ".env not found"
    exit 1
fi

[ -z "$HF_TOKEN" ] && { echo "HF_TOKEN missing"; exit 1; }

echo "=== LLM Pipeline ==="
echo "Model      : $MODEL_NAME"
echo "Batch size : $BATCH_SIZE"
echo "Max samples: ${MAX_SAMPLES:-All}"
echo "==================="

# Start vLLM
vllm serve "$MODEL_PATH" \
  --served-model-name "$MODEL_NAME" \
  --trust-remote-code \
  --port "$PORT" \
  > "$LOG_DIR/vllm_${MODEL_NAME}.log" 2>&1 &

VLLM_PID=$!
trap "kill $VLLM_PID" EXIT

# Wait for server
for i in {1..30}; do
  if curl -s http://localhost:$PORT/v1/models >/dev/null; then
    echo "vLLM ready"
    break
  fi
  sleep 60
done

# Run unified pipeline
python src/local_script/llm_local.py \
  --model_name "$MODEL_NAME" \
  --model_path "$MODEL_PATH" \
  --port "$PORT" \
  --hf_token "$HF_TOKEN" \
  --batch_size "$BATCH_SIZE" \
  --suc_only \
  ${MAX_SAMPLES:+--limit "$MAX_SAMPLES"} \
  > "$LOG_DIR/${MODEL_NAME}_pipeline.log" 2>&1

echo "Done"