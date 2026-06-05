#!/bin/bash
set -e
set -o pipefail

MODEL_NAME="TableLlama"
MODEL_PATH="osunlp/TableLlama"
PORT=8005
BATCH_SIZE="${2:-8}"
MAX_SAMPLES="${1:-$MAX_SAMPLES}"

export CUDA_VISIBLE_DEVICES=0
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
CHAT_TEMPLATE="{% for message in messages %}{% if message['role'] == 'user' %}[INST] {{ message['content'] }} [/INST]{% elif message['role'] == 'assistant' %}{{ message['content'] }}{% endif %}{% endfor %}"

vllm serve "$MODEL_PATH" \
  --served-model-name "$MODEL_NAME" \
  --trust-remote-code \
  --chat-template "$CHAT_TEMPLATE" \
  --port "$PORT" \
  > "$LOG_DIR/vllm_${MODEL_NAME}.log" 2>&1 &

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
  ${MAX_SAMPLES:+--limit "$MAX_SAMPLES"} \
  --suc_only \
  > "$LOG_DIR/${MODEL_NAME}_pipeline.log" 2>&1

echo "Done"
