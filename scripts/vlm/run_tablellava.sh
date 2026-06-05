#!/bin/bash
set -e
set -o pipefail

# ================= Model Configuration ================= #
MODEL_NAME="table-llava-v1.5-7b-hf"
MODEL_PATH="SpursgoZmy/table-llava-v1.5-7b-hf"
PORT=8003

# ================= Runtime Configuration ================= #
MAX_SAMPLES="${1:-$MAX_SAMPLES}"

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PATH=$HOME/.local/bin:$PATH

LOG_DIR="logs"
mkdir -p "$LOG_DIR"

# ================= Load HF Token ================= #
if [ -f ".env" ]; then
    export $(grep -v '^#' .env | xargs)
else
    echo "Error: .env file not found."
    exit 1
fi

if [ -z "$HF_TOKEN" ]; then
    echo "Error: HF_TOKEN not found in .env."
    exit 1
fi

echo "=== Unified VLM Pipeline ==="
echo "Model Name     : $MODEL_NAME"
echo "Model Path     : $MODEL_PATH"
echo "Port           : $PORT"
echo "Max Samples    : ${MAX_SAMPLES:-All}"
echo "GPU            : $CUDA_VISIBLE_DEVICES"
echo "============================"

# ================= Start vLLM ================= #
vllm serve "$MODEL_PATH" \
  --served-model-name "$MODEL_NAME" \
  --tensor-parallel-size 1 \
  --dtype half \
  --disable-log-requests \
  --host 0.0.0.0 \
  --trust-remote-code \
  --port "$PORT" \
  > "$LOG_DIR/vllm_server_${MODEL_NAME}.log" 2>&1 &

VLLM_PID=$!
trap "kill $VLLM_PID" EXIT

# ================= Health Check ================= #
echo "Waiting for vLLM server..."
for i in {1..30}; do
  if curl -s http://localhost:$PORT/v1/models >/dev/null; then
    echo "vLLM server is ready."
    break
  fi
  echo "Waiting... ($i)"
  sleep 60
done

# ================= Run Unified VLM Pipeline ================= #
PY_ARGS=(
  --model_name "$MODEL_NAME"
  --port "$PORT"
  --hf_token "$HF_TOKEN"
)

[ -n "$MAX_SAMPLES" ] && PY_ARGS+=(--limit "$MAX_SAMPLES")
PY_ARGS+=(--suc_only)
echo "Running unified VLM pipeline (Task → SUC)..."

python src/local_script/vlm_local.py "${PY_ARGS[@]}" \
  > "$LOG_DIR/${MODEL_NAME}_vlm_pipeline.log" 2>&1

echo "VLM pipeline completed."
echo "Results saved to: results/vlmpipeline/$MODEL_NAME/"


# ================= Run Unified VLM_text Pipeline ================= #
PY_ARGS=(
  --model_name "$MODEL_NAME"
  --port "$PORT"
  --hf_token "$HF_TOKEN"
)

[ -n "$MAX_SAMPLES" ] && PY_ARGS+=(--limit "$MAX_SAMPLES")
PY_ARGS+=(--suc_only)
echo "Running unified VLM_text pipeline (Task → SUC)..."

python src/local_script/vlm_text_local.py "${PY_ARGS[@]}" \
  > "$LOG_DIR/${MODEL_NAME}_vlm_text_pipeline.log" 2>&1

echo "VLM_text pipeline completed."
echo "Results saved to: results/vlm_text_pipeline/$MODEL_NAME/"

# # ================= Run Unified Generation Pipeline ================= #
# PY_ARGS=(
#   --model_name "$MODEL_NAME"
#   --port "$PORT"
#   --hf_token "$HF_TOKEN"
# )

# [ -n "$MAX_SAMPLES" ] && PY_ARGS+=(--limit "$MAX_SAMPLES")

# echo "Running unified Generation pipeline (Task → SUC)..."

# python src/gen_local.py "${PY_ARGS[@]}" \
#   > "$LOG_DIR/${MODEL_NAME}_gen_pipeline.log" 2>&1

# echo "Generation pipeline completed."
# echo "Results saved to: results/gen_pipeline/$MODEL_NAME/"
