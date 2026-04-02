#!/bin/bash

# 建议将模型路径设置为环境变量，或者直接在这里修改
model=${MODEL_PATH:-"/public/home/xlwang/genalyu/models/Qwen2.5-VL-3B-Instruct"}
model_name=qwen2-5-vl-3b
num_images=16

port=9000

# Function to clean up processes on exit
cleanup() {
    echo "Stopping all processes..."
    pkill -P $$  # Kill all child processes of this script
    exit 0
}

# Trap SIGINT (Ctrl+C) and SIGTERM to run cleanup function
trap cleanup SIGINT SIGTERM

# Start processes
# 设置你想使用的 GPU 数量，默认为 1
NUM_GPUS=${1:-1}

for i in $(seq 0 $((NUM_GPUS - 1))); do
    CUDA_VISIBLE_DEVICES=$i python -m vllm.entrypoints.openai.api_server \
        --served-model-name $model_name \
        --model $model \
        --limit-mm-per-prompt image=$num_images \
        --max-model-len 32768 \
        --tensor-parallel-size 1 \
        --port $((9000 + i)) &
done

# Wait to keep the script running
wait