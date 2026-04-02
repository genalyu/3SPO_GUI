#!/bin/bash

# --- 关键：环境初始化 (Conda 激活) ---
# 1. 显式加载 Conda 的基础配置
CONDA_BASE=$(conda info --base 2>/dev/null || echo "$HOME/anaconda3")
if [ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]; then
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
else
    echo "Warning: conda.sh not found at ${CONDA_BASE}/etc/profile.d/conda.sh"
fi

# 2. 激活环境
conda activate 3spo || echo "Warning: failed to activate conda environment 3spo"

# 3. 确保当前目录在项目根目录
# 获取脚本所在目录的绝对路径，然后切换到父目录（项目根目录）
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR/.." || exit

# 定义清理函数
cleanup() {
    echo "Stopping model server (PID: $SERVER_PID)..."
    if [ -n "$SERVER_PID" ]; then
        # 杀掉 start_server.sh 及其子进程
        pkill -P $SERVER_PID
        kill $SERVER_PID
    fi
    exit 0
}

# 捕获信号以运行清理
trap cleanup SIGINT SIGTERM

# --- 步骤 1: 启动模型推理服务 ---
echo "Starting model server on 8 GPUs..."
mkdir -p logs
# 使用当前时间戳作为日志名
LOG_FILE="logs/vllm_debug_$(date +%Y%m%d_%H%M%S).log"
# 传入 8 以便启动 8 个服务，对应 8 个 GPU
bash start_server.sh 8 > "$LOG_FILE" 2>&1 &
SERVER_PID=$!

echo "Waiting for vLLM to be ready (logging to $LOG_FILE)..."
for i in {1..50}; do
    # 假设你的 vLLM 跑在 9000 端口
    if curl -s http://127.0.0.1:9000/v1/models > /dev/null; then
        echo "vLLM is READY!"
        break
    fi
    if [ $i -eq 50 ]; then
        echo "Error: vLLM server failed to start in time."
        cleanup
        exit 1
    fi
    echo "Wait $i: Server not ready yet..."
    sleep 10
done

# --- 步骤 2: 运行 OSWorld 评估脚本 ---
echo "Starting OSWorld evaluation..."
cd OSWorld || exit

# 注意：LOCAL_IMAGE 变量需要预先设置，或者在这里指定默认值
if [ -z "$LOCAL_IMAGE" ]; then
    echo "Warning: LOCAL_IMAGE is not set. Using default 'osworld:latest'"
    LOCAL_IMAGE="osworld:latest"
fi

python run_multienv_uitars.py \
    --headless \
    --observation_type screenshot \
    --max_steps 15 \
    --max_trajectory_length 15 \
    --temperature 0.6 \
    --model qwen2-5-vl-3b \
    --action_space pyautogui \
    --num_envs 8 \
    --result_dir ./results/ \
    --test_all_meta_path ./evaluation_examples/test_all.json \
    --trial-id 0 \
    --path_to_vm "$LOCAL_IMAGE" \
    --provider docker \
    --server_ip http://127.0.0.1

# --- 步骤 3: 清理 ---
echo "Evaluation finished. Final cleanup..."
cleanup
