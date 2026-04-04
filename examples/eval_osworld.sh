#!/bin/bash
source /public/home/genalyu/miniconda3/etc/profile.d/conda.sh
conda activate 3spo
# --- 步骤 1: 启动模型推理服务 ---
echo "Starting model server on 8 GPUs..."
# 使用当前时间戳作为日志名
LOG_FILE="logs/vllm_debug_$(date +%Y%m%d_%H%M%S).log"
# 传入 8 以便启动 8 个服务，对应 8 个 GPU
bash start_server.sh 1 > "$LOG_FILE" 2>&1 &
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

# 注意：LOCAL_IMAGE 变量需要指向本地 .qcow2 镜像文件的绝对路径
if [ -z "$LOCAL_IMAGE" ]; then
    LOCAL_IMAGE="/public/home/genalyu/project/3SPO/OSWorld/docker_vm_data/Ubuntu.qcow2"
fi

# 注意：SIF_IMAGE 变量需要指向本地 .sif 镜像文件或 Sandbox 目录的绝对路径
if [ -z "$SIF_IMAGE" ]; then
    # 如果 .sif 文件报错 "unsquashfs not found"，建议先将其解压为 Sandbox 目录并上传
    # 解压命令: apptainer build --sandbox osworld-sandbox osworld.sif
    SIF_IMAGE="/public/home/genalyu/osworld.sandbox"
fi

# 检查镜像文件或目录是否存在
if [ ! -f "$LOCAL_IMAGE" ]; then
    echo "Error: VM image file not found at $LOCAL_IMAGE"
    exit 1
fi

if [ ! -e "$SIF_IMAGE" ]; then
    echo "Error: SIF image or Sandbox directory not found at $SIF_IMAGE"
    exit 1
fi

python run_multienv_uitars.py \
    --headless \
    --observation_type screenshot \
    --max_steps 15 \
    --max_trajectory_length 15 \
    --temperature 0.6 \
    --model qwen2-5-vl-3b \
    --action_space pyautogui \
    --num_envs 1 \
    --result_dir ./results/ \
    --test_all_meta_path ./evaluation_examples/test_all.json \
    --trial-id 0 \
    --path_to_vm "$LOCAL_IMAGE" \
    --sif_path "$SIF_IMAGE" \
    --provider apptainer \
    --server_ip http://127.0.0.1

# --- 步骤 3: 清理 ---
echo "Evaluation finished. Final cleanup..."
cleanup
