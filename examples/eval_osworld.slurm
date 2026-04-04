#!/bin/bash
#SBATCH --job-name=osworld_eval
#SBATCH --partition=H100          # 根据你的分区名修改
#SBATCH --nodes=1                 # 4 个节点即可
#SBATCH --gres=gpu:1              # 评估通常需要多张卡来跑 vLLM
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --output=/public/home/genalyu/project/3SPO/logs/eval_%j.out
#SBATCH --error=/public/home/genalyu/project/3SPO/logs/eval_%j.err

source ~/.bashrc
# --- 关键：环境初始化 (Conda 激活) ---
# 1. 显式加载 Conda 的基础配置
source $(conda info --base)/etc/profile.d/conda.sh

# 2. 激活环境
conda activate 3spo

# 3. 确保当前目录在项目根目录
cd /public/home/genalyu/project/3SPO  # 替换为你的真实绝对路径

# --- 步骤 1: 启动模型推理服务 ---
echo "Starting model server..."
# 注意：在 Slurm 内部不需要 nohup，因为整个脚本就是在后台运行的
# 但为了让它在后台跑，我们用 & 并记录 PID
bash start_server.sh > logs/vllm_debug_${SLURM_JOB_ID}.log 2>&1 &
SERVER_PID=$!
echo "Waiting for vLLM to be ready..."
for i in {1..50}; do
    # 假设你的 vLLM 跑在 9000 端口
    if curl -s http://127.0.0.1:9000/v1/models > /dev/null; then
        echo "vLLM is READY!"
        break
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
    SIF_IMAGE="/public/home/genalyu/osworld.sif"
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
