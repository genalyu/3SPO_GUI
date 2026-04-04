#!/bin/bash
#SBATCH --job-name=osworld_eval
#SBATCH --partition=H200          # 根据 sinfo 输出，分区名是 H200
#SBATCH --nodes=1
#SBATCH --gres=gpu:1              # 使用 1 张 GPU 跑 vLLM
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --output=/public/home/genalyu/project/3SPO/logs/eval_%j.out
#SBATCH --error=/public/home/genalyu/project/3SPO/logs/eval_%j.err

# --- 关键：环境初始化 (Conda 激活) ---
# 显式加载 Conda 的基础配置
CONDA_BASE=$(conda info --base 2>/dev/null || echo "/public/home/genalyu/miniconda3")
source ${CONDA_BASE}/etc/profile.d/conda.sh

# 激活环境
conda activate 3spo
echo "1. 存在性: "; [ -e /dev/kvm ] && echo "OK" || echo "FAIL"; \
echo "2. 权限: "; [ -w /dev/kvm ] && echo "OK (可写)" || echo "FAIL (无权)"; \
echo "3. 硬件支持: "; grep -Eoc '(vmx|svm)' /proc/cpuinfo
# 确保当前目录在项目根目录
PROJECT_ROOT="/public/home/genalyu/project/3SPO"
cd $PROJECT_ROOT

# 创建日志目录
mkdir -p logs

# 定义清理函数
cleanup() {
    echo "Cleaning up..."
    [ -n "$SERVER_PID" ] && kill $SERVER_PID
}
trap cleanup EXIT

# --- 步骤 1: 启动模型推理服务 ---
echo "Starting model server on GPU..."
# 启动 vLLM 推理服务，使用 start_server.sh
bash start_server.sh 1 > logs/vllm_debug_${SLURM_JOB_ID}.log 2>&1 &
SERVER_PID=$!

echo "Waiting for vLLM to be ready..."
for i in {1..50}; do
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

# 动态指定路径 (优先使用环境变量，否则使用默认值)
LOCAL_IMAGE=${LOCAL_IMAGE:-"/public/home/genalyu/project/3SPO/OSWorld/docker_vm_data/Ubuntu.qcow2"}
SIF_IMAGE=${SIF_IMAGE:-"/public/home/genalyu/osworld_sandbox"}

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
# cleanup 函数会通过 trap EXIT 自动调用

