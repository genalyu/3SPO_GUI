#!/bin/bash
#SBATCH --job-name=osworld_eval
#SBATCH --partition=a100          # 根据你的分区名修改
#SBATCH --nodes=1                 # 1 个节点即可
#SBATCH -w gpu[22-23]      # 指定在 gpu22 或 gpu23 节点运行
#SBATCH --gres=gpu:2              # 评估通常需要多张卡来跑 vLLM
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=256G
#SBATCH --output=/public/home/xlwang/genalyu/3SPO/logs/eval_%j.out
#SBATCH --error=/public/home/xlwang/genalyu/3SPO/logs/eval_%j.err

source ~/.bashrc
# --- 关键：环境初始化 (Conda 激活) ---
# 1. 显式加载 Conda 的基础配置
source /public/home/xlwang/jyy/anaconda/etc/profile.d/conda.sh  # 路径根据你安装 anaconda/miniconda 的位置修改
# 或者使用: source $(conda info --base)/etc/profile.d/conda.sh

# 2. 激活环境
conda activate 3spo

# 3. 确保当前目录在项目根目录
cd /public/home/xlwang/genalyu/3SPO  # 替换为你的真实绝对路径

# --- 强力清理函数 ---
cleanup() {
    echo "Caught signal! Cleaning up..."
    pkill -9 -u $USER -f vllm || true
    pkill -9 -u $USER -f singularity || true
    pkill -9 -u $USER -f qemu || true
    [ -d "$LOCAL_WORK_DIR" ] && rm -rf "$LOCAL_WORK_DIR"
    rm -rf /tmp/osworld_cache 2>/dev/null || true
    echo "Cleanup complete."
}
trap cleanup EXIT SIGINT SIGTERM

echo "Checking KVM acceleration..."
if [ -e /dev/kvm ]; then
    echo "KVM device exists."
    if [ -r /dev/kvm ] && [ -w /dev/kvm ]; then
        echo "KVM permissions are CORRECT."
    else
        echo "KVM permissions are MISSING for user $USER."
        echo "Try adding '--bind /dev/kvm' to Singularity or contact admin."
    fi
else
    echo "KVM device NOT FOUND. Expect extremely slow performance."
fi
# --- 步骤 0: 环境强力清理与本地化 I/O ---
echo "Cleaning up previous zombie processes..."
pkill -9 -u $USER -f singularity || true
pkill -9 -u $USER -f qemu || true
sleep 2


LOCAL_WORK_DIR="/tmp/$USER/osworld_runtime_$SLURM_JOB_ID"
mkdir -p "$LOCAL_WORK_DIR"

echo "Using Local Work Directory: $LOCAL_WORK_DIR"
REMOTE_IMAGE="/public/home/xlwang/genalyu/3SPO/OSWorld/docker_vm_data/Ubuntu.qcow2"
LOCAL_IMAGE="$LOCAL_WORK_DIR/Ubuntu.qcow2"

echo "Copying image to local SSD for maximum I/O performance..."
cp "$REMOTE_IMAGE" "$LOCAL_IMAGE"

# 清理过往残留的临时文件
rm -rf /tmp/fake_id_* /tmp/singularity_run_* /tmp/osworld_cache 2>/dev/null || true

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
cd OSWorld

python run_multienv_uitars.py \
    --headless \
    --observation_type screenshot \
    --max_steps 15 \
    --max_trajectory_length 15 \
    --temperature 0.6 \
    --model qwen2-5-vl-3b\
    --action_space pyautogui \
    --num_envs 1 \
    --result_dir ./results/ \
    --test_all_meta_path ./evaluation_examples/test_all.json \
    --trial-id 0 \
    --path_to_vm "$LOCAL_IMAGE" \
    --server_ip http://127.0.0.1

# --- 步骤 3: 清理 ---
echo "Evaluation finished. Final cleanup..."
cleanup