#!/bin/bash
#SBATCH --job-name=3spo_multinode
#SBATCH --partition=a100          # 根据你的分区名修改
#SBATCH --nodes=2                 # 使用 2 个节点
#SBATCH --nodelist=node1,node2    # 【关键】在这里明确指定节点名，例如 node1,node2
#SBATCH --gres=gpu:2              # 每个节点 2 张 A100
#SBATCH --ntasks-per-node=1       # 每个节点启动一个任务
#SBATCH --cpus-per-task=32
#SBATCH --mem=512G
#SBATCH --output=logs/3spo_multinode_%j.out
#SBATCH --error=logs/3spo_multinode_%j.err

source ~/.bashrc
conda activate 3spo
cd /public/home/xlwang/genalyu/3SPO

# --- 强力清理函数 ---
cleanup() {
    echo "Cleaning up..."
    ray stop --force
    pkill -9 -u $USER -f vllm || true
    pkill -9 -u $USER -f singularity || true
    pkill -9 -u $USER -f qemu || true
}
trap cleanup EXIT SIGINT SIGTERM

# --- 步骤 1: 环境分发 (可选) ---
# 将虚拟机镜像分发到每个节点的本地 /tmp 目录，提升 I/O 性能
REMOTE_IMAGE="/public/home/xlwang/genalyu/3SPO/OSWorld/docker_vm_data/Ubuntu.qcow2"
LOCAL_IMAGE="/tmp/Ubuntu.qcow2"
echo "Distributing image to all nodes..."
srun --ntasks=$SLURM_JOB_NUM_NODES cp "$REMOTE_IMAGE" "$LOCAL_IMAGE"

# --- 步骤 2: 启动 Ray 集群 ---
# 获取主节点 IP
head_node=$(hostname -I | cut -d' ' -f1)
echo "Head node IP: $head_node"

# 启动 Head 节点
ray start --head --node-ip-address=$head_node --port=6379 \
    --resources="{\"docker:$head_node\": 4}" --block &
sleep 10

# 启动 Worker 节点
worker_nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST | grep -v $(hostname))
for node in $worker_nodes; do
    echo "Starting worker on $node"
    srun --nodes=1 --ntasks=1 -w $node \
        ray start --address="$head_node:6379" \
        --resources="{\"docker:$(srun --nodes=1 --ntasks=1 -w $node hostname -I | cut -d' ' -f1)\": 4}" --block &
done
sleep 20

# --- 步骤 3: 启动 vLLM 模型服务 (在主节点) ---
bash start_server.sh > logs/vllm_debug_${SLURM_JOB_ID}.log 2>&1 &
echo "Waiting for vLLM..."
until curl -s http://localhost:9000/v1/models > /dev/null; do sleep 5; done
echo "vLLM is READY!"

# --- 步骤 4: 运行 3SPO 训练 ---
# 修改 n_gpus_per_node=2 以匹配你的硬件 (2x A100)
python3 -m verl.trainer.main \
    config=examples/config.yaml \
    algorithm.adv_estimator=3spo \
    trainer.nnodes=$SLURM_JOB_NUM_NODES \
    trainer.n_gpus_per_node=2 \
    env.num_envs=8 \
    trainer.experiment_name=osworld_3spo_multinode \
    trainer.save_checkpoint_path="./checkpoints/3spo_multinode"
