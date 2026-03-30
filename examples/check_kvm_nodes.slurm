#!/bin/bash
#SBATCH --job-name=check_kvm
#SBATCH --partition=a100          # 根据你的分区名修改
#SBATCH --nodes=4                 # 想要检查的节点数量
#SBATCH --ntasks-per-node=1       # 每个节点运行一个任务
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=logs/kvm_check_%j.out
#SBATCH --error=logs/kvm_check_%j.err

echo "Starting KVM support check across nodes (using SSH)..."
echo "------------------------------------------"

# 获取分配到的所有节点列表
nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST)

for node in $nodes; do
    # 使用 ssh 在每个节点上执行检查（通常 Slurm 节点间互信）
    result=$(ssh -o StrictHostKeyChecking=no "$node" '
        kvm_exists="NO"
        kvm_rw="NO"
        if [ -e /dev/kvm ]; then
            kvm_exists="YES"
            if [ -r /dev/kvm ] && [ -w /dev/kvm ]; then
                kvm_rw="YES"
            fi
        fi
        printf "Node: %-15s | KVM Device: %-3s | KVM RW: %-3s" "$(hostname)" "$kvm_exists" "$kvm_rw"
    ')
    echo "$result"
done | sort

echo "------------------------------------------"
echo "Check finished. Detailed logs in logs/kvm_check_${SLURM_JOB_ID}.out"
