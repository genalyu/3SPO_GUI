#!/bin/bash
#SBATCH --job-name=kvm_array
#SBATCH --partition=a100          # 你的分区名
#SBATCH --nodes=1                 # 每个任务只占 1 个节点
#SBATCH --ntasks=1                # 每个任务 1 个进程
#SBATCH --array=0-19%20
#SBATCH --output=logs/kvm_res_%A_%a.out
#SBATCH --error=logs/kvm_res_%A_%a.err

# 获取当前节点名称
node=$(hostname)

# 执行 KVM 检查
kvm_exists="NO"
kvm_rw="NO"
if [ -e /dev/kvm ]; then
    kvm_exists="YES"
    if [ -r /dev/kvm ] && [ -w /dev/kvm ]; then
        kvm_rw="YES"
    fi
fi

# 输出结果到各自的日志文件
printf "Node: %-15s | KVM Device: %-3s | KVM RW: %-3s\n" "$node" "$kvm_exists" "$kvm_rw"