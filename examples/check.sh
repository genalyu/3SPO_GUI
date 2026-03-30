#!/bin/bash
#SBATCH --job-name=SuperDiag
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --nodelist=gpu22          # 指定在 gpu22 节点运行
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=super_diag_%j.log

echo "==================== [STEP 1: Host Environment] ===================="
echo "Node: $(hostname)"
echo "User: $(whoami)"
echo "Kernel: $(uname -r)"
echo "Uptime: $(uptime)"

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
echo -e "\n--- Checking KVM ---"
ls -l /dev/kvm
[ -w /dev/kvm ] && echo "KVM Access: YES" || echo "KVM Access: NO (CRITICAL)"

echo -e "\n--- Checking Disk Space ---"
df -h /tmp
df -h /public/home/xlwang

echo "==================== [STEP 2: IO & Image Integrity] ===================="
REMOTE_IMG="/public/home/xlwang/genalyu/3SPO/OSWorld/docker_vm_data/Ubuntu.qcow2"
LOCAL_DIR="/tmp/xlwang/diag_$SLURM_JOB_ID"
LOCAL_IMG="$LOCAL_DIR/Ubuntu.qcow2"

mkdir -p "$LOCAL_DIR"
echo "Copying image to local SSD..."
time cp "$REMOTE_IMG" "$LOCAL_IMG"

echo -e "\n--- Checking Image File ---"
du -sh "$LOCAL_IMG"
file "$LOCAL_IMG"
# singularity exec /public/home/xlwang/genalyu/3SPO/osworld-sandbox qemu-img info "$LOCAL_IMG" || echo "ERROR: Image header corrupted"

echo "==================== [STEP 3: Low-Level Container Test] ===================="
echo "Attempting to run a simple command inside the sandbox..."
singularity exec --writable-tmpfs "$SANDBOX_PATH" uptime || echo "Singularity basic exec failed"

echo "==================== [STEP 4: Manual VM Boot Test] ===================="
echo "Starting raw QEMU boot test (no GUI, 300s timeout)..."

singularity exec --nv --writable-tmpfs \
    --bind /dev/kvm:/dev/kvm \
    --bind "$LOCAL_IMG":/System.qcow2 \
    /public/home/xlwang/genalyu/3SPO/osworld-sandbox \
    qemu-system-x86_64 \
    -m 4096 \
    -enable-kvm \
    -drive file=/System.qcow2,if=virtio \
    -display none \
    -device virtio-net-pci,netdev=net0 \
    -netdev user,id=net0,hostfwd=tcp::5010-:22 \
    -vnc :1 &

QEMU_PID=$!
sleep 60 
if ps -p $QEMU_PID > /dev/null; then
    echo "SUCCESS: QEMU process is still running after 60s."
    netstat -tulpn | grep 5010 && echo "Port 5010 is LISTENING." || echo "Port 5010 NOT found."
    kill $QEMU_PID
else
    echo "FAILURE: QEMU process died immediately. Check dmesg."
    dmesg | tail -n 20
fi

echo "==================== [STEP 5: Cleanup] ===================="
rm -rf "$LOCAL_DIR"
echo "Diagnostic Finished."