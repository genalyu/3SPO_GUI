#!/bin/bash
#SBATCH --job-name=SuperDiag
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --nodelist=gpu23
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=super_diag_%j.log

set -u

TIMEOUT_BIN=$(command -v timeout || true)
JOB_ID="${SLURM_JOB_ID:-manual}"
RUN_DIR="/tmp/${USER}/super_diag_${JOB_ID}"
LOG_FILE="${RUN_DIR}/diag.log"
mkdir -p "$RUN_DIR"

SANDBOX_DIR="/public/home/xlwang/genalyu/3SPO/osworld-sandbox"
SANDBOX_SIF="/public/home/xlwang/genalyu/3SPO/osworld_uitars.sif"
REMOTE_IMG="/public/home/xlwang/genalyu/3SPO/OSWorld/docker_vm_data/Ubuntu.qcow2"

if [ -f "$SANDBOX_SIF" ]; then
    SANDBOX_TARGET="$SANDBOX_SIF"
elif [ -d "$SANDBOX_DIR" ]; then
    SANDBOX_TARGET="$SANDBOX_DIR"
else
    echo "FATAL: sandbox not found: $SANDBOX_DIR or $SANDBOX_SIF"
    exit 2
fi

exec > >(tee -a "$LOG_FILE") 2>&1

echo "===== SuperDiag Start ====="
echo "Node: $(hostname)"
echo "User: $(whoami)"
echo "Kernel: $(uname -r)"
echo "Date: $(date '+%F %T')"
echo "Sandbox: $SANDBOX_TARGET"
echo "Log: $LOG_FILE"

# --- Host Checks ---
echo
echo "===== Host Checks ====="
echo "Current User Info: $(id)"
echo "KVM device info: $(ls -l /dev/kvm 2>/dev/null || echo 'Not found')"
if [ -e /dev/kvm ]; then
    [ -r /dev/kvm ] && [ -w /dev/kvm ] && echo "KVM_ACCESS=YES" || echo "KVM_ACCESS=NO"
    # Basic KVM init test
    if python3 -c "import os; os.open('/dev/kvm', os.O_RDWR)" 2>/dev/null; then
        echo "KVM_RDWR_TEST=OK"
    else
        echo "KVM_RDWR_TEST=FAILED"
    fi
fi

# --- Python Environment Setup ---
echo
echo "===== Python Environment Setup ====="
source /public/home/xlwang/jyy/anaconda/etc/profile.d/conda.sh
conda activate 3spo
cd /public/home/xlwang/genalyu/3SPO

# --- Prepare Local Image ---
echo
echo "===== Prepare Local Image ====="
LOCAL_WORK_DIR="/tmp/${USER}/osworld_diag_${JOB_ID}"
mkdir -p "$LOCAL_WORK_DIR"
LOCAL_IMAGE="${LOCAL_WORK_DIR}/Ubuntu.qcow2"
echo "Copying image to local SSD: $LOCAL_IMAGE"
cp "$REMOTE_IMG" "$LOCAL_IMAGE"

# --- RUN PYTHON PROVIDER TEST ---
echo
echo "===== RUN PYTHON PROVIDER TEST ====="
python examples/test_singularity_provider.py "$LOCAL_IMAGE"
rc=$?

echo
echo "===== Summary ====="
if [ $rc -eq 0 ]; then
    echo "FINAL_RESULT=PASS"
else
    echo "FINAL_RESULT=FAIL"
fi

# --- Cleanup ---
rm -rf "$LOCAL_WORK_DIR"
exit $rc
