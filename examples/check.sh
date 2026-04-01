#!/bin/bash
#SBATCH --job-name=SuperDiag
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --nodelist=gpu22
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

if [ -z "$TIMEOUT_BIN" ]; then
    echo "FATAL: timeout command not found."
    exit 2
fi

run_with_timeout() {
    local name="$1"
    local sec="$2"
    shift 2
    echo
    echo "===== TEST: $name (timeout ${sec}s) ====="
    set +e
    "$TIMEOUT_BIN" -k 5 "${sec}s" "$@"
    local rc=$?
    set -e
    if [ $rc -eq 0 ]; then
        echo "RESULT: PASS - $name"
    elif [ $rc -eq 124 ]; then
        echo "RESULT: TIMEOUT - $name"
    else
        echo "RESULT: FAIL($rc) - $name"
    fi
    return $rc
}

PASS_COUNT=0
FAIL_COUNT=0

mark_result() {
    local rc="$1"
    if [ "$rc" -eq 0 ]; then
        PASS_COUNT=$((PASS_COUNT + 1))
    else
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
}

echo "===== Host Checks ====="
df -h /tmp || true
df -h /public/home/xlwang || true
if [ -e /dev/kvm ]; then
    ls -l /dev/kvm || true
    [ -r /dev/kvm ] && [ -w /dev/kvm ] && echo "KVM_ACCESS=YES" || echo "KVM_ACCESS=NO"
else
    echo "KVM_ACCESS=NO_DEVICE"
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
echo "This will test the actual SingularityProvider class logic (preflight, binds, nginx, etc.)"
python examples/test_singularity_provider.py "$LOCAL_IMAGE"
rc=$?

echo
echo "===== Summary ====="
if [ $rc -eq 0 ]; then
    echo "FINAL_RESULT=PASS: SingularityProvider successfully started and cleaned up."
else
    echo "FINAL_RESULT=FAIL: SingularityProvider failed. Check logs above."
fi

# --- Cleanup ---
rm -rf "$LOCAL_WORK_DIR"
exit $rc
