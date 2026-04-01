#!/bin/bash
#SBATCH --job-name=SuperDiag
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --nodelist=gpu22
#SBATCH --gres=gpu:1
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

echo
echo "===== Host Checks ====="
df -h /tmp || true
df -h /public/home/xlwang || true
if [ -e /dev/kvm ]; then
    ls -l /dev/kvm || true
    [ -r /dev/kvm ] && [ -w /dev/kvm ] && echo "KVM_ACCESS=YES" || echo "KVM_ACCESS=NO"
else
    echo "KVM_ACCESS=NO_DEVICE"
fi

run_with_timeout "singularity version" 10 singularity --version
mark_result $?

run_with_timeout "singularity simple host command" 30 singularity exec "$SANDBOX_TARGET" /bin/sh -c 'echo host_exec_ok'
mark_result $?

# If simple command failed, try with --debug for next one
if [ $? -ne 0 ]; then
    run_with_timeout "singularity simple command (DEBUG)" 45 singularity -d exec "$SANDBOX_TARGET" /bin/sh -c 'echo host_exec_debug_ok'
    mark_result $?
fi

MODES=(
    "--cleanenv --no-home --writable-tmpfs --no-mount overlay"
    "--cleanenv --no-home --writable-tmpfs"
    "--cleanenv --no-home --no-mount overlay"
    "--cleanenv --no-home"
    "--cleanenv --containall"
    "--cleanenv --no-privs"
    "--cleanenv"
    ""
)

MODE_PASS=0
for mode in "${MODES[@]}"; do
    run_with_timeout "preflight mode: [${mode:-none}]" 45 bash -lc "singularity exec $mode '$SANDBOX_TARGET' /bin/sh -c 'echo preflight_ok'"
    rc=$?
    mark_result $rc
    if [ $rc -eq 0 ]; then
        MODE_PASS=$((MODE_PASS + 1))
    fi
done

if [ -f "$REMOTE_IMG" ]; then
    run_with_timeout "bind qcow2 into container" 30 bash -lc "singularity exec --cleanenv --no-home --bind '$REMOTE_IMG':/System.qcow2 '$SANDBOX_TARGET' /bin/sh -c 'test -f /System.qcow2 && echo bind_ok'"
    mark_result $?
else
    echo "SKIP: remote image not found at $REMOTE_IMG"
fi

if [ -e /dev/kvm ]; then
    run_with_timeout "kvm bind probe" 30 bash -lc "singularity exec --cleanenv --bind /dev/kvm:/dev/kvm '$SANDBOX_TARGET' /bin/sh -c 'test -e /dev/kvm && echo kvm_bind_ok'"
    mark_result $?
fi

echo
echo "===== Summary ====="
echo "PASS_COUNT=$PASS_COUNT"
echo "FAIL_COUNT=$FAIL_COUNT"
echo "MODE_PASS=$MODE_PASS"

if [ "$MODE_PASS" -eq 0 ]; then
    echo "FINAL_RESULT=FAIL: all preflight modes failed or timed out"
    exit 1
fi

echo "FINAL_RESULT=PASS: at least one preflight mode works"
exit 0
