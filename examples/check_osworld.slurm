#!/bin/bash
#SBATCH --job-name=osworld_debug
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1   
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G       
#SBATCH --time=01:00:00 
#SBATCH --output=osworld_diag_%j.log   # Standard output log

echo "--- [1/5] System Environment Check ---"
echo "Running on node: $(hostname)"
echo "Current time: $(date)"
echo "GPU Status:"
nvidia-smi -L || echo "ERROR: No GPU found or nvidia-smi failed"

echo -e "\n--- [2/5] Virtualization (KVM) Check ---"
if [ -e /dev/kvm ]; then
    echo "SUCCESS: /dev/kvm device found."
    ls -l /dev/kvm
    [ -r /dev/kvm ] && [ -w /dev/kvm ] && echo "Access: Read/Write permissions OK" || echo "WARNING: No R/W permissions for /dev/kvm"
else
    echo "CRITICAL ERROR: /dev/kvm not found."
    echo "Without KVM, the VM will run in software emulation mode (extremely slow)."
    echo "This is the most likely cause of your 600s timeouts."
fi

echo -e "\n--- [4/5] Launching Single Singularity Instance ---"
SANDBOX_PATH="/public/home/xlwang/genalyu/3SPO/osworld-sandbox"

# We bind /dev/kvm explicitly to ensure the container sees hardware acceleration
# We use the LOCAL_IMAGE to ensure maximum I/O performance
singularity run --nv --writable-tmpfs \
    --bind "$LOCAL_IMAGE":/System.qcow2 \
    --bind /dev/kvm:/dev/kvm \
    "$SANDBOX_PATH"

echo -e "\n--- [5/5] Debug Session Finished ---"
# Optional: Uncomment the next line to clean up the large image file after testing
# rm -rf "$LOCAL_DIR"