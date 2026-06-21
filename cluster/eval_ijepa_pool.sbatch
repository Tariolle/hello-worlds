#!/bin/bash
# Evaluate a frozen EEG checkpoint with a chosen pooling mode on a compute node.
#
# Usage:
#   CKPT=/path/to/latest.pth.tar POOL=meanstd sbatch cluster/eval_ijepa_pool.sbatch
#SBATCH --partition=defq
#SBATCH --reservation=Vivatech
#SBATCH --account=vivatech-helloworlds
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --time=00:30:00
#SBATCH --job-name=hw-eeg-eval-pool
#SBATCH --output=/lustre/work/vivatech-helloworlds/yfrayce/logs/eval_pool_%j.out
set -e

PROJECT=${PROJECT:-/lustre/work/vivatech-helloworlds/shared/hello-worlds}
VENV=${VENV:-/lustre/work/vivatech-helloworlds/ftariolle/venvs/hw_aarch64}
POOL=${POOL:-meanstd}

if [ -z "$CKPT" ]; then
  echo "Missing CKPT=/path/to/latest.pth.tar"
  exit 2
fi

source "$VENV/bin/activate"
cd "$PROJECT"
export PYTHONPATH="$PROJECT:$PYTHONPATH"

echo "CKPT=$CKPT"
echo "POOL=$POOL"
python -u -m examples.eeg.eval --ckpt "$CKPT" --pool "$POOL"
