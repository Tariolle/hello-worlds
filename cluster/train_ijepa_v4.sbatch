#!/bin/bash
# EEG I-JEPA v4 — cross-corpus TUAB+TUEV + attention pooler.
#
# Combine les deux levers positifs :
#   v3 (TUAB+TUEV, avg pool)  -> 0.834
#   v2 (TUAB, attn pool)      -> 0.816
#   v4 (TUAB+TUEV, attn pool) -> ?
#
# Submit:
#   sbatch /lustre/work/vivatech-helloworlds/shared/hello-worlds/cluster/train_ijepa_v4.sbatch
#SBATCH --partition=defq
#SBATCH --reservation=Vivatech
#SBATCH --account=vivatech-helloworlds
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00
#SBATCH --job-name=hw-eeg-ijepa-v4
#SBATCH --output=/lustre/work/vivatech-helloworlds/yfrayce/logs/ijepa_v4_%j.out
set -e

VENV=/lustre/work/vivatech-helloworlds/ftariolle/venvs/hw_aarch64
PROJECT=/lustre/work/vivatech-helloworlds/shared/hello-worlds
CKPT_DIR=$PROJECT/checkpoints/eeg_ijepa_v4

export XDG_CACHE_HOME=/lustre/work/vivatech-helloworlds/yfrayce/.cache
export TORCH_HOME=/lustre/work/vivatech-helloworlds/yfrayce/.cache/torch

mkdir -p /lustre/work/vivatech-helloworlds/yfrayce/logs
mkdir -p /lustre/work/vivatech-helloworlds/yfrayce/.cache/torch
mkdir -p "$CKPT_DIR"

source "$VENV/bin/activate"
cd "$PROJECT"
export PYTHONPATH="$PROJECT:$PYTHONPATH"

echo "=== env ==="
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

echo "=== I-JEPA v4 TUAB+TUEV + attn pool (50ep) ==="
python -u -m examples.eeg.main_ijepa --fname "$PROJECT/examples/eeg/cfgs/train_ijepa_v4.yaml"

echo "=== eval: I-JEPA v4 ==="
python -u -m examples.eeg.eval \
  --ckpt "$CKPT_DIR/latest.pth.tar" \
  --floor

echo "IJEPA_V4_DONE"
echo "Loss log -> $CKPT_DIR/train_log.json"
