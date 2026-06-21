#!/bin/bash
# EEG I-JEPA v3 100ep — cross-corpus TUAB+TUEV, 100 epochs.
#
# v3 (50ep) -> 0.834 balanced_acc, loss encore decroissante a ep49 (0.128).
# Hypothese : 100 epochs donnent +0.5-1% supplementaire.
#
# Submit:
#   sbatch /lustre/work/vivatech-helloworlds/shared/hello-worlds/cluster/train_ijepa_v3_100ep.sbatch
#SBATCH --partition=defq
#SBATCH --reservation=Vivatech
#SBATCH --account=vivatech-helloworlds
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --time=06:00:00
#SBATCH --job-name=hw-ijepa-v3-100
#SBATCH --output=/lustre/work/vivatech-helloworlds/yfrayce/logs/ijepa_v3_100ep_%j.out
set -e

VENV=/lustre/work/vivatech-helloworlds/ftariolle/venvs/hw_aarch64
PROJECT=/lustre/work/vivatech-helloworlds/shared/hello-worlds
CKPT_DIR=$PROJECT/checkpoints/eeg_ijepa_v3_100ep

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

echo "=== I-JEPA v3 100ep cross-corpus TUAB+TUEV ==="
python -u -m examples.eeg.main_ijepa \
  --fname "$PROJECT/examples/eeg/cfgs/train_ijepa_v3_100ep.yaml"

echo "=== eval: I-JEPA v3 100ep ==="
python -u -m examples.eeg.eval \
  --ckpt "$CKPT_DIR/latest.pth.tar" \
  --floor

echo "IJEPA_V3_100EP_DONE"
