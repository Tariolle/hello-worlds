#!/bin/bash
# EEG I-JEPA v2 — attention pooler + masquage de bandes de frequence.
#
# vs v1 (train_ijepa.sbatch) :
#   - pool_type: attn   : l'encodeur apprend a ponderer les instants temporels
#   - freq_mask_p: 0.5  : masquage de bandes EEG (delta/theta/alpha/beta/gamma)
#                         comme tache predictive supplementaire
#
# Submit from Dalia:
#   sbatch /lustre/work/vivatech-helloworlds/shared/hello-worlds/cluster/train_ijepa_v2.sbatch
#SBATCH --partition=defq
#SBATCH --reservation=Vivatech
#SBATCH --account=vivatech-helloworlds
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00
#SBATCH --job-name=hw-eeg-ijepa-v2
#SBATCH --output=/lustre/work/vivatech-helloworlds/yfrayce/logs/ijepa_v2_%j.out
set -e

VENV=/lustre/work/vivatech-helloworlds/ftariolle/venvs/hw_aarch64
PROJECT=/lustre/work/vivatech-helloworlds/shared/hello-worlds
CKPT_DIR=$PROJECT/checkpoints/eeg_ijepa_v2

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

echo "=== I-JEPA v2 SSL train on TUAB (50ep, attn pool + freq masking) ==="
python -u -m examples.eeg.main_ijepa --fname "$PROJECT/examples/eeg/cfgs/train_ijepa_v2.yaml"

echo "=== eval: random encoder floor ==="
python -u -m examples.eeg.eval \
  --ckpt "$CKPT_DIR/latest.pth.tar" \
  --floor

echo "=== eval: I-JEPA v2 trained encoder ==="
python -u -m examples.eeg.eval \
  --ckpt "$CKPT_DIR/latest.pth.tar"

echo "IJEPA_V2_DONE"
echo "Loss log -> $CKPT_DIR/train_log.json"
