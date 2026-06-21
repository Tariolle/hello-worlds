#!/bin/bash
# EEG I-JEPA pretraining on TUAB + eval.
# vs. SIGReg: adds EMA target encoder, latent predictor, cosine LR, temporal masking.
#
# Submit from Dalia:
#   sbatch /lustre/work/vivatech-helloworlds/shared/hello-worlds/cluster/train_ijepa.sbatch
#SBATCH --partition=defq
#SBATCH --reservation=Vivatech
#SBATCH --account=vivatech-helloworlds
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00
#SBATCH --job-name=hw-eeg-ijepa
#SBATCH --output=/lustre/work/vivatech-helloworlds/yfrayce/logs/ijepa_%j.out
set -e

VENV=/lustre/work/vivatech-helloworlds/ftariolle/venvs/hw_aarch64
PROJECT=/lustre/work/vivatech-helloworlds/shared/hello-worlds
CKPT_DIR=$PROJECT/checkpoints/eeg_ijepa
TUAB_ROOT=/lustre/work/pdl17890/udl806719/datasets/Neuro/TUAB-TUEV/TUAB_PREPROCESSED

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

echo "=== I-JEPA SSL train on TUAB (50 epochs, EMA + cosine LR) ==="
python -u -m examples.eeg.main_ijepa --fname "$PROJECT/examples/eeg/cfgs/train_ijepa.yaml"

echo "=== eval: random encoder floor ==="
python -u -m examples.eeg.eval \
  --ckpt "$CKPT_DIR/latest.pth.tar" \
  --floor

echo "=== eval: I-JEPA trained encoder ==="
python -u -m examples.eeg.eval \
  --ckpt "$CKPT_DIR/latest.pth.tar"

echo "IJEPA_DONE"
echo "Loss log -> $CKPT_DIR/train_log.json"
