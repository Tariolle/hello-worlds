#!/bin/bash
# EEG I-JEPA v3 — pretraining cross-corpus TUAB + TUEV.
#
# vs v1 (TUAB seul, 0.828) : meme archi, double source de donnees.
# TUEV ajoute ~600 sessions cliniques longues, diversite pathologique.
#
# Submit:
#   sbatch /lustre/work/vivatech-helloworlds/shared/hello-worlds/cluster/train_ijepa_v3.sbatch
#SBATCH --partition=defq
#SBATCH --reservation=Vivatech
#SBATCH --account=vivatech-helloworlds
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00
#SBATCH --job-name=hw-eeg-ijepa-v3
#SBATCH --output=/lustre/work/vivatech-helloworlds/yfrayce/logs/ijepa_v3_%j.out
set -e

VENV=/lustre/work/vivatech-helloworlds/ftariolle/venvs/hw_aarch64
PROJECT=/lustre/work/vivatech-helloworlds/shared/hello-worlds
CKPT_DIR=$PROJECT/checkpoints/eeg_ijepa_v3

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

echo "=== I-JEPA v3 cross-corpus TUAB+TUEV (50ep, avg pool) ==="
python -u -m examples.eeg.main_ijepa --fname "$PROJECT/examples/eeg/cfgs/train_ijepa_v3.yaml"

echo "=== eval: random encoder floor ==="
python -u -m examples.eeg.eval \
  --ckpt "$CKPT_DIR/latest.pth.tar" \
  --floor

echo "=== eval: I-JEPA v3 cross-corpus encoder ==="
python -u -m examples.eeg.eval \
  --ckpt "$CKPT_DIR/latest.pth.tar"

echo "IJEPA_V3_DONE"
echo "Loss log -> $CKPT_DIR/train_log.json"
