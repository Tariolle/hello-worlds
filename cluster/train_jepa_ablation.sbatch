#!/bin/bash
# JEPA two-view vs single-view ablation — run sequentially on one GPU.
#
# Two-view:   masked v1 -> predict EMA(v2)          [augmentation invariance + masking]
# Single-view: masked v1 -> predict EMA(clean v1)   [masking only, no second view]
#
# Submit:
#   sbatch /lustre/work/vivatech-helloworlds/shared/hello-worlds/cluster/train_jepa_ablation.sbatch
#SBATCH --partition=defq
#SBATCH --reservation=Vivatech
#SBATCH --account=vivatech-helloworlds
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --time=06:00:00
#SBATCH --job-name=hw-jepa-ablation
#SBATCH --output=/lustre/work/vivatech-helloworlds/yfrayce/logs/jepa_ablation_%j.out
set -e

VENV=/lustre/work/vivatech-helloworlds/ftariolle/venvs/hw_aarch64
PROJECT=/lustre/work/vivatech-helloworlds/shared/hello-worlds
CKPT_2V=$PROJECT/checkpoints/eeg_jepa_abl_2view
CKPT_1V=$PROJECT/checkpoints/eeg_jepa_abl_1view

export XDG_CACHE_HOME=/lustre/work/vivatech-helloworlds/yfrayce/.cache
export TORCH_HOME=/lustre/work/vivatech-helloworlds/yfrayce/.cache/torch

mkdir -p /lustre/work/vivatech-helloworlds/yfrayce/logs
mkdir -p /lustre/work/vivatech-helloworlds/yfrayce/.cache/torch
mkdir -p "$CKPT_2V"
mkdir -p "$CKPT_1V"

source "$VENV/bin/activate"
cd "$PROJECT"
export PYTHONPATH="$PROJECT:$PYTHONPATH"

echo "=== env ==="
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

# ── Two-view JEPA (30 ep) ──────────────────────────────────────────────────
echo "=== [1/4] Two-view JEPA — training 30ep ==="
python -u -m examples.eeg.main_ijepa \
  --fname "$PROJECT/examples/eeg/cfgs/train_jepa_ablation_2view.yaml"

echo "=== [2/4] Two-view JEPA — eval ==="
python -u -m examples.eeg.eval \
  --ckpt "$CKPT_2V/latest.pth.tar" \
  --floor
python -u -m examples.eeg.eval \
  --ckpt "$CKPT_2V/latest.pth.tar"

# ── Single-view JEPA (30 ep) ───────────────────────────────────────────────
echo "=== [3/4] Single-view JEPA — training 30ep ==="
python -u -m examples.eeg.main_ijepa \
  --fname "$PROJECT/examples/eeg/cfgs/train_jepa_ablation_1view.yaml"

echo "=== [4/4] Single-view JEPA — eval ==="
python -u -m examples.eeg.eval \
  --ckpt "$CKPT_1V/latest.pth.tar" \
  --floor
python -u -m examples.eeg.eval \
  --ckpt "$CKPT_1V/latest.pth.tar"

echo "=== ABLATION DONE ==="
echo "Two-view  log -> $CKPT_2V/train_log.json"
echo "Single-view log -> $CKPT_1V/train_log.json"
