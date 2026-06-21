#!/bin/bash
# EEG World Model (3-geometry JEPA) on TUEV — train + eval.
# Submit from Dalia:
#   sbatch /lustre/work/vivatech-helloworlds/shared/hello-worlds/cluster/train_tuev_wm.sbatch
#SBATCH --partition=defq
#SBATCH --reservation=Vivatech
#SBATCH --account=vivatech-helloworlds
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00
#SBATCH --job-name=hw-tuev-wm
#SBATCH --output=/lustre/work/vivatech-helloworlds/yfrayce/logs/tuev_wm_%j.out
set -e

VENV=/lustre/work/vivatech-helloworlds/ftariolle/venvs/hw_aarch64
PROJECT=/lustre/work/vivatech-helloworlds/shared/hello-worlds
CKPT_DIR=$PROJECT/checkpoints/eeg_tuev_wm
TUEV_ROOT=/lustre/work/pdl17890/udl806719/datasets/Neuro/TUAB-TUEV/TUEV_RAW_DATA

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

echo "=== TUEV data check ==="
for split in train eval; do
  n=$(find "$TUEV_ROOT/$split" -name "*.htk" 2>/dev/null | wc -l)
  echo "  $split: $n .htk files"
done

echo "=== World Model train on TUEV (3-geometry JEPA, 30 epochs) ==="
python -u -m examples.eeg.main_tuev_wm \
  --fname "$PROJECT/examples/eeg/cfgs/train_tuev_wm.yaml"

echo "=== TUEV eval: random encoder floor ==="
python -u -m examples.eeg.eval_tuev \
  --floor \
  --data_root "$TUEV_ROOT"

echo "=== TUEV eval: trained world model encoder ==="
python -u -m examples.eeg.eval_tuev \
  --ckpt "$CKPT_DIR/latest.pth.tar" \
  --data_root "$TUEV_ROOT"

echo "WM_TUEV_DONE"
echo "Loss log -> $CKPT_DIR/train_log.json"
