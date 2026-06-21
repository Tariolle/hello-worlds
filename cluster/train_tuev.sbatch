#!/bin/bash
# Train JEPA on TUEV (6-class events) then eval: random floor + trained encoder.
# Uses ftariolle's venv + shared/hello-worlds project dir.
# Submit from Dalia:  sbatch /lustre/work/vivatech-helloworlds/shared/hello-worlds/cluster/train_tuev.sbatch
#SBATCH --partition=defq
#SBATCH --reservation=Vivatech
#SBATCH --account=vivatech-helloworlds
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00
#SBATCH --job-name=hw-tuev-train
#SBATCH --output=/lustre/work/vivatech-helloworlds/yfrayce/logs/tuev_train_%j.out
set -e

VENV=/lustre/work/vivatech-helloworlds/ftariolle/venvs/hw_aarch64
PROJECT=/lustre/work/vivatech-helloworlds/shared/hello-worlds
CKPT_DIR=$PROJECT/checkpoints/eeg_tuev_sigreg
TUEV_ROOT=/lustre/work/pdl17890/udl806719/datasets/Neuro/TUAB-TUEV/TUEV_RAW_DATA
# data_root in train_tuev.yaml must match this path

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
ls "$TUEV_ROOT/train/" | head -5
for split in train eval; do
  n=$(find "$TUEV_ROOT/$split" -name "*.htk" 2>/dev/null | wc -l)
  echo "  $split: $n .htk files"
done

echo "=== SSL train on TUEV (30 epochs) ==="
python -u -m examples.eeg.main_tuev --fname "$PROJECT/examples/eeg/cfgs/train_tuev.yaml"

echo "=== TUEV eval: random encoder floor ==="
python -u -m examples.eeg.eval_tuev \
  --floor \
  --data_root "$TUEV_ROOT"

echo "=== TUEV eval: trained encoder ==="
python -u -m examples.eeg.eval_tuev \
  --ckpt "$CKPT_DIR/latest.pth.tar" \
  --floor \
  --data_root "$TUEV_ROOT"

echo "TUEV_DONE"
echo "Loss log -> $CKPT_DIR/train_log.json"
