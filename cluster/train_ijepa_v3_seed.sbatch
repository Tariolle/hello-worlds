#!/bin/bash
# EEG I-JEPA v3 seed run — cross-corpus TUAB + TUEV, mean pooling, 50 epochs.
#
# Submit one seed:
#   SEED=1000 CKPT_DIR=/lustre/work/vivatech-helloworlds/shared/hello-worlds/checkpoints/eeg_ijepa_v3_s1000 \
#     sbatch cluster/train_ijepa_v3_seed.sbatch
#
# Defaults are safe but intentionally write to a seed-specific checkpoint.
#SBATCH --partition=defq
#SBATCH --reservation=Vivatech
#SBATCH --account=vivatech-helloworlds
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00
#SBATCH --job-name=hw-eeg-ijepa-v3s
#SBATCH --output=/lustre/work/vivatech-helloworlds/yfrayce/logs/ijepa_v3_seed_%j.out
set -e

VENV=${VENV:-/lustre/work/vivatech-helloworlds/ftariolle/venvs/hw_aarch64}
PROJECT=${PROJECT:-/lustre/work/vivatech-helloworlds/shared/hello-worlds}
CFG=${CFG:-$PROJECT/examples/eeg/cfgs/train_ijepa_v3.yaml}
SEED=${SEED:-1}
CKPT_DIR=${CKPT_DIR:-$PROJECT/checkpoints/eeg_ijepa_v3_s$SEED}

export XDG_CACHE_HOME=/lustre/work/vivatech-helloworlds/yfrayce/.cache
export TORCH_HOME=/lustre/work/vivatech-helloworlds/yfrayce/.cache/torch
export TRITON_CACHE_DIR=/lustre/work/vivatech-helloworlds/yfrayce/.cache/triton

mkdir -p /lustre/work/vivatech-helloworlds/yfrayce/logs
mkdir -p /lustre/work/vivatech-helloworlds/yfrayce/.cache/torch
mkdir -p "$CKPT_DIR"

source "$VENV/bin/activate"
cd "$PROJECT"
export PYTHONPATH="$PROJECT:$PYTHONPATH"

echo "=== env ==="
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
echo "CFG=$CFG"
echo "SEED=$SEED"
echo "CKPT_DIR=$CKPT_DIR"

echo "=== I-JEPA v3 cross-corpus TUAB+TUEV seed $SEED ==="
python -u -c "from examples.eeg.main_ijepa import run; run(fname='$CFG', **{'meta.seed': '$SEED', 'meta.ckpt_dir': '$CKPT_DIR'})"

echo "=== eval: trained I-JEPA v3 encoder seed $SEED ==="
python -u -m examples.eeg.eval --ckpt "$CKPT_DIR/latest.pth.tar"

echo "=== eval: random encoder floor seed $SEED ==="
python -u -m examples.eeg.eval --ckpt "$CKPT_DIR/latest.pth.tar" --floor

echo "IJEPA_V3_SEED_DONE seed=$SEED ckpt=$CKPT_DIR"
