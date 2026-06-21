#!/bin/bash
# TUEV 6-class eval: random encoder floor + trained encoder probe.
# Submit from repo root:  sbatch cluster/eval_tuev.sbatch
# To eval a specific checkpoint:
#   sbatch cluster/eval_tuev.sbatch --ckpt checkpoints/eeg_ambient_sigreg/latest.pth.tar
#SBATCH --partition=defq
#SBATCH --reservation=Vivatech
#SBATCH --account=vivatech-helloworlds
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --job-name=hw-tuev-eval
#SBATCH --output=/lustre/work/vivatech-helloworlds/%u/logs/tuev_eval_%j.out
set -e
WORK=/lustre/work/vivatech-helloworlds/$USER
export XDG_CACHE_HOME=$WORK/.cache
source "$WORK/venvs/hw_aarch64/bin/activate"
cd "$WORK/hello-worlds"
git pull --ff-only

TUEV_ROOT=/lustre/work/pdl17890/udl806719/datasets/Neuro/TUAB-TUEV/TUEV_PREPROCESSED

# first: sanity check — list what's in TUEV
echo "=== TUEV directory structure ==="
ls "$TUEV_ROOT/train/" 2>/dev/null || echo "TUEV_PREPROCESSED not found — check path"
echo "================================"

# random encoder floor (no checkpoint needed, ~5 min)
python -u -m examples.eeg.eval_tuev \
  --floor \
  --data_root "$TUEV_ROOT" \
  "$@"
