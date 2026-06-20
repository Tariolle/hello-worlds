#!/bin/bash
# Supervised-from-scratch baseline on TUAB -> end-to-end number + frozen-probe number.
# One job = one seed. Smoke test:   sbatch cluster/train_supervised.sbatch
# Other seed:  SEED=1000 sbatch --export=ALL,SEED cluster/train_supervised.sbatch
#SBATCH --partition=defq
#SBATCH --reservation=Vivatech
#SBATCH --account=vivatech-helloworlds
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --time=03:00:00
#SBATCH --job-name=hw-sup
#SBATCH --output=/lustre/work/vivatech-helloworlds/%u/logs/sup_%j.out
set -e
WORK=/lustre/work/vivatech-helloworlds/$USER
export XDG_CACHE_HOME=$WORK/.cache TORCH_HOME=$WORK/.cache/torch TRITON_CACHE_DIR=$WORK/.cache/triton
source "$WORK/venvs/hw_aarch64/bin/activate"
cd "$WORK/hello-worlds"
CFG="${CFG:-examples/eeg/cfgs/train_supervised.yaml}"
SEED="${SEED:-1}"
CKPT="${CKPT:-$WORK/checkpoints/sup_tuab_s${SEED}}"
echo "=== SUPERVISED train: seed=$SEED  (ckpt -> $CKPT) ==="
python -u -m examples.eeg.main_supervised --fname "$CFG" meta.seed=$SEED meta.ckpt_dir=$CKPT
echo "=== FROZEN linear probe on the supervised encoder (held-out patients) ==="
python -u -m examples.eeg.eval --ckpt "$CKPT/latest.pth.tar"
echo "SUP_TRAIN_EVAL_DONE"
