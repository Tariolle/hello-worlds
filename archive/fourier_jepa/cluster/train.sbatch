#!/bin/bash
# SSL pretrain one cell -> frozen linear probe on TUAB, from the ISOLATED fourier
# checkout (NOT the shared $WORK/hello-worlds working tree). One job = one number.
# Defaults to the Fourier-JEPA cell; point CFG at the conv baseline for the
# matched head-to-head:
#   sbatch archive/fourier_jepa/cluster/train.sbatch                         # Fourier
#   CFG=examples/eeg/cfgs/train.yaml sbatch --export=ALL,CFG archive/fourier_jepa/cluster/train.sbatch  # conv baseline
#SBATCH --partition=defq
#SBATCH --reservation=Vivatech
#SBATCH --account=vivatech-helloworlds
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --time=03:00:00
#SBATCH --job-name=hw-fourier-train
#SBATCH --output=/lustre/work/vivatech-helloworlds/%u/logs/fourier_train_%j.out
set -e
WORK=/lustre/work/vivatech-helloworlds/$USER
export XDG_CACHE_HOME=$WORK/.cache TORCH_HOME=$WORK/.cache/torch TRITON_CACHE_DIR=$WORK/.cache/triton
source "$WORK/venvs/hw_aarch64/bin/activate"
REPO_DIR="${REPO_DIR:-$WORK/fourierjepa/hello-worlds}"
cd "$REPO_DIR"
CFG="${CFG:-archive/fourier_jepa/train.yaml}"
DATA="${DATA:-/lustre/work/pdl17890/udl806719/datasets/Neuro/TUAB-TUEV/TUAB_PREPROCESSED}"
CKPT=$(python -c "from omegaconf import OmegaConf;print(OmegaConf.load('$CFG').meta.ckpt_dir)")
echo "=== SSL train: $CFG  (data=$DATA, ckpt -> $CKPT) ==="
python -u -m examples.eeg.main --fname "$CFG" data.data_root="$DATA"
echo "=== frozen linear probe (held-out patients) ==="
python -u -m examples.eeg.eval --ckpt "$CKPT/latest.pth.tar" --data-root "$DATA"
echo "FOURIER_TRAIN_EVAL_DONE"
