#!/bin/bash
# TUEV cross-TASK frozen probe (P2 bonus): frozen TUAB SIGReg encoder -> 6-class
# event probe on TUEV (held-out patients). Cross-task, same-site (TUH montage).
#   CKPT=<.../latest.pth.tar> sbatch --export=ALL,CKPT cluster/tuev.sbatch
#SBATCH --partition=defq
#SBATCH --reservation=Vivatech
#SBATCH --account=vivatech-helloworlds
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --time=00:30:00
#SBATCH --job-name=hw-tuev
#SBATCH --output=/lustre/work/vivatech-helloworlds/%u/logs/tuev_%j.out
set -e
W=/lustre/work/vivatech-helloworlds/$USER
export XDG_CACHE_HOME=$W/.cache TORCH_HOME=$W/.cache/torch TRITON_CACHE_DIR=$W/.cache/triton
source "$W/venvs/hw_aarch64/bin/activate"
cd "$W/hello-worlds"
CKPT="${CKPT:-$W/checkpoints/c1_sigreg_ambient_s1/latest.pth.tar}"
TUEV="${TUEV:-/lustre/work/pdl17890/udl806719/datasets/Neuro/TUAB-TUEV-PREPROCESSED/TUEV_PREPROCESSED}"
echo "=== TUEV cross-task frozen probe ==="
echo "ckpt=$CKPT"
echo "tuev=$TUEV"
python -u -m examples.eeg.tuev_probe --ckpt "$CKPT" --tuev-root "$TUEV" --floor --per-class-cap 1500
echo "TUEV_DONE"
