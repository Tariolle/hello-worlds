#!/bin/bash
# Smoke test the Fourier-STFT encoder (forward+backward, all reg cells) on a B200.
# Runs from the ISOLATED fourier checkout, never the shared $WORK/hello-worlds tree.
#   sbatch archive/fourier_jepa/cluster/smoke.sbatch   ->   look for FOURIER_SMOKE_DONE in the log
#SBATCH --partition=defq
#SBATCH --reservation=Vivatech
#SBATCH --account=vivatech-helloworlds
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --time=00:10:00
#SBATCH --job-name=hw-fourier-smoke
#SBATCH --output=/lustre/work/vivatech-helloworlds/%u/logs/fourier_smoke_%j.out
set -e
WORK=/lustre/work/vivatech-helloworlds/$USER
export XDG_CACHE_HOME=$WORK/.cache TORCH_HOME=$WORK/.cache/torch TRITON_CACHE_DIR=$WORK/.cache/triton
source "$WORK/venvs/hw_aarch64/bin/activate"
REPO_DIR="${REPO_DIR:-$WORK/fourierjepa/hello-worlds}"
cd "$REPO_DIR"
python -u -m archive.fourier_jepa.smoke
