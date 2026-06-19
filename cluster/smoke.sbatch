#!/bin/bash
# Smoke test a forward+backward pass of all 6 (reg_type x reg_space) cells on a B200.
# Needs the venv built first (cluster/build_venv.sbatch). Submit from the repo root:
#   sbatch cluster/smoke.sbatch   ->   check $WORK/logs/smoke_*.out for SMOKE_DONE
#SBATCH --partition=defq
#SBATCH --reservation=Vivatech
#SBATCH --account=vivatech-helloworlds
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --time=00:10:00
#SBATCH --job-name=hw-smoke
#SBATCH --output=/lustre/work/vivatech-helloworlds/%u/logs/smoke_%j.out
set -e
WORK=/lustre/work/vivatech-helloworlds/$USER
export XDG_CACHE_HOME=$WORK/.cache TORCH_HOME=$WORK/.cache/torch TRITON_CACHE_DIR=$WORK/.cache/triton
source "$WORK/venvs/hw_aarch64/bin/activate"
cd "$WORK/hello-worlds" && python examples/eeg/smoke.py
