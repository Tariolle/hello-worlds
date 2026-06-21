#!/bin/bash
# Render the EEG benchmark comparison once both checkpoints exist: conv-baseline vs
# Fourier-JEPA (both SIGReg-ambient, seed 1) + random-encoder floor + Riemannian
# baseline, alongside the published references. Writes table/CSV/plots to results/.
# Submit AFTER the two train jobs (e.g. --dependency=afterok:<jobA>:<jobB>):
#   sbatch --dependency=afterok:JOBA:JOBB archive/fourier_jepa/cluster/benchmark.sbatch
#SBATCH --partition=defq
#SBATCH --reservation=Vivatech
#SBATCH --account=vivatech-helloworlds
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00
#SBATCH --job-name=hw-fourier-bench
#SBATCH --output=/lustre/work/vivatech-helloworlds/%u/logs/fourier_bench_%j.out
set -e
WORK=/lustre/work/vivatech-helloworlds/$USER
export XDG_CACHE_HOME=$WORK/.cache TORCH_HOME=$WORK/.cache/torch TRITON_CACHE_DIR=$WORK/.cache/triton
source "$WORK/venvs/hw_aarch64/bin/activate"
REPO_DIR="${REPO_DIR:-$WORK/fourierjepa/hello-worlds}"
cd "$REPO_DIR"
DATA="${DATA:-/lustre/work/pdl17890/udl806719/datasets/Neuro/TUAB-TUEV/TUAB_PREPROCESSED}"
CONV_CKPT="${CONV_CKPT:-checkpoints/eeg_ambient_sigreg/latest.pth.tar}"
FOURIER_CKPT="${FOURIER_CKPT:-checkpoints/eeg_fourier_sigreg_ambient/latest.pth.tar}"
echo "=== benchmark render: conv=$CONV_CKPT  fourier=$FOURIER_CKPT  (data=$DATA) ==="
python -u -m examples.eeg.benchmark \
  --data-root "$DATA" \
  --run-riemann --run-random-floor \
  --checkpoint "jepa_sigreg_ambient=$CONV_CKPT" \
  --checkpoint "jepa_fourier_sigreg_ambient=$FOURIER_CKPT"
echo "FOURIER_BENCH_DONE"
