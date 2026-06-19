#!/bin/bash
# One-time: build the aarch64 compute-node venv with torch+cu128 and our deps.
# Runs on a COMPUTE node (aarch64) so pip grabs the right wheels. ~10 min.
# Submit from the repo root:  sbatch cluster/build_venv.sbatch
# (ensure $WORK/logs exists first:  mkdir -p /lustre/work/vivatech-helloworlds/$USER/logs)
#SBATCH --partition=defq
#SBATCH --reservation=Vivatech
#SBATCH --account=vivatech-helloworlds
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:40:00
#SBATCH --job-name=hw-venv
#SBATCH --output=/lustre/work/vivatech-helloworlds/%u/logs/venv_%j.out
set -e
WORK=/lustre/work/vivatech-helloworlds/$USER
# keep every cache off the tiny /lustre/home quota
export PIP_CACHE_DIR=$WORK/.cache/pip XDG_CACHE_HOME=$WORK/.cache
mkdir -p "$PIP_CACHE_DIR" "$WORK/venvs"
module load python312 2>/dev/null || true
PY=$(command -v python3.12 || command -v python3)
echo "python: $($PY --version 2>&1) on $(uname -m)"
rm -rf "$WORK/venvs/hw_aarch64"
"$PY" -m venv "$WORK/venvs/hw_aarch64"
source "$WORK/venvs/hw_aarch64/bin/activate"
pip install --upgrade pip -q
echo "=== torch cu128 (aarch64) ==="
pip install torch --index-url https://download.pytorch.org/whl/cu128
echo "=== deps ==="
pip install numpy scipy scikit-learn omegaconf pyyaml einops pyedflib pyriemann matplotlib tqdm wandb
python -c "import torch,numpy,sklearn,omegaconf,pyriemann,pyedflib;print('torch',torch.__version__,'cuda',torch.cuda.is_available())"
echo "VENV_DONE"
