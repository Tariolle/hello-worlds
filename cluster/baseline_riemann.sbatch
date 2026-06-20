#!/bin/bash
# Classical Riemannian EEG baseline = covariance-manifold yardstick and a real
# pipeline sanity check (reads all 2993 .edf via pyedflib). CPU job, ~15-30 min.
# Submit from the repo root:  sbatch cluster/baseline_riemann.sbatch
#SBATCH --partition=defq
#SBATCH --reservation=Vivatech
#SBATCH --account=vivatech-helloworlds
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:30:00
#SBATCH --job-name=hw-riemann
#SBATCH --output=/lustre/work/vivatech-helloworlds/%u/logs/riemann_%j.out
set -e
WORK=/lustre/work/vivatech-helloworlds/$USER
export XDG_CACHE_HOME=$WORK/.cache
source "$WORK/venvs/hw_aarch64/bin/activate"
cd "$WORK/hello-worlds"
python -u -m examples.eeg.baseline_riemann \
  --data-root /lustre/work/pdl17890/udl806719/datasets/Neuro/TUAB-TUEV/TUAB_PREPROCESSED \
  --classifier tangent-logreg \
  --aggregation riemann
