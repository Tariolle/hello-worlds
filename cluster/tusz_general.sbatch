#!/bin/bash
# FAIR-COMPARISON experiment (jury defense): GENERAL TUH pretrain -> frozen TUAB.
# Pretrain SSL on TUSZ (TUH Seizure corpus, 19ch@200Hz, same montage as TUAB),
# with TUAB-eval patients EXCLUDED (patient-disjoint), then freeze and linear-probe
# TUAB. This matches the foundation-models' regime (general pretrain -> frozen TUAB)
# and removes our in-domain advantage, so "we hold / they collapse" becomes
# apples-to-apples. (TUSZ is same-SITE general, not multi-site like the FMs.)
#   CFG=<tusz cfg> sbatch --export=ALL,CFG cluster/tusz_general.sbatch
#SBATCH --partition=defq
#SBATCH --reservation=Vivatech
#SBATCH --account=vivatech-helloworlds
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00
#SBATCH --job-name=hw-tusz
#SBATCH --output=/lustre/work/vivatech-helloworlds/%u/logs/tusz_%j.out
set -e
W=/lustre/work/vivatech-helloworlds/$USER
export XDG_CACHE_HOME=$W/.cache TORCH_HOME=$W/.cache/torch TRITON_CACHE_DIR=$W/.cache/triton
source "$W/venvs/hw_aarch64/bin/activate"
cd "$W/hello-worlds"
CFG="${CFG:-$W/tusz_cfgs/tusz_general.yaml}"
TUAB="${TUAB:-/lustre/work/pdl17890/udl806719/datasets/Neuro/TUAB-TUEV/TUAB_PREPROCESSED}"
CKPT=$(python -c "from omegaconf import OmegaConf;print(OmegaConf.load('$CFG').meta.ckpt_dir)")
echo "=== GENERAL pretrain (TUSZ, ex-TUAB-eval patients): $CFG -> $CKPT ==="
python -u -m examples.eeg.main --fname "$CFG"
echo "=== frozen linear-probe on TUAB (held-out patients) + random floor ==="
python -u -m examples.eeg.eval --ckpt "$CKPT/latest.pth.tar" --data-root "$TUAB" --label-scheme tuab --floor
echo "TUSZ_GENERAL_DONE"
