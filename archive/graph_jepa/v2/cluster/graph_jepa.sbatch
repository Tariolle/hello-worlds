#!/bin/bash
# TCP-Graph-JEPA: SSL pretrain on NORMAL TUAB windows -> file-level anomaly eval.
# Self-contained, isolated from the main two-view JEPA track. Submit from the
# repo root of a CLEAN checkout (NOT the shared $WORK/hello-worlds working tree):
#   sbatch archive/graph_jepa/v2/cluster/graph_jepa.sbatch
# Optionally point CFG at a different config:
#   CFG=archive/graph_jepa/v2/config.yaml sbatch --export=ALL,CFG archive/graph_jepa/v2/cluster/graph_jepa.sbatch
#SBATCH --partition=defq
#SBATCH --reservation=Vivatech
#SBATCH --account=vivatech-helloworlds
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --job-name=hw-graphjepa
#SBATCH --output=/lustre/work/vivatech-helloworlds/%u/logs/graphjepa_%j.out
set -e
WORK=/lustre/work/vivatech-helloworlds/$USER
export XDG_CACHE_HOME=$WORK/.cache TORCH_HOME=$WORK/.cache/torch TRITON_CACHE_DIR=$WORK/.cache/triton
source "$WORK/venvs/hw_aarch64/bin/activate"
# REPO_DIR defaults to an isolated graph-JEPA checkout; override if needed.
REPO_DIR="${REPO_DIR:-$WORK/graphjepa/hello-worlds}"
cd "$REPO_DIR"
CFG="${CFG:-archive/graph_jepa/v2/config.yaml}"
DATA="${DATA:-/lustre/work/pdl17890/udl806719/datasets/Neuro/TUAB-TUEV/TUAB_PREPROCESSED}"
CKPT=$(python -c "from archive.graph_jepa.v2.core.config import load_yaml; print(load_yaml('$CFG').get('meta',{}).get('ckpt_dir','./checkpoints/graph_jepa'))")

echo "=== TCP-Graph-JEPA pretrain: $CFG (data=$DATA, ckpt=$CKPT) ==="
python -u -m archive.graph_jepa.v2.scripts.train_graph_jepa --config "$CFG" data.data_root="$DATA"

echo "=== file-level anomaly evaluation (eval split) ==="
python -u -m archive.graph_jepa.v2.scripts.evaluate_graph_jepa \
  --ckpt "$CKPT/latest.pth.tar" --data-root "$DATA" \
  --threshold-from normal --quantile 0.95 \
  --out "$REPO_DIR/archive/graph_jepa/v2/results/eval_metrics.json"

echo "GRAPH_JEPA_SBATCH_DONE"
