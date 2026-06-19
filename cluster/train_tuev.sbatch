#!/bin/bash
# Train JEPA on TUEV (6-class events) then eval: random floor + trained encoder.
# Submit from repo root:  sbatch cluster/train_tuev.sbatch
#SBATCH --partition=defq
#SBATCH --reservation=Vivatech
#SBATCH --account=vivatech-helloworlds
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00
#SBATCH --job-name=hw-tuev-train
#SBATCH --output=/lustre/work/vivatech-helloworlds/%u/logs/tuev_train_%j.out
set -e
WORK=/lustre/work/vivatech-helloworlds/$USER
export XDG_CACHE_HOME=$WORK/.cache TORCH_HOME=$WORK/.cache/torch TRITON_CACHE_DIR=$WORK/.cache/triton
source "$WORK/venvs/hw_aarch64/bin/activate"
cd "$WORK/hello-worlds"
git pull --ff-only

TUEV_ROOT=/lustre/work/pdl17890/udl806719/datasets/Neuro/TUAB-TUEV/TUEV_PREPROCESSED

echo "=== TUEV directory check ==="
ls "$TUEV_ROOT/train/" 2>/dev/null || { echo "ERROR: TUEV_PREPROCESSED not found"; exit 1; }
for cls in bckg spsw eyem artf gped pled; do
  n=$(find "$TUEV_ROOT/train/$cls" -name "*.edf" 2>/dev/null | wc -l)
  echo "  train/$cls: $n edf files"
done
echo "============================"

CFG=examples/eeg/cfgs/train_tuev.yaml
CKPT=$(python -c "from omegaconf import OmegaConf;print(OmegaConf.load('$CFG').meta.ckpt_dir)")

echo "=== SSL train on TUEV: $CFG -> $CKPT ==="
python -u -m examples.eeg.main --fname "$CFG"

echo "=== TUEV eval: random floor ==="
python -u -m examples.eeg.eval_tuev \
  --floor \
  --data_root "$TUEV_ROOT"

echo "=== TUEV eval: trained encoder ==="
python -u -m examples.eeg.eval_tuev \
  --ckpt "$CKPT/latest.pth.tar" \
  --floor \
  --data_root "$TUEV_ROOT"

echo "TUEV_TRAIN_EVAL_DONE"
echo "Loss log: $CKPT/train_log.json"
echo "To get loss log: cat $CKPT/train_log.json"
