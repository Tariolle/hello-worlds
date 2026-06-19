#!/bin/bash
# SSL pretrain a JEPA cell -> frozen linear probe on TUAB. One job = one number.
# Pick the cell by pointing CFG at a config (default = sigreg x ambient = Rung 0):
#   sbatch cluster/train.sbatch                              # uses cfgs/train.yaml
#   CFG=examples/eeg/cfgs/train.yaml sbatch --export=ALL,CFG cluster/train.sbatch
#SBATCH --partition=defq
#SBATCH --reservation=Vivatech
#SBATCH --account=vivatech-helloworlds
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --time=03:00:00
#SBATCH --job-name=hw-train
#SBATCH --output=/lustre/work/vivatech-helloworlds/%u/logs/train_%j.out
set -e
WORK=/lustre/work/vivatech-helloworlds/$USER
export XDG_CACHE_HOME=$WORK/.cache TORCH_HOME=$WORK/.cache/torch TRITON_CACHE_DIR=$WORK/.cache/triton
source "$WORK/venvs/hw_aarch64/bin/activate"
cd "$WORK/hello-worlds"
CFG="${CFG:-examples/eeg/cfgs/train.yaml}"
CKPT=$(python -c "from omegaconf import OmegaConf;print(OmegaConf.load('$CFG').meta.ckpt_dir)")
echo "=== SSL train: $CFG  (ckpt -> $CKPT) ==="
python -u -m examples.eeg.main --fname "$CFG"
echo "=== frozen linear probe (held-out patients) ==="
python -u -m examples.eeg.eval --ckpt "$CKPT/latest.pth.tar"
echo "TRAIN_EVAL_DONE"
