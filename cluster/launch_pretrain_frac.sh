#!/bin/bash
# Pretrain-data-efficiency (panel 2 of the "value of SSL" figure): SSL-pretrain
# SIGReg-ambient on {5,10,25,50,100}% of TUAB-train recordings, freeze, probe full
# eval. Run from the repo root:  bash cluster/launch_pretrain_frac.sh
set -e
WORK=/lustre/work/vivatech-helloworlds/$USER
SRC=examples/eeg/cfgs/train.yaml
GEN=$WORK/frac_cfgs; mkdir -p "$GEN" "$WORK/logs"
for frac in 0.05 0.1 0.25 0.5 1.0; do
  tag=$(echo "$frac" | tr -d '.')
  cfg="$GEN/pdf_${tag}.yaml"
  sed -e "s#reg_type: .*#reg_type: sigreg#" \
      -e "s#reg_space: .*#reg_space: ambient#" \
      -e "s#  frac: .*#  frac: $frac#" \
      -e "s#ckpt_dir: .*#ckpt_dir: $WORK/checkpoints/pdf_${tag}#" "$SRC" > "$cfg"
  J=$(sbatch --parsable --export=ALL,CFG="$cfg" cluster/train.sbatch)
  echo "frac=$frac -> job $J (ckpt pdf_${tag})"
done
echo "--- queue ---"; squeue -u "$USER" -o "%.8i %.9T %.20j" 2>/dev/null | head
