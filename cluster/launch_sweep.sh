#!/bin/bash
# Launch the 2x2 factorial (+ VICReg reference) as parallel single-GPU jobs.
# Each cell = SSL pretrain -> frozen probe (~3 min). Run from the repo root:
#   bash cluster/launch_sweep.sh [SEED]      # SEED default 1
set -e
WORK=/lustre/work/vivatech-helloworlds/$USER
SEED=${1:-1}
SRC=examples/eeg/cfgs/train.yaml
GEN=$WORK/sweep_cfgs; mkdir -p "$GEN" "$WORK/logs"
# name:reg_type:reg_space  (C0 reference + the 2x2 core C1..C4)
cells="c0_vicreg_ambient:vicreg:ambient \
c1_sigreg_ambient:sigreg:ambient \
c2_sigreg_tangent:sigreg:tangent \
c3_peira_ambient:peira:ambient \
c4_peira_tangent:peira:tangent"
for cell in $cells; do
  name=${cell%%:*}; rest=${cell#*:}; rt=${rest%%:*}; rs=${rest##*:}
  cfg="$GEN/${name}_s${SEED}.yaml"
  sed -e "s#reg_type: .*#reg_type: $rt#" \
      -e "s#reg_space: .*#reg_space: $rs#" \
      -e "s#  seed: .*#  seed: $SEED#" \
      -e "s#ckpt_dir: .*#ckpt_dir: $WORK/checkpoints/${name}_s${SEED}#" "$SRC" > "$cfg"
  J=$(sbatch --parsable --export=ALL,CFG="$cfg" cluster/train.sbatch)
  echo "$name (seed $SEED) -> job $J"
done
echo "--- queue ---"; squeue -u "$USER" -o "%.8i %.9T %.20j" 2>/dev/null | head -12
