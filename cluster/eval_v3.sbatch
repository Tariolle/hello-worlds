#!/bin/bash
#SBATCH --partition=defq
#SBATCH --reservation=Vivatech
#SBATCH --account=vivatech-helloworlds
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=00:20:00
#SBATCH --job-name=hw-eval-v3
#SBATCH --output=/lustre/work/vivatech-helloworlds/yfrayce/logs/eval_v3_%j.out
set -e
VENV=/lustre/work/vivatech-helloworlds/ftariolle/venvs/hw_aarch64
PROJECT=/lustre/work/vivatech-helloworlds/shared/hello-worlds
source $VENV/bin/activate
cd $PROJECT
export PYTHONPATH=$PROJECT:$PYTHONPATH
python -u -m examples.eeg.eval --ckpt $PROJECT/checkpoints/eeg_ijepa_v3/latest.pth.tar --floor
echo EVAL_V3_DONE
