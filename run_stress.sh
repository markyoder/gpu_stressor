#!/bin/bash
#
#SBATCH --job-name=gpu-stress-test
#SBATCH --output=gpu_stress_%A.out
#SBATCH --partition=bprogers
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#

# for use on Sherlock; there's a venv, but probably nothing in it.
module load py-pytorch/2.4.1_py312
#
GPU_ID=${1:-0}
DURATION=${2:-25}
LOGFILE="gpu${GPU_ID}_$(date +%s).csv"

# set up nvidia-smi (to run in background):
nvidia-smi -i $GPU_ID \
  --query-gpu=timestamp,temperature.gpu,temperature.memory,power.draw,clocks.sm,clocks.mem,clocks_throttle_reasons.active \
  --format=csv -l 1 > "$LOGFILE" &
MONITOR_PID=$!

#
# now launch your stresor:
CUDA_VISIBLE_DEVICES=$GPU_ID python stress.py --duration $DURATION
# CUDA_VISIBLE_DEVICES=$GPU_ID python per_gpu_bench.py --duration $DURATION

kill $MONITOR_PID
echo "Log written to $LOGFILE"