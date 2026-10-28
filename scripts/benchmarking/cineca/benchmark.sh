#!/bin/bash

#SBATCH --job-name=mlpf_gpu

#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --account=try26_CERN

#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

#SBATCH --output=logs_slurm/log_%x_%j.out
#SBATCH --error=logs_slurm/log_%x_%j.err

set -euo pipefail

echo "################ Job submission script ################"
cat "$0"
echo "########################################################"


echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

module --force purge
module load python gcc cmake cuda

nvidia-smi
export TORCH_LOGS="recompiles"


CKPT="$FAST/colliderml/checkpoint-40000.pth"
KWRGS="$FAST/colliderml/model_kwargs.pkl"

DATA_DIR="$FAST/"

uv run --project envs/ort-gpu --no-sync python scripts/benchmarking/benchmark.py \
    --checkpoint $CKPT \
    --data-dir $DATA_DIR \
    --dataset dataset \
    --num-events 100 \
    --outdir test_new_baseline_test_1_single_test_B2 \
    --model-kwargs $KWRGS \
    --allow-partial-state \
    --warmup-iters 1 \
    --pdf \
    --repeats 7 \
    --batch-sizes 2 \
    --pad-bin-size 8 \
    --compile-mode default \
    --warmup-seq-len 20000 \
    --variants fp16_flash \
    --compile