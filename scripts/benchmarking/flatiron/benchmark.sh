#!/bin/sh

# Walltime limit
#SBATCH -t 1:30:00
#SBATCH -N 1
#SBATCH --tasks-per-node=1
#SBATCH -p gpu
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --constraint=a100-80gb


# Job name
#SBATCH -J onnx_gpu

# Output and error logs
#SBATCH -o logs_slurm/log_%x_%j.out
#SBATCH -e logs_slurm/log_%x_%j.err

export TORCH_LOGS="recompiles"

# Add jobscript to job output
echo "#################### Job submission script. #############################"
cat $0
echo "################# End of job submission script. #########################"


module --force purge; module load modules/2.4-20250724
module load slurm gcc cmake

nvidia-smi

CKPT="$FAST/colliderml/checkpoint-40000.pth"
KWRGS="$FAST/colliderml/model_kwargs.pkl"

DATA_DIR="$FAST/"

uv run --project envs/ort-gpu --no-sync python scripts/benchmarking/benchmark.py \
    --checkpoint $CKPT \
    --data-dir $DATA_DIR \
    --dataset dataset \
    --num-events 50 \
    --outdir test_new_baseline_EXTRA_LARGE \
    --model-kwargs $KWRGS \
    --allow-partial-state \
    --warmup-iters 3 \
    --pdf \
    --repeats 10 \
    --batch-sizes 1 2 4 \
    --pad-bin-size 8 \
    --compile-mode default \
    --warmup-seq-len 20000 \
    --variants fp16_flash bf16_flash \
    --compile