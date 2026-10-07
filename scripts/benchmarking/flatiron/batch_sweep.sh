#!/bin/sh

# Walltime limit
#SBATCH -t 0:30:00
#SBATCH -N 1
#SBATCH --tasks-per-node=1
#SBATCH -p gpu
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --constraint=h100


# Job name
#SBATCH -J onnx_gpu

# Output and error logs
#SBATCH -o logs_slurm/log_%x_%j.out
#SBATCH -e logs_slurm/log_%x_%j.err

# Add jobscript to job output
echo "#################### Job submission script. #############################"
cat $0
echo "################# End of job submission script. #########################"


module --force purge; module load modules/2.4-20250724
module load slurm gcc cmake

nvidia-smi

uv run --project envs/ort-gpu --no-sync \
    python scripts/benchmarking/batch_sweep.py \
    --runner scripts/benchmarking/inference.py \
    --onnx-model ./onnx_benchmarks/accuracy_check/model_fused_fp16.onnx \
    --data-dir ~/ceph \
    --dataset cms_pf_ttbar \
    --num-events 500 \
    --batch-sizes 1 2 4 8 16 32 \
    --warmup-runs 2 \
    --benchmark-repeats 3 \
    --continue-on-error \
    --pad-bin-size 8 \
    --outdir ./batch_16