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

BATCH_SIZE=${1:-1}

set -euxo pipefail 

# Add jobscript to job output
echo "#################### Job submission script. #############################"
cat $0
echo "################# End of job submission script. #########################"


module --force purge; module load modules/2.4-20250724
module load slurm gcc cmake

nvidia-smi

srun uv run --project envs/ort-gpu --no-sync \
    python scripts/benchmarking/test_all.py \
    --checkpoint ~/MLPF_data/latest-cms/onnx_ref_pyg-cms-v1_cms_run3_20260930_105918_812327/checkpoints/checkpoint-500.pth \
    --model-kwargs ~/MLPF_data/latest-cms/onnx_ref_pyg-cms-v1_cms_run3_20260930_105918_812327/model_kwargs.pkl \
    --data-dir ~/ceph \
    --batch-size "$BATCH_SIZE" \
    --outdir ./b06-10_1 \
    --num-events 500 \
    --pad-bin-size 8 \
    --num-warmup 3 \
    --sort-by-length \
    --num-threads 8 \
    --compile \
    --num-repeats 3

