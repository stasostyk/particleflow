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

set -euxo pipefail 

module --force purge; module load modules/2.4-20250724
module load slurm gcc cmake

nvidia-smi

srun uv run --project envs/ort-gpu \
    python scripts/benchmarking/orchestrator.py \
    --runner scripts/benchmarking/test_all.py \
    --vary batch-size --values 1 2 4 \
    --title "A100, Simons" \
    --output profiling/sweep_batch_size.pdf \
    --checkpoint ~/MLPF_data/latest-cms/onnx_ref_pyg-cms-v1_cms_run3_20260930_105918_812327/checkpoints/checkpoint-500.pth \
    --model-kwargs ~/MLPF_data/latest-cms/onnx_ref_pyg-cms-v1_cms_run3_20260930_105918_812327/model_kwargs.pkl \
    --data-dir ~/ceph \
    --dataset cms_pf_ttbar \
    --num-events 500 \
    --batch-size 1 \
    --num-threads 1 \
    --num-repeats 3 \
    --num-warmup 1 \
    --pad-bin-size 8 \
    --sort-by-length \
    --compile