#!/bin/sh

# Walltime limit
#SBATCH -t 0:30:00
#SBATCH -N 1
#SBATCH --tasks-per-node=1
#SBATCH -p gpu
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G



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



# uv run --project envs/ort-gpu --no-sync \
#     python scripts/benchmarking/batch.py \
#   --checkpoint ~/checkpoints/checkpoint-10000.pth \
#   --model-kwargs ~/model_kwargs.pkl \
#   --data-dir ~/ceph \
#   --dataset cms_pf_ttbar \
#   --num-events 200 \
#   --device cuda \
#   --outdir onnx_benchmarks/gpu \
#   --num-threads 4 \
#   --batch-size 16

# uv run --project envs/ort-gpu --no-sync \
#     python scripts/benchmarking/inference.py \
#     --onnx-model ./onnx_benchmarks/accuracy_check/model_fused_fp16.onnx \
#     --data-dir ~/ceph \
#     --dataset cms_pf_ttbar \
#     --num-events 500 \
#     --batch-size 4 \
#     --warmup-runs 2 \
#     --benchmark-repeats 3 \
#     --outdir ./benchmark_test2

# uv run --project envs/ort-gpu --no-sync \
#     python scripts/benchmarking/inference.py \
#     --onnx-model ./onnx_benchmarks/gpu/model_fused_fp32_fp16.onnx \
#     --data-dir /mnt/ceph/users/ewulff/tensorflow_datasets/cld/ \
#     --dataset cld_edm_ttbar_pf \
#     --num-events 500 \
#     --batch-size 4 \
#     --warmup-runs 2 \
#     --benchmark-repeats 3 \
#     --outdir ./benchmark_1

uv run --project envs/ort-gpu \
    python scripts/cms-validate-onnx.py \
    --checkpoint ~/MLPF_data/latest-cms/onnx_ref_pyg-cms-v1_cms_run3_20260930_105918_812327/checkpoints/checkpoint-500.pth \
    --model-kwargs ~/MLPF_data/latest-cms/onnx_ref_pyg-cms-v1_cms_run3_20260930_105918_812327/model_kwargs.pkl \
    --data-dir ~/ceph \
    --dataset cms_pf_ttbar \
    --num-events 500 \
    --device cuda \
    --outdir onnx_benchmarks/accuracy_check2\ \
    --num-threads 1