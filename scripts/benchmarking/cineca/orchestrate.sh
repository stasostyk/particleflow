#!/bin/bash

#SBATCH --job-name=onnx_gpu

#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --account=try26_CERN

#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G

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

srun uv run --project envs/ort-gpu --no-sync \
    python scripts/benchmarking/orchestrator.py \
    --runner scripts/benchmarking/test_all.py \
    --vary batch-size --values 1 4 8 16 32 \
    --title "A100, Cineca" \
    --output profiling/sweep_batch_no_sorting.pdf \
    --checkpoint ~/latest-cms/onnx_ref_pyg-cms-v1_cms_run3_20260930_105918_812327/checkpoints/checkpoint-500.pth \
    --model-kwargs ~/latest-cms/onnx_ref_pyg-cms-v1_cms_run3_20260930_105918_812327/model_kwargs.pkl \
    --data-dir ~/ \
    --dataset cms_pf_ttbar \
    --num-events 500 \
    --batch-size 8 \
    --num-threads 4 \
    --num-repeats 5 \
    --num-warmup 1 \
    --pad-bin-size 8 \
    --compile \
    --onnx-dir onnx_models