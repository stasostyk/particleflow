#!/bin/bash

#SBATCH --job-name=onnx_gpu

#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --account=try26_CERN

#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=8G

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


# uv run --project envs/ort-gpu --no-sync \
#     python scripts/cms-validate-onnx.py \
#     --checkpoint ~/latest-cms/onnx_ref_pyg-cms-v1_cms_run3_20260930_105918_812327/checkpoints/checkpoint-500.pth \
#     --dataset cms_pf_ttbar \
#     --model-kwargs ~/latest-cms/onnx_ref_pyg-cms-v1_cms_run3_20260930_105918_812327/model_kwargs.pkl \
#     --data-dir ~/ \
#     --outdir test1 \
#     --num-events 10 \
#     --device cuda

# uv run --project envs/ort-gpu --no-sync \
#     python scripts/benchmarking/batch_sweep.py \
#     --runner scripts/benchmarking/inference.py \
#     --onnx-model ./test1/model_fused_fp16.onnx \
#     --data-dir ~/ \
#     --dataset cms_pf_ttbar \
#     --num-events 500 \
#     --batch-sizes 1 2 4 8 16 32 \
#     --warmup-runs 2 \
#     --benchmark-repeats 3 \
#     --continue-on-error \
#     --pad-bin-size 8 \
#     --outdir ./batch_8

srun uv run --project envs/ort-gpu --no-sync \
    python scripts/benchmarking/tentative_script.py \
    --checkpoint ~/latest-cms/onnx_ref_pyg-cms-v1_cms_run3_20260930_105918_812327/checkpoints/checkpoint-500.pth \
    --dataset cms_pf_ttbar \
    --model-kwargs ~/latest-cms/onnx_ref_pyg-cms-v1_cms_run3_20260930_105918_812327/model_kwargs.pkl \
    --data-dir ~/ \
    --outdir test1 \
    --num-events 10 \
    --device cuda