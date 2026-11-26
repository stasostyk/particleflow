#!/bin/bash

#SBATCH --job-name=mlpf_fa3
#SBATCH --partition=gpu
#SBATCH --account=ehpc-ben-2026b12-038-gpu
#SBATCH --time=02:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --output=logs_slurm/log_%x_%j.out
#SBATCH --error=logs_slurm/log_%x_%j.err

set -euo pipefail

module purge

module load GPU/buildenv-gcccuda/2026.03-cu13.0
module load GPU/Python/3.13.5-bare-gcc-2025b-eb

export PATH="$HOME/.local/bin-arm64:$PATH"

echo "Node: $(hostname)"
echo "Python: $(which python)"
python --version
nvidia-smi

deactivate 2>/dev/null || true

unset VIRTUAL_ENV
unset CONDA_PREFIX
unset UV_PROJECT_ENVIRONMENT
unset UV_PYTHON
export UV_PROJECT_ENVIRONMENT="$PWD/envs/ort-gh200/.venv"
export UV_PYTHON_DOWNLOADS=never

export TORCH_CUDA_ARCH_LIST="9.0"
export MAX_JOBS=4
export UV_PYTHON_DOWNLOADS=never

# uv sync \
#     --project envs/ort-gh200 \
#     --python "$(which python)"

# Avoid accidental Python/user-site contamination
export PYTHONNOUSERSITE=1

# Restrict locally compiled CUDA code to Hopper
export TORCH_CUDA_ARCH_LIST="9.0a"


CKPT="/nobackup/proj/disk/ehpc-ben-2026b12-038/shared/colliderml/checkpoint-40000.pth"

KWRGS="/nobackup/proj/disk/ehpc-ben-2026b12-038/shared/colliderml/model_kwargs.pkl"

DATA_DIR="/nobackup/proj/disk/ehpc-ben-2026b12-038/shared/"

uv run --project envs/ort-gh200 --no-sync python scripts/benchmarking/input_size.py \
    --checkpoint $CKPT \
    --model-kwargs $KWRGS \
    --pad-bin-size 128 \
    --reference fp32_efficient \
    --output-pdf input_size_fixed5.pdf \
    --num-events 3000 \
    --load-reference /nobackup/proj/disk/ehpc-ben-2026b12-038/shared/colliderml/references