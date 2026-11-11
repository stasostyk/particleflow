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

# ncu \
#     -f \
#     --target-processes all \
#     --nvtx \
#     --nvtx-include "fwd_pass/" \
#     --kernel-name "regex:flash" \
#     --section SchedulerStats \
#     --section WarpStateStats \
#     -o extensive_batch_4 \
#         uv run --project envs/ort-gpu --no-sync python scripts/benchmarking/benchmark.py \
#             --checkpoint $CKPT \
#             --data-dir $DATA_DIR \
#             --dataset small \
#             --num-events 40 \
#             --outdir test_small1 \
#             --model-kwargs $KWRGS \
#             --allow-partial-state \
#             --warmup-iters 1 \
#             --pdf \
#             --repeats 2 \
#             --batch-sizes 4\
#             --pad-bin-size 8 \
#             --compile-mode default \
#             --warmup-seq-len 15000 \
#             --variants fp16_flash_compiled \
#             --compile-dynamic true \
#             --no-profile \
#             --reuse-reference small-small

uv run --project envs/ort-gpu --no-sync python - <<'PY'
import torch
from torch.nn.attention import SDPBackend

print("PyTorch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("cuDNN:", torch.backends.cudnn.version())
print("cuDNN backend enum:", hasattr(SDPBackend, "CUDNN_ATTENTION"))

try:
    from torch.nn.attention.varlen import varlen_attn
    print("PyTorch varlen_attn: available")
except ImportError:
    print("PyTorch varlen_attn: unavailable")
PY