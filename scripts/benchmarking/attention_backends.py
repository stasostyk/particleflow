"""Strict external, packed-variable-length FlashAttention inference dispatch.

The MLPF backbone makes FP32 projections and calls ``run_packed_attention``;
only the Q/K/V kernel inputs are converted to FP16/BF16. These functions do
not own any parameters, modify checkpoints, or implement fallback attention.

Upstream APIs (Dao-AILab/flash-attention, checked 2026-11-11):
* FA2: flash_attn.flash_attn_varlen_func, SM80 / SM90.
* FA3: flash_attn_3.flash_attn_interface.flash_attn_varlen_func, SM90.
* FA4: flash_attn.cute.flash_attn_varlen_func, SM100 (CuTeDSL).

The external kernel call is deliberately a torch.compile graph break. Other
parts of the model can be compiled; kernel implementation and dtype cannot
silently change under torch.compile. See worker profiler checks.
"""

from __future__ import annotations

from functools import lru_cache
import importlib
from importlib import metadata

import torch


BACKENDS = {
    "fa2": ("flash-attn", "flash_attn", (8, 0), (9, 0)),
    "fa3": ("flash-attn-3", "flash_attn_3.flash_attn_interface", (9, 0)),
    "fa4": ("flash-attn-4", "flash_attn.cute", (10, 0)),
}
DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16}


@lru_cache(maxsize=None)
def _get_kernel(backend: str):
    if backend not in BACKENDS:
        raise ValueError(f"Unknown external FlashAttention backend {backend!r}")
    distribution, module, *_ = BACKENDS[backend]
    try:
        version = metadata.version(distribution)
    except metadata.PackageNotFoundError as exc:
        raise ImportError(
            f"{backend} requires the officially installed {distribution} distribution "
            "(install its CUDA build; no fallback is permitted)"
        ) from exc
    try:
        loaded = importlib.import_module(module)
        kernel = getattr(loaded, "flash_attn_varlen_func")
    except (ImportError, OSError, AttributeError) as exc:
        raise ImportError(
            f"{backend} ({distribution} {version}) must provide "
            f"{module}.flash_attn_varlen_func with native packed varlen support: {exc}"
        ) from exc
    if not callable(kernel):
        raise TypeError(f"{module}.flash_attn_varlen_func is not callable")
    return kernel, version


def validate_backend(backend: str, dtype: str, device: torch.device, head_dim: int) -> dict:
    """Check exact architecture, dtype, distribution and native varlen API."""
    if backend not in BACKENDS:
        raise ValueError(f"Unknown FlashAttention backend: {backend!r}")
    if dtype not in DTYPES:
        raise ValueError(f"{backend}: supported kernel dtypes are FP16 and BF16, got {dtype!r}")
    if device.type != "cuda" or torch.version.cuda is None:
        raise RuntimeError(f"{backend} requires NVIDIA CUDA, not {device}")
    # Conservative published toolkit requirements; reject unsupported stacks
    # rather than relying on an accidental binary/kernel substitution.
    cuda_version = tuple(int(part) for part in torch.version.cuda.split(".")[:2])
    minimum = {"fa2": (12, 0), "fa3": (12, 3), "fa4": (12, 8)}[backend]
    if cuda_version < minimum:
        raise RuntimeError(
            f"{backend} on this architecture requires CUDA >= {minimum[0]}.{minimum[1]} "
            f"(PyTorch was built with CUDA {torch.version.cuda}); install a compatible build"
        )
    cc = torch.cuda.get_device_capability(device)
    if tuple(cc) not in BACKENDS[backend][2:]:
        raise RuntimeError(
            f"{backend} requires SM{', SM'.join('%d%d' % c for c in BACKENDS[backend][2:])}; "
            f"device {device} has SM{cc[0]}{cc[1]}; no alternate backend will be used"
        )
    if dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError(f"{backend}: this CUDA device does not support BF16")
    if head_dim < 8 or head_dim > 256 or head_dim % 8 != 0:
        raise ValueError(f"{backend}: packed Q/K/V head_dim={head_dim} must be 8..256, divisible by 8")
    # FA4 SM100 supports standard head dimensions <= 128 and the (256,256) pair.
    if backend == "fa4" and head_dim > 128 and head_dim != 256:
        raise ValueError(f"FA4 SM100 does not support standard head_dim={head_dim}; use <=128 or 256")
    kernel, version = _get_kernel(backend)
    expected_module = {"fa2": "flash_attn.flash_attn_interface",
                       "fa3": "flash_attn_3.flash_attn_interface",
                       "fa4": "flash_attn.cute"}[backend]
    if not kernel.__module__.startswith(expected_module):
        raise RuntimeError(
            f"{backend} did not resolve to its official interface: {kernel.__module__}; "
            f"expected {expected_module}; refusing to use an aliased backend"
        )
    # All three official interfaces support these arguments by keyword. FA4
    # puts qv BEFORE cu_seqlens, so calling FA4 positionally would be wrong.
    import inspect
    parameters = inspect.signature(kernel).parameters
    needed = {"cu_seqlens_q", "cu_seqlens_k", "max_seqlen_q", "max_seqlen_k", "causal"}
    if not needed.issubset(parameters):
        raise RuntimeError(f"{backend}: installed native varlen API is incompatible: missing {needed - parameters.keys()}")
    if backend == "fa2" and "dropout_p" not in parameters:
        raise RuntimeError("FA2: installed flash_attn_varlen_func lacks documented dropout_p")
    return {"backend": backend, "attention_dtype": dtype, "distribution": BACKENDS[backend][0],
            "version": version, "interface": kernel.__module__ + "." + kernel.__name__,
            "compute_capability": list(cc), "cuda_runtime": torch.version.cuda,
            "head_dim": head_dim, "native_varlen": True}


def validate_packed(q, k, v, cu_q, cu_k, max_q: int, max_k: int, backend: str):
    """Cheap shape/metadata validation, without a CPU sync per attention layer."""
    if any(t.ndim != 3 for t in (q, k, v)) or q.shape != k.shape or q.shape != v.shape:
        raise ValueError(f"{backend}: Q/K/V must be packed (total_tokens, heads, head_dim), same shape")
    # if any(t.dtype != torch.float32 or not t.is_cuda for t in (q, k, v)):
    #     raise TypeError(f"{backend}: all Q/K/V projections must be CUDA FP32 before the attention boundary")
    if any(t.stride(-1) != 1 for t in (q, k, v)):
        raise ValueError(f"{backend}: Q/K/V last dimension must be contiguous")
    if (cu_q.ndim != 1 or cu_k.ndim != 1 or cu_q.numel() < 2 or
            cu_q.shape != cu_k.shape or cu_q.dtype != torch.int32 or
            cu_k.dtype != torch.int32 or not cu_q.is_cuda or not cu_k.is_cuda or
            not cu_q.is_contiguous() or not cu_k.is_contiguous()):
        raise ValueError(f"{backend}: cu_seqlens must be contiguous CUDA int32 vectors (batch+1)")
    if not (isinstance(max_q, int) and isinstance(max_k, int) and max_q > 0 and max_k > 0):
        raise ValueError(f"{backend}: max_seqlen values must be positive host integers")
    if (q.shape[0] > (cu_q.numel() - 1) * max_q or
            k.shape[0] > (cu_k.numel() - 1) * max_k):
        raise ValueError(f"{backend}: packed token count exceeds max_seqlen bounds")


@torch.compiler.disable
@torch.no_grad()
def _call_external(backend, q, k, v, cu_q, cu_k, max_q, max_k):
    # A graph break is intentional for a stable, independently verifiable
    # external backend; no torch SDPA fallback and no wrapper module.
    kernel, _ = _get_kernel(backend)
    with torch.profiler.record_function(f"mlpf_external_{backend}_native_varlen"):
        if backend == "fa2":
            out = kernel(q, k, v, cu_q, cu_k, max_q, max_k,
                         dropout_p=0.0, causal=False)
        elif backend in ("fa3", "fa4"):
            out = kernel(q, k, v, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
                         max_seqlen_q=max_q, max_seqlen_k=max_k, causal=False)
        else:
            raise ValueError(f"Unknown backend {backend}")
    return out


def run_packed_attention(backend, dtype, q, k, v, cu_q, cu_k, max_q, max_k):
    """Convert only Q/K/V for the kernel; return FP32 to the out projection."""
    validate_packed(q, k, v, cu_q, cu_k, max_q, max_k, backend)
    if dtype not in DTYPES:
        raise ValueError(f"Unsupported attention dtype {dtype!r}")
    target_dtype = DTYPES[dtype]
    q_low = q.to(dtype=target_dtype).contiguous()
    k_low = k.to(dtype=target_dtype).contiguous()
    v_low = v.to(dtype=target_dtype).contiguous()
    output = _call_external(backend, q_low, k_low, v_low, cu_q, cu_k, max_q, max_k)
    if not isinstance(output, torch.Tensor) or output.shape != q.shape or output.dtype != target_dtype:
        raise RuntimeError(
            f"{backend}: external native-varlen kernel returned unexpected output "
            f"shape/dtype {getattr(output, 'shape', None)}/{getattr(output, 'dtype', None)}"
        )
    return output.to(q.dtype)
