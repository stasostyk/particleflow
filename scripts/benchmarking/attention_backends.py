"""Strict external, packed-variable-length FlashAttention inference dispatch.

The MLPF backbone makes FP32 projections and calls ``run_packed_attention``;
only the Q/K/V kernel inputs are converted to FP16/BF16/FP8 (FP8: FA3 only,
optionally with static calibrated scales and a folded Hadamard rotation). These functions do
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
DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float8_e4m3fn": torch.float8_e4m3fn}
FP8_DTYPE = "float8_e4m3fn"
FP8_MAX = 448.0  # Largest finite float8_e4m3fn value
FP8_BACKENDS = ("fa3",)  # Only FA3 is wired up with FP8 descale factors here
# Keyword names FA3 builds have used for the FP8 descale factors (q, k, v).
FP8_DESCALE_NAMES = (("q_descale", "k_descale", "v_descale"),
                     ("descale_q", "descale_k", "descale_v"))


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


def _module_owned_by(distribution: str, module_name: str):
    """True/False if the module file is (not) listed in the distribution's
    installed files; None if the distribution records no file list (e.g. some
    editable installs), in which case only the module-name check applies."""
    import sys
    from pathlib import Path
    module = sys.modules.get(module_name) or importlib.import_module(module_name)
    path = getattr(module, "__file__", None)
    files = metadata.files(distribution)
    if not path or not files:
        return None
    target = Path(path).resolve()
    return any(Path(f.locate()).resolve() == target for f in files)


def validate_backend(backend: str, dtype: str, device: torch.device, head_dim: int) -> dict:
    """Check exact architecture, dtype, distribution and native varlen API."""
    if backend not in BACKENDS:
        raise ValueError(f"Unknown FlashAttention backend: {backend!r}")
    if dtype not in DTYPES:
        raise ValueError(f"{backend}: supported kernel dtypes are {list(DTYPES)}, got {dtype!r}")
    if dtype == FP8_DTYPE and backend not in FP8_BACKENDS:
        raise ValueError(f"{backend}: FP8 attention is only supported for {FP8_BACKENDS}")
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
    # FA3 builds from the hopper/ directory install a top-level
    # ``flash_attn_interface`` module; ``flash_attn_3.flash_attn_interface``
    # re-exports it. Both are official, as long as the file belongs to the
    # flash-attn-3 distribution (checked below), not to another package.
    expected_modules = {"fa2": ("flash_attn.flash_attn_interface",),
                        "fa3": ("flash_attn_3.flash_attn_interface", "flash_attn_interface"),
                        "fa4": ("flash_attn.cute",)}[backend]
    if not any(kernel.__module__ == m or kernel.__module__.startswith(m + ".")
               for m in expected_modules):
        raise RuntimeError(
            f"{backend} did not resolve to its official interface: {kernel.__module__}; "
            f"expected one of {expected_modules}; refusing to use an aliased backend"
        )
    owner = _module_owned_by(BACKENDS[backend][0], kernel.__module__)
    if owner is False:
        raise RuntimeError(
            f"{backend}: {kernel.__module__} is not a file of the installed "
            f"{BACKENDS[backend][0]} distribution; refusing to use an aliased backend"
        )
    # All three official interfaces support these arguments by keyword. FA4
    # puts qv BEFORE cu_seqlens, so calling FA4 positionally would be wrong.
    import inspect
    parameters = inspect.signature(kernel).parameters
    needed = {"cu_seqlens_q", "cu_seqlens_k", "max_seqlen_q", "max_seqlen_k", "causal"}
    if not needed.issubset(parameters):
        raise RuntimeError(f"{backend}: installed native varlen API is incompatible: missing {needed - parameters.keys()}")
    if dtype == FP8_DTYPE:
        _fp8_descale_names(backend)
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


@lru_cache(maxsize=None)
def _fp8_descale_names(backend: str) -> tuple[str, str, str]:
    """Find which descale keyword names the installed kernel accepts."""
    import inspect
    kernel, version = _get_kernel(backend)
    parameters = inspect.signature(kernel).parameters
    for names in FP8_DESCALE_NAMES:
        if all(name in parameters for name in names):
            return names
    raise RuntimeError(
        f"{backend} ({BACKENDS[backend][0]} {version}): {kernel.__module__}."
        f"{kernel.__name__} accepts none of the FP8 descale arguments "
        f"{[n for group in FP8_DESCALE_NAMES for n in group]}; this build cannot "
        "run scaled FP8 attention. Available arguments: " + ", ".join(parameters)
    )


# ---------------------------------------------------------------------------
# FP8 attention: static (calibrated) per-layer, per-head scales + optional
# Hadamard rotation folded into the projection weights. Nothing here depends on
# the data at run time, so it stays inside torch.compile and fuses with the Q/K/V
# bias add; results never depend on which other events share a batch.
# ---------------------------------------------------------------------------
class Fp8Calibration:
    """Placed on a layer during calibration: records per-head |max| of Q, K, V."""

    def __init__(self):
        self.amax = None  # (3, heads): rows q, k, v

    def observe(self, q, k, v):
        current = torch.stack([t.detach().float().abs().amax(dim=(0, 2)) for t in (q, k, v)])
        self.amax = current if self.amax is None else torch.maximum(self.amax, current)


def fp8_attention_layers(model) -> list:
    """Attention modules configured for FP8 (after configure_backbone_attention)."""
    return [m for m in model.modules()
            if getattr(m, "attention_kernel_dtype", None) == FP8_DTYPE]


def _orthogonal_matrix(dim: int, seed: int) -> torch.Tensor:
    """Randomized Hadamard (power-of-two dim) or random orthogonal matrix, FP64."""
    generator = torch.Generator().manual_seed(seed)
    if dim & (dim - 1) == 0:
        hadamard = torch.ones((1, 1), dtype=torch.float64)
        while hadamard.shape[0] < dim:
            hadamard = torch.cat([torch.cat([hadamard, hadamard], 1),
                                  torch.cat([hadamard, -hadamard], 1)], 0)
        signs = torch.randint(0, 2, (dim,), generator=generator).to(torch.float64) * 2 - 1
        return hadamard * signs / dim ** 0.5
    q, r = torch.linalg.qr(torch.randn((dim, dim), generator=generator, dtype=torch.float64))
    return q * torch.sign(torch.diagonal(r))


@torch.no_grad()
def apply_hadamard_rotation(model, seed: int = 0) -> int:
    """Fold a per-head orthogonal rotation R into every FP8 attention layer.

    q, k, v of each head become qR, kR, vR (so QK^T and softmax are unchanged)
    and out_proj absorbs R^T, so the layer computes the same function in exact
    arithmetic. Outlier channels get spread across the head dimension, which
    makes per-head FP8 scales much tighter. Zero runtime cost: weights only.
    """
    layers = fp8_attention_layers(model)
    for mha in layers:
        heads, dim, embed = mha.num_heads, mha.head_dim, mha.embed_dim
        rotation = _orthogonal_matrix(dim, seed).to(mha.in_proj_weight.device)
        weight = mha.in_proj_weight.double().view(3, heads, dim, embed)
        bias = mha.in_proj_bias.double().view(3, heads, dim)
        weight = torch.einsum("ji,shjE->shiE", rotation, weight)  # W_h' = R^T W_h
        bias = bias @ rotation                                      # b_h' = b_h R
        out_weight = mha.out_proj.weight.double().view(embed, heads, dim)
        out_weight = out_weight @ rotation                          # C_h' = C_h R
        mha.in_proj_weight.copy_(weight.reshape(3 * embed, embed).to(mha.in_proj_weight.dtype))
        mha.in_proj_bias.copy_(bias.reshape(3 * embed).to(mha.in_proj_bias.dtype))
        mha.out_proj.weight.copy_(out_weight.reshape(embed, embed).to(mha.out_proj.weight.dtype))
    return len(layers)


@torch.no_grad()
def calibrate_fp8_scales(model, run_calibration, margin: float = 1.0) -> dict:
    """Run `run_calibration()` (eager forwards) and freeze static FP8 scales.

    Each FP8 layer gets one scale per head for Q, K and V, mapping the largest
    observed |value| (times `margin`) to FP8_MAX. Larger unseen values are
    clamped, never NaN. Stored on the module as plain tensors (no state_dict keys).
    """
    layers = fp8_attention_layers(model)
    if not layers:
        raise RuntimeError("No FP8 attention layers to calibrate")
    recorders = []
    for mha in layers:
        mha.attention_fp8_scales = Fp8Calibration()
        recorders.append(mha.attention_fp8_scales)
    try:
        run_calibration()
    finally:
        for mha in layers:
            mha.attention_fp8_scales = None
    summary = []
    for mha, recorder in zip(layers, recorders):
        if recorder.amax is None:
            raise RuntimeError("Calibration never reached an FP8 attention layer")
        amax = recorder.amax.clamp_min(1e-12) * margin
        scale = FP8_MAX / amax  # (3, heads)
        mha.attention_fp8_scales = (
            tuple(scale[i].view(1, -1, 1).contiguous() for i in range(3))
            + tuple((1.0 / scale[i]).view(1, -1).contiguous() for i in range(3))
        )
        summary.append({name: recorder.amax[i].cpu().tolist() for i, name in enumerate("qkv")})
    return {"layers": len(layers), "margin": margin, "amax_by_layer": summary}


def _fp8_cast(t: torch.Tensor) -> torch.Tensor:
    return t.clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).contiguous()


@torch.compiler.disable
@torch.no_grad()
def _call_external(backend, q, k, v, cu_q, cu_k, max_q, max_k, descales=None):
    # A graph break is intentional for a stable, independently verifiable
    # external backend; no torch SDPA fallback and no wrapper module.
    kernel, _ = _get_kernel(backend)
    with torch.profiler.record_function(f"mlpf_external_{backend}_native_varlen"):
        if backend == "fa2":
            out = kernel(q, k, v, cu_q, cu_k, max_q, max_k,
                         dropout_p=0.0, causal=False)
        elif backend in ("fa3", "fa4"):
            extra = {}
            if descales is not None:
                if backend not in FP8_BACKENDS:
                    raise ValueError(f"{backend}: FP8 descale factors are not supported")
                extra = dict(zip(_fp8_descale_names(backend),
                                 (d.contiguous() for d in descales)))
            out = kernel(q, k, v, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
                         max_seqlen_q=max_q, max_seqlen_k=max_k, causal=False, **extra)
        else:
            raise ValueError(f"Unknown backend {backend}")
    return out


def run_packed_attention(backend, dtype, q, k, v, cu_q, cu_k, max_q, max_k, fp8_scales=None):
    """Convert only Q/K/V for the kernel; return the input dtype to the out projection.

    FP8: `fp8_scales` is None (plain cast, clamped to the FP8 range), the static
    (q_scale, k_scale, v_scale, q_descale, k_descale, v_descale) tuple from
    calibrate_fp8_scales, or an Fp8Calibration recorder during calibration.
    """
    validate_packed(q, k, v, cu_q, cu_k, max_q, max_k, backend)
    if dtype not in DTYPES:
        raise ValueError(f"Unsupported attention dtype {dtype!r}")
    target_dtype = DTYPES[dtype]
    descales = None
    if dtype == FP8_DTYPE:
        if backend not in FP8_BACKENDS:
            raise ValueError(f"{backend}: FP8 attention is only supported for {FP8_BACKENDS}")
        if isinstance(fp8_scales, Fp8Calibration):
            fp8_scales.observe(q, k, v)
            fp8_scales = None
        if fp8_scales is None:
            q_low, k_low, v_low = _fp8_cast(q), _fp8_cast(k), _fp8_cast(v)
        else:
            q_scale, k_scale, v_scale, q_descale, k_descale, v_descale = fp8_scales
            q_low, k_low, v_low = _fp8_cast(q * q_scale), _fp8_cast(k * k_scale), _fp8_cast(v * v_scale)
            batch = cu_q.numel() - 1
            descales = tuple(d.expand(batch, -1) for d in (q_descale, k_descale, v_descale))
    else:
        q_low = q.to(dtype=target_dtype).contiguous()
        k_low = k.to(dtype=target_dtype).contiguous()
        v_low = v.to(dtype=target_dtype).contiguous()
    output = _call_external(backend, q_low, k_low, v_low, cu_q, cu_k, max_q, max_k, descales)
    return output.to(q.dtype)