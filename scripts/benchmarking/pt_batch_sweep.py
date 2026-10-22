#!/usr/bin/env python3
"""Benchmark MLPF PyTorch inference on an NVIDIA GPU (no ONNX).

Two timing scopes are measured separately:
  e2e:      pinned CPU tensors -> CUDA copies -> forward -> CUDA synchronization
  gpu_only: inputs already on CUDA -> forward (CUDA event elapsed time)

All timed inferences run with torch.inference_mode(). Warmup and FlashAttention
verification use SYNTHETIC inputs only. Dataset I/O, CPU preprocessing,
compilation, warmup, and numeric validation are excluded from measurements.

Run from the MLPF repository root so `mlpf` can be imported. Dependencies:
  torch (CUDA build), numpy, scipy, tensorflow-datasets, matplotlib (for --pdf),
  and the dependencies required by the MLPF repository.

To add a precision/attention variant, append a Variant to VARIANTS below.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import pickle
import sys
import time
from typing import Any

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel


@dataclass(frozen=True)
class Variant:
    """Benchmark configuration. `mode` is 'native' or 'autocast'."""

    name: str
    attention: str                 # 'math', 'flash', or 'efficient'
    dtype: torch.dtype            # torch.float32, torch.float16, torch.bfloat16
    mode: str = "native"           # native: convert model/input; autocast: FP32 model/input
    jagged: bool = False            # MLPF internal packed/jagged attention path
    varlen: bool = False            # Require flash-attn variable-length kernels

    @property
    def input_dtype(self) -> torch.dtype:
        return torch.float32 if self.mode == "autocast" else self.dtype


# Accuracy reference first; remaining variants are timing sweep candidates:
REFERENCE_NAME = "fp32_efficient"

VARIANTS = (
    # Strict FP32 mathematical attention: accuracy reference only (B=1, untimed).
    # Variant(REFERENCE_NAME, "math", torch.float32),
    Variant("fp16_flash", "flash", torch.float16, jagged=True, varlen=True),
    Variant("bf16_flash", "flash", torch.bfloat16, jagged=True, varlen=True),
    # Optional FP32 memory-efficient SDPA comparison (no custom varlen path).
    Variant("fp32_efficient", "efficient", torch.float32, jagged=True, varlen=True),
    # Optional comparison matching the original script's autocast approach:
    Variant("fp16_flash_autocast", "flash", torch.float16, mode="autocast", jagged=True, varlen=True),
    Variant("bf16_flash_autocast", "flash", torch.bfloat16, mode="autocast", jagged=True, varlen=True),
)

HEADS = {"id": 1, "momentum": 2, "pu": 3}  # Same output indices as original.
MODES = ("e2e", "gpu_only")
MIB = 1024.0 * 1024.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--model-kwargs", type=Path, required=True,
                   help="Pickled MLPFConfig or dict, as in the original script")
    p.add_argument("--dataset", default="cms_pf_ttbar")
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--split", default="train")
    p.add_argument("--outdir", type=Path, default=Path("./pt_inference_benchmark"))
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8],
                   help="Exact batch sizes to sweep")
    p.add_argument("--num-events", type=int, default=64,
                   help="Number of dataset events cached on CPU; incomplete final batches discarded")
    p.add_argument("--repeats", type=int, default=20,
                   help="Independent sweeps per timing scope (>=2 for a meaningful CI)")
    p.add_argument("--warmup-iters", type=int, default=5,
                   help="Synthetic forwards once per model, at batch size 1")
    p.add_argument("--baseline-cache-clear-every", type=int, default=1,
                   help="Release unused CUDA allocator cache after every N reference events (0 disables)")
    p.add_argument("--baseline-memory-log-every", type=int, default=25,
                   help="Print reference CUDA memory before/after cache cleanup every N events (0 disables)")
    p.add_argument("--warmup-seq-len", type=int, default=256,
                   help="Synthetic warmup sequence length before required padding; independent of dataset")
    p.add_argument("--pad-bin-size", type=int, default=0)
    p.add_argument("--fixed-seq-len", type=int, default=None,
                   help="Optional fixed padded sequence length; fail rather than truncate an event")
    p.add_argument("--device", default="cuda:0", help="NVIDIA CUDA device, e.g. cuda:0")
    p.add_argument("--variants", nargs="+", default=[REFERENCE_NAME, "fp16_flash"],
                   choices=[v.name for v in VARIANTS])
    p.add_argument("--compile", action="store_true",
                   help="Enable torch.compile after verifying eager-mode FlashAttention")
    p.add_argument("--compile-mode", default="default",
                   choices=["default", "reduce-overhead", "max-autotune"])
    p.add_argument("--no-pin-memory", action="store_true",
                   help="Disable pinned host buffers / asynchronous CUDA copies")
    p.add_argument("--allow-partial-state", action="store_true",
                   help="Use strict=False on checkpoint loading; normally prefer strict=True")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pdf", action="store_true",
                   help="Also produce a SINGLE-PAGE comparison.pdf")
    args = p.parse_args()
    if (len(set(args.batch_sizes)) != len(args.batch_sizes)
            or any(b <= 0 for b in args.batch_sizes)
            or args.num_events < max(args.batch_sizes)
            or args.repeats < 2 or args.warmup_iters < 1
            or args.pad_bin_size < 0 or args.warmup_seq_len <= 0
            or args.baseline_cache_clear_every < 0 or args.baseline_memory_log_every < 0
            or (args.fixed_seq_len is not None and args.fixed_seq_len <= 0)):
        p.error("Invalid batch sizes, event count, repeats, warmup, or padding")
    if REFERENCE_NAME not in args.variants:
        p.error(f"{REFERENCE_NAME} must be included as the accuracy reference")
    args.batch_sizes = sorted(args.batch_sizes)
    if not args.device.startswith("cuda"):
        p.error("NVIDIA CUDA is required")
    return args


def select_variants(names: list[str]) -> list[Variant]:
    registry = {v.name: v for v in VARIANTS}
    ordered = list(dict.fromkeys(names))
    ordered.remove(REFERENCE_NAME)
    return [registry[REFERENCE_NAME]] + [registry[name] for name in ordered]


def load_config_and_state(args: argparse.Namespace) -> tuple[Any, dict, int]:
    # Module imports are intentionally deferred, so --help is usable on systems
    # without the MLPF dependencies installed.
    from mlpf.conf import MLPFConfig, ModelType

    with args.model_kwargs.open("rb") as f:
        config_obj = pickle.load(f)  # Trusted local project artifact ONLY.
    raw = config_obj if isinstance(config_obj, dict) else config_obj.model_dump()
    config = MLPFConfig.model_validate(raw)
    if config.model.type != ModelType.ATTENTION:
        raise ValueError("These variants require MLPF ModelType.ATTENTION")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    state = checkpoint.get("model_state_dict", checkpoint)
    if not isinstance(state, dict):
        raise TypeError("Checkpoint does not contain a model state_dict")
    pad_multiple = max(args.pad_bin_size, config.pad_to_multiple_elements or 0)
    if args.fixed_seq_len is not None and pad_multiple:
        if args.fixed_seq_len % pad_multiple:
            raise ValueError("--fixed-seq-len must be divisible by required pad multiple")
    return config, state, pad_multiple


def build_model(config: Any, state: dict, variant: Variant,
                device: torch.device, partial: bool) -> torch.nn.Module:
    from mlpf.conf import AttentionType
    from mlpf.model.mlpf import MLPF

    cfg = config.model_copy(deep=True)
    # EFFICIENT_ATTENTION selects a PyTorch SDPA kernel, so use MLPF's SIMPLE
    # (SDPA-based) implementation rather than its custom flash-attn path.
    attention_types = {
        "math": AttentionType.MATH,
        "flash": AttentionType.FLASH,
        "efficient": AttentionType.SIMPLE,
    }
    cfg.model.attention.attention_type = attention_types[variant.attention]
    cfg.model.attention.export_onnx_fused = False
    cfg.model.attention.save_attention = False
    # MLPF itself packs valid elements for jagged attention. Keep the padded
    # [B,L,F] input and boolean mask: its model forward owns the conversion.
    # FP32 math reference and FP32 efficient SDPA use non-jagged paths.
    if not hasattr(cfg.model.attention, "use_jagged_attention"):
        if variant.jagged:
            raise RuntimeError(
                "This MLPF checkout does not expose attention.use_jagged_attention; "
                "update MLPF to a version with the jagged attention path."
            )
    else:
        cfg.model.attention.use_jagged_attention = variant.jagged
    # This is a separate MLPF switch from use_jagged_attention.  A FLASH-only
    # SDPA context does not guarantee a packed variable-length flash-attn call.
    # Fail explicitly if an older MLPF configuration lacks the requested field.
    if variant.varlen and not variant.jagged:
        raise ValueError(f"{variant.name}: varlen requires jagged attention")
    if variant.attention == "flash":
        if not hasattr(cfg.model.attention, "use_flash_attn_varlen"):
            raise RuntimeError(
                f"{variant.name}: MLPF config lacks attention.use_flash_attn_varlen. "
                "Install an MLPF implementation that supports the varlen path."
            )
        cfg.model.attention.use_flash_attn_varlen = variant.varlen
    elif hasattr(cfg.model.attention, "use_flash_attn_varlen"):
        cfg.model.attention.use_flash_attn_varlen = False
    model = MLPF(config=cfg).eval()
    # Some training checkpoints serialize loss-calibration state that is not
    # part of the inference-only model. Discard ONLY these unexpected entries;
    # keep strict validation for all other weights and buffers.
    training_only = {
        "task_loss_weighter.weights",
        "task_loss_weighter.loss_sums",
        "task_loss_weighter.calibration_count",
        "task_loss_weighter.calibration_observations",
        "task_loss_weighter.calibrated",
    }
    expected = set(model.state_dict())
    ignored = sorted(key for key in state if key in training_only and key not in expected)
    inference_state = {key: value for key, value in state.items() if key not in ignored}
    if ignored:
        print(f"INFO: {variant.name}: ignoring training-only checkpoint keys: {ignored}")
    incompatible = model.load_state_dict(inference_state, strict=not partial)
    if partial and (incompatible.missing_keys or incompatible.unexpected_keys):
        print(f"WARNING: {variant.name}: missing={incompatible.missing_keys}, "
              f"unexpected={incompatible.unexpected_keys}")
    model = model.to(device=device)
    if variant.mode == "native":
        model = model.to(dtype=variant.dtype)
    return model


def attention_context(variant: Variant):
    """Prohibit silent SDPA fallback to a different backend."""
    backends = {
        "math": SDPBackend.MATH,
        "flash": SDPBackend.FLASH_ATTENTION,
        "efficient": SDPBackend.EFFICIENT_ATTENTION,
    }
    backend = backends[variant.attention]
    return sdpa_kernel(backend)


def autocast_context(variant: Variant):
    if variant.mode == "autocast":
        return torch.autocast(device_type="cuda", dtype=variant.dtype)
    return nullcontext()


def forward(model: torch.nn.Module, variant: Variant,
            features: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, ...]:
    with attention_context(variant), autocast_context(variant):
        out = model(features, mask)
    if not isinstance(out, (tuple, list)) or len(out) < 4:
        raise TypeError("Expected MLPF outputs (bid, id, momentum, pu)")
    return tuple(out)


def load_events(args: argparse.Namespace, input_dim: int) -> list[torch.Tensor]:
    import tensorflow_datasets as tfds

    source = tfds.builder(args.dataset, data_dir=str(args.data_dir)).as_data_source(
        split=args.split
    )
    if len(source) < args.num_events:
        raise ValueError(f"Dataset has only {len(source)} events; requested {args.num_events}")
    events = []
    for i in range(args.num_events):
        event = np.asarray(source[i]["X"], dtype=np.float32)
        if event.ndim != 2 or event.shape[1] != input_dim or event.shape[0] == 0:
            raise ValueError(f"Event {i}: expected (elements, {input_dim}), got {event.shape}")
        if not np.all(np.isfinite(event)):
            raise ValueError(f"Event {i} has non-finite input features")
        # Match the original script's FP16 overflow protection.
        events.append(torch.from_numpy(np.clip(event, -60000, 60000).copy()))
    return events


def make_batches(events: list[torch.Tensor], batch_size: int,
                 pad_multiple: int, fixed_seq_len: int | None) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """CPU FP32 batches and boolean padding masks, shared by all variants."""
    batches = []
    for start in range(0, len(events) - batch_size + 1, batch_size):
        subset = events[start:start + batch_size]
        lengths = [x.shape[0] for x in subset]
        seq = max(lengths)
        if fixed_seq_len is not None:
            if seq > fixed_seq_len:
                raise ValueError(f"Event length {seq} > --fixed-seq-len {fixed_seq_len}")
            seq = fixed_seq_len
        elif pad_multiple:
            seq = ((seq + pad_multiple - 1) // pad_multiple) * pad_multiple
        features = torch.zeros((batch_size, seq, subset[0].shape[1]), dtype=torch.float32)
        mask = torch.zeros((batch_size, seq), dtype=torch.bool)
        for j, item in enumerate(subset):
            features[j, :lengths[j], :] = item
            mask[j, :lengths[j]] = True
        batches.append((features, mask))
    return batches


def prepare_host_batches(
    batches: list[tuple[torch.Tensor, torch.Tensor]], variant: Variant, pin: bool
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Pre-convert/pin input before timing. GPU copies remain INSIDE e2e timing."""
    ready = []
    for features, mask in batches:
        x = features.to(dtype=variant.input_dtype).contiguous()
        m = mask.contiguous()
        if pin:
            x = x.pin_memory()
            m = m.pin_memory()
        ready.append((x, m))
    return ready


def to_cuda(batch: tuple[torch.Tensor, torch.Tensor],
            device: torch.device, pin: bool) -> tuple[torch.Tensor, torch.Tensor]:
    x, mask = batch
    return (x.to(device=device, non_blocking=pin),
            mask.to(device=device, non_blocking=pin))


def fake_cuda_batch(batch: tuple[torch.Tensor, torch.Tensor],
                    variant: Variant, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Match shape/dtype but NOT actual dataset features or mask values."""
    x, mask = batch
    fake_x = torch.randn(x.shape, device=device, dtype=variant.input_dtype)
    fake_mask = torch.zeros(mask.shape, device=device, dtype=torch.bool)
    # A synthetic, partly padded mask (not copied from the dataset).
    fake_mask[:, :max(1, int(mask.shape[1] * 0.85))] = True
    return fake_x, fake_mask


def verify_flash(model: torch.nn.Module, variant: Variant,
                 sample: tuple[torch.Tensor, torch.Tensor],
                 device: torch.device) -> list[str]:
    """Require a real flash-attn VARLEN operator for packed Flash variants.

    Use an untimed synthetic *unequal-length* B=3 batch, not the B=1 warmup:
    B=1 cannot verify segmented variable-length attention and is not sufficient
    evidence that the packed path is used for batches.  The CPU profiler sees
    custom flash-attn Python/autograd op names (as in MLPF's own test).
    """
    from torch.profiler import ProfilerActivity, profile

    if variant.varlen:
        _, source_mask = sample
        length = int(source_mask.shape[1])
        if length < 3:
            raise ValueError("Varlen verification requires synthetic sequence length >= 3")
        # Nonzero and nonuniform lengths.  Mask is synthetic, not drawn from data.
        lengths = (length, max(2, length // 2), 1)
        x = torch.randn((3, length, sample[0].shape[2]), device=device,
                        dtype=variant.input_dtype)
        m = torch.arange(length, device=device)[None, :] < torch.tensor(
            lengths, device=device
        )[:, None]
    else:
        x, m = fake_cuda_batch(sample, variant, device)

    with torch.inference_mode():
        # Untimed initial call to initialize backend / extension.
        forward(model, variant, x, m)
        torch.cuda.synchronize(device)
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            forward(model, variant, x, m)
            torch.cuda.synchronize(device)
    keys = sorted({event.key for event in prof.key_averages()})
    if variant.varlen:
        # Strict requirement: generic dense SDPA/flash is NOT an acceptable match.
        found = [key for key in keys if "flash_attn_varlen" in key.lower()]
        if not found:
            raise RuntimeError(
                f"{variant.name}: variable-length FlashAttention NOT observed on "
                f"synthetic unequal-length B=3 batch (lengths={lengths}). "
                "Expected a profiler operator containing 'flash_attn_varlen'; "
                "dense FlashAttention is not accepted. Verify your MLPF version "
                "implements use_flash_attn_varlen and flash-attn is installed. "
                f"Observed flash-related operators: "
                f"{[key for key in keys if 'flash' in key.lower()][:20]}"
            )
    else:
        markers = ("_scaled_dot_product_flash_attention", "_flash_attention_forward",
                   "flash_attn", "flash_fwd")
        found = [key for key in keys if any(marker in key.lower() for marker in markers)]
        if not found:
            raise RuntimeError(
                f"{variant.name}: no FlashAttention operator observed. "
                f"Sample ops: {keys[:20]}"
            )
    del x, m
    return found


def verify_efficient(model: torch.nn.Module, variant: Variant,
                     sample: tuple[torch.Tensor, torch.Tensor],
                     device: torch.device) -> list[str]:
    """Verify FP32 efficient SDPA on synthetic inputs; never accept math fallback.

    This fails explicitly on GPUs/PyTorch builds that do not support efficient
    SDPA with the exact MLPF tensor shapes and mask semantics.
    """
    from torch.profiler import ProfilerActivity, profile

    x, mask = fake_cuda_batch(sample, variant, device)
    try:
        with torch.inference_mode():
            forward(model, variant, x, mask)
            torch.cuda.synchronize(device)
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                forward(model, variant, x, mask)
                torch.cuda.synchronize(device)
    except RuntimeError as exc:
        raise RuntimeError(
            f"{variant.name}: FP32 EFFICIENT_ATTENTION unavailable for this "
            "MLPF model/GPU/PyTorch build. No math fallback is allowed. "
            f"Original failure: {exc}"
        ) from exc
    finally:
        del x, mask
    keys = sorted({event.key for event in prof.key_averages()})
    found = [key for key in keys if "_efficient_attention_forward" in key.lower()]
    if not found:
        raise RuntimeError(
            f"{variant.name}: efficient SDPA operator was not observed; "
            "cannot validate FP32 efficient attention dispatch. "
            f"SDPA-related ops: {[k for k in keys if 'attention' in k.lower()][:30]}"
        )
    return found


def warmup(model: torch.nn.Module, variant: Variant,
           sample: tuple[torch.Tensor, torch.Tensor],
           device: torch.device, iterations: int) -> None:
    """One untimed synthetic warmup per model, always with batch size 1.

    Other batch sizes and sequence lengths are deliberately not pre-warmed.
    With torch.compile they may trigger lazy compilation during measurement.
    """
    x, m = fake_cuda_batch(sample, variant, device)
    with torch.inference_mode():
        for _ in range(iterations):
            y = forward(model, variant, x, m)
        torch.cuda.synchronize(device)
    del x, m, y


def clear_caches(device: torch.device) -> None:
    """Between repeat sweeps; never inside the measured inference region."""
    gc.collect()
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()  # Allocator cache, NOT GPU L2 cache or compiler cache.


def measure_one(model: torch.nn.Module, variant: Variant,
                batch: tuple[torch.Tensor, torch.Tensor], scope: str,
                device: torch.device, pin: bool) -> float:
    """Return milliseconds per batch. Runs one forward and discards outputs."""
    if scope == "e2e":
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        x, m = to_cuda(batch, device, pin)
        y = forward(model, variant, x, m)
        torch.cuda.synchronize(device)  # Includes async H2D and all GPU work.
        milliseconds = (time.perf_counter() - t0) * 1000.0
    else:
        x, m = to_cuda(batch, device, pin)
        torch.cuda.synchronize(device)  # Transfer is outside the GPU-only clock.
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        y = forward(model, variant, x, m)
        end.record()
        end.synchronize()
        milliseconds = begin.elapsed_time(end)
    del y, x, m
    return milliseconds


def benchmark_scope(model: torch.nn.Module, variant: Variant,
                    batches: list[tuple[torch.Tensor, torch.Tensor]], scope: str,
                    device: torch.device, pin: bool, repeats: int) -> dict[str, Any]:
    runs = []
    with torch.inference_mode():
        for repeat in range(repeats):
            clear_caches(device)
            torch.cuda.reset_peak_memory_stats(device)
            batch_ms = []
            for batch_index, batch in enumerate(batches):
                try:
                    batch_ms.append(measure_one(model, variant, batch, scope, device, pin))
                except torch.cuda.OutOfMemoryError:
                    free, total = torch.cuda.mem_get_info(device)
                    x, mask = batch
                    print(
                        f"OOM: variant={variant.name} mode={scope} repeat={repeat} "
                        f"batch_index={batch_index} shape={tuple(x.shape)} "
                        f"valid_elements_per_event={mask.sum(dim=1).tolist()} "
                        f"allocated_mib={torch.cuda.memory_allocated(device)/MIB:.0f} "
                        f"reserved_mib={torch.cuda.memory_reserved(device)/MIB:.0f} "
                        f"free_device_mib={free/MIB:.0f} total_device_mib={total/MIB:.0f}",
                        file=sys.stderr, flush=True,
                    )
                    raise
            torch.cuda.synchronize(device)
            peak_allocated = torch.cuda.max_memory_allocated(device) / MIB
            peak_reserved = torch.cuda.max_memory_reserved(device) / MIB
            n_events = sum(batch[0].shape[0] for batch in batches)
            seconds = sum(batch_ms) / 1000.0
            runs.append({
                "repeat": repeat,
                "batch_latencies_ms": batch_ms,
                "p95_latency_ms": float(np.percentile(batch_ms, 95)),
                "throughput_events_per_s": n_events / seconds,
                "total_timed_s": seconds,
                "events": n_events,
                "peak_allocated_mib": peak_allocated,
                "peak_reserved_mib": peak_reserved,
            })
    return {"repeats": runs}


def student_t_mean_ci(values: list[float]) -> dict[str, float]:
    """Two-sided 95% Student's t CI for the mean of independent repeat metrics.

    CI = sample_mean +/- t_{0.975, n-1} * sample_std / sqrt(n).
    The assumption concerns the repeat-level metric's mean, not individual
    batch latency quantiles. With fewer than two repeats the CI is undefined.
    """
    from scipy.stats import t  # Imported on demand so --help needs no SciPy.

    a = np.asarray(values, dtype=np.float64)
    n = len(a)
    if n < 2 or not np.isfinite(a).all():
        raise ValueError("A Student's t confidence interval needs >=2 finite repeats")
    mean = float(a.mean())
    std = float(a.std(ddof=1))
    sem = std / np.sqrt(n)
    critical = float(t.ppf(0.975, df=n - 1))
    half_width = critical * sem
    return {"mean": mean, "ci95_low": mean - half_width,
            "ci95_high": mean + half_width, "std": std,
            "sem": float(sem), "n": n, "df": n - 1,
            "t_critical": critical}


def summarize_scope(block: dict[str, Any]) -> None:
    metrics = ("p95_latency_ms", "throughput_events_per_s", "peak_allocated_mib",
               "peak_reserved_mib")
    block["summary"] = {
        name: student_t_mean_ci([r[name] for r in block["repeats"]])
        for name in metrics
    }


def reference_predictions(model: torch.nn.Module, variant: Variant,
                          batches: list[tuple[torch.Tensor, torch.Tensor]],
                          device: torch.device, pin: bool,
                          cache_clear_every: int = 1,
                          memory_log_every: int = 25) -> list[dict[str, torch.Tensor]]:
    """Untimed B=1 mixed-precision accuracy reference; optional cache cleanup.

    Reference outputs are copied to CPU for comparison; GPU outputs are freed.
    This reference uses BF16 varlen attention, not quadratic FP32 math attention.
    """
    reference = []
    with torch.inference_mode():
        for event_id, batch in enumerate(batches):
            assert batch[0].shape[0] == 1
            x, m = to_cuda(batch, device, pin)
            try:
                out = forward(model, variant, x, m)
                valid = batch[1][0]
                predictions = {
                    name: out[idx][0].detach().float().cpu()[valid].clone()
                    for name, idx in HEADS.items()
                }
                for name, tensor in predictions.items():
                    if not torch.isfinite(tensor).all():
                        raise ValueError(f"Non-finite reference outputs in {name}")
                reference.append(predictions)
            finally:
                # Delete live references BEFORE empty_cache(). A cache flush
                # cannot reclaim memory belonging to a live output or input.
                if "out" in locals():
                    del out
                del x, m

            log_now = (memory_log_every > 0 and
                       (event_id == 0 or (event_id + 1) % memory_log_every == 0))
            cleanup_now = (cache_clear_every > 0 and
                           (event_id + 1) % cache_clear_every == 0)
            if log_now or cleanup_now:
                torch.cuda.synchronize(device)
                allocated_before = torch.cuda.memory_allocated(device) / (1024 ** 3)
                reserved_before = torch.cuda.memory_reserved(device) / (1024 ** 3)
                if cleanup_now:
                    torch.cuda.empty_cache()
                if log_now:
                    allocated_after = torch.cuda.memory_allocated(device) / (1024 ** 3)
                    reserved_after = torch.cuda.memory_reserved(device) / (1024 ** 3)
                    free, total = torch.cuda.mem_get_info(device)
                    valid_len = int(batch[1][0].sum())
                    print(
                        f"[REF] event={event_id} valid_len={valid_len} "
                        f"padded_len={batch[0].shape[1]} "
                        f"allocated={allocated_before:.2f}->{allocated_after:.2f} GiB "
                        f"reserved={reserved_before:.2f}->{reserved_after:.2f} GiB "
                        f"gpu_free={free / (1024 ** 3):.2f} GiB",
                        flush=True,
                    )
        # Flush any final partial interval as well.
        if cache_clear_every > 0 and len(batches) % cache_clear_every:
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
    return reference


def mae_to_reference(model: torch.nn.Module, variant: Variant,
                     batches: list[tuple[torch.Tensor, torch.Tensor]],
                     reference: list[dict[str, torch.Tensor]],
                     batch_event_ids: list[list[int]],
                     device: torch.device, pin: bool) -> dict[str, Any]:
    """Compare each valid event to its B=1 FP32 math reference output."""
    sums = {name: 0.0 for name in HEADS}
    counts = {name: 0 for name in HEADS}
    invalid = {name: 0 for name in HEADS}
    events_compared = 0
    if len(batches) != len(batch_event_ids):
        raise ValueError("Batch/event-ID mapping length mismatch")
    with torch.inference_mode():
        for batch, ids in zip(batches, batch_event_ids):
            if len(ids) != batch[0].shape[0]:
                raise ValueError("Batch/event-ID mapping shape mismatch")
            x, m = to_cuda(batch, device, pin)
            pred = forward(model, variant, x, m)
            for j, event_index in enumerate(ids):
                valid = batch[1][j]
                ref = reference[event_index]
                for name, idx in HEADS.items():
                    actual = pred[idx][j].detach().float().cpu()[valid]
                    if actual.shape != ref[name].shape:
                        raise ValueError(
                            f"{name} shape mismatch for event {event_index}: "
                            f"prediction={tuple(actual.shape)}, reference={tuple(ref[name].shape)}"
                        )
                    difference = (actual - ref[name]).abs()
                    finite = torch.isfinite(difference)
                    invalid[name] += int((~finite).sum().item())
                    sums[name] += float(difference[finite].double().sum().item())
                    counts[name] += difference.numel()
                events_compared += 1
            del pred, x, m
    all_invalid = sum(invalid.values())
    valid_total = sum(counts.values())
    return {
        "mae": (sum(sums.values()) / valid_total if valid_total and not all_invalid else None),
        "mae_by_head": {name: (sums[name] / counts[name]
                               if counts[name] and not invalid[name] else None)
                        for name in HEADS},
        "invalid_by_head": invalid,
        "elements_compared": valid_total,
        "events_compared": events_compared,
    }


def print_summary(result: dict[str, Any], name: str, batch_size: int) -> None:
    e2e = result["timings"]["e2e"]["summary"]
    gpu = result["timings"]["gpu_only"]["summary"]
    fmt = lambda d: f'{d["mean"]:.3f} [{d["ci95_low"]:.3f}, {d["ci95_high"]:.3f}]'
    print(f"{name:13s} B={batch_size:<3d} | p95 ms e2e={fmt(e2e['p95_latency_ms'])} "
          f"gpu={fmt(gpu['p95_latency_ms'])} | GPU throughput/s="
          f"{fmt(gpu['throughput_events_per_s'])} | "
          f"peak allocated MiB={fmt(gpu['peak_allocated_mib'])} | "
          f"MAE={result['accuracy']['mae']}", flush=True)


def plot_report(results: dict[str, Any], out_path: Path) -> None:
    """Plots swept variants only; FP32 math B=1 reference is not timed."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [name for name in results["variants"] if name != REFERENCE_NAME]
    batch_sizes = results["batch_sizes"]
    data = results["results"]
    palette = plt.get_cmap("tab10")
    colors = {name: palette(i % 10) for i, name in enumerate(names)}
    fig = plt.figure(figsize=(16.5, 12.5), layout="constrained")
    grid = fig.add_gridspec(3, 2, height_ratios=[1.1, 0.85, 0.8])
    ax_latency = fig.add_subplot(grid[0, 0])
    ax_rate = fig.add_subplot(grid[0, 1])
    ax_mem = fig.add_subplot(grid[1, :])
    ax_mae = fig.add_subplot(grid[2, :])

    for name in names:
        for mode, line in (("e2e", "-"), ("gpu_only", "--")):
            def series(metric: str, part: str) -> list[float]:
                return [data[str(b)][name]["timings"][mode]["summary"][metric][part]
                        for b in batch_sizes]
            for ax, metric in ((ax_latency, "p95_latency_ms"),
                               (ax_rate, "throughput_events_per_s")):
                yy, low, high = (series(metric, field)
                                 for field in ("mean", "ci95_low", "ci95_high"))
                label = f"{name} - {mode.replace('_', ' ')}"
                ax.plot(batch_sizes, yy, color=colors[name], linestyle=line,
                        marker="o", label=label, linewidth=1.8)
                ax.fill_between(batch_sizes, low, high, color=colors[name], alpha=0.14)

        mm = data
        yy = [mm[str(b)][name]["timings"]["gpu_only"]["summary"]["peak_allocated_mib"]
              for b in batch_sizes]
        ax_mem.plot(batch_sizes, [m["mean"] for m in yy], marker="o", color=colors[name],
                    linewidth=2, label=name)
        ax_mem.fill_between(batch_sizes, [m["ci95_low"] for m in yy],
                            [m["ci95_high"] for m in yy], color=colors[name], alpha=0.14)

    ax_latency.set(title="95th percentile batch latency", ylabel="p95 latency (ms / batch)",
                   xlabel="Batch size")
    ax_rate.set(title="Throughput", ylabel="Events / second", xlabel="Batch size")
    ax_mem.set(title="Peak GPU allocated memory (GPU-only runs)",
               ylabel="Peak allocated (MiB; includes model)", xlabel="Batch size")
    for ax in (ax_latency, ax_rate, ax_mem):
        ax.set_xticks(batch_sizes)
        ax.grid(True, alpha=0.2)
        ax.set_axisbelow(True)
    ax_latency.legend(loc="best", fontsize=8)
    ax_rate.legend(loc="best", fontsize=8)
    ax_mem.legend(loc="best", ncol=min(4, len(names)), fontsize=8)

    compared = [name for name in names if name != REFERENCE_NAME]
    xs = np.arange(len(batch_sizes))
    if compared:
        width = 0.78 / len(compared)
        for i, name in enumerate(compared):
            values = [data[str(b)][name]["accuracy"]["mae"] for b in batch_sizes]
            positions = xs - 0.39 + (i + 0.5) * width
            bars = ax_mae.bar(positions, [0.0 if v is None else v for v in values],
                              width=width, label=name, color=colors[name], alpha=0.85)
            for bar, val in zip(bars, values):
                if val is None:
                    ax_mae.text(bar.get_x() + bar.get_width() / 2, 0, "INVALID",
                                ha="center", va="bottom", fontsize=8, rotation=45)
        ax_mae.legend(loc="best", fontsize=8)
    else:
        ax_mae.text(0.5, 0.5, "Reference model only: MAE = 0",
                    transform=ax_mae.transAxes, ha="center")
    ax_mae.set_xticks(xs, [str(b) for b in batch_sizes])
    ax_mae.set(title="MAE vs FP32 math reference (id + momentum + pu)",
               xlabel="Batch size", ylabel="Mean absolute error")
    ax_mae.grid(axis="y", alpha=0.2)
    ax_mae.set_axisbelow(True)
    fig.suptitle("MLPF PyTorch GPU inference benchmark", fontsize=18, weight="bold")
    fig.supxlabel("Shading = Student's t 95% CI of the mean across repeat sweeps. "
                  "Warmup uses synthetic tensors only.", fontsize=9)
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def write_json(data: dict[str, Any], path: Path) -> None:
    """Atomic write; serializable and safe even when --pdf is omitted."""
    tmp = path.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, allow_nan=False)
    tmp.replace(path)


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    json_path = args.outdir / "summary.json"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA-enabled PyTorch build and NVIDIA GPU are required")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    # Disable TF32 so FP32 math attention is an FP32 accuracy reference.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    if hasattr(torch.backends.cudnn, "benchmark"):
        torch.backends.cudnn.benchmark = True  # Shapes repeat; may help convolution models.

    variants = select_variants(args.variants)
    if any(v.dtype == torch.bfloat16 for v in variants) and not torch.cuda.is_bf16_supported():
        raise RuntimeError("Selected BF16 variant requires a CUDA GPU with BF16 support")
    config, state, pad_multiple = load_config_and_state(args)
    print("Loading identical CPU dataset events for every variant / batch size...")
    events = load_events(args, config.input_dim)
    # Preserve original event order. Event IDs map to unbatched reference predictions.
    grouped = {b: make_batches(events, b, pad_multiple, args.fixed_seq_len)
               for b in args.batch_sizes}
    batch_event_ids = {
        b: [list(range(start, start + b))
            for start in range(0, len(events) - b + 1, b)]
        for b in args.batch_sizes
    }
    baseline_batches = make_batches(events, 1, pad_multiple, args.fixed_seq_len)
    report: dict[str, Any] = {
        "status": "running",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "variants": [v.name for v in variants],
        "variant_specs": {v.name: {"attention": v.attention, "dtype": str(v.dtype),
                                    "mode": v.mode, "use_jagged_attention": v.jagged,
                                    "use_flash_attn_varlen": v.varlen} for v in variants},
        "batch_sizes": args.batch_sizes,
        "baseline": {"variant": REFERENCE_NAME, "batch_size": 1, "passes_per_event": 1,
                     "purpose": "Strict FP32 MATH accuracy reference, B=1 once per event; not a performance sweep"},
        "settings": {
            "checkpoint": str(args.checkpoint), "model_kwargs": str(args.model_kwargs),
            "dataset": args.dataset, "data_dir": str(args.data_dir), "split": args.split,
            "num_events_loaded": len(events), "repeats": args.repeats,
            "warmup_iters_per_model_batch1": args.warmup_iters,
            "baseline_cache_clear_every": args.baseline_cache_clear_every,
            "baseline_memory_log_every": args.baseline_memory_log_every,
            "warmup_seq_len_requested": args.warmup_seq_len,
            "pad_multiple": pad_multiple, "fixed_seq_len": args.fixed_seq_len,
            "pinned_memory": not args.no_pin_memory,
            "compile": args.compile, "compile_mode": args.compile_mode if args.compile else None,
            "seed": args.seed, "strict_state_dict": not args.allow_partial_state,
            "precision": "TF32 OFF; reference uses FP32 weights and inputs with PyTorch MATH SDPA. Native variants convert weights/input to their declared dtype. FP32 MATH may OOM on very long events.",
            "timing": {
                "e2e": "CPU wall time: host->GPU copies + forward + CUDA synchronize; excludes CPU preprocessing, dataset I/O and GPU->host outputs",
                "gpu_only": "CUDA event elapsed time around forward after GPU data residency, including in-forward dispatch gaps",
                "throughput": "events / sum(per-batch measured latency) per repeat; serialized batches, no copy/compute overlap",
                "p95": "95th percentile of per-batch latency within each repeat",
                "ci95": "Two-sided Student's t CI for the mean of repeat-level metrics: mean +/- t(0.975, n-1) * sample_std / sqrt(n)",
                "memory": "PyTorch peak CUDA allocator allocated/reserved MiB, including resident weights; excludes external GPU allocations",
            },
            "jagged_attention": "Flash variants require jagged+varlen profiler verification. fp32_efficient uses MLPF SIMPLE SDPA with forced efficient backend and separate operator verification.",
            "warmup_data": "one synthetic CUDA input of batch size 1 per model; never dataset feature values",
            "cache_clearing": "Untimed accuracy reference: empty_cache after every configured N events (default 1); other variants: clear between repeats; does not clear GPU L2 or live allocations",
        },
        "system": {
            "pytorch": torch.__version__, "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
            "gpu_capability": list(torch.cuda.get_device_capability(device)),
            "device": str(device),
        },
        "shapes": {
            str(b): {"num_batches": len(grouped[b]), "events_per_repeat": b * len(grouped[b]),
                     "event_ids_per_batch": batch_event_ids[b],
                     "sequence_lengths": [int(batch[0].shape[1]) for batch in grouped[b]],
                     "valid_elements": sum(int(mask.sum()) for _, mask in grouped[b]),
                     "padded_elements": sum(int(mask.numel()) for _, mask in grouped[b]),
                     "padding_efficiency": (
                         sum(int(mask.sum()) for _, mask in grouped[b]) /
                         sum(int(mask.numel()) for _, mask in grouped[b])
                     )}
            for b in args.batch_sizes
        },
        "flash_verified_ops": {},
        "results": {str(b): {} for b in args.batch_sizes},
    }
    write_json(report, json_path)
    # Process one variant at a time; keep reference predictions on CPU for MAE.
    references: list[dict[str, torch.Tensor]] = []
    try:
        for variant in variants:
            print(f"\nLoading {variant.name}...", flush=True)
            model = build_model(config, state, variant, device, args.allow_partial_state)
            # Warmup must NOT inherit event sizes from the real dataset.
            # Keep B=1; satisfy any model-required block-size padding.
            seq_len = (
                (args.warmup_seq_len + pad_multiple - 1) // pad_multiple * pad_multiple
                if pad_multiple else args.warmup_seq_len
            )
            synthetic_shape = (
                torch.empty((1, seq_len, config.input_dim), dtype=variant.input_dtype),
                torch.empty((1, seq_len), dtype=torch.bool),
            )
            if variant.attention == "flash":
                ops = verify_flash(model, variant, synthetic_shape, device)
                report["flash_verified_ops"][variant.name] = {"unequal_batch3_eager": ops}
                print(f"  Verified varlen FlashAttention on synthetic unequal-length B=3: {ops[:3]}")
            elif variant.attention == "efficient":
                ops = verify_efficient(model, variant, synthetic_shape, device)
                report.setdefault("efficient_verified_ops", {})[variant.name] = {"eager": ops}
                print(f"  Verified FP32 efficient SDPA: {ops[:3]}")
            if args.compile:
                model = torch.compile(model, mode=args.compile_mode)
            print(f"  {variant.name}: synthetic warmup once at B=1 "
                  f"(sequence length {seq_len}, iterations {args.warmup_iters})", flush=True)
            warmup(model, variant, synthetic_shape, device, args.warmup_iters)
            if args.compile and variant.attention == "flash":
                ops = verify_flash(model, variant, synthetic_shape, device)
                report["flash_verified_ops"][variant.name]["unequal_batch3_compiled"] = ops
            elif args.compile and variant.attention == "efficient":
                ops = verify_efficient(model, variant, synthetic_shape, device)
                report["efficient_verified_ops"][variant.name]["compiled"] = ops
            if variant.name == REFERENCE_NAME:
                # No repeated reference timing; strict FP32 math at B=1.
                host = prepare_host_batches(baseline_batches, variant, not args.no_pin_memory)
                references = reference_predictions(
                    model, variant, host, device, not args.no_pin_memory,
                    cache_clear_every=args.baseline_cache_clear_every,
                    memory_log_every=args.baseline_memory_log_every
                )
                report["baseline"].update({
                    "events_evaluated": len(references),
                    "elements_compared": sum(sum(v.numel() for v in r.values()) for r in references),
                    "accuracy_mae": 0.0,
                    "precision_note": "FP32 model and input with forced PyTorch MATH SDPA; TF32 disabled",
                })
                write_json(report, json_path)
                print(f"  FP32 math reference captured once: {len(references)} events (B=1)")
                del host
            else:
                for b in args.batch_sizes:
                    host = prepare_host_batches(grouped[b], variant, not args.no_pin_memory)
                    entry: dict[str, Any] = {"timings": {}}
                    for mode in MODES:
                        entry["timings"][mode] = benchmark_scope(
                            model, variant, host, mode, device, not args.no_pin_memory,
                            args.repeats
                        )
                        summarize_scope(entry["timings"][mode])
                    entry["accuracy"] = mae_to_reference(
                        model, variant, host, references, batch_event_ids[b],
                        device, not args.no_pin_memory
                    )
                    report["results"][str(b)][variant.name] = entry
                    write_json(report, json_path)
                    print_summary(entry, variant.name, b)
                    del host
                    clear_caches(device)
            del model
            clear_caches(device)
        report["status"] = "complete"
        write_json(report, json_path)
        if args.pdf:
            pdf_path = args.outdir / "comparison.pdf"
            plot_report(report, pdf_path)
            print(f"Single-page PDF: {pdf_path}")
        print(f"JSON results: {json_path}")
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        write_json(report, json_path)
        raise


if __name__ == "__main__":
    main()
