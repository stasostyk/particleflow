#!/usr/bin/env python3
"""One-shot, CUDA-owning worker for exactly ONE MLPF benchmark variant.

Internal entrypoint: orchestrator launches a fresh interpreter per variant.
The process exits after writing its result, destroying its CUDA context.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import nullcontext
from datetime import datetime, timezone
import gc
from fnmatch import fnmatchcase
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import re
import sys
import threading
import time
import traceback
from typing import Any

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from benchmark_variants import Variant, get_variant

MIB = 1024 * 1024


def dtype_of(name: str) -> torch.dtype:
    dtype = getattr(torch, name, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"Unsupported PyTorch dtype name: {name!r}")
    return dtype
HEADS = {"id": 1, "momentum": 2, "pu": 3}
MODES = ("e2e", "gpu_only")


def atomic_json(obj: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as out:
        json.dump(obj, out, indent=2, allow_nan=False)
    temp.replace(path)


def load_config_and_state(settings: dict[str, Any]):
    from mlpf.conf import MLPFConfig, ModelType

    # Pickle is only safe for a trusted, locally produced project artifact.
    with open(settings["model_kwargs"], "rb") as fp:
        obj = pickle.load(fp)
    raw = obj if isinstance(obj, dict) else obj.model_dump()
    config = MLPFConfig.model_validate(raw)
    if config.model.type != ModelType.ATTENTION:
        raise ValueError("Only MLPF ModelType.ATTENTION models are supported")

    checkpoint = torch.load(settings["checkpoint"], map_location="cpu", weights_only=True)
    state = checkpoint.get("model_state_dict", checkpoint)
    if not isinstance(state, dict):
        raise TypeError("Checkpoint does not contain a state_dict")

    # A requested bucketing multiple must satisfy BOTH the model's own
    # padding multiple and the benchmark's. max(8, 250) was incorrect.
    configured_multiple = int(config.pad_to_multiple_elements or 0)
    requested_multiple = int(settings["pad_bin_size"])
    multiple = (math.lcm(configured_multiple or 1, requested_multiple or 1)
                if configured_multiple or requested_multiple else 0)
    fixed = settings["fixed_seq_len"]
    if fixed is not None and multiple and fixed % multiple:
        raise ValueError("--fixed-seq-len must be divisible by the model's pad multiple")
    return config, state, multiple


def set_nested_attribute(obj: Any, dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    current = obj
    for part in parts[:-1]:
        if not hasattr(current, part):
            raise ValueError(f"Unknown model configuration path: {dotted}")
        current = getattr(current, part)
    if not hasattr(current, parts[-1]):
        raise ValueError(f"Unknown model configuration path: {dotted}")
    setattr(current, parts[-1], value)


def _cast_floating(value: Any, dtype: torch.dtype) -> Any:
    if isinstance(value, torch.Tensor) and value.is_floating_point():
        return value.to(dtype=dtype)
    if isinstance(value, tuple):
        return tuple(_cast_floating(x, dtype) for x in value)
    if isinstance(value, list):
        return [_cast_floating(x, dtype) for x in value]
    if isinstance(value, dict):
        return {k: _cast_floating(v, dtype) for k, v in value.items()}
    return value


def _first_float_dtype(value: Any) -> torch.dtype | None:
    if isinstance(value, torch.Tensor) and value.is_floating_point():
        return value.dtype
    if isinstance(value, (tuple, list)):
        for item in value:
            found = _first_float_dtype(item)
            if found is not None:
                return found
    if isinstance(value, dict):
        for item in value.values():
            found = _first_float_dtype(item)
            if found is not None:
                return found
    return None


class ExplicitPrecisionAdapter(torch.nn.Module):
    """Wrap a submodule with deterministic, native (non-autocast) dtype edges."""
    def __init__(self, inner: torch.nn.Module, dtype: torch.dtype, restore: bool):
        super().__init__()
        self.inner = inner.to(dtype=dtype)
        self.target_dtype = dtype
        self.restore = restore

    def forward(self, *args, **kwargs):
        original = _first_float_dtype((args, kwargs))
        output = self.inner(*_cast_floating(args, self.target_dtype),
                            **_cast_floating(kwargs, self.target_dtype))
        if self.restore and original is not None:
            output = _cast_floating(output, original)
        return output


def apply_precision_rules(model: torch.nn.Module, variant: Variant) -> torch.nn.Module:
    if not variant.module_precision_rules:
        return model
    names = {name: module for name, module in model.named_modules() if name}
    selected: dict[str, Any] = {}
    for rule in variant.module_precision_rules:
        matched = [name for name in names if fnmatchcase(name, rule.module_glob)]
        if not matched:
            raise ValueError(f"Module precision rule matched nothing: {rule.module_glob!r}")
        for name in matched:
            if name in selected:
                raise ValueError(f"Multiple module precision rules target {name!r}")
            selected[name] = rule
    targets = sorted(selected)
    for i, name in enumerate(targets):
        if any(other.startswith(name + ".") for other in targets[i + 1:]):
            raise ValueError(f"Overlapping nested module precision rules for {name!r}")
    for name in targets:
        parent_name, _, child_name = name.rpartition(".")
        parent = names[parent_name] if parent_name else model
        inner = parent._modules[child_name]
        rule = selected[name]
        parent._modules[child_name] = ExplicitPrecisionAdapter(
            inner, dtype_of(rule.dtype), rule.restore_output_dtype)
        print(f"  native mixed precision: {name} -> {rule.dtype} "
              f"(restore={rule.restore_output_dtype})", flush=True)
    return model


def build_model(config: Any, state: dict, variant: Variant, device: torch.device,
                partial: bool) -> torch.nn.Module:
    from mlpf.conf import AttentionType
    from mlpf.model.mlpf import MLPF

    cfg = config.model_copy(deep=True)
    attn = cfg.model.attention
    spec = variant.attention
    if spec.varlen and (not spec.jagged or spec.implementation != "FLASH"):
        raise ValueError(f"{variant.name}: varlen requires FLASH + jagged")
    try:
        attn.attention_type = getattr(AttentionType, spec.implementation)
    except AttributeError as exc:
        raise ValueError(f"No AttentionType.{spec.implementation} in this MLPF checkout") from exc
    attn.export_onnx_fused = False
    attn.save_attention = False
    if spec.jagged and not hasattr(attn, "use_jagged_attention"):
        raise RuntimeError("This MLPF checkout lacks attention.use_jagged_attention")
    if hasattr(attn, "use_jagged_attention"):
        attn.use_jagged_attention = spec.jagged
    if spec.varlen and not hasattr(attn, "use_flash_attn_varlen"):
        raise RuntimeError("This MLPF checkout lacks attention.use_flash_attn_varlen")
    if hasattr(attn, "use_flash_attn_varlen"):
        attn.use_flash_attn_varlen = spec.varlen
    for key, value in variant.config_overrides.items():
        set_nested_attribute(cfg, key, value)

    model = MLPF(config=cfg).eval()
    training_only = {
        "task_loss_weighter.weights", "task_loss_weighter.loss_sums",
        "task_loss_weighter.calibration_count",
        "task_loss_weighter.calibration_observations",
        "task_loss_weighter.calibrated",
    }
    expected = set(model.state_dict())
    ignored = [k for k in state if k in training_only and k not in expected]
    safe_state = {k: v for k, v in state.items() if k not in ignored}
    if ignored:
        print(f"{variant.name}: ignored inference-irrelevant checkpoint keys: {ignored}", flush=True)
    incompat = model.load_state_dict(safe_state, strict=not partial)
    if partial and (incompat.missing_keys or incompat.unexpected_keys):
        print(f"WARNING: missing={incompat.missing_keys}, unexpected={incompat.unexpected_keys}", flush=True)
    model = model.to(device=device, dtype=dtype_of(variant.precision.weights))
    if variant.model_transform is not None:
        model = variant.model_transform(model)
    if not isinstance(model, torch.nn.Module):
        raise TypeError("model_transform must return a torch.nn.Module")
    model = apply_precision_rules(model, variant)
    return model.eval()


def attention_context(variant: Variant):
    name = variant.attention.sdpa_backend
    if name is None:
        return nullcontext()
    mapping = {
        "math": SDPBackend.MATH,
        "flash": SDPBackend.FLASH_ATTENTION,
        "efficient": SDPBackend.EFFICIENT_ATTENTION,
        "cudnn": getattr(SDPBackend, "CUDNN_ATTENTION", None),
    }
    if name not in mapping or mapping[name] is None:
        raise ValueError(f"SDPA backend {name!r} unsupported by this PyTorch build")
    return sdpa_kernel(mapping[name])  # Enforce only this SDPA backend; never silently fall back.


def forward(model: torch.nn.Module, variant: Variant, x: torch.Tensor,
            mask: torch.Tensor) -> tuple[torch.Tensor, ...]:
    with attention_context(variant):
        out = (variant.forward_fn(model, x, mask) if variant.forward_fn is not None
               else model(x, mask))
    if not isinstance(out, (tuple, list)) or len(out) < 4:
        raise TypeError("MLPF must return at least (bid, id, momentum, pu)")
    return tuple(out)


def load_events(settings: dict[str, Any], feature_dim: int) -> list[torch.Tensor]:
    import tensorflow_datasets as tfds

    source = tfds.builder(settings["dataset"], data_dir=settings["data_dir"]).as_data_source(
        split=settings["split"])
    requested = settings["num_events"]
    if len(source) < requested:
        raise ValueError(f"Only {len(source)} dataset events; requested {requested}")
    events = []
    for i in range(requested):
        arr = np.asarray(source[i]["X"], dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != feature_dim or arr.shape[0] == 0:
            raise ValueError(f"Event {i}: expected (elements, {feature_dim}), got {arr.shape}")
        if not np.isfinite(arr).all():
            raise ValueError(f"Event {i} contains non-finite features")
        # Do not silently modify physical inputs. The old benchmark clamped every
        # feature to +/-60000, even for the FP32 accuracy reference.
        if settings.get("input_clip") is not None:
            limit = settings["input_clip"]
            arr = np.clip(arr, -limit, limit)
        events.append(torch.from_numpy(np.ascontiguousarray(arr).copy()))
    return events


def event_fingerprint(events: list[torch.Tensor]) -> str:
    """Verify that separate OS workers compare the very same ordered events."""
    digest = hashlib.sha256()
    for event in events:
        digest.update(np.asarray(event.shape, dtype=np.int64).tobytes())
        digest.update(event.numpy().tobytes())
    return digest.hexdigest()


def make_batches(events: list[torch.Tensor], b: int, multiple: int,
                 fixed: int | None, order: str = "sequential"):
    batches = []
    # Explicit event IDs let us compare references correctly even when sorting
    # events by length to reduce padding. Never drop the last partial batch.
    ids = list(range(len(events)))
    if order == "length-sorted":
        ids.sort(key=lambda i: len(events[i]))
    elif order != "sequential":
        raise ValueError(f"Unknown batch order: {order}")
    for start in range(0, len(events), b):
        batch_ids = ids[start:start + b]
        subset = [events[i] for i in batch_ids]
        actual_batch_size = len(subset)
        lengths = [len(x) for x in subset]
        seq_len = max(lengths)
        if fixed is not None:
            if seq_len > fixed:
                raise ValueError(f"Event length {seq_len} exceeds --fixed-seq-len {fixed}")
            seq_len = fixed
        elif multiple:
            seq_len = (seq_len + multiple - 1) // multiple * multiple
        # The final batch must use its ACTUAL batch size; zero-length ghost
        # events can break flash-attn varlen and distort the latency benchmark.
        x = torch.zeros((actual_batch_size, seq_len, subset[0].shape[1]), dtype=torch.float32)
        mask = torch.zeros((actual_batch_size, seq_len), dtype=torch.bool)
        for i, event in enumerate(subset):
            x[i, :lengths[i]] = event
            mask[i, :lengths[i]] = True
        batches.append((x, mask, batch_ids))
    return batches


def host_batches(batches, variant: Variant, pinned: bool):
    out = []
    for x, m, batch_ids in batches:
        input_dtype = dtype_of(variant.precision.input_dtype)
        if input_dtype == torch.float16 and torch.any(x.abs() > torch.finfo(torch.float16).max):
            raise ValueError(
                "Dataset contains features outside FP16 representable range. "
                "Use BF16, explicit feature normalization, or --input-clip (which alters data)."
            )
        x, m = x.to(input_dtype).contiguous(), m.contiguous()
        if variant.input_transform is not None:
            x, m = variant.input_transform(x, m)
            if x.device.type != "cpu" or m.device.type != "cpu":
                raise ValueError("input_transform must return CPU tensors; transfers are benchmarked")
        x, m = ((x.pin_memory(), m.pin_memory()) if pinned else (x, m))
        out.append((x, m, batch_ids))
    return out


def to_gpu(batch, device: torch.device, pinned: bool):
    x, mask = batch[:2]
    return (x.to(device, non_blocking=pinned), mask.to(device, non_blocking=pinned))


def synthetic_batch(shape, variant: Variant, device: torch.device, lengths=None):
    b, seq_len, d = shape
    # Important: CUDA tensors transferred under inference_mode() have a
    # different dispatch-key set from tensors created outside inference_mode().
    # Mixing these was a cause of unnecessary torch.compile recompilation.
    with torch.inference_mode():
        x = torch.randn((b, seq_len, d), dtype=dtype_of(variant.precision.input_dtype), device=device)
        mask = torch.zeros((b, seq_len), device=device, dtype=torch.bool)
        if lengths is None:
            lengths = [max(1, int(seq_len * (0.85 - 0.05 * (i % 3)))) for i in range(b)]
        for i, n in enumerate(lengths):
            mask[i, :n] = True
        if variant.input_transform is not None:
            # The hook is intentionally permitted to be CPU-specific; explicit
            # GPU support must be implemented for synthetic verification paths.
            x, mask = variant.input_transform(x, mask)
    return x, mask


def cuda_cleanup(device: torch.device):
    gc.collect()
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()  # Allocator only; process exit destroys the CUDA context.


def profiler_keys(prof) -> list[str]:
    return sorted({e.key for e in prof.key_averages()})


def profile_forward(model, variant, batch, device, pinned, trace_path: Path | None = None):
    """One UNTIMED forward on real input, separate from latency measurements."""
    from torch.profiler import ProfilerActivity, profile

    with torch.inference_mode():
        x, m = to_gpu(batch, device, pinned)
        forward(model, variant, x, m)  # First-call initialization is NOT profiled.
        torch.cuda.synchronize(device)
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                     record_shapes=True, profile_memory=True, with_flops=True,
                     acc_events=True) as prof:
            forward(model, variant, x, m)
            torch.cuda.synchronize(device)
    if trace_path is not None:
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        prof.export_chrome_trace(str(trace_path))

    def attr(row, primary, legacy):
        # Do NOT evaluate the deprecated attribute as a default expression:
        # getattr(row, primary, getattr(row, legacy, 0)) still reads legacy!
        value = getattr(row, primary, None)
        if value is None:
            value = getattr(row, legacy, 0)
        return float(value or 0)

    rows = []
    for row in prof.key_averages():
        rows.append({
            "op": row.key,
            "calls": int(row.count),
            "gpu_self_ms": attr(row, "self_device_time_total", "self_cuda_time_total") / 1000,
            "gpu_total_ms": attr(row, "device_time_total", "cuda_time_total") / 1000,
            "cpu_self_ms": float(row.self_cpu_time_total) / 1000,
            "device_memory_net_mib": attr(row, "self_device_memory_usage", "self_cuda_memory_usage") / MIB,
        })
    rows.sort(key=lambda item: item["gpu_self_ms"], reverse=True)
    keys = profiler_keys(prof)
    # Event-level GPU kernel names may differ from operator names; retain a sample.
    gpu_kernels = defaultdict(lambda: {"calls": 0, "gpu_total_ms": 0.0})
    for event in prof.events():
        if "CUDA" not in str(getattr(event, "device_type", "")).upper():
            continue
        # CompiledFxGraph activities are profiling ranges / graph wrappers,
        # NOT compiler invocations or individual CUDA kernels. They can overlap
        # contained flash-attn/GEMM kernels and must not be summed with them.
        if "CompiledFxGraph" in event.name or event.name.startswith("## "):
            continue
        entry = gpu_kernels[event.name]
        entry["calls"] += 1
        entry["gpu_total_ms"] += attr(event, "device_time_total", "cuda_time_total") / 1000
    kernels = sorted(({"kernel": k, **v} for k, v in gpu_kernels.items()),
                     key=lambda row: row["gpu_total_ms"], reverse=True)
    del x, m
    return {
        "input_shape": list(batch[0].shape),
        "top_ops": rows[:20],
        "top_cuda_kernels": kernels[:20],
        "total_self_gpu_op_ms": sum(r["gpu_self_ms"] for r in rows),
        "top_ops_note": (
            "Real input, instrumented totals, not comparable to CUDA-event latency. "
            "CompiledFxGraph wrappers are excluded from the kernel breakdown. "
            "GPU operator activities can overlap."
        ),
        "trace": str(trace_path) if trace_path else None,
    }, keys


def verify_dispatch(model, variant, input_dim: int, seq_len: int, device,
                    expected_ops: tuple[str, ...] | None = None):
    """Verify backend on synthetic data; for varlen, unequal B=3 segments are mandatory."""
    patterns = variant.attention.expected_ops if expected_ops is None else expected_ops
    if not patterns:
        return {"status": "unchecked", "reason": "No expected_ops supplied"}
    if variant.attention.varlen and seq_len < 3:
        raise ValueError("Varlen verification needs synthetic sequence length >= 3")
    shape = ((3 if variant.attention.varlen else 1), seq_len, input_dim)
    lengths = ([seq_len, max(2, seq_len // 2), 1]
               if variant.attention.varlen else None)
    # Use a verification-specific B=3 mask to prove segmented varlen dispatch.
    from torch.profiler import ProfilerActivity, profile
    x, mask = synthetic_batch(shape, variant, device, lengths=lengths)
    with torch.inference_mode():
        forward(model, variant, x, mask)
        torch.cuda.synchronize(device)
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                     acc_events=True) as prof:
            forward(model, variant, x, mask)
            torch.cuda.synchronize(device)
    keys = profiler_keys(prof)
    observed = [k for k in keys if any(re.search(regex, k, flags=re.I)
                for regex in patterns)]
    if not observed:
        nearby = [k for k in keys if any(t in k.lower() for t in ("attn", "flash", "attention"))]
        raise RuntimeError(
            f"{variant.name}: no required kernel op {patterns} on synthetic "
            f"shape {shape}. Nearby profiler ops: {nearby[:30]}. No SDPA fallback accepted.")
    del x, mask
    return {"status": "verified", "matched_ops": observed, "verification_shape": list(shape),
            "unequal_lengths": lengths}


def warmup(model, variant, shape, device, iterations):
    x, mask = synthetic_batch(shape, variant, device)
    with torch.inference_mode():
        for _ in range(iterations):
            result = forward(model, variant, x, mask)
        torch.cuda.synchronize(device)
    del x, mask, result


class NvmlSampler:
    """Optional NVML time sampler; reports device-wide metrics, not per-process use.

    NVML utilization rates are driver-window samples, not instantaneous SM occupancy.
    Sampling includes host gaps between serial batches, and may alias short sweeps.
    """

    def __init__(self, cuda_index: int, period_ms: int,
                 nvml_index: int | None = None):
        self.period_s = period_ms / 1000.0
        self.samples: list[dict[str, float]] = []
        self.available = False
        self.reason: str | None = None
        self.mapping: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._nvml = None
        self._handle = None
        self._initialized = False
        stage = "import pynvml"
        try:
            import pynvml
            stage = "nvmlInit (NVIDIA host driver access)"
            pynvml.nvmlInit()
            self._initialized = True
            self._nvml = pynvml
            stage = "CUDA to NVML GPU mapping"
            properties = torch.cuda.get_device_properties(cuda_index)
            attempts = []

            def choose(label, fn):
                if self._handle is not None:
                    return
                try:
                    self._handle = fn()
                    self.mapping = label
                except Exception as err:
                    attempts.append(f"{label}: {type(err).__name__}: {err}")

            if nvml_index is not None:
                choose("explicit NVML index", lambda: pynvml.nvmlDeviceGetHandleByIndex(nvml_index))

            uuid = getattr(properties, "uuid", None)
            if uuid:
                uuid_str = str(uuid)
                if not uuid_str.startswith(("GPU-", "MIG-")) and hasattr(uuid, "bytes"):
                    import uuid as uuidlib
                    uuid_str = "GPU-" + str(uuidlib.UUID(bytes=bytes(uuid.bytes)))
                if uuid_str.startswith(("GPU-", "MIG-")):
                    choose("CUDA GPU UUID", lambda: pynvml.nvmlDeviceGetHandleByUUID(uuid_str))

            pci_bus = getattr(properties, "pci_bus_id", None)
            if isinstance(pci_bus, str) and ":" in pci_bus:
                choose("CUDA PCI bus ID", lambda: pynvml.nvmlDeviceGetHandleByPciBusId(pci_bus))
            elif pci_bus is not None:
                # Torch CUDA device properties frequently expose numeric PCI
                # domain, bus and device fields; str(pci_bus_id) was WRONG.
                domain = getattr(properties, "pci_domain_id", None)
                dev = getattr(properties, "pci_device_id", None)
                if domain is not None and dev is not None:
                    bus_id = f"{int(domain):08x}:{int(pci_bus):02x}:{int(dev):02x}.0"
                    choose("CUDA PCI bus ID", lambda: pynvml.nvmlDeviceGetHandleByPciBusId(bus_id))

            visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
            visible_tokens = [x.strip() for x in visible.split(",") if x.strip()]
            if len(visible_tokens) > cuda_index:
                token = visible_tokens[cuda_index]
                if token.startswith(("GPU-", "MIG-")):
                    choose("CUDA_VISIBLE_DEVICES UUID",
                           lambda: pynvml.nvmlDeviceGetHandleByUUID(token))
            # Never assume a Slurm-remapped numeric CUDA_VISIBLE_DEVICES index
            # is the same as the physical NVML/nvidia-smi GPU index.
            if self._handle is None and pynvml.nvmlDeviceGetCount() == 1:
                choose("single NVML GPU", lambda: pynvml.nvmlDeviceGetHandleByIndex(0))
            if self._handle is None:
                raise RuntimeError(
                    "No safe CUDA -> NVML device mapping. "
                    "Use --nvml-index as shown by nvidia-smi if necessary. "
                    f"CUDA_VISIBLE_DEVICES={visible!r}; attempts={attempts}"
                )
            self.available = True
        except Exception as exc:
            self.reason = f"NVML unavailable at {stage}: {type(exc).__name__}: {exc}"
            print(f"WARNING: {self.reason}; utilization/physical VRAM omitted", flush=True)

    def _sample(self):
        try:
            nv = self._nvml
            utilization = nv.nvmlDeviceGetUtilizationRates(self._handle)
            mem = nv.nvmlDeviceGetMemoryInfo(self._handle)
            row = {
                "gpu_util_pct": float(utilization.gpu),
                "memory_controller_util_pct": float(utilization.memory),
                "device_used_mib": float(mem.used / MIB),
            }
            try:
                row["power_w"] = float(nv.nvmlDeviceGetPowerUsage(self._handle)) / 1000
            except Exception:
                pass
            self.samples.append(row)
        except Exception as exc:
            if self.reason is None:
                self.reason = f"NVML sampling error: {exc}"

    def _loop(self):
        while not self._stop.is_set():
            self._sample()
            self._stop.wait(self.period_s)

    def start(self):
        self.samples = []
        if self.available:
            self._stop.clear()
            self._sample()
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()

    def stop(self) -> dict[str, Any]:
        if self.available:
            self._stop.set()
            if self._thread:
                self._thread.join(timeout=5)
            self._sample()
        samples = list(self.samples)
        result: dict[str, Any] = {"nvml_sample_count": len(samples)}
        if not samples:
            return result
        for key, metric in (("gpu_util_pct", "gpu_util_pct"),
                            ("memory_controller_util_pct", "mem_controller_util_pct"),
                            ("device_used_mib", "device_used_mib"),
                            ("power_w", "power_w")):
            values = [item[key] for item in samples if key in item]
            if values:
                result[f"mean_{metric}"] = float(np.mean(values))
                result[f"peak_{metric}"] = float(np.max(values))
        return result

    def close(self):
        if self._initialized:
            try:
                self._nvml.nvmlShutdown()
            except Exception:
                pass


class NoTelemetrySampler:
    available = False
    reason = "Disabled by --no-telemetry"
    mapping = None

    def start(self):
        pass

    def stop(self):
        return {"nvml_sample_count": 0}

    def close(self):
        pass


def time_batch(model, variant, batch, mode: str, device, pinned):
    if mode == "e2e":
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        x, mask = to_gpu(batch, device, pinned)
        out = forward(model, variant, x, mask)
        torch.cuda.synchronize(device)
        result_ms = 1000 * (time.perf_counter() - t0)
    else:
        x, mask = to_gpu(batch, device, pinned)
        torch.cuda.synchronize(device)
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        out = forward(model, variant, x, mask)
        end.record()
        end.synchronize()
        result_ms = begin.elapsed_time(end)
    del out, x, mask
    return result_ms


def student_ci(values: list[float]) -> dict[str, float]:
    from scipy.stats import t

    a = np.asarray(values, dtype=np.float64)
    if len(a) < 2 or not np.isfinite(a).all():
        raise ValueError("CI requires >=2 finite repeat-level values")
    mean, sd = float(a.mean()), float(a.std(ddof=1))
    half = float(t.ppf(0.975, len(a) - 1) * sd / np.sqrt(len(a)))
    return {"mean": mean, "ci95_low": mean - half, "ci95_high": mean + half,
            "std": sd, "n": len(a)}


def compiled_graph_count() -> int:
    """Best-effort count of graph compilations in the current worker process."""
    try:
        from torch._dynamo.utils import counters
        return int(counters["stats"].get("unique_graphs", 0))
    except (AttributeError, ImportError, KeyError, TypeError):
        return 0


def prewarm_real_batches(model, variant, batches, device, pinned):
    """Warm real input/mask shapes and dispatch keys, never random stand-ins."""
    graph_before = compiled_graph_count()
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    with torch.inference_mode():
        for batch in batches:
            x, mask = to_gpu(batch, device, pinned)
            out = forward(model, variant, x, mask)
            del out, x, mask
        torch.cuda.synchronize(device)
    seconds = time.perf_counter() - t0
    return {
        "wall_s": seconds,
        "new_compiled_graphs": compiled_graph_count() - graph_before,
        "unique_padded_shapes": len({tuple(batch[0].shape) for batch in batches}),
        "batches_prewarmed": len(batches),
        "note": "Prewarm wall time INCLUDES inference and lazy compilation, not compilation alone",
    }


def benchmark_scope(model, variant, batches, mode, device, pinned, repeats, sampler):
    runs = []
    with torch.inference_mode():
        for repeat in range(repeats):
            # empty_cache()/gc.collect() here used to destroy allocator
            # steady state before every repeat. Keep the warm allocator instead.
            torch.cuda.reset_peak_memory_stats(device)
            elapsed = []
            sampler.start()
            graph_before = compiled_graph_count()
            torch.cuda.synchronize(device)
            sweep_start = time.perf_counter()
            try:
                for index, batch in enumerate(batches):
                    try:
                        elapsed.append(time_batch(model, variant, batch, mode, device, pinned))
                    except torch.cuda.OutOfMemoryError:
                        free, total = torch.cuda.mem_get_info(device)
                        print(f"OOM: {variant.name} {mode} repeat={repeat} batch={index} "
                              f"shape={tuple(batch[0].shape)} free={free/MIB:.0f}MiB "
                              f"total={total/MIB:.0f}MiB", file=sys.stderr, flush=True)
                        raise
                torch.cuda.synchronize(device)
                sweep_wall_s = time.perf_counter() - sweep_start
            finally:
                telemetry = sampler.stop()
            seconds = sum(elapsed) / 1000
            events_count = sum(len(batch[2]) for batch in batches)
            runs.append({
                "repeat": repeat,
                "batch_latencies_ms": elapsed,
                "p95_latency_ms": float(np.percentile(elapsed, 95)),
                "p50_latency_ms": float(np.percentile(elapsed, 50)),
                "throughput_events_per_s": events_count / seconds,
                "observed_throughput_events_per_s": events_count / sweep_wall_s,
                "sweep_wall_s": sweep_wall_s,
                "total_timed_s": seconds,
                "events": events_count,
                "new_compiled_graphs": compiled_graph_count() - graph_before,
                "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / MIB,
                "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / MIB,
                **telemetry,
            })
    always = ("p95_latency_ms", "p50_latency_ms", "throughput_events_per_s",
              "observed_throughput_events_per_s", "sweep_wall_s",
              "peak_allocated_mib", "peak_reserved_mib")
    optional = ("mean_gpu_util_pct", "peak_gpu_util_pct", "mean_mem_controller_util_pct",
                "mean_device_used_mib", "peak_device_used_mib", "mean_power_w")
    summary = {key: student_ci([row[key] for row in runs]) for key in always}
    for key in optional:
        rows_with_metric = [row[key] for row in runs if key in row]
        summary[key] = student_ci(rows_with_metric) if len(rows_with_metric) >= 2 else None
    total_graphs = sum(row["new_compiled_graphs"] for row in runs)
    return {"repeats": runs, "summary": summary,
            "new_compiled_graphs_during_timing": total_graphs,
            "steady_state_valid": total_graphs == 0,
            "notes": {
                "throughput_events_per_s": "events / sum of measured per-batch intervals",
                "observed_throughput_events_per_s":
                    "events / whole repeat wall time (includes host dispatch and untimed H2D for gpu_only)",
                "steady_state_valid": ("False if Dynamo counters report graph creation during "
                                       "timed repeats; does not rule out an eager fallback"),
            }}


def capture_reference(model, variant, batches, settings, device, pinned):
    reference = [None] * settings["num_events"]
    clear_every = settings["baseline_cache_clear_every"]
    log_every = settings["baseline_memory_log_every"]
    with torch.inference_mode():
        for i, batch in enumerate(batches):
            x, mask = to_gpu(batch, device, pinned)
            pred = None
            try:
                pred = forward(model, variant, x, mask)
                heads = extract_event_heads(pred, batch[1], 0, batch[2][0])
                for name, value in heads.items():
                    if not torch.isfinite(value).all():
                        raise ValueError(f"Reference has nonfinite {name} outputs at event {i}")
                reference[batch[2][0]] = heads
            finally:
                if pred is not None:
                    del pred
                del x, mask
            if clear_every and (i + 1) % clear_every == 0:
                cuda_cleanup(device)
            if log_every and ((i + 1) % log_every == 0 or i == 0):
                print(f"[REFERENCE] event={i+1}/{len(batches)} "
                      f"allocated={torch.cuda.memory_allocated(device)/MIB:.0f}MiB "
                      f"reserved={torch.cuda.memory_reserved(device)/MIB:.0f}MiB", flush=True)
    if any(x is None for x in reference):
        raise RuntimeError("Reference did not capture every event")
    return reference


def extract_event_heads(out, batch_mask, j: int, event_id: int):
    """MLPF elementwise contract: outputs must align with the input mask.

    Slot/set-prediction models require a domain-specific output mask and
    permutation-invariant matcher; an input-element mask is NOT appropriate.
    """
    valid = batch_mask[j]
    heads = {}
    for head, idx in HEADS.items():
        tensor = out[idx]
        if (not isinstance(tensor, torch.Tensor) or tensor.ndim < 2 or
                tuple(tensor.shape[:2]) != tuple(batch_mask.shape)):
            shape = getattr(tensor, "shape", None)
            raise ValueError(
                f"Event {event_id}, head {head}: output shape {shape} is not aligned "
                f"with padded input mask {tuple(batch_mask.shape)}. This benchmark "
                "compares ELEMENTWISE MLPF predictions only. For output slots "
                "of a different length/order, add a domain-specific matcher."
            )
        heads[head] = tensor[j].detach().float().cpu()[valid].clone()
    return heads


def capture_selected_b1(model, variant, events, ids, multiple, fixed, device, pinned):
    selected = {}
    if not ids:
        return selected
    with torch.inference_mode():
        for event_id in ids:
            batch = host_batches(make_batches([events[event_id]], 1, multiple, fixed),
                                 variant, pinned)[0]
            x, mask = to_gpu(batch, device, pinned)
            out = forward(model, variant, x, mask)
            selected[event_id] = extract_event_heads(out, batch[1], 0, event_id)
            del x, mask, out
    return selected


def compare_compiled_to_eager(compiled_model, eager_model, variant, events, ids,
                              multiple, fixed, device, pinned):
    """Isolate an Inductor numerical error from FP32-vs-FP16 differences."""
    totals = {h: 0.0 for h in HEADS}
    counts = {h: 0 for h in HEADS}
    worst = []
    with torch.inference_mode():
        for event_id in ids:
            batch = host_batches(make_batches([events[event_id]], 1, multiple, fixed),
                                 variant, pinned)[0]
            x, mask = to_gpu(batch, device, pinned)
            compiled = extract_event_heads(forward(compiled_model, variant, x, mask),
                                           batch[1], 0, event_id)
            eager = extract_event_heads(forward(eager_model, variant, x, mask),
                                        batch[1], 0, event_id)
            scores = {}
            for head in HEADS:
                diff = (compiled[head] - eager[head]).abs()
                if not torch.isfinite(diff).all():
                    raise ValueError(f"Compiled/eager numerical nonfinite at {event_id} {head}")
                totals[head] += float(diff.double().sum())
                counts[head] += diff.numel()
                scores[head] = float(diff.double().mean())
            worst.append({"event_id": event_id, "mae_by_head": scores})
            del x, mask
    return {
        "mae_by_head": {h: totals[h] / counts[h] if counts[h] else None for h in HEADS},
        "events": worst,
        "description": "Same variant, B=1, eager vs torch.compile on selected worst FP32-reference discrepancies",
    }


def accuracy(model, variant, batches, reference, device, pinned,
             same_variant_b1: dict[int, Any] | None = None):
    sums = {k: 0.0 for k in HEADS}
    counts = {k: 0 for k in HEADS}
    invalid = {k: 0 for k in HEADS}
    maximum = {k: 0.0 for k in HEADS}
    self_sums = {k: 0.0 for k in HEADS}
    self_counts = {k: 0 for k in HEADS}
    per_event = []
    with torch.inference_mode():
        for batch in batches:
            x, mask = to_gpu(batch, device, pinned)
            out = forward(model, variant, x, mask)
            for j, event_id in enumerate(batch[2]):
                actual_heads = extract_event_heads(out, batch[1], j, event_id)
                event_sum = 0.0
                event_count = 0
                event_head_mae = {}
                for head in HEADS:
                    actual = actual_heads[head]
                    correct = reference[event_id][head]
                    if actual.shape != correct.shape:
                        raise ValueError(f"{head} output shape mismatch for event {event_id}: "
                                         f"{actual.shape} vs {correct.shape}")
                    diff = (actual - correct).abs()
                    valid_values = torch.isfinite(diff)
                    invalid[head] += int((~valid_values).sum())
                    diff_sum = float(diff[valid_values].double().sum())
                    sums[head] += diff_sum
                    counts[head] += diff.numel()
                    maximum[head] = max(maximum[head], float(diff[valid_values].max())
                                        if valid_values.any() else 0.0)
                    event_sum += diff_sum
                    event_count += diff.numel()
                    event_head_mae[head] = (diff_sum / diff.numel()
                                            if diff.numel() and valid_values.all() else None)
                    if same_variant_b1 is not None and event_id in same_variant_b1:
                        other = same_variant_b1[event_id][head]
                        if actual.shape != other.shape:
                            raise ValueError(f"B=1 self-check shape differs for event {event_id}, {head}")
                        self_diff = (actual - other).abs()
                        if not torch.isfinite(self_diff).all():
                            raise ValueError(f"B=1 self-check non-finite at event {event_id}, {head}")
                        self_sums[head] += float(self_diff.double().sum())
                        self_counts[head] += self_diff.numel()
                per_event.append({"event_id": event_id, "input_length": int(batch[1][j].sum()),
                                  "mae": event_sum / event_count if event_count else None,
                                  "mae_by_head": event_head_mae})
            del x, mask, out
    total = sum(counts.values())
    has_invalid = any(invalid.values())
    macro = [row["mae"] for row in per_event if row["mae"] is not None]
    per_event.sort(key=lambda row: row["event_id"])
    return {
        "mae": (sum(sums.values()) / total if total and not has_invalid else None),
        "mae_by_head": {k: (sums[k] / counts[k] if counts[k] and not invalid[k] else None)
                        for k in HEADS},
        "max_abs_error_by_head": maximum,
        "invalid_by_head": invalid,
        "elements_compared": total,
        "events_compared": len(per_event),
        "macro_event_mae": float(np.mean(macro)) if macro and not has_invalid else None,
        "worst_events": sorted(per_event, key=lambda row: row["mae"] or -1,
                               reverse=True)[:10],
        "per_event": per_event,
        "mae_vs_same_variant_b1_by_head": {
            k: (self_sums[k] / self_counts[k] if self_counts[k] else None)
            for k in HEADS
        },
        "b1_self_comparison_event_count": sum(
            1 for row in per_event if same_variant_b1 is not None
            and row["event_id"] in same_variant_b1),
    }


def run_job(job: dict[str, Any], variant_name: str, output: Path):
    settings = job["settings"]
    variant = get_variant(variant_name)
    is_reference = variant_name == job["reference"]
    seed = settings["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA-enabled PyTorch build and an NVIDIA GPU are required")
    device = torch.device(settings["device"])
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = variant.precision.allow_tf32
    torch.backends.cudnn.allow_tf32 = variant.precision.allow_tf32
    torch.set_float32_matmul_precision("high" if variant.precision.allow_tf32 else "highest")
    # Shape-changing data cause cuDNN to autotune repeatedly when enabled.
    # Keep off unless user wants that explicit throughput trade-off.
    torch.backends.cudnn.benchmark = settings.get("cudnn_benchmark", False)
    if any(dt == "bfloat16" for dt in (variant.precision.weights, variant.precision.input_dtype)):
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16 variant requested, but GPU does not support BF16")

    print(f"[WORKER] {variant_name} | PID={os.getpid()} | {torch.cuda.get_device_name(device)}",
          flush=True)
    config, state, multiple = load_config_and_state(settings)
    events = load_events(settings, config.input_dim)
    input_fingerprint = event_fingerprint(events)
    model = build_model(config, state, variant, device, settings["allow_partial_state"])
    del state
    warm_len = settings["warmup_seq_len"]
    if multiple:
        warm_len = (warm_len + multiple - 1) // multiple * multiple
    shape = (1, warm_len, config.input_dim)
    verification = verify_dispatch(model, variant, config.input_dim, warm_len, device)
    # Initialize eager libraries BEFORE compilation. Do not compile a synthetic
    # 256-token B=1 graph that the actual dataset never uses.
    warmup(model, variant, shape, device, settings["warmup_iters"])
    eager_model = model
    # A B=1 numerical reference should remain eager even when --compile
    # requests compilation of ALL timed variants.
    compiled = not is_reference and bool(variant.compile or settings["compile"])
    compile_mode = (settings["compile_mode"] if settings["compile"] else variant.compile_mode)
    compile_dynamic = {"auto": None, "true": True, "false": False}[
        settings.get("compile_dynamic", "auto")]
    if compiled:
        model = torch.compile(model, mode=compile_mode, dynamic=compile_dynamic)
    compiled_verification = None
    sysinfo = {
        "pytorch": torch.__version__, "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device),
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "device": str(device),
        "pid": os.getpid(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "slurm_job_gpus": os.environ.get("SLURM_JOB_GPUS"),
    }
    pinned = not settings["no_pin_memory"]
    result: dict[str, Any] = {
        "variant": variant_name, "is_reference": is_reference,
        "variant_spec": variant.describe(), "system": sysinfo,
        "dispatch_verification": verification,
        "compiled_dispatch_verification": compiled_verification,
        "compiled": compiled, "compile_mode": compile_mode if compiled else None,
        "compile_dynamic": settings.get("compile_dynamic", "auto") if compiled else None,
        "pad_multiple": multiple,
        "dataset_fingerprint_sha256": input_fingerprint,
        "input_lengths": {
            "min": min(len(x) for x in events),
            "p50": float(np.median([len(x) for x in events])),
            "p95": float(np.percentile([len(x) for x in events], 95)),
            "max": max(len(x) for x in events),
        },
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    if is_reference:
        batches = host_batches(make_batches(events, 1, multiple, settings["fixed_seq_len"]),
                               variant, pinned)
        refs = capture_reference(model, variant, batches, settings, device, pinned)
        reference_file = Path(job["reference_file"])
        reference_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"predictions": refs, "dataset_fingerprint_sha256": input_fingerprint},
                   reference_file)
        result["baseline"] = {
            "variant": variant_name, "batch_size": 1, "events_evaluated": len(refs),
            "elements_compared": sum(t.numel() for r in refs for t in r.values()),
            "accuracy_mae": 0.0,
            "description": "Untimed B=1 CPU-stored reference; backend is exactly the variant_spec above",
            "reference_file": str(reference_file),
            "dataset_fingerprint_sha256": input_fingerprint,
        }
        print(f"Captured {len(refs)} CPU reference events", flush=True)
    else:
        reference_payload = torch.load(job["reference_file"], map_location="cpu", weights_only=True)
        if reference_payload.get("dataset_fingerprint_sha256") != input_fingerprint:
            raise ValueError("Loaded TFDS events differ from the reference worker: dataset fingerprint mismatch")
        refs = reference_payload["predictions"]
        if len(refs) != settings["num_events"]:
            raise ValueError("Reference predictions missing or event count mismatch")
        sampler = (NoTelemetrySampler() if settings.get("no_telemetry") else
                   NvmlSampler(device.index if device.index is not None else 0,
                               settings["telemetry_ms"], settings.get("nvml_index")))
        result["telemetry"] = {
            "available": sampler.available,
            "reason": sampler.reason,
            "gpu_mapping": sampler.mapping,
            "sample_period_ms": settings["telemetry_ms"],
            "scope": "device-wide NVML, sampled during each repeat, including inter-batch gaps",
        }
        result["results"] = {}
        selected_b1 = None
        try:
            for b in settings["batch_sizes"]:
                print(f"[WORKER] {variant_name} B={b} | benchmarking", flush=True)
                batches = host_batches(
                    make_batches(events, b, multiple, settings["fixed_seq_len"],
                                 settings.get("batch_order", "sequential")),
                    variant, pinned)
                valid_tokens = sum(int(mask.sum()) for _, mask, _ in batches)
                padded_tokens = sum(mask.numel() for _, mask, _ in batches)
                entry: dict[str, Any] = {
                    "timings": {},
                    "batching": {
                        "requested_batch_size": b,
                        "num_batches": len(batches),
                        "actual_batch_sizes": [len(batch[2]) for batch in batches],
                        "events": sum(len(batch[2]) for batch in batches),
                        "valid_input_tokens": valid_tokens,
                        "padded_input_tokens": padded_tokens,
                        "padding_efficiency": valid_tokens / padded_tokens,
                        "padded_lengths": [int(batch[0].shape[1]) for batch in batches],
                        "order": settings.get("batch_order", "sequential"),
                    },
                }
                # Real inputs: identical masks, dispatch keys and sequences as
                # timing. The first pass also pays any lazy compiler startup.
                entry["prewarm"] = prewarm_real_batches(model, variant, batches, device, pinned)
                print(f"  prewarm: {entry['prewarm']['unique_padded_shapes']} shapes, "
                      f"{entry['prewarm']['wall_s']:.2f}s incl. compile, "
                      f"{entry['prewarm']['new_compiled_graphs']} new Dynamo graphs", flush=True)
                torch.cuda.synchronize(device)
                for mode in MODES:
                    entry["timings"][mode] = benchmark_scope(
                        model, variant, batches, mode, device, pinned, settings["repeats"], sampler)
                    if not entry["timings"][mode]["steady_state_valid"]:
                        print(f"WARNING: {variant_name} B={b} {mode} created Dynamo graphs "
                              "DURING timed repeats; these results are not steady-state!",
                              flush=True)
                entry["accuracy"] = accuracy(model, variant, batches, refs, device,
                                              pinned, same_variant_b1=selected_b1)
                if b == 1 and settings.get("batch_consistency_events", 0):
                    worst = entry["accuracy"]["worst_events"]
                    selected_ids = [row["event_id"] for row in
                                    worst[:settings["batch_consistency_events"]]]
                    selected_b1 = capture_selected_b1(
                        model, variant, events, selected_ids, multiple,
                        settings["fixed_seq_len"], device, pinned)
                    entry["accuracy"]["b1_selected_diagnostic_event_ids"] = selected_ids
                    if compiled:
                        entry["accuracy"]["compiled_vs_eager_b1"] = compare_compiled_to_eager(
                            model, eager_model, variant, events, selected_ids,
                            multiple, settings["fixed_seq_len"], device, pinned)
                result["results"][str(b)] = entry
                atomic_json(result, output)  # Keep completed batch sizes on disk on failure.
                ideal = entry["timings"]["gpu_only"]["summary"]["throughput_events_per_s"]["mean"]
                rate = entry["timings"]["e2e"]["summary"]["observed_throughput_events_per_s"]["mean"]
                util = entry["timings"]["gpu_only"]["summary"]["mean_gpu_util_pct"]
                print(f"  B={b} e2e_wall_rate={rate:.1f} events/s "
                      f"(ideal GPU-event sum={ideal:.1f}), "
                      f"GPU util={util['mean'] if util else 'N/A'}%, "
                      f"MAE={entry['accuracy']['mae']}, "
                      f"by_head={entry['accuracy']['mae_by_head']}, "
                      f"worst={[(e['event_id'], round(e['mae'], 5)) for e in entry['accuracy']['worst_events'][:3]]}",
                      flush=True)
                del batches
            # Profiling and its extra graph specializations must NOT interfere
            # with the timed sweeps. Defer all profiler work until the end.
            if compiled:
                compiled_patterns = (variant.attention.compiled_expected_ops
                                     if variant.attention.compiled_expected_ops is not None
                                     else variant.attention.expected_ops)
                compiled_verification = verify_dispatch(
                    model, variant, config.input_dim, warm_len, device,
                    compiled_patterns)
                result["compiled_dispatch_verification"] = compiled_verification
            if settings["profile"]:
                for b in settings["batch_sizes"]:
                    print(f"[WORKER] {variant_name} B={b} | untimed real-batch profile", flush=True)
                    first_batch = host_batches(
                        make_batches(events, b, multiple, settings["fixed_seq_len"],
                                     settings.get("batch_order", "sequential")),
                        variant, pinned)[0]
                    trace_path = (Path(job["trace_dir"]) / f"{variant_name}_b{b}.json"
                                  if settings["profile_traces"] else None)
                    result["results"][str(b)]["profile"], _ = profile_forward(
                        model, variant, first_batch, device, pinned, trace_path)
                    atomic_json(result, output)
        finally:
            result["telemetry"]["reason"] = sampler.reason
            if sampler.reason:
                result["telemetry"]["available"] = False
            sampler.close()
    result["status"] = "complete"
    atomic_json(result, output)
    print(f"[WORKER] {variant_name} finished", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    job = json.loads(args.job.read_text(encoding="utf-8"))
    try:
        run_job(job, args.variant, args.output)
    except BaseException as exc:
        # Leave a failure marker even if no batch result was written.
        try:
            partial = (json.loads(args.output.read_text(encoding="utf-8"))
                       if args.output.exists() else {"variant": args.variant})
            partial.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}",
                            "traceback": traceback.format_exc()})
            atomic_json(partial, args.output)
        finally:
            raise


if __name__ == "__main__":
    main()
