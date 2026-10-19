#!/usr/bin/env python3
"""Minimal MLPF FP32 FLASH benchmark using ONNX Runtime CUDA only.

Requirements:
  pip install numpy tensorflow-datasets tqdm onnxruntime-gpu psutil

Key benchmark rule:
  The reported inference time contains only ONNX Runtime execution with CUDA-resident
  inputs and CUDA-resident outputs. CPU batching, H2D copies, D2H copies, and output
  statistics are all outside the timed region.

The input ONNX model is expected to be the FP32 fused/FLASH model from the original
script (normally model_fused_fp32.onnx) with inputs:
  Xfeat_normed, mask
and outputs such as:
  bid, id, momentum, pu
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import time
from typing import Any

import numpy as np
import onnxruntime as ort
ort.preload_dlls(directory="")

import tensorflow_datasets as tfds
from tqdm import tqdm



X_NAME = "Xfeat_normed"
MASK_NAME = "mask"

SEED = 12345


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark FP32 FLASH MLPF with ONNX Runtime CUDA")
    p.add_argument("--onnx-model", required=True, help="Path to model_fused_fp32.onnx")
    p.add_argument("--data-dir", required=True, help="TFDS data directory")
    p.add_argument("--dataset", default="cms_pf_ttbar", help="TFDS dataset name")
    p.add_argument("--num-events", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument(
        "--pad-bin-size",
        type=int,
        default=0,
        help="Round each batch max sequence length to this multiple; 0 disables it",
    )
    p.add_argument("--device-id", type=int, default=0)
    p.add_argument("--num-threads", type=int, default=1)
    p.add_argument("--warmup-runs", type=int, default=1, help="Untimed warmups per prepared batch")
    p.add_argument(
        "--benchmark-repeats",
        type=int,
        default=1,
        help="Timed repeats per batch; the batch mean is used in aggregate throughput",
    )
    p.add_argument("--outdir", default="./onnx_flash_fp32_benchmark")
    return p.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.num_events <= 0:
        raise ValueError("--num-events must be > 0")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if args.pad_bin_size < 0:
        raise ValueError("--pad-bin-size must be >= 0")
    if args.warmup_runs < 0:
        raise ValueError("--warmup-runs must be >= 0")
    if args.benchmark_repeats <= 0:
        raise ValueError("--benchmark-repeats must be > 0")
    if not os.path.isfile(args.onnx_model):
        raise FileNotFoundError(args.onnx_model)


def round_up(n: int, multiple: int) -> int:
    if multiple <= 0:
        return n
    return ((n + multiple - 1) // multiple) * multiple


def get_cpu_info() -> str:
    try:
        out = subprocess.check_output(["lscpu"], text=True, stderr=subprocess.DEVNULL)
        for line in out.splitlines():
            if line.startswith("Model name:"):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "Unknown CPU"


def get_gpu_info(device_id: int) -> dict[str, Any]:
    ret: dict[str, Any] = {"device_id": device_id, "name": "Unknown GPU", "memory_total_mib": None}
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={device_id}",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        name, mem = [x.strip() for x in out.split(",", 1)]
        ret["name"] = name
        ret["memory_total_mib"] = float(mem)
    except Exception:
        pass
    return ret


def gpu_memory_used_mib(device_id: int) -> float | None:
    """Best-effort device-wide snapshot; always called outside timed inference."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={device_id}",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return float(out.splitlines()[0])
    except Exception:
        return None


def process_rss_mib() -> float | None:
    try:
        import psutil

        return psutil.Process().memory_info().rss / (1024**2)
    except Exception:
        return None


def create_session(model_path: str, device_id: int, num_threads: int) -> ort.InferenceSession:
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError(
            "CUDAExecutionProvider is not available. Use onnxruntime-gpu. "
            f"Available providers: {ort.get_available_providers()}"
        )

    so = ort.SessionOptions()
    so.intra_op_num_threads = num_threads
    so.inter_op_num_threads = num_threads
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    sess = ort.InferenceSession(
        model_path,
        sess_options=so,
        providers=[("CUDAExecutionProvider", {"device_id": device_id})],
    )
    sess.disable_fallback()

    inputs = {x.name for x in sess.get_inputs()}
    missing = {X_NAME, MASK_NAME} - inputs
    if missing:
        raise ValueError(f"Missing expected model inputs {sorted(missing)}; model has {sorted(inputs)}")
    return sess


def ort_dtype(type_string: str):
    table = {
        "tensor(float)": np.float32,
        "tensor(float16)": np.float16,
        "tensor(double)": np.float64,
        "tensor(int64)": np.int64,
        "tensor(int32)": np.int32,
        "tensor(bool)": np.bool_,
    }
    if type_string not in table:
        raise TypeError(f"Unsupported ONNX output type: {type_string}")
    return table[type_string]


def output_shape(meta_shape: list[Any], batch: int, seq: int) -> tuple[int, ...]:
    """Resolve the MLPF output shapes, whose first two axes are batch and sequence."""
    shape: list[int] = []
    for axis, dim in enumerate(meta_shape):
        # MLPF outputs follow [batch, sequence, ...]. Override these two axes even
        # if an older export accidentally left 1 / dummy_sequence in ValueInfo.
        if axis == 0:
            shape.append(batch)
        elif axis == 1:
            shape.append(seq)
        elif isinstance(dim, int) and dim >= 0:
            shape.append(dim)
        else:
            raise ValueError(f"Cannot resolve dynamic output shape: {meta_shape}")
    return tuple(shape)


def make_run_options() -> ort.RunOptions:
    ro = ort.RunOptions()
    # We synchronize explicitly after Run. With device-local I/O this keeps transfers
    # out of the timed interval while still waiting for CUDA execution to finish.
    try:
        ro.add_run_config_entry("disable_synchronize_execution_providers", "1")
    except Exception:
        pass
    return ro


def load_events(ds, num_events: int) -> list[tuple[int, np.ndarray]]:
    events: list[tuple[int, np.ndarray]] = []
    for i in range(num_events):
        x = np.asarray(ds[i]["X"], dtype=np.float32)
        np.clip(x, -60000.0, 60000.0, out=x)
        events.append((i, np.ascontiguousarray(x)))
    return events


def make_batches(events: list[tuple[int, np.ndarray]], batch_size: int):
    # Length sorting is the key batching optimization: similar event sizes are grouped,
    # reducing wasted padded tokens in the O(N^2) attention workload.
    events = sorted(events, key=lambda item: item[1].shape[0], reverse=True)
    for start in range(0, len(events), batch_size):
        yield events[start : start + batch_size]


def pack_batch(batch, pad_bin_size: int, x_dtype, mask_dtype):
    indices = [idx for idx, _ in batch]
    sizes = [int(x.shape[0]) for _, x in batch]
    input_dim = int(batch[0][1].shape[1])
    seq = round_up(max(sizes), pad_bin_size)

    x = np.zeros((len(batch), seq, input_dim), dtype=x_dtype)
    mask = np.zeros((len(batch), seq), dtype=mask_dtype)
    for row, (_, event) in enumerate(batch):
        n = event.shape[0]
        x[row, :n] = event
        mask[row, :n] = 1.0
    return indices, sizes, np.ascontiguousarray(x), np.ascontiguousarray(mask)

def get_input_dtypes(sess: ort.InferenceSession):
    inputs = {inp.name: inp for inp in sess.get_inputs()}

    x_dtype = ort_dtype(inputs[X_NAME].type)
    mask_dtype = ort_dtype(inputs[MASK_NAME].type)

    return x_dtype, mask_dtype

def run_gpu_batch(
    sess: ort.InferenceSession,
    x_cpu: np.ndarray,
    mask_cpu: np.ndarray,
    device_id: int,
    warmup_runs: int,
    repeats: int,
):
    """Return timed GPU inference samples and one CPU output copy for statistics."""
    batch, seq, _ = x_cpu.shape

    # H2D happens BEFORE the timer.
    x_gpu = ort.OrtValue.ortvalue_from_numpy(x_cpu, "cuda", device_id)
    mask_gpu = ort.OrtValue.ortvalue_from_numpy(mask_cpu, "cuda", device_id)

    io = sess.io_binding()
    io.bind_ortvalue_input(X_NAME, x_gpu)
    io.bind_ortvalue_input(MASK_NAME, mask_gpu)

    # Preallocate outputs on CUDA BEFORE the timer. This avoids both D2H copies and
    # first-run output allocation from contaminating the benchmark interval.
    for meta in sess.get_outputs():
        shape = output_shape(meta.shape, batch, seq)
        out = ort.OrtValue.ortvalue_from_shape_and_type(shape, ort_dtype(meta.type), "cuda", device_id)
        io.bind_ortvalue_output(meta.name, out)

    try:
        io.synchronize_inputs()
    except Exception:
        pass

    if warmup_runs > 0:
        rng = np.random.default_rng(SEED)

        x_warm_cpu = rng.standard_normal(
            x_cpu.shape
        ).astype(x_cpu.dtype)

        mask_warm_cpu = np.ones(
            mask_cpu.shape,
            dtype=mask_cpu.dtype,
        )

        x_warm_gpu = ort.OrtValue.ortvalue_from_numpy(
            np.ascontiguousarray(x_warm_cpu),
            "cuda",
            device_id,
        )
        mask_warm_gpu = ort.OrtValue.ortvalue_from_numpy(
            np.ascontiguousarray(mask_warm_cpu),
            "cuda",
            device_id,
        )

        warm_io = sess.io_binding()
        warm_io.bind_ortvalue_input(X_NAME, x_warm_gpu)
        warm_io.bind_ortvalue_input(MASK_NAME, mask_warm_gpu)

        warm_outputs = []

        for meta in sess.get_outputs():
            shape = output_shape(meta.shape, batch, seq)

            out = ort.OrtValue.ortvalue_from_shape_and_type(
                shape,
                ort_dtype(meta.type),
                "cuda",
                device_id,
            )

            warm_outputs.append(out)
            warm_io.bind_ortvalue_output(meta.name, out)

        try:
            warm_io.synchronize_inputs()
        except Exception:
            pass

        warm_opts = make_run_options()

        for _ in range(warmup_runs):
            sess.run_with_iobinding(
                warm_io,
                warm_opts,
            )
            warm_io.synchronize_outputs()

        del warm_io
        del warm_outputs
        del x_warm_gpu
        del mask_warm_gpu
        del x_warm_cpu
        del mask_warm_cpu

    run_opts = make_run_options()
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        sess.run_with_iobinding(io, run_opts)
        io.synchronize_outputs()
        samples.append(time.perf_counter() - t0)

    # D2H happens AFTER all timing is finished.
    outputs_cpu = io.copy_outputs_to_cpu()
    return samples, outputs_cpu


def merge_output_stats(acc: dict[str, list[np.ndarray]], output_names, outputs, valid_mask):
    for name, arr in zip(output_names, outputs):
        values = arr[valid_mask] if arr.ndim >= 2 and arr.shape[:2] == valid_mask.shape else arr.reshape(-1)
        acc.setdefault(name, []).append(np.asarray(values).reshape(-1))


def finalize_output_stats(acc: dict[str, list[np.ndarray]]):
    ret = {}
    for name, chunks in acc.items():
        data = np.concatenate(chunks).astype(np.float64, copy=False) if chunks else np.array([], dtype=np.float64)
        finite = np.isfinite(data)
        vals = data[finite]
        ret[name] = {
            "count": int(data.size),
            "invalid": int((~finite).sum()),
            "mean": float(vals.mean()) if vals.size else None,
            "std": float(vals.std()) if vals.size else None,
            "min": float(vals.min()) if vals.size else None,
            "max": float(vals.max()) if vals.size else None,
        }
    return ret


def main() -> None:
    args = parse_args()
    validate_args(args)
    os.makedirs(args.outdir, exist_ok=True)

    print(f"ONNX Runtime:  {ort.__version__}")
    print(f"Model:         {args.onnx_model}")
    print(f"Dataset:       {args.dataset}")
    print(f"Events:        {args.num_events}")
    print(f"Batch size:    {args.batch_size}")
    print(f"CUDA device:   {args.device_id}")
    print(f"Warmup runs:   {args.warmup_runs}")
    print(f"Timed repeats: {args.benchmark_repeats}")

    builder = tfds.builder(args.dataset, data_dir=args.data_dir)
    ds = builder.as_data_source(split="train")
    events = load_events(ds, args.num_events)
    batches = list(make_batches(events, args.batch_size))

    sess = create_session(args.onnx_model, args.device_id, args.num_threads)
    x_dtype, mask_dtype = get_input_dtypes(sess)

    output_names = [x.name for x in sess.get_outputs()]

    event_sizes = np.asarray([x.shape[0] for _, x in events], dtype=np.int64)
    gpu_mem_before = gpu_memory_used_mib(args.device_id)

    batch_records = []
    event_ms = []
    output_chunks: dict[str, list[np.ndarray]] = {}

    for batch_idx, batch in enumerate(batches):
        indices, sizes, x_cpu, mask_cpu = pack_batch(batch, args.pad_bin_size, x_dtype, mask_dtype)
        valid_mask = mask_cpu.astype(bool, copy=False)

        samples_s, outputs = run_gpu_batch(
            sess,
            x_cpu,
            mask_cpu,
            args.device_id,
            args.warmup_runs,
            args.benchmark_repeats,
        )
        merge_output_stats(output_chunks, output_names, outputs, valid_mask)

        mean_s = float(np.mean(samples_s))
        std_s = float(np.std(samples_s))
        bsz = len(batch)
        ms_per_event = mean_s * 1000.0 / bsz
        throughput = bsz / mean_s
        padding_efficiency = float(sum(sizes) / (bsz * x_cpu.shape[1]))
        event_ms.extend([ms_per_event] * bsz)

        batch_records.append(
            {
                "batch_idx": batch_idx,
                "event_indices": indices,
                "event_sizes": sizes,
                "batch_size": bsz,
                "padded_sequence_length": int(x_cpu.shape[1]),
                "padding_efficiency": padding_efficiency,
                "timed_samples_s": samples_s,
                "inference_mean_s": mean_s,
                "inference_std_s": std_s,
                "inference_ms_per_event": ms_per_event,
                "throughput_events_per_s": throughput,
                "gpu_memory_used_mib_after_batch": gpu_memory_used_mib(args.device_id),
            }
        )

    output_stats = finalize_output_stats(output_chunks)
    gpu_mem_after = gpu_memory_used_mib(args.device_id)

    total_gpu_s = float(sum(b["inference_mean_s"] for b in batch_records))
    throughput = args.num_events / total_gpu_s
    inference_ms_per_event = 1000.0 * total_gpu_s / args.num_events
    event_ms_arr = np.asarray(event_ms, dtype=np.float64)
    batch_ms_arr = np.asarray([b["inference_mean_s"] * 1000.0 for b in batch_records])
    padding_arr = np.asarray([b["padding_efficiency"] for b in batch_records])

    timing = {
        "definition": "run_with_iobinding + synchronize_outputs only; CPU batching, H2D, D2H, and stats excluded",
        "total_gpu_inference_s": total_gpu_s,
        "throughput_events_per_s": throughput,
        "inference_ms_per_event": inference_ms_per_event,
        "batch_ms_mean": float(batch_ms_arr.mean()),
        "batch_ms_std": float(batch_ms_arr.std()),
        "event_normalized_ms_mean": float(event_ms_arr.mean()),
        "event_normalized_ms_std": float(event_ms_arr.std()),
        "event_normalized_ms_median": float(np.median(event_ms_arr)),
        "event_normalized_ms_p95": float(np.percentile(event_ms_arr, 95)),
        "event_normalized_ms_p99": float(np.percentile(event_ms_arr, 99)),
    }

    summary = {
        "model": {
            "name": "ONNX_ATTN_FLASH_FP32",
            "path": os.path.abspath(args.onnx_model),
            "provider": "CUDAExecutionProvider",
        },
        "num_processed_events": args.num_events,
        "batch_size": args.batch_size,
        "num_batches": len(batch_records),
        "pad_bin_size": args.pad_bin_size,
        "warmup_runs": args.warmup_runs,
        "benchmark_repeats": args.benchmark_repeats,
        "event_size_distribution": event_sizes.tolist(),
        "event_size_statistics": {
            "min": int(event_sizes.min()),
            "max": int(event_sizes.max()),
            "mean": float(event_sizes.mean()),
            "median": float(np.median(event_sizes)),
            "std": float(event_sizes.std()),
        },
        "padding_statistics": {
            "mean_efficiency": float(padding_arr.mean()),
            "min_efficiency": float(padding_arr.min()),
        },
        "timing": timing,
        "memory": {
            "gpu_memory_used_mib_before": gpu_mem_before,
            "gpu_memory_used_mib_after": gpu_mem_after,
            "process_rss_mib_after": process_rss_mib(),
            "note": "GPU numbers are nvidia-smi snapshots outside the timed interval, not allocator peak memory.",
        },
        "output_statistics": output_stats,
        "system": {
            "cpu": get_cpu_info(),
            "gpu": get_gpu_info(args.device_id),
            "python": platform.python_version(),
            "onnxruntime_version": ort.__version__,
            "providers": sess.get_providers(),
            "num_threads": args.num_threads,
        },
        "batches": batch_records,
    }

    summary_path = os.path.join(args.outdir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== ONNX GPU BENCHMARK ===")
    print(f"Processed events:        {args.num_events}")
    print(f"Batches:                 {len(batch_records)} (batch-size={args.batch_size})")
    print(
        f"Event sizes:             mean={event_sizes.mean():.1f}, median={np.median(event_sizes):.1f}, "
        f"min={event_sizes.min()}, max={event_sizes.max()}"
    )
    print(f"Mean padding efficiency: {100.0 * padding_arr.mean():.2f}%")
    print(f"GPU inference total:     {1000.0 * total_gpu_s:.3f} ms")
    print(f"Inference / event:       {inference_ms_per_event:.3f} ms/event")
    print(f"Throughput:              {throughput:.2f} events/s")
    print(
        f"Normalized event time:   mean={event_ms_arr.mean():.3f} ms, median={np.median(event_ms_arr):.3f} ms, "
        f"p95={np.percentile(event_ms_arr, 95):.3f} ms, p99={np.percentile(event_ms_arr, 99):.3f} ms"
    )
    print(f"Batch inference:         {batch_ms_arr.mean():.3f} +/- {batch_ms_arr.std():.3f} ms")
    print("Timing excludes:         CPU padding/batching + H2D + D2H + statistics")

    if gpu_mem_before is not None or gpu_mem_after is not None:
        print(f"GPU memory snapshots:    before={gpu_mem_before} MiB, after={gpu_mem_after} MiB")

    print("\nOutput statistics (unpadded elements):")
    for name, s in output_stats.items():
        print(
            f"  {name:10s} count={s['count']:,} invalid={s['invalid']:,} "
            f"mean={s['mean']} std={s['std']} min={s['min']} max={s['max']}"
        )

    print(f"\nSummary JSON: {summary_path}")


if __name__ == "__main__":
    main()
