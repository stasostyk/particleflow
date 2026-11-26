#!/usr/bin/env python3
r"""Benchmark single-event MLPF inference on synthetic inputs of increasing size.

Example (first run computes and saves the FP32 reference; later runs reuse it):
    python input_size.py \
        --model-kwargs model_kwargs.pkl \
        --checkpoint checkpoint.pt \
        --pad-bin-size 128 \
        --num-events 5000 \
        --mae-events 100 \
        --save-reference fp32_reference/ \
        --output-pdf event_size_scaling.pdf

    python input_size.py ... --load-reference fp32_reference/ --output-pdf run2.pdf

Configure VARIANT_NAMES and the event-size sweep below. For every event size,
all variants process the SAME deterministic synthetic events (B=1): event i of
size S is identical in every process and every run on the same setup.

* Timing uses --num-events events per size, generated on the GPU in chunks of
  --chunk-events. The clock is paused while each chunk is generated, so GPU
  memory stays bounded and input preparation is never timed.
* Accuracy uses the first --mae-events of those events. The golden reference
  (fp32_efficient, never compiled) runs untimed on just those events and writes
  its predictions to disk, one file per size; variants read one size at a time.
  --load-reference skips the reference run entirely and performs NO checks that
  the saved reference matches the current settings.

The PDF has four charts (reference variant not plotted):

* Throughput: timed events / summed chunk wall time, including Python dispatch
  and CUDA execution, excluding input generation and MAE.
* Mean latency: mean per-event CUDA-event forward time, 95% Student-t CI.
* p95 latency: 95th percentile of the same per-event forward times.
* Accuracy: mean per-event, element-weighted MAE across ID/momentum/PU heads,
  with 95% Student-t confidence intervals over the MAE events.

Inputs are always generated as FP32; any lower precision is applied by the
variant/worker itself. With no warmup, first-use/torch.compile startup affects
the first events of each size. Before the sweep, each process runs the same
untimed backend/dispatch checks as benchmark_worker.py on the *eager* model
(set VERIFY_BEFORE_SWEEP = False to skip them). Each variant runs in a fresh
CUDA-owning process, like benchmark_worker.py.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import multiprocessing as mp
from pathlib import Path
import tempfile
from time import perf_counter

from benchmark_variants import REFERENCE_NAME, REGISTRY


# ---------------------------------------------------------------------------
# User configuration: benchmarked variants and event-size sweep
# ---------------------------------------------------------------------------
# Names must exist in benchmark_variants.REGISTRY.
VARIANT_NAMES = (
    "flash_attention_3_fp16_compiled",
    # "flash_attention_3_bf16_compiled",
    "flash_attention_3_fp8_clamp_compiled",    # FP8: plain clamped cast
    "flash_attention_3_fp8_static_compiled",   # FP8: static per-head scales
    "flash_attention_3_fp8_compiled",          # FP8: static scales + Hadamard
)

MIN_EVENT_SIZE = 100    # Minimum number of real input elements
MAX_EVENT_SIZE = 25000  # Maximum number of real input elements
NUM_SAMPLES = 50        # Number of distinct, linearly spaced x-axis points

DEVICE = "cuda:0"
SEED = 1234

HEAD_NAMES = ("id", "momentum", "pu")  # Must match benchmark_worker.HEADS
# Untimed eager backend/dispatch verification, mirroring benchmark_worker.run_job.
VERIFY_BEFORE_SWEEP = True
# Static FP8 scales are calibrated once per variant process on synthetic events
# drawn with a DIFFERENT seed than the timed/MAE events, at sizes spread over
# the sweep (so the largest sizes' value ranges are covered).
FP8_CALIBRATION_SIZES = 8
FP8_CALIBRATION_EVENTS_PER_SIZE = 4
CALIBRATION_SEED_OFFSET = 7_777_777
NAME_WIDTH = max(len(name) for name in REGISTRY)


@dataclass(frozen=True)
class RunSettings:
    model_kwargs: Path
    checkpoint: Path
    pad_bin_size: int
    num_events: int
    mae_events: int
    chunk_events: int


@dataclass(frozen=True)
class Measurement:
    variant: str
    event_size: int
    padded_size: int
    num_events: int
    mae_events: int
    mean_latency_ms: float
    mean_latency_ci95_low: float
    mean_latency_ci95_high: float
    p95_latency_ms: float
    throughput_events_s: float
    mae_mean: float
    mae_ci95_low: float
    mae_ci95_high: float
    gpu: str


def event_sizes() -> tuple[int, ...]:
    """Produce exactly NUM_SAMPLES distinct integer sizes, inclusive."""
    if MIN_EVENT_SIZE < 1 or MAX_EVENT_SIZE < MIN_EVENT_SIZE:
        raise ValueError("Require 1 <= MIN_EVENT_SIZE <= MAX_EVENT_SIZE")
    if NUM_SAMPLES < 2 or NUM_SAMPLES > MAX_EVENT_SIZE - MIN_EVENT_SIZE + 1:
        raise ValueError("NUM_SAMPLES must be >= 2 and fit the integer size range")
    steps = NUM_SAMPLES - 1
    return tuple(
        MIN_EVENT_SIZE + (i * (MAX_EVENT_SIZE - MIN_EVENT_SIZE) + steps // 2) // steps
        for i in range(NUM_SAMPLES)
    )


def reference_path(directory: Path, size: int) -> Path:
    return directory / f"reference_size_{size:06d}.pt"


class EventStream:
    """Deterministic synthetic B=1 events of one size, generated on the GPU.

    Each size has its own seeded CUDA generator, and every event is drawn with
    its own call in order, so event i is identical regardless of chunk size,
    --num-events, or which process generates it. Padding rows stay zero.
    A reusable buffer holds at most `capacity` events; it must not be refilled
    while the previous chunk is still running (callers synchronize first).
    """

    def __init__(self, size: int, padded: int, input_dim: int, capacity: int,
                 device, transform, seed_offset: int = 0):
        import torch

        self.size = size
        self.transform = transform
        self.generator = torch.Generator(device=device).manual_seed(
            SEED * 1_000_003 + size + seed_offset)
        self.buffer = torch.zeros((capacity, 1, padded, input_dim),
                                  dtype=torch.float32, device=device)
        self.mask = torch.zeros((1, padded), dtype=torch.bool, device=device)
        self.mask[:, :size] = True

    def next_chunk(self, count: int) -> list:
        import torch

        if count > self.buffer.shape[0]:
            raise ValueError("Chunk larger than the event buffer")
        events = []
        for j in range(count):
            x = self.buffer[j]
            x[0, :self.size].normal_(generator=self.generator)
            mask = self.mask
            if self.transform is not None:
                x, mask = self.transform(x, mask)
                if (not isinstance(x, torch.Tensor) or not isinstance(mask, torch.Tensor)
                        or x.dtype != torch.float32 or mask.dtype != torch.bool
                        or x.ndim != 3 or mask.ndim != 2 or x.shape[:2] != mask.shape
                        or x.shape[0] != 1):
                    raise ValueError("input_transform must return FP32 (1, L, D) and bool (1, L) tensors")
            events.append((x, mask))
        return events


def iterate_events(stream: EventStream, total: int, chunk: int):
    """Yield chunks of (x, mask) pairs until `total` events have been produced."""
    for start in range(0, total, chunk):
        yield stream.next_chunk(min(chunk, total - start))


def calibration_events(input_dim: int, pad_multiple: int, device, transform):
    """Yield synthetic FP8 calibration events (separate seed from timed/MAE events)."""
    steps = FP8_CALIBRATION_SIZES - 1
    sizes = sorted({MIN_EVENT_SIZE + (i * (MAX_EVENT_SIZE - MIN_EVENT_SIZE)) // max(steps, 1)
                    for i in range(FP8_CALIBRATION_SIZES)})
    for size in sizes:
        padded = ((size + pad_multiple - 1) // pad_multiple * pad_multiple
                  if pad_multiple else size)
        stream = EventStream(size, padded, input_dim, FP8_CALIBRATION_EVENTS_PER_SIZE,
                             device, transform, seed_offset=CALIBRATION_SEED_OFFSET)
        yield from stream.next_chunk(FP8_CALIBRATION_EVENTS_PER_SIZE)


def compute_mae(predicted: dict, reference: dict) -> float:
    """Worker-compatible, element-weighted MAE across valid ID/momentum/PU."""
    import torch

    total_difference = 0.0
    total_elements = 0
    for head in HEAD_NAMES:
        actual, expected = predicted[head], reference[head]
        if actual.shape != expected.shape:
            raise ValueError(
                f"{head} output shape differs from reference: {actual.shape} != {expected.shape}"
            )
        if actual.numel() == 0:
            raise ValueError(f"{head} has no valid output elements")
        difference = (actual.float() - expected.float()).abs()
        if not bool(torch.isfinite(difference).all()):
            raise ValueError(f"{head} output difference contains nonfinite values")
        total_difference += float(difference.double().sum())
        total_elements += difference.numel()
    return total_difference / total_elements


def mean_ci95(values, *, clip_at_zero: bool = False) -> tuple[float, float, float]:
    """Mean and a two-sided 95% Student-t confidence interval."""
    import numpy as np
    from scipy.stats import t

    array = np.asarray(values, dtype=np.float64)
    if array.size < 2 or not np.isfinite(array).all():
        raise ValueError("Confidence interval requires >= 2 finite values")
    mean = float(array.mean())
    sem = float(array.std(ddof=1) / np.sqrt(array.size))
    half_width = float(t.ppf(0.975, array.size - 1)) * sem
    low = mean - half_width
    return mean, (max(0.0, low) if clip_at_zero else low), mean + half_width


def verify_variant(model, variant, config, pad_multiple: int, device) -> None:
    """Untimed eager checks mirroring benchmark_worker.run_job (no silent fallback)."""
    import torch

    from benchmark_worker import (
        FULL_FP16,
        verify_dispatch,
        verify_external_kernel_event_isolation,
        verify_fp32_layers,
        verify_fp32_parameters,
    )

    attention = config.model.attention
    if variant.attention.backend is not None and not FULL_FP16:
        from attention_backends import validate_backend
        validate_backend(variant.attention.backend, variant.attention.kernel_dtype,
                         device, int(attention.head_dim))
    verify_fp32_parameters(model)
    verify_external_kernel_event_isolation(variant, device, int(attention.num_heads),
                                           int(attention.head_dim))
    seq_len = max(MIN_EVENT_SIZE, 3)
    if pad_multiple:
        seq_len = (seq_len + pad_multiple - 1) // pad_multiple * pad_multiple
    verify_fp32_layers(model, variant, int(config.input_dim), seq_len, device)
    result = verify_dispatch(model, variant, int(config.input_dim), seq_len, device)
    torch.cuda.synchronize(device)
    print(f"  {variant.name}: dispatch verified -> {result.get('matched_ops')}", flush=True)


def benchmark_variant(name: str, sizes: tuple[int, ...], settings: RunSettings,
                      reference_dir: Path, is_reference: bool) -> list[Measurement]:
    """Run in a fresh process.

    Reference: untimed, writes one prediction file per size to reference_dir and
    returns []. Variant: returns small per-size summaries only; reference
    predictions are read from disk one size at a time, so neither inter-process
    transfer nor CPU memory grows with the number of sizes or events.
    """
    import numpy as np
    import torch

    from benchmark_variants import get_variant
    from benchmark_worker import (
        HEADS,
        build_model,
        compiled_graph_count,
        extract_event_heads,
        forward,
        load_config_and_state,
        prepare_fp8,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA and an NVIDIA GPU")
    device = torch.device(DEVICE)
    torch.cuda.set_device(device)
    variant = get_variant(name)
    torch.manual_seed(SEED)
    if tuple(HEADS) != HEAD_NAMES:
        raise RuntimeError(f"HEAD_NAMES {HEAD_NAMES} differ from benchmark_worker.HEADS {tuple(HEADS)}")

    # Same golden-reference contract that benchmark_worker.run_job enforces.
    if is_reference and name != REFERENCE_NAME:
        raise RuntimeError(f"The only golden reference is {REFERENCE_NAME}")
    if not is_reference and name == REFERENCE_NAME:
        raise ValueError(f"{REFERENCE_NAME} must run as the golden reference, not a timed variant")
    if is_reference and variant.compile:
        raise RuntimeError("The golden FP32 reference must never be compiled")

    torch.backends.cuda.matmul.allow_tf32 = variant.precision.allow_tf32
    torch.backends.cudnn.allow_tf32 = variant.precision.allow_tf32
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision(
        "high" if variant.precision.allow_tf32 else "highest"
    )
    if variant.attention.kernel_dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError(f"{name}: BF16 is not supported on this GPU")

    config, state, pad_multiple = load_config_and_state({
        "model_kwargs": str(settings.model_kwargs),
        "checkpoint": str(settings.checkpoint),
        "pad_bin_size": settings.pad_bin_size,
        "fixed_seq_len": None,
    })
    model = build_model(config, state, variant, device, partial=False)
    del state
    # FP8 only: fold rotation / calibrate static scales on the eager model.
    prepare_fp8(model, variant,
                calibration_events(int(config.input_dim), pad_multiple, device,
                                   variant.input_transform), device)
    if VERIFY_BEFORE_SWEEP:
        verify_variant(model, variant, config, pad_multiple, device)
    # Like the worker: verify eager first, then compile the timed variant.
    if variant.compile:
        model = torch.compile(model, mode=variant.compile_mode)

    input_dim = int(config.input_dim)
    gpu_name = torch.cuda.get_device_name(device)
    measurements: list[Measurement] = []
    chunk = settings.chunk_events

    with torch.inference_mode():
        for size in sizes:
            padded = ((size + pad_multiple - 1) // pad_multiple * pad_multiple
                      if pad_multiple else size)

            if is_reference:
                # Untimed: only the MAE events, outputs concatenated per head.
                stream = EventStream(size, padded, input_dim,
                                     min(chunk, settings.mae_events), device,
                                     variant.input_transform)
                captured = {head: [] for head in HEAD_NAMES}
                index = 0
                for events in iterate_events(stream, settings.mae_events, chunk):
                    for x, mask in events:
                        output = forward(model, variant, x, mask)
                        heads = extract_event_heads(output, mask.cpu(), 0, index)
                        for head, value in heads.items():
                            if value.shape[0] != size:
                                raise ValueError(f"{head}: expected {size} output rows, got {value.shape[0]}")
                            if not bool(torch.isfinite(value).all()):
                                raise ValueError(
                                    f"Reference outputs contain nonfinite values: size={size}, event={index}"
                                )
                            captured[head].append(value)
                        index += 1
                        del output, heads
                    torch.cuda.synchronize(device)
                torch.save({"size": size, "events": settings.mae_events,
                            "heads": {h: torch.cat(v) for h, v in captured.items()}},
                           reference_path(reference_dir, size))
                print(f"  {name:{NAME_WIDTH}s} size={size:6d} padded={padded:6d} "
                      f"saved {settings.mae_events} reference events (untimed)", flush=True)
                del stream, captured
                continue

            # ---- Timed sweep: chunks of GPU-generated events; clock paused
            # while each chunk is generated. ----
            stream = EventStream(size, padded, input_dim,
                                 min(chunk, settings.num_events), device,
                                 variant.input_transform)
            latencies: list[float] = []
            elapsed_s = 0.0
            for events in iterate_events(stream, settings.num_events, chunk):
                markers = [(torch.cuda.Event(enable_timing=True),
                            torch.cuda.Event(enable_timing=True)) for _ in events]
                torch.cuda.synchronize(device)
                chunk_start = perf_counter()
                for (x, mask), (start, end) in zip(events, markers):
                    start.record()
                    output = forward(model, variant, x, mask)
                    end.record()
                    del output
                torch.cuda.synchronize(device)
                elapsed_s += perf_counter() - chunk_start
                latencies.extend(start.elapsed_time(end) for start, end in markers)
                del events, markers
            if elapsed_s <= 0 or any(ms <= 0 for ms in latencies):
                raise RuntimeError(f"{name}: invalid inference timing at size {size}")
            del stream

            # ---- Accuracy: the first MAE events, regenerated identically, in a
            # separate untimed pass so output copies never affect timing. ----
            saved = torch.load(reference_path(reference_dir, size), map_location="cpu",
                               weights_only=True)
            mae_events = int(saved["events"])
            reference_heads = saved["heads"]
            stream = EventStream(size, padded, input_dim, min(chunk, mae_events),
                                 device, variant.input_transform)
            event_errors = []
            index = 0
            for events in iterate_events(stream, mae_events, chunk):
                for x, mask in events:
                    output = forward(model, variant, x, mask)
                    heads = extract_event_heads(output, mask.cpu(), 0, index)
                    rows = slice(index * size, (index + 1) * size)
                    expected = {h: reference_heads[h][rows] for h in HEAD_NAMES}
                    event_errors.append(compute_mae(heads, expected))
                    index += 1
                    del output, heads, expected
                torch.cuda.synchronize(device)
            del stream, saved, reference_heads

            mae_mean, mae_low, mae_high = mean_ci95(event_errors, clip_at_zero=True)
            mean_ms, mean_low, mean_high = mean_ci95(latencies)
            p95_ms = float(np.percentile(latencies, 95))
            events_per_s = settings.num_events / elapsed_s
            measurements.append(Measurement(
                variant=name,
                event_size=size,
                padded_size=padded,
                num_events=settings.num_events,
                mae_events=mae_events,
                mean_latency_ms=mean_ms,
                mean_latency_ci95_low=mean_low,
                mean_latency_ci95_high=mean_high,
                p95_latency_ms=p95_ms,
                throughput_events_s=events_per_s,
                mae_mean=mae_mean,
                mae_ci95_low=mae_low,
                mae_ci95_high=mae_high,
                gpu=gpu_name,
            ))
            print(f"  {name:{NAME_WIDTH}s} size={size:6d} padded={padded:6d} "
                  f"mean={mean_ms:8.3f} ms  p95={p95_ms:8.3f} ms  "
                  f"observed={events_per_s:8.1f} events/s  "
                  f"MAE={mae_mean:.5g} [95% CI: {mae_low:.5g}, {mae_high:.5g}]",
                  flush=True)
            del latencies, event_errors

    if variant.compile and compiled_graph_count() == 0:
        raise RuntimeError(f"{name}: torch.compile produced zero graphs; "
                           "compiled variant is not actually compiled")
    return measurements


def plot_pdf(rows: list[Measurement], names: tuple[str, ...], reference: str,
             path: Path) -> None:
    """Plot throughput, mean latency, p95 latency and MAE on one page."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(15.4, 11.4))
    grid = fig.add_gridspec(2, 2, left=0.085, right=0.975, top=0.86,
                            bottom=0.2, hspace=0.36, wspace=0.23)
    ax_rate = fig.add_subplot(grid[0, 0])
    ax_mean = fig.add_subplot(grid[0, 1])
    ax_p95 = fig.add_subplot(grid[1, 0])
    ax_mae = fig.add_subplot(grid[1, 1])
    palette = plt.get_cmap("tab10")

    for index, name in enumerate(names):
        series = sorted((row for row in rows if row.variant == name),
                        key=lambda row: row.event_size)
        xs = [row.event_size for row in series]
        style = dict(color=palette(index % 10), linewidth=1.7,
                     marker="o", markersize=3.5, label=name)
        ax_rate.plot(xs, [row.throughput_events_s for row in series], **style)
        ax_p95.plot(xs, [row.p95_latency_ms for row in series], **style)
        ax_mean.errorbar(
            xs,
            [row.mean_latency_ms for row in series],
            yerr=[
                [row.mean_latency_ms - row.mean_latency_ci95_low for row in series],
                [row.mean_latency_ci95_high - row.mean_latency_ms for row in series],
            ],
            capsize=2.5, elinewidth=0.9, **style,
        )
        ax_mae.errorbar(
            xs,
            [row.mae_mean for row in series],
            yerr=[
                [row.mae_mean - row.mae_ci95_low for row in series],
                [row.mae_ci95_high - row.mae_mean for row in series],
            ],
            capsize=2.5, elinewidth=0.9, **style,
        )

    ax_rate.set(title="Observed throughput vs. event size",
                ylabel="Timed events / summed chunk wall time (events/s)")
    ax_mean.set(title="Mean latency vs. event size",
                ylabel="Mean GPU forward latency (ms/event, 95% CI)")
    ax_p95.set(title="p95 tail latency vs. event size",
               ylabel="p95 GPU forward latency (ms/event)")
    ax_mae.set(title=f"Mean prediction MAE vs. event size  |  reference: {reference}",
               ylabel="Mean per-event MAE (95% CI)")
    for ax in (ax_rate, ax_mean, ax_p95, ax_mae):
        ax.set_xlabel("Event size (real input elements)")
        ax.set_xlim(min(row.event_size for row in rows), max(row.event_size for row in rows))
        ax.grid(True, alpha=0.25)
        ax.set_axisbelow(True)
        ax.margins(y=0.08)
    ax_mae.set_ylim(bottom=0)

    fig.suptitle("MLPF  |  Single-event inference scaling", fontsize=18,
                 fontweight="bold", y=0.965)
    fig.text(0.5, 0.918,
             f"{rows[0].gpu}  |  batch size 1  |  {rows[0].num_events} timed events/size  |  "
             f"MAE on {rows[0].mae_events} events/size  |  no warmup",
             ha="center", fontsize=11, color="#566273")
    handles, labels = ax_rate.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.09),
               ncol=min(4, len(names)), frameon=False, fontsize=9.5)
    fig.text(0.5, 0.032,
             "Reference: untimed and unplotted. Throughput: summed wall time of GPU-resident event "
             "chunks (no input generation or MAE).",
             ha="center", fontsize=8.5, color="#566273")
    fig.text(0.5, 0.014,
             "Latency: per-event CUDA-event forward time (mean with 95% Student-t CI, and p95). "
             "MAE: per-event mean over ID/momentum/PU; error bars: 95% Student-t CI.",
             ha="center", fontsize=8.5, color="#566273")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference", choices=(REFERENCE_NAME,), default=REFERENCE_NAME,
                        help=f"Golden reference variant (untimed, MAE only; default: {REFERENCE_NAME})")
    parser.add_argument("--pad-bin-size", type=int, required=True,
                        help="Additional padding multiple (0 means model-config padding only)")
    parser.add_argument("--model-kwargs", type=Path, required=True,
                        help="Pickled MLPF model configuration")
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="PyTorch model checkpoint")
    parser.add_argument("--output-pdf", type=Path, required=True,
                        help="Destination for the four-chart PDF")
    parser.add_argument("--num-events", type=int, default=5000,
                        help="Timed synthetic events at EACH size (default: 5000; min: 2)")
    parser.add_argument("--mae-events", type=int, default=100,
                        help="First N events per size used for MAE and the reference "
                             "(default: 100; min: 2; ignored with --load-reference)")
    parser.add_argument("--chunk-events", type=int, default=256,
                        help="Events kept on the GPU at once during timing (default: 256)")
    reference_io = parser.add_mutually_exclusive_group()
    reference_io.add_argument("--save-reference", type=Path, metavar="DIR",
                              help="Compute the FP32 reference and save it to DIR for reuse")
    reference_io.add_argument("--load-reference", type=Path, metavar="DIR",
                              help="Reuse a reference saved with --save-reference "
                                   "(no compatibility checks; skips the reference run)")
    args = parser.parse_args()

    if args.pad_bin_size < 0:
        parser.error("--pad-bin-size must be nonnegative")
    if args.num_events < 2:
        parser.error("--num-events must be >= 2 to calculate latency confidence intervals")
    if args.mae_events < 2:
        parser.error("--mae-events must be >= 2 to calculate MAE confidence intervals")
    if args.chunk_events < 1:
        parser.error("--chunk-events must be >= 1")
    if args.output_pdf.suffix.lower() != ".pdf":
        parser.error("--output-pdf must end in .pdf")
    for model_path in (args.model_kwargs, args.checkpoint):
        if not model_path.is_file():
            parser.error(f"Model file not found: {model_path}")
    if args.output_pdf.resolve() in {args.model_kwargs.resolve(), args.checkpoint.resolve()}:
        parser.error("--output-pdf must not overwrite a model file")
    if args.load_reference is not None and not args.load_reference.is_dir():
        parser.error(f"--load-reference directory not found: {args.load_reference}")

    # Never benchmark or plot the reference, even if it occurs in VARIANT_NAMES.
    names = tuple(dict.fromkeys(name for name in VARIANT_NAMES if name != args.reference))
    unknown = [name for name in names if name not in REGISTRY]
    if unknown:
        parser.error(f"Unknown names in VARIANT_NAMES: {unknown}; available: {list(REGISTRY)}")
    if not names:
        parser.error("VARIANT_NAMES must include at least one non-reference variant")
    sizes = event_sizes()
    settings = RunSettings(args.model_kwargs, args.checkpoint, args.pad_bin_size,
                           args.num_events, args.mae_events, args.chunk_events)
    print(f"Reference (untimed): {args.reference} | sizes: {sizes} | GPU: {DEVICE}", flush=True)
    print(f"Measured variants: {', '.join(names)} | timed events/size: {args.num_events} | "
          f"chunk: {args.chunk_events}", flush=True)

    rows: list[Measurement] = []
    context = mp.get_context("spawn")
    # Each OS process destroys its CUDA context before the next variant runs.
    # Reference predictions travel through files, never through the process pool.
    with tempfile.TemporaryDirectory(prefix="mlpf_reference_",
                                     dir=args.output_pdf.resolve().parent) as scratch:
        if args.load_reference is not None:
            reference_dir = args.load_reference
            print(f"\n[reference] reusing {reference_dir} (not recomputed)", flush=True)
        else:
            reference_dir = args.save_reference if args.save_reference else Path(scratch)
            reference_dir.mkdir(parents=True, exist_ok=True)
            print(f"\n[reference] {args.reference} -> {reference_dir} "
                  f"({args.mae_events} events/size)", flush=True)
            with ProcessPoolExecutor(max_workers=1, mp_context=context) as executor:
                executor.submit(benchmark_variant, args.reference, sizes, settings,
                                reference_dir, True).result()
        for name in names:
            print(f"\n[variant] {name}", flush=True)
            with ProcessPoolExecutor(max_workers=1, mp_context=context) as executor:
                rows.extend(executor.submit(
                    benchmark_variant, name, sizes, settings, reference_dir, False
                ).result())

    plot_pdf(rows, names, args.reference, args.output_pdf)
    print(f"\nPDF: {args.output_pdf.resolve()}", flush=True)


if __name__ == "__main__":
    main()