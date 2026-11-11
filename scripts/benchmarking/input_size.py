#!/usr/bin/env python3
r"""Benchmark single-event MLPF inference on synthetic inputs of increasing size.

Example:
    python benchmark_event_scaling.py fp32_efficient \
        --model-kwargs model_kwargs.pkl \
        --checkpoint checkpoint.pt \
        --pad-bin-size 256 \
        --num-events 64 \
        --output-pdf event_size_scaling.pdf

Configure VARIANT_NAMES and the event-size sweep below. For every event size,
all variants process the SAME independently generated synthetic events (B=1).
The reference variant runs first, untimed, and its predictions remain in memory.
Only non-reference variants appear in the three PDF charts:

* Latency: p95 of per-event CUDA forward times, excluding input preparation.
* Throughput: completed events / elapsed sweep wall time, including Python
  dispatch and CUDA execution, but excluding input preparation and MAE.
* Accuracy: mean per-event, element-weighted MAE across ID/momentum/PU heads,
  with 95% Student-t confidence intervals over independent events.

No warmup, pinned memory, CSV, or reference-prediction files. The benchmark
keeps each size's inputs GPU-resident for the measured sweep, and runs accuracy
inference separately so output copying does not contaminate throughput.
With no warmup, first-use/torch.compile startup can affect observed throughput.
Each variant runs in a fresh CUDA-owning process, like benchmark_worker.py.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import multiprocessing as mp
from pathlib import Path
from time import perf_counter

from benchmark_variants import REGISTRY


# ---------------------------------------------------------------------------
# User configuration: benchmarked variants and event-size sweep
# ---------------------------------------------------------------------------
VARIANT_NAMES = (
    "fp16_flash",
    "bf16_flash",
    "fp16_flash_compiled",
    "bf16_flash_compiled",
)

MIN_EVENT_SIZE = 100   # Minimum number of real input elements
MAX_EVENT_SIZE = 20000  # Maximum number of real input elements
NUM_SAMPLES = 50       # Number of distinct, linearly spaced x-axis points

DEVICE = "cuda:0"
SEED = 1234

HEAD_NAMES = ("id", "momentum", "pu")


@dataclass(frozen=True)
class RunSettings:
    model_kwargs: Path
    checkpoint: Path
    pad_bin_size: int
    num_events: int


@dataclass(frozen=True)
class Measurement:
    variant: str
    event_size: int
    padded_size: int
    num_events: int
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


def synthetic_event(size: int, padded: int, input_dim: int, generator, dtype, transform):
    """Make a distinct, reproducible event with zero padding on the CPU.

    Generate *only* real FP32 rows with the dedicated generator, then pad and
    convert. Therefore, differences in model padding or input dtype cannot
    change the generated values for subsequent events/variants.
    """
    import torch

    real = torch.randn((1, size, input_dim), generator=generator)
    x = torch.zeros((1, padded, input_dim), dtype=torch.float32)
    x[:, :size] = real
    x = x.to(dtype=dtype)
    mask = torch.zeros((1, padded), dtype=torch.bool)
    mask[:, :size] = True
    if transform is not None:
        x, mask = transform(x, mask)
    if (not isinstance(x, torch.Tensor) or not isinstance(mask, torch.Tensor)
            or x.device.type != "cpu" or mask.device.type != "cpu"
            or mask.dtype != torch.bool or mask.ndim != 2 or x.ndim != 3
            or x.shape[:2] != mask.shape or x.shape[0] != 1):
        raise ValueError("input_transform must return CPU (B=1, L, D) and bool (B=1, L) tensors")
    if int(mask.sum()) != size:
        raise ValueError("input_transform changed the event's number of valid elements")
    return x, mask


def prepare_events(size: int, padded: int, dim: int, count: int,
                   generator, dtype, transform, device):
    """Prepare GPU-resident B=1 inputs before timing the sweep.

    Stacking into one allocation per tensor minimizes transfers and overhead.
    Each slice retains B=1, and each source event has independent random data.
    Keep CPU masks for extracting element-aligned output heads after inference.
    """
    import torch

    batches = [synthetic_event(size, padded, dim, generator, dtype, transform)
               for _ in range(count)]
    shapes = {(tuple(x.shape), tuple(mask.shape)) for x, mask in batches}
    if len(shapes) != 1:
        raise ValueError("input_transform produced inconsistent shapes within one event size")
    xs = torch.stack([x for x, _ in batches]).to(device)
    masks = torch.stack([mask for _, mask in batches]).to(device)
    cpu_masks = [mask for _, mask in batches]
    return xs, masks, cpu_masks


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


def mean_mae_ci95(values: list[float]) -> tuple[float, float, float]:
    """Mean per-event MAE and a two-sided 95% Student-t confidence interval."""
    import numpy as np
    from scipy.stats import t

    array = np.asarray(values, dtype=np.float64)
    if array.size < 2 or not np.isfinite(array).all():
        raise ValueError("MAE confidence interval requires >= 2 finite events")
    mean = float(array.mean())
    sem = float(array.std(ddof=1) / np.sqrt(array.size))
    half_width = float(t.ppf(0.975, array.size - 1)) * sem
    return mean, max(0.0, mean - half_width), mean + half_width


def benchmark_variant(name: str, sizes: tuple[int, ...], settings: RunSettings,
                      reference_heads: dict[int, list[dict]] | None = None,
                      ) -> dict[int, list[dict]] | list[Measurement]:
    """Run in a fresh process; return untimed reference or measured summaries.

    Reference head tensors are copied to CPU and returned to the parent in RAM;
    nothing is saved to disk except the final PDF in the parent process.
    """
    import numpy as np
    import torch

    from benchmark_variants import get_variant
    from benchmark_worker import (
        build_model,
        dtype_of,
        extract_event_heads,
        forward,
        load_config_and_state,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA and an NVIDIA GPU")
    device = torch.device(DEVICE)
    torch.cuda.set_device(device)
    variant = get_variant(name)
    torch.manual_seed(SEED)

    torch.backends.cuda.matmul.allow_tf32 = variant.precision.allow_tf32
    torch.backends.cudnn.allow_tf32 = variant.precision.allow_tf32
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision(
        "high" if variant.precision.allow_tf32 else "highest"
    )
    if ("bfloat16" in (variant.precision.weights, variant.precision.input_dtype)
            and not torch.cuda.is_bf16_supported()):
        raise RuntimeError(f"{name}: BF16 is not supported on this GPU")

    config, state, pad_multiple = load_config_and_state({
        "model_kwargs": str(settings.model_kwargs),
        "checkpoint": str(settings.checkpoint),
        "pad_bin_size": settings.pad_bin_size,
        "fixed_seq_len": None,
    })
    model = build_model(config, state, variant, device, partial=False)
    del state
    if variant.compile:
        model = torch.compile(model, mode=variant.compile_mode)

    is_reference = reference_heads is None
    if not is_reference:
        if set(reference_heads) != set(sizes):
            raise ValueError("Reference event sizes do not match the selected samples")
        if any(len(reference_heads[size]) != settings.num_events for size in sizes):
            raise ValueError("Reference event counts do not match --num-events")

    input_dtype = dtype_of(variant.precision.input_dtype)
    input_dim = int(config.input_dim)
    gpu_name = torch.cuda.get_device_name(device)
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    measurements: list[Measurement] = []
    captured: dict[int, list[dict]] = {}

    with torch.inference_mode():
        for size in sizes:
            padded = ((size + pad_multiple - 1) // pad_multiple * pad_multiple
                      if pad_multiple else size)
            xs, masks, cpu_masks = prepare_events(
                size, padded, input_dim, settings.num_events, generator,
                input_dtype, variant.input_transform, device,
            )

            if is_reference:
                # No CUDA timing events, or performance metrics for the reference.
                reference_events = []
                for index in range(settings.num_events):
                    output = forward(model, variant, xs[index], masks[index])
                    heads = extract_event_heads(output, cpu_masks[index], 0, index)
                    if any(not bool(torch.isfinite(head).all()) for head in heads.values()):
                        raise ValueError(
                            f"Reference outputs contain nonfinite values: size={size}, event={index}"
                        )
                    reference_events.append(heads)
                    del output, heads
                captured[size] = reference_events
                print(f"  {name:23s} size={size:6d} padded={padded:6d} "
                      f"captured {settings.num_events} reference events (untimed)", flush=True)
            else:
                # GPU inputs already prepared; output transfer and accuracy are
                # done in a completely separate, untimed pass below.
                markers = [
                    (torch.cuda.Event(enable_timing=True),
                     torch.cuda.Event(enable_timing=True))
                    for _ in range(settings.num_events)
                ]
                torch.cuda.synchronize(device)
                sweep_start = perf_counter()
                for index, (start, end) in enumerate(markers):
                    start.record()
                    output = forward(model, variant, xs[index], masks[index])
                    end.record()
                    del output
                torch.cuda.synchronize(device)
                elapsed_s = perf_counter() - sweep_start
                latencies = [start.elapsed_time(end) for start, end in markers]
                if elapsed_s <= 0 or any(ms <= 0 for ms in latencies):
                    raise RuntimeError(f"{name}: invalid inference timing at size {size}")

                # Re-run the same event data for MAE. This is deliberately
                # outside the wall-clock sweep and CUDA event markers.
                event_errors = []
                for index in range(settings.num_events):
                    output = forward(model, variant, xs[index], masks[index])
                    heads = extract_event_heads(output, cpu_masks[index], 0, index)
                    event_errors.append(compute_mae(heads, reference_heads[size][index]))
                    del output, heads
                mae_mean, ci_low, ci_high = mean_mae_ci95(event_errors)
                p95_ms = float(np.percentile(latencies, 95))
                events_per_s = settings.num_events / elapsed_s
                measurements.append(Measurement(
                    variant=name,
                    event_size=size,
                    padded_size=padded,
                    num_events=settings.num_events,
                    p95_latency_ms=p95_ms,
                    throughput_events_s=events_per_s,
                    mae_mean=mae_mean,
                    mae_ci95_low=ci_low,
                    mae_ci95_high=ci_high,
                    gpu=gpu_name,
                ))
                print(f"  {name:23s} size={size:6d} padded={padded:6d} "
                      f"p95={p95_ms:9.3f} ms  "
                      f"observed={events_per_s:9.1f} events/s  "
                      f"MAE={mae_mean:.5g} [95% CI: {ci_low:.5g}, {ci_high:.5g}]",
                      flush=True)
                del markers, latencies, event_errors
            del xs, masks, cpu_masks

    return captured if is_reference else measurements


def plot_pdf(rows: list[Measurement], names: tuple[str, ...], reference: str,
             path: Path) -> None:
    """Plot observed throughput, p95 latency and MAE with 95% CI on one page."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(15.4, 10.4))
    grid = fig.add_gridspec(2, 2, left=0.085, right=0.975, top=0.845,
                            bottom=0.225, hspace=0.43, wspace=0.23)
    ax_rate = fig.add_subplot(grid[0, 0])
    ax_latency = fig.add_subplot(grid[0, 1])
    ax_mae = fig.add_subplot(grid[1, :])
    palette = plt.get_cmap("tab10")

    for index, name in enumerate(names):
        series = sorted((row for row in rows if row.variant == name),
                        key=lambda row: row.event_size)
        xs = [row.event_size for row in series]
        style = dict(color=palette(index % 10), linewidth=1.7,
                     marker="o", markersize=3.5, label=name)
        ax_rate.plot(xs, [row.throughput_events_s for row in series], **style)
        ax_latency.plot(xs, [row.p95_latency_ms for row in series], **style)
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
                ylabel="Completed events / sweep wall time (events/s)")
    ax_latency.set(title="p95 tail latency vs. event size",
                   ylabel="p95 GPU forward latency (ms/event)")
    ax_mae.set(title=f"Mean prediction MAE vs. event size  |  reference: {reference}",
               ylabel="Mean per-event MAE (95% CI)")
    for ax in (ax_rate, ax_latency, ax_mae):
        ax.set_xlabel("Event size (real input elements)")
        ax.set_xlim(min(row.event_size for row in rows), max(row.event_size for row in rows))
        ax.grid(True, alpha=0.25)
        ax.set_axisbelow(True)
        ax.margins(y=0.08)
    ax_mae.set_ylim(bottom=0)

    fig.suptitle("MLPF  |  Single-event inference scaling", fontsize=18,
                 fontweight="bold", y=0.965)
    fig.text(0.5, 0.913,
             f"{rows[0].gpu}  |  batch size 1  |  {rows[0].num_events} events/size  |  no warmup",
             ha="center", fontsize=11, color="#566273")
    handles, labels = ax_rate.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.108),
               ncol=min(4, len(names)), frameon=False, fontsize=9.5)
    fig.text(0.5, 0.032,
             "Reference: untimed and unplotted. Throughput: full GPU-resident sweep wall time "
             "(no input setup or MAE).",
             ha="center", fontsize=8.5, color="#566273")
    fig.text(0.5, 0.014,
             "Latency: p95 CUDA-event inference. MAE: per-event mean over ID/momentum/PU; "
             "error bars: 95% Student-t CI.",
             ha="center", fontsize=8.5, color="#566273")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference", choices=tuple(REGISTRY),
                        help="Reference variant: untimed, used only for prediction MAE")
    parser.add_argument("--pad-bin-size", type=int, required=True,
                        help="Additional padding multiple (0 means model-config padding only)")
    parser.add_argument("--model-kwargs", type=Path, required=True,
                        help="Pickled MLPF model configuration")
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="PyTorch model checkpoint")
    parser.add_argument("--output-pdf", type=Path, required=True,
                        help="Destination for the three-chart PDF")
    parser.add_argument("--num-events", type=int, default=64,
                        help="Independent synthetic events at EACH size (default: 64; min: 2)")
    args = parser.parse_args()

    if args.pad_bin_size < 0:
        parser.error("--pad-bin-size must be nonnegative")
    if args.num_events < 2:
        parser.error("--num-events must be >= 2 to calculate MAE confidence intervals")
    if args.output_pdf.suffix.lower() != ".pdf":
        parser.error("--output-pdf must end in .pdf")
    for model_path in (args.model_kwargs, args.checkpoint):
        if not model_path.is_file():
            parser.error(f"Model file not found: {model_path}")
    if args.output_pdf.resolve() in {args.model_kwargs.resolve(), args.checkpoint.resolve()}:
        parser.error("--output-pdf must not overwrite a model file")

    # Never benchmark or plot the reference, even if it occurs in VARIANT_NAMES.
    names = tuple(dict.fromkeys(name for name in VARIANT_NAMES if name != args.reference))
    unknown = [name for name in names if name not in REGISTRY]
    if unknown:
        parser.error(f"Unknown names in VARIANT_NAMES: {unknown}; available: {list(REGISTRY)}")
    if not names:
        parser.error("VARIANT_NAMES must include at least one non-reference variant")
    sizes = event_sizes()
    settings = RunSettings(args.model_kwargs, args.checkpoint, args.pad_bin_size,
                           args.num_events)
    print(f"Reference (untimed): {args.reference} | sizes: {sizes} | GPU: {DEVICE}", flush=True)
    print(f"Measured variants: {', '.join(names)} | events per size: {args.num_events}",
          flush=True)

    rows: list[Measurement] = []
    context = mp.get_context("spawn")
    # Each OS process destroys its CUDA context before the next variant runs.
    # CPU reference predictions remain only in parent memory, never on disk.
    print(f"\n[reference] {args.reference}", flush=True)
    with ProcessPoolExecutor(max_workers=1, mp_context=context) as executor:
        reference_heads = executor.submit(
            benchmark_variant, args.reference, sizes, settings
        ).result()
    for name in names:
        print(f"\n[variant] {name}", flush=True)
        with ProcessPoolExecutor(max_workers=1, mp_context=context) as executor:
            rows.extend(executor.submit(
                benchmark_variant, name, sizes, settings, reference_heads
            ).result())

    plot_pdf(rows, names, args.reference, args.output_pdf)
    print(f"\nPDF: {args.output_pdf.resolve()}", flush=True)


if __name__ == "__main__":
    main()
