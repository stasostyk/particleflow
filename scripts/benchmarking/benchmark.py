#!/usr/bin/env python3
"""MLPF CUDA benchmark orchestrator. Run from the MLPF repository root.

Launches EXACTLY ONE worker process per variant, sequentially. The parent never
uses CUDA; once a worker exits its CUDA context, allocated memory and allocator
caches from that process are released before the next worker starts.

Outputs: summary.json, optional comparison.pdf, per-variant worker JSON files,
optional Chrome traces, and CPU reference_predictions.pt.

Example:
  python /path/to/benchmark.py --checkpoint model.pt --model-kwargs config.pkl \
      --data-dir /path/to/tfds --variants fp16_flash bf16_flash --pdf
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import tempfile
import sys
from typing import Any

from benchmark_variants import REFERENCE_NAME, REGISTRY


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--model-kwargs", type=Path, required=True,
                   help="Trusted pickled MLPFConfig/dict from the MLPF project")
    p.add_argument("--dataset", default="cms_pf_ttbar")
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--split", default="train")
    p.add_argument("--outdir", type=Path, default=Path("./pt_inference_benchmark"))
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--num-events", type=int, default=64)
    p.add_argument("--repeats", type=int, default=20)
    p.add_argument("--warmup-iters", type=int, default=5)
    p.add_argument("--warmup-seq-len", type=int, default=256)
    p.add_argument("--pad-bin-size", type=int, default=0)
    p.add_argument("--batch-order", choices=["sequential", "length-sorted"],
                   default="sequential",
                   help="Length sorting reduces padding but assumes offline bucketing")
    p.add_argument("--input-clip", type=float, default=None,
                   help="Explicit optional input clipping. Default: never alter dataset values")
    p.add_argument("--fixed-seq-len", type=int, default=None)
    p.add_argument("--baseline-cache-clear-every", type=int, default=0,
                   help="Reference-only emergency allocator clear interval; 0 keeps memory warm")
    p.add_argument("--baseline-memory-log-every", type=int, default=25)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--reference", choices=list(REGISTRY), default=REFERENCE_NAME,
                   help="Untimed B=1 reference model (never included in timing sweep)")
    p.add_argument("--variants", nargs="+", choices=list(REGISTRY), default=["fp16_flash"],
                   help="Variants to time; reference automatically runs first")
    p.add_argument("--compile", action="store_true", help="Override and compile all TIMED variants (reference stays eager)")
    p.add_argument("--compile-mode", choices=["default", "reduce-overhead", "max-autotune"],
                   default="default")
    p.add_argument("--compile-dynamic", choices=["auto", "true", "false"],
                   default="auto", help="torch.compile(dynamic=...) strategy")
    p.add_argument("--cudnn-benchmark", action="store_true",
                   help="Enable cuDNN autotuning (usually not advantageous for many shapes)")
    p.add_argument("--compiler-cache-root", type=Path, default=None,
                   help="Reuse per-variant persistent compiler caches; default isolates cold caches")
    p.add_argument("--no-pin-memory", action="store_true")
    p.add_argument("--allow-partial-state", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pdf", action="store_true", help="Create the two-page comparison PDF")
    p.add_argument("--no-profile", action="store_true",
                   help="Disable untimed PyTorch operator/kernel profiling")
    p.add_argument("--profile-traces", action="store_true",
                   help="Export Chrome traces per variant/batch size (may be large)")
    p.add_argument("--telemetry-ms", type=int, default=100,
                   help="NVML device-utilization/VRAM sampling interval in milliseconds")
    p.add_argument("--no-telemetry", action="store_true",
                   help="Disable NVML sampler entirely to measure without sampling overhead")
    p.add_argument("--nvml-index", type=int, default=None,
                   help="Explicit nvidia-smi/NVML index when CUDA UUID/PCI mapping fails")
    p.add_argument("--batch-consistency-events", type=int, default=4,
                   help="Compare B>1 against this variant's B=1 output on worst-error events")
    args = p.parse_args()
    if (len(set(args.batch_sizes)) != len(args.batch_sizes)
            or any(b <= 0 for b in args.batch_sizes)
            or args.num_events < max(args.batch_sizes)
            or args.repeats < 2 or args.warmup_iters < 1
            or args.pad_bin_size < 0 or args.warmup_seq_len < 3
            or args.baseline_cache_clear_every < 0 or args.baseline_memory_log_every < 0
            or (args.fixed_seq_len is not None and args.fixed_seq_len <= 0)
             or args.telemetry_ms < 20 or args.batch_consistency_events < 0
             or (args.nvml_index is not None and args.nvml_index < 0)
             or (args.input_clip is not None and (not 0 < args.input_clip < float("inf")))):
        p.error("Invalid batch sizes, repeats, warmup, padding, baseline cleanup, or telemetry interval")
    if not args.device.startswith("cuda:") or not args.device.split(":", 1)[1].isdigit():
        p.error("--device must be an explicit CUDA ordinal, e.g. cuda:0")
    if args.no_profile and args.profile_traces:
        p.error("--profile-traces requires profiling (remove --no-profile)")
    if args.reference in args.variants:
        print(f"NOTE: {args.reference} is reference-only; omitting it from timed variants", flush=True)
    args.variants = list(dict.fromkeys(v for v in args.variants if v != args.reference))
    if not args.variants:
        p.error("Select at least one timed variant other than --reference")
    args.batch_sizes = sorted(args.batch_sizes)
    if not args.checkpoint.is_file() or not args.model_kwargs.is_file():
        p.error("--checkpoint and --model-kwargs must exist")
    return args


def atomic_json(data: dict[str, Any], path: Path) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as fp:
        json.dump(data, fp, indent=2, allow_nan=False)
    temp.replace(path)


def render_pdf(report: dict[str, Any], path: Path) -> None:
    """Performance, accuracy, kernels, and optional available NVML telemetry."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    import numpy as np

    variants = [v for v in report["variants"] if v != report["reference"]]
    batches = report["batch_sizes"]
    results = report["results"]
    colors = {v: plt.get_cmap("tab10")(i % 10) for i, v in enumerate(variants)}

    def stat(b: int, variant: str, scope: str, key: str, field="mean"):
        try:
            timing = results[str(b)][variant]["timings"][scope]
            if timing.get("steady_state_valid") is False:
                return float("nan")
            item = timing["summary"].get(key)
            return float(item[field]) if item is not None else float("nan")
        except (KeyError, TypeError):
            return float("nan")

    def value(b: int, variant: str, section: str, key: str):
        try:
            raw = results[str(b)][variant][section][key]
            return float(raw) if raw is not None else float("nan")
        except (KeyError, ValueError, TypeError):
            return float("nan")

    def plot_lines(ax, key, scope="gpu_only", secondary_scope=False, confidence=False):
        for variant in variants:
            for selected_scope, line in (("gpu_only", "-"), ("e2e", "--")):
                if not secondary_scope and selected_scope != scope:
                    continue
                means = [stat(b, variant, selected_scope, key) for b in batches]
                label = variant + (f" ({selected_scope})" if secondary_scope else "")
                ax.plot(batches, means, "o", color=colors[variant], linestyle=line,
                        linewidth=1.7, markersize=4, label=label)
                if confidence:
                    lower = [stat(b, variant, selected_scope, key, "ci95_low") for b in batches]
                    upper = [stat(b, variant, selected_scope, key, "ci95_high") for b in batches]
                    ax.fill_between(batches, lower, upper, color=colors[variant], alpha=0.12)
        ax.set_xticks(batches)
        ax.grid(alpha=0.18)
        ax.set_axisbelow(True)

    with PdfPages(path) as pdf:
        fig, axes = plt.subplots(3, 2, figsize=(16.5, 12.0), layout="constrained")
        fig.suptitle("MLPF  |  isolated-process CUDA inference benchmark", fontsize=17,
                     fontweight="bold")
        ax_latency, ax_rate, ax_padding, ax_mem, ax_compile, ax_accuracy = axes.flat
        plot_lines(ax_latency, "p95_latency_ms", secondary_scope=True, confidence=True)
        ax_latency.set(title="p95 batch latency (95% confidence bands)", ylabel="ms / batch",
                       xlabel="Batch size")
        # GPU-only CUDA-event sums are NOT actual sustained wall throughput:
        # they exclude CPU dispatch, sync and H2D. Put both in clear context.
        for variant in variants:
            wall = [stat(b, variant, "e2e", "observed_throughput_events_per_s")
                    for b in batches]
            ideal = [stat(b, variant, "gpu_only", "throughput_events_per_s")
                     for b in batches]
            ax_rate.plot(batches, wall, "o-", color=colors[variant],
                         label=f"{variant} e2e wall", linewidth=1.8)
            ax_rate.plot(batches, ideal, "s:", color=colors[variant],
                         label=f"{variant} GPU-event sum", alpha=0.7)
        ax_rate.set(title="Actual e2e wall throughput vs idealized GPU-event sum",
                    ylabel="Events / second", xlabel="Batch size")
        for variant in variants:
            pad_efficiency = [100 * value(b, variant, "batching", "padding_efficiency")
                              for b in batches]
            ax_padding.plot(batches, pad_efficiency, "o-", color=colors[variant],
                            label=variant)
        ax_padding.set(title="Useful input tokens (variable-length padding overhead)",
                       ylabel="Valid / padded tokens (%)", xlabel="Batch size", ylim=(0, 105))
        plot_lines(ax_mem, "peak_allocated_mib", confidence=True)
        for name in variants:
            values = [stat(b, name, "gpu_only", "peak_device_used_mib") for b in batches]
            ax_mem.plot(batches, values, marker="s", linestyle=":", color=colors[name],
                        linewidth=1.5, markersize=4)
        ax_mem.set(title="GPU peak memory: solid = PyTorch allocated, dotted = NVML physical used",
                   ylabel="MiB", xlabel="Batch size")
        for name in variants:
            overhead = [value(b, name, "prewarm", "wall_s") for b in batches]
            ax_compile.plot(batches, overhead, "o-", color=colors[name], label=name)
        ax_compile.set(title="Real-batch prewarm wall time (compile + inference)",
                       ylabel="Seconds (UNTIMED)", xlabel="Batch size")

        markers = {"id": "o", "momentum": "s", "pu": "^"}
        styles = {"id": "-", "momentum": "--", "pu": ":"}
        for name in variants:
            for head in markers:
                per_head = [results.get(str(b), {}).get(name, {}).get("accuracy", {})
                            .get("mae_by_head", {}).get(head) for b in batches]
                ax_accuracy.plot(batches, [float(x) if x is not None else np.nan
                                           for x in per_head], marker=markers[head],
                                 linestyle=styles[head], color=colors[name],
                                 label=f"{name} {head}", markersize=4)
        # symlog preserves exact zero agreement instead of silently hiding it.
        ax_accuracy.set_yscale("symlog", linthresh=1e-4)
        ax_accuracy.set(title=f"Per-head error vs {report['reference']} (symlog)",
                        xlabel="Batch size", ylabel="Mean |prediction - reference|")
        ax_accuracy.grid(axis="y", alpha=0.2)

        for ax in (ax_latency, ax_rate, ax_padding, ax_mem, ax_compile):
            ax.set_xticks(batches)
            ax.grid(alpha=0.18)
            ax.legend(fontsize=7, ncol=2, loc="best")
        ax_accuracy.set_xticks(batches)
        ax_accuracy.legend(fontsize=7, ncol=2, loc="best")
        fig.supxlabel(
            "e2e = pinned H2D + forward + sync, not dataset loading. GPU-only = CUDA-event forward. "
            "Prewarm is UNTIMED and includes lazy compilation.\n"
            "All batch sizes use the same events (including the final short batch). "
            "Timed recompilation points are omitted.\n"
            "CI reflects repeat-to-repeat variation on this fixed event set, not event-population uncertainty. "
            "NVML data/per-event errors in summary.json.", fontsize=8)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Second page: operator hotspots, with an explicit warning about instrumented time.
        columns = 2
        rows = max(1, (len(variants) + columns - 1) // columns)
        fig, axes = plt.subplots(rows, columns, figsize=(16.5, max(7.6, rows * 5.4)),
                                 layout="constrained", squeeze=False)
        max_b = max(batches)
        for i, name in enumerate(variants):
            ax = axes[i // columns, i % columns]
            profile = results.get(str(max_b), {}).get(name, {}).get("profile", {})
            kernels = profile.get("top_cuda_kernels", [])
            # Some torch versions only expose operator-level device timings.
            op_data = (kernels if kernels else profile.get("top_ops", []))[:9]
            durations = [row.get("gpu_total_ms", row.get("gpu_self_ms", 0.0))
                         for row in op_data]
            names = [row.get("kernel", row.get("op", "unknown")) for row in op_data]
            display_names = [s if len(s) <= 65 else s[:24] + "..." + s[-35:] for s in names]
            if op_data:
                yy = np.arange(len(display_names))
                ax.barh(yy, durations, color=colors[name], alpha=0.85)
                ax.set_yticks(yy, display_names, fontsize=7)
                ax.invert_yaxis()
                ax.set_xlabel("Instrumented GPU kernel / op time (ms)")
            else:
                ax.text(0.5, 0.5, "No profile (disabled or unavailable)",
                        ha="center", transform=ax.transAxes)
            ax.set_title(f"{name}  |  B={max_b}  |  first real batch")
            ax.grid(axis="x", alpha=0.2)
        for i in range(len(variants), rows * columns):
            axes[i // columns, i % columns].set_axis_off()
        fig.suptitle("CUDA kernel / PyTorch operator hotspots (untimed profiling pass)",
                     fontweight="bold", fontsize=16)
        fig.supxlabel("Profiling runs on the first REAL batch. CompiledFxGraph wrappers are NOT compiler work "
                      "and are excluded from individual kernel bars. "
                      "Instrumented device times are NOT unprofiled inference latency. "
                      "Full top-20 ops and timing statistics are in summary.json.", fontsize=8)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        has_nvml = any(
            np.isfinite(stat(b, variant, "gpu_only", "mean_gpu_util_pct"))
            for variant in variants for b in batches
        )
        if has_nvml:
            fig, panels = plt.subplots(2, 2, figsize=(14, 10), layout="constrained")
            for ax, metric, name, unit in (
                (panels[0, 0], "mean_gpu_util_pct", "GPU busy", "%"),
                (panels[0, 1], "mean_mem_controller_util_pct", "Memory-controller busy", "%"),
                (panels[1, 0], "peak_device_used_mib", "Device physical memory used", "MiB"),
                (panels[1, 1], "mean_power_w", "Device power", "W"),
            ):
                for variant in variants:
                    means = [stat(b, variant, "gpu_only", metric) for b in batches]
                    ax.plot(batches, means, "o-", color=colors[variant], label=variant)
                ax.set_title(name)
                ax.set_ylabel(unit)
                ax.set_xlabel("Batch size")
                ax.set_xticks(batches)
                ax.grid(alpha=0.18)
                ax.legend(fontsize=8)
            fig.suptitle("NVML telemetry (only available metrics; device-wide)",
                         fontsize=16, fontweight="bold")
            fig.supxlabel("Driver sampling is coarse and includes inter-batch gaps; "
                          "values are NOT SM occupancy and may include other GPU jobs.", fontsize=8)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


def main():
    args = parse_args()
    root = args.outdir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    result_path = root / "summary.json"
    settings = {
        "checkpoint": str(args.checkpoint.resolve()),
        "model_kwargs": str(args.model_kwargs.resolve()),
        "dataset": args.dataset,
        "data_dir": str(args.data_dir.resolve()),
        "split": args.split,
        "batch_sizes": args.batch_sizes,
        "num_events": args.num_events,
        "repeats": args.repeats,
        "warmup_iters": args.warmup_iters,
        "warmup_seq_len": args.warmup_seq_len,
        "pad_bin_size": args.pad_bin_size,
        "batch_order": args.batch_order,
        "input_clip": args.input_clip,
        "fixed_seq_len": args.fixed_seq_len,
        "baseline_cache_clear_every": args.baseline_cache_clear_every,
        "baseline_memory_log_every": args.baseline_memory_log_every,
        "device": args.device,
        "compile": args.compile,
        "compile_mode": args.compile_mode,
        "compile_dynamic": args.compile_dynamic,
        "cudnn_benchmark": args.cudnn_benchmark,
        "compiler_cache_root": (str(args.compiler_cache_root.resolve())
                                if args.compiler_cache_root else None),
        "no_pin_memory": args.no_pin_memory,
        "allow_partial_state": args.allow_partial_state,
        "seed": args.seed,
        "profile": not args.no_profile,
        "profile_traces": args.profile_traces,
        "telemetry_ms": args.telemetry_ms,
        "no_telemetry": args.no_telemetry,
        "nvml_index": args.nvml_index,
        "batch_consistency_events": args.batch_consistency_events,
    }
    variants = [args.reference, *args.variants]
    job = {
        "reference": args.reference,
        "reference_file": str(root / "reference_predictions.pt"),
        "trace_dir": str(root / "traces"),
        "settings": settings,
    }
    job_path = root / "job.json"
    atomic_json(job, job_path)
    report: dict[str, Any] = {
        "status": "running",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "reference": args.reference,
        "variants": variants,
        "variant_specs": {name: REGISTRY[name].describe() for name in variants},
        "batch_sizes": args.batch_sizes,
        "settings": settings,
        "baseline": {},
        "worker_metadata": {},
        "results": {str(b): {} for b in args.batch_sizes},
        "timing_notes": {
            "isolation": "Separate sequential OS process for each variant and per-worker compiler-cache directories; exit destroys that process's CUDA context. Does not flush hardware L2 or caches belonging to other processes",
            "accuracy": ("Untimed B=1 reference outputs saved on CPU and compared per valid "
                         "ELEMENTWISE model output; classification logit MAE is not a physics metric. "
                         "Per-event and per-head errors and B=1 self-consistency diagnostics saved."),
            "e2e": "CPU wall-clock H2D + forward + CUDA sync, excludes pre-pinning/CPU processing",
            "gpu_only": "CUDA-event elapsed duration of forward; H2D copy happens outside timed interval",
            "memory": "PyTorch peak allocator allocated/reserved per repeat; optional NVML physical device-wide VRAM used and utilization samples",
            "utilization": "NVML busy-time samples; NOT per-kernel SM occupancy; short sweeps can be under-sampled",
            "profiling": "Separate unmeasured profiler forward on first REAL batch; compile prewarm uses real events and masks",
            "compilation": "Prewarm wall time includes compilation and forward work; never misread profiler CompiledFxGraph as compile time",
            "batching": "All events included, including last short batch. Optional offline length-sorted batching reduces padding",
            "throughput": "Observed sweep throughput is whole-sweep wall events/s; CUDA-event-only idealized throughput excludes CPU dispatch",
        },
    }
    atomic_json(report, result_path)
    worker_path = Path(__file__).with_name("benchmark_worker.py").resolve()
    if not worker_path.is_file():
        raise FileNotFoundError(f"Expected adjacent worker file: {worker_path}")
    env = os.environ.copy()
    # Running from the MLPF repository root remains supported even when these
    # scripts live elsewhere (Python scripts otherwise change sys.path[0]).
    env["PYTHONPATH"] = os.pathsep.join(x for x in [str(Path.cwd()),
                              str(worker_path.parent), env.get("PYTHONPATH", "")] if x)

    for name in variants:
        output = root / "workers" / f"{name}.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.unlink(missing_ok=True)
        command = [sys.executable, "-u", str(worker_path), "--job", str(job_path),
                   "--variant", name, "--output", str(output)]
        print(f"\n=== Launch isolated worker: {name} ===", flush=True)
        print("CUDA process lifetime begins here; next worker starts only after exit.", flush=True)
        # Per-worker compiler caches prevent accidental cross-variant reuse of
        # Inductor/Triton/PTX and extension artifacts. The temporary cache tree
        # is removed AFTER subprocess exit (also on worker failure).
        # Cold caches are isolated, and on Slurm preferentially placed on fast
        # node-local scratch. Opt into a persistent per-variant cache for warm
        # starts in production (never mix cold and warm data in one comparison).
        scratch = os.environ.get("SLURM_TMPDIR")
        cache_parent = Path(scratch) if scratch and Path(scratch).is_dir() else None
        if args.compiler_cache_root:
            cache = args.compiler_cache_root.resolve() / name
            cache.mkdir(parents=True, exist_ok=True)
            cache_context = nullcontext(str(cache))
        else:
            cache_context = tempfile.TemporaryDirectory(
                prefix=f"mlpf_{name}_", dir=cache_parent)
        with cache_context as cache_string:
            cache = Path(cache_string)
            worker_env = env.copy()
            for key, directory in (
                ("TORCHINDUCTOR_CACHE_DIR", "inductor"),
                ("TRITON_CACHE_DIR", "triton"),
                ("CUDA_CACHE_PATH", "cuda"),
                ("TORCH_EXTENSIONS_DIR", "extensions"),
            ):
                location = cache / directory
                location.mkdir(parents=True, exist_ok=True)
                worker_env[key] = str(location)
            completed = subprocess.run(command, cwd=Path.cwd(), env=worker_env, check=False)
        if output.exists():
            payload = json.loads(output.read_text(encoding="utf-8"))
            report["worker_metadata"][name] = {
                key: payload.get(key) for key in
                ("status", "system", "dispatch_verification", "compiled_dispatch_verification", "telemetry", "compiled", "dataset_fingerprint_sha256", "input_lengths", "error")
            }
            previous = report["worker_metadata"].get(args.reference, {})
            if name != args.reference and previous.get("dataset_fingerprint_sha256"):
                if (previous["dataset_fingerprint_sha256"] !=
                        payload.get("dataset_fingerprint_sha256")):
                    report["status"] = "failed"
                    report["error"] = f"Dataset fingerprint mismatch: {name} vs {args.reference}"
                    atomic_json(report, result_path)
                    raise RuntimeError(report["error"])
            if name == args.reference and "baseline" in payload:
                report["baseline"] = payload["baseline"]
            for batch_size, data in payload.get("results", {}).items():
                report["results"][batch_size][name] = data
        if completed.returncode != 0:
            report["status"] = "failed"
            report["failed_variant"] = name
            report["error"] = (report["worker_metadata"].get(name, {}).get("error") or
                               f"Worker exited with status {completed.returncode}")
            atomic_json(report, result_path)
            print(f"FAILED: {name}: {report['error']} (see {output})", file=sys.stderr)
            raise SystemExit(completed.returncode)
        atomic_json(report, result_path)
        print(f"=== Worker {name} exited; its CUDA context is released ===", flush=True)

    report["status"] = "complete"
    report["completed_utc"] = datetime.now(timezone.utc).isoformat()
    atomic_json(report, result_path)
    if args.pdf:
        pdf_path = root / "comparison.pdf"
        render_pdf(report, pdf_path)
        print(f"Report: {pdf_path}")
    print(f"Machine-readable results: {result_path}")


if __name__ == "__main__":
    main()
