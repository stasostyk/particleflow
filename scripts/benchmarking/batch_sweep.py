#!/usr/bin/env python3
"""Sweep ONNX benchmark batch sizes and generate an aggregate PDF report.

This script repeatedly calls an existing benchmark runner (e.g. inference.py), once
per requested batch size. Each run writes its own summary.json. Those summaries
are then aggregated into CSV/JSON and a multi-page PDF.

Example:
    uv run --project envs/ort-gpu --no-sync \
      python scripts/benchmarking/benchmark_batch_sweep.py \
      --runner scripts/benchmarking/single.py \
      --onnx-model ./onnx_benchmarks/gpu/model_fused_fp32.onnx \
      --data-dir ~/ceph \
      --dataset cms_pf_ttbar \
      --num-events 200 \
      --batch-sizes 1 2 4 8 16 \
      --warmup-runs 0 \
      --benchmark-repeats 3 \
      --outdir ./batch_sweep_fused_fp32
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run an ONNX benchmark over many batch sizes and create an aggregate PDF."
    )
    p.add_argument(
        "--runner",
        required=True,
        help="Path to the existing benchmark script (for example scripts/benchmarking/inference.py)",
    )
    p.add_argument("--onnx-model", required=True, help="Path to the ONNX model")
    p.add_argument("--data-dir", required=True, help="TFDS data directory")
    p.add_argument("--dataset", default="cms_pf_ttbar", help="TFDS dataset name")
    p.add_argument("--num-events", type=int, default=200)
    p.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16],
        help="Batch sizes to benchmark",
    )
    p.add_argument("--pad-bin-size", type=int, default=0)
    p.add_argument("--device-id", type=int, default=0)
    p.add_argument("--num-threads", type=int, default=1)
    p.add_argument(
        "--warmup-runs",
        type=int,
        default=0,
        help="Forwarded to the benchmark runner. Use 0 if you do not want per-batch warmup on real data.",
    )
    p.add_argument("--benchmark-repeats", type=int, default=3)
    p.add_argument(
        "--checkpoint",
        default=None,
        help="Optional checkpoint argument forwarded to the runner (metadata only in the simplified runner)",
    )
    p.add_argument("--outdir", default="./onnx_batch_sweep")
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse an existing bs_<N>/summary.json instead of rerunning that batch size",
    )
    p.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue the sweep if one batch size fails (for example due to OOM)",
    )
    p.add_argument(
        "--extra-runner-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Any additional arguments to append to the underlying benchmark command",
    )
    return p.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.num_events <= 0:
        raise ValueError("--num-events must be > 0")
    if args.benchmark_repeats <= 0:
        raise ValueError("--benchmark-repeats must be > 0")
    if args.warmup_runs < 0:
        raise ValueError("--warmup-runs must be >= 0")
    if not args.batch_sizes:
        raise ValueError("At least one --batch-sizes value is required")
    if any(b <= 0 for b in args.batch_sizes):
        raise ValueError("All --batch-sizes values must be > 0")

    # De-duplicate while keeping the requested order, then sort for plotting.
    args.batch_sizes = sorted(set(args.batch_sizes))


def tee_subprocess(cmd: list[str], log_path: Path) -> int:
    print("\n$ " + " ".join(cmd), flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            log.write(line)
        return proc.wait()


def build_runner_command(args: argparse.Namespace, batch_size: int, run_dir: Path) -> list[str]:
    cmd = [
        sys.executable,
        args.runner,
        "--onnx-model",
        args.onnx_model,
        "--data-dir",
        os.path.expanduser(args.data_dir),
        "--dataset",
        args.dataset,
        "--num-events",
        str(args.num_events),
        "--batch-size",
        str(batch_size),
        "--pad-bin-size",
        str(args.pad_bin_size),
        "--device-id",
        str(args.device_id),
        "--num-threads",
        str(args.num_threads),
        "--warmup-runs",
        str(args.warmup_runs),
        "--benchmark-repeats",
        str(args.benchmark_repeats),
        "--outdir",
        str(run_dir),
    ]

    if args.checkpoint:
        cmd += ["--checkpoint", os.path.expanduser(args.checkpoint)]

    if args.extra_runner_args:
        cmd += args.extra_runner_args

    return cmd


def load_summary(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def safe_float(value: Any) -> float:
    if value is None:
        return math.nan
    return float(value)


def summarize_run(summary: dict[str, Any], summary_path: Path) -> dict[str, Any]:
    timing = summary["timing"]
    padding = summary.get("padding_statistics", {})
    memory = summary.get("memory", {})
    event_stats = summary.get("event_size_statistics", {})

    # Attention work is approximately quadratic in sequence length.  The original
    # runner reports token padding efficiency. Derive an N^2-aware efficiency from
    # per-batch event sizes when possible.
    batches = summary.get("batches", [])
    attention_num = 0.0
    attention_den = 0.0
    for batch in batches:
        seq = int(batch.get("padded_sequence_length", 0))
        sizes = [int(x) for x in batch.get("event_sizes", [])]
        if seq > 0 and sizes:
            attention_num += sum(n * n for n in sizes)
            attention_den += len(sizes) * seq * seq
    attention_eff = attention_num / attention_den if attention_den > 0 else math.nan

    return {
        "batch_size": int(summary["batch_size"]),
        "num_events": int(summary["num_processed_events"]),
        "num_batches": int(summary["num_batches"]),
        "throughput_events_per_s": safe_float(timing.get("throughput_events_per_s")),
        "inference_ms_per_event": safe_float(timing.get("inference_ms_per_event")),
        "total_gpu_inference_s": safe_float(timing.get("total_gpu_inference_s")),
        "batch_ms_mean": safe_float(timing.get("batch_ms_mean")),
        "batch_ms_std": safe_float(timing.get("batch_ms_std")),
        "event_normalized_ms_median": safe_float(timing.get("event_normalized_ms_median")),
        "event_normalized_ms_p95": safe_float(timing.get("event_normalized_ms_p95")),
        "event_normalized_ms_p99": safe_float(timing.get("event_normalized_ms_p99")),
        "token_padding_efficiency": safe_float(padding.get("mean_efficiency")),
        "attention_padding_efficiency": attention_eff,
        "gpu_memory_before_mib": safe_float(memory.get("gpu_memory_used_mib_before")),
        "gpu_memory_after_mib": safe_float(memory.get("gpu_memory_used_mib_after")),
        "event_size_mean": safe_float(event_stats.get("mean")),
        "event_size_median": safe_float(event_stats.get("median")),
        "event_size_min": safe_float(event_stats.get("min")),
        "event_size_max": safe_float(event_stats.get("max")),
        "summary_json": str(summary_path),
    }


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_aggregate_json(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    failed: list[dict[str, Any]],
    path: Path,
) -> None:
    payload = {
        "runner": os.path.abspath(args.runner),
        "onnx_model": os.path.abspath(args.onnx_model),
        "dataset": args.dataset,
        "data_dir": os.path.abspath(os.path.expanduser(args.data_dir)),
        "num_events": args.num_events,
        "requested_batch_sizes": args.batch_sizes,
        "warmup_runs": args.warmup_runs,
        "benchmark_repeats": args.benchmark_repeats,
        "successful_runs": rows,
        "failed_runs": failed,
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _set_batch_axis(ax: plt.Axes, batch_sizes: np.ndarray) -> None:
    ax.set_xlabel("Batch size")
    ax.set_xticks(batch_sizes)
    ax.set_xticklabels([str(int(x)) for x in batch_sizes])
    ax.grid(True, alpha=0.25)
    if len(batch_sizes) > 1 and np.all(batch_sizes > 0):
        ratios = batch_sizes[1:] / batch_sizes[:-1]
        if np.allclose(ratios, ratios[0]) and ratios[0] > 1.0:
            ax.set_xscale("log", base=ratios[0])


def make_pdf(rows: list[dict[str, Any]], model_path: str, output_pdf: Path) -> None:
    rows = sorted(rows, key=lambda r: r["batch_size"])

    bs = np.asarray([r["batch_size"] for r in rows], dtype=float)
    throughput = np.asarray([r["throughput_events_per_s"] for r in rows], dtype=float)
    ms_per_event = np.asarray([r["inference_ms_per_event"] for r in rows], dtype=float)
    batch_ms = np.asarray([r["batch_ms_mean"] for r in rows], dtype=float)
    batch_ms_std = np.asarray([r["batch_ms_std"] for r in rows], dtype=float)
    p50 = np.asarray([r["event_normalized_ms_median"] for r in rows], dtype=float)
    p95 = np.asarray([r["event_normalized_ms_p95"] for r in rows], dtype=float)
    p99 = np.asarray([r["event_normalized_ms_p99"] for r in rows], dtype=float)
    token_eff = 100.0 * np.asarray([r["token_padding_efficiency"] for r in rows], dtype=float)
    attn_eff = 100.0 * np.asarray([r["attention_padding_efficiency"] for r in rows], dtype=float)
    gpu_before = np.asarray([r["gpu_memory_before_mib"] for r in rows], dtype=float) / 1024.0
    gpu_after = np.asarray([r["gpu_memory_after_mib"] for r in rows], dtype=float) / 1024.0

    model_name = Path(model_path).name

    with PdfPages(output_pdf) as pdf:
        # Page 1: main performance scaling.
        fig, axs = plt.subplots(2, 2, figsize=(11.0, 8.5), constrained_layout=True)
        fig.suptitle(f"ONNX GPU batch-size sweep - {model_name}", fontsize=15)

        ax = axs[0, 0]
        ax.plot(bs, throughput, marker="o", label="Measured")
        ax.set_ylabel("Throughput [events/s]")
        ax.set_title("Throughput")
        ax.legend()
        _set_batch_axis(ax, bs)

        ax = axs[0, 1]
        ax.plot(bs, ms_per_event, marker="o")
        ax.set_ylabel("GPU inference [ms/event]")
        ax.set_title("Inference time per event")
        _set_batch_axis(ax, bs)

        ax = axs[1, 0]
        ax.errorbar(bs, batch_ms, yerr=batch_ms_std, marker="o", capsize=4)
        ax.set_ylabel("Batch inference [ms]")
        ax.set_title("Mean batch latency (+/- 1 std)")
        _set_batch_axis(ax, bs)

        ax = axs[1, 1]
        base = throughput[0]
        speedup = throughput / base if np.isfinite(base) and base != 0 else np.full_like(throughput, np.nan)
        ax.plot(bs, speedup, marker="o", label="Measured throughput speedup")
        ax.set_ylabel("Speedup vs first batch size")
        ax.set_title("Scaling efficiency")
        ax.legend()
        _set_batch_axis(ax, bs)

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Page 2: padding, tails, and memory.
        fig, axs = plt.subplots(2, 2, figsize=(11.0, 8.5), constrained_layout=True)
        fig.suptitle(f"Benchmark diagnostics - {model_name}", fontsize=15)

        ax = axs[0, 0]
        ax.plot(bs, token_eff, marker="o", label="Token efficiency")
        ax.plot(bs, attn_eff, marker="o", label="Approx. N^2 attention efficiency")
        ax.set_ylabel("Padding efficiency [%]")
        ax.set_ylim(0, 101)
        ax.set_title("Padding efficiency")
        ax.legend()
        _set_batch_axis(ax, bs)

        ax = axs[0, 1]
        ax.plot(bs, p50, marker="o", label="Median")
        ax.plot(bs, p95, marker="o", label="p95")
        ax.plot(bs, p99, marker="o", label="p99")
        ax.set_ylabel("Batch-normalized time [ms/event]")
        ax.set_title("Normalized timing distribution")
        ax.legend()
        _set_batch_axis(ax, bs)

        ax = axs[1, 0]
        if np.any(np.isfinite(gpu_before)):
            ax.plot(bs, gpu_before, marker="o", label="Before")
        if np.any(np.isfinite(gpu_after)):
            ax.plot(bs, gpu_after, marker="o", label="After")
        ax.set_ylabel("GPU memory snapshot [GiB]")
        ax.set_title("GPU memory snapshots")
        ax.legend()
        _set_batch_axis(ax, bs)

        ax = axs[1, 1]
        total_s = np.asarray([r["total_gpu_inference_s"] for r in rows], dtype=float)
        ax.plot(bs, total_s, marker="o")
        ax.set_ylabel("Total measured GPU inference [s]")
        ax.set_title("Total GPU inference for fixed event count")
        _set_batch_axis(ax, bs)

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Page 3: compact numerical summary table.
        fig, ax = plt.subplots(figsize=(11.0, 8.5))
        ax.axis("off")
        ax.set_title(f"Numerical summary - {model_name}", fontsize=15, pad=18)

        columns = [
            "Batch",
            "Events/s",
            "ms/event",
            "Batch ms",
            "Token pad %",
            "Attn pad %",
            "GPU after GiB",
        ]
        table_rows = []
        for r in rows:
            table_rows.append(
                [
                    str(r["batch_size"]),
                    f"{r['throughput_events_per_s']:.2f}",
                    f"{r['inference_ms_per_event']:.3f}",
                    f"{r['batch_ms_mean']:.3f}",
                    f"{100.0 * r['token_padding_efficiency']:.2f}",
                    f"{100.0 * r['attention_padding_efficiency']:.2f}",
                    f"{r['gpu_memory_after_mib'] / 1024.0:.2f}" if np.isfinite(r["gpu_memory_after_mib"]) else "n/a",
                ]
            )

        table = ax.table(
            cellText=table_rows,
            colLabels=columns,
            loc="center",
            cellLoc="center",
            colLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.0, 1.5)
        ax.text(
            0.5,
            0.08,
            "Timing values come directly from each runner summary.json. H2D/D2H and CPU batching are excluded by the underlying benchmark.",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=9,
            wrap=True,
        )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def print_summary(rows: list[dict[str, Any]]) -> None:
    print("\n=== Batch sweep summary ===")
    print(f"{'batch':>7} {'events/s':>12} {'ms/event':>12} {'batch ms':>12} {'token pad':>12} {'attn pad':>12}")
    for r in sorted(rows, key=lambda x: x["batch_size"]):
        print(
            f"{r['batch_size']:7d} "
            f"{r['throughput_events_per_s']:12.2f} "
            f"{r['inference_ms_per_event']:12.3f} "
            f"{r['batch_ms_mean']:12.3f} "
            f"{100.0 * r['token_padding_efficiency']:11.2f}% "
            f"{100.0 * r['attention_padding_efficiency']:11.2f}%"
        )


def main() -> None:
    args = parse_args()
    validate_args(args)

    root = Path(args.outdir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    successful: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []

    print("=== ONNX batch-size sweep ===")
    print(f"Runner:              {args.runner}")
    print(f"Model:               {args.onnx_model}")
    print(f"Batch sizes:         {args.batch_sizes}")
    print(f"Events / run:        {args.num_events}")
    print(f"Warmup runs:         {args.warmup_runs}")
    print(f"Benchmark repeats:   {args.benchmark_repeats}")
    print(f"Output:              {root}")

    for batch_size in args.batch_sizes:
        run_dir = root / f"bs_{batch_size}"
        summary_path = run_dir / "summary.json"
        log_path = run_dir / "run.log"
        run_dir.mkdir(parents=True, exist_ok=True)

        if args.skip_existing and summary_path.exists():
            print(f"\n[batch {batch_size}] Reusing {summary_path}")
        else:
            cmd = build_runner_command(args, batch_size, run_dir)
            rc = tee_subprocess(cmd, log_path)
            if rc != 0:
                failure = {
                    "batch_size": batch_size,
                    "return_code": rc,
                    "log": str(log_path),
                }
                failed.append(failure)
                print(f"[batch {batch_size}] FAILED with return code {rc}", file=sys.stderr)
                if args.continue_on_error:
                    continue
                raise SystemExit(rc)

        if not summary_path.exists():
            failure = {
                "batch_size": batch_size,
                "return_code": None,
                "log": str(log_path),
                "error": f"Missing {summary_path}",
            }
            failed.append(failure)
            if args.continue_on_error:
                continue
            raise FileNotFoundError(summary_path)

        summary = load_summary(summary_path)
        successful.append(summarize_run(summary, summary_path))

    if not successful:
        raise RuntimeError("No batch sizes completed successfully; no report can be produced")

    successful.sort(key=lambda r: r["batch_size"])
    csv_path = root / "batch_sweep.csv"
    json_path = root / "batch_sweep.json"
    pdf_path = root / "batch_sweep.pdf"

    write_csv(successful, csv_path)
    write_aggregate_json(args, successful, failed, json_path)
    make_pdf(successful, args.onnx_model, pdf_path)
    print_summary(successful)

    if failed:
        print("\nFailed batch sizes:")
        for item in failed:
            print(f"  batch={item['batch_size']} log={item['log']}")

    print("\nOutputs:")
    print(f"  PDF:  {pdf_path}")
    print(f"  CSV:  {csv_path}")
    print(f"  JSON: {json_path}")
    print(f"  Per-batch directories: {root}/bs_<batch_size>/")


if __name__ == "__main__":
    main()
