#!/usr/bin/env python
"""
Run the benchmark with ``--num-repeats`` of at least 2 to get confidence bands.

The intervals use the Student t-distribution with ``n - 1`` degrees of freedom, which is
the appropriate choice for the small sample counts a benchmark produces.

Usage
-----
    python sweep_benchmark.py \
        --runner benchmark_batched_inference.py \
        --vary batch-size --values 1 2 4 8 16 32 \
        --title "MLPF attention inference, A100" \
        --output sweep_batch_size.pdf \
        --checkpoint ... --model-kwargs ... --data-dir ... --num-events 200 --num-repeats 5

Any runner argument may be passed through unchanged. If the varied parameter is also
given as a fixed value, the fixed value is ignored with a warning.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field

import matplotlib
import numpy as np
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------

# Parameters that can be swept: CLI name -> axis label.
VARIABLE_PARAMS = {
    "batch-size": "Batch size",
    "num-threads": "Number of CPU threads",
    "pad-bin-size": "Padding bin size (elements)",
}

# Runner arguments passed through unchanged: argparse dest -> kind.
#   "value": single value, "flag": boolean switch, "list": space-separated list.
PASSTHROUGH_ARGS = {
    "checkpoint": "value",
    "model_kwargs": "value",
    "dataset": "value",
    "data_dir": "value",
    "num_events": "value",
    "batch_size": "value",
    "pad_bin_size": "value",
    "num_warmup": "value",
    "num_repeats": "value",
    "num_threads": "value",
    "compile": "flag",
    "sort_by_length": "flag",
    "configs": "list",
}

CONFIDENCE = 0.95


# --------------------------------------------------------------------------------------
# Argument parsing
# --------------------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    sweep = parser.add_argument_group("sweep")
    sweep.add_argument("--runner", required=True, help="Path to benchmark_batched_inference.py")
    sweep.add_argument("--vary", required=True, choices=sorted(VARIABLE_PARAMS), help="Runner parameter to sweep")
    sweep.add_argument("--values", required=True, type=int, nargs="+", help="Values of the swept parameter")
    sweep.add_argument("--title", required=True, help="Report title")
    sweep.add_argument("--output", required=True, help="Output PDF path (the only artifact of this script)")
    sweep.add_argument("--python", default=sys.executable, help="Python interpreter used to launch the runner")

    runner = parser.add_argument_group("runner (passed through unchanged)")
    runner.add_argument("--checkpoint", required=True)
    runner.add_argument("--model-kwargs", required=True)
    runner.add_argument("--data-dir", required=True)
    runner.add_argument("--dataset")
    runner.add_argument("--num-events", type=int)
    runner.add_argument("--batch-size", type=int)
    runner.add_argument("--pad-bin-size", type=int)
    runner.add_argument("--num-warmup", type=int)
    runner.add_argument("--num-repeats", type=int)
    runner.add_argument("--num-threads", type=int)
    runner.add_argument("--compile", action="store_true")
    runner.add_argument("--sort-by-length", action="store_true")
    runner.add_argument("--configs", nargs="+")

    args = parser.parse_args()
    if len(set(args.values)) != len(args.values):
        parser.error("--values contains duplicates")
    return args


def warn_if_fixed_and_varied(args: argparse.Namespace) -> None:
    """Print a warning when the swept parameter was also given a fixed value."""
    dest = args.vary.replace("-", "_")
    if getattr(args, dest) is not None:
        print(
            f"WARNING: --{args.vary}={getattr(args, dest)} was provided but --vary {args.vary} "
            f"is set; ignoring the fixed value and sweeping over {args.values} instead."
        )


# --------------------------------------------------------------------------------------
# Running the benchmark script
# --------------------------------------------------------------------------------------


def build_command(args: argparse.Namespace, value: int, outdir: str) -> list[str]:
    """Assemble the runner command line for one sweep point."""
    cmd = [args.python, args.runner]
    vary_dest = args.vary.replace("-", "_")

    for dest, kind in PASSTHROUGH_ARGS.items():
        if dest == vary_dest:
            continue  # replaced by the swept value below
        val = getattr(args, dest)
        if val is None or val is False:
            continue
        flag = "--" + dest.replace("_", "-")
        if kind == "flag":
            cmd.append(flag)
        elif kind == "list":
            cmd.extend([flag, *map(str, val)])
        else:
            cmd.extend([flag, str(val)])

    cmd.extend([f"--{args.vary}", str(value), "--outdir", outdir])
    return cmd


def run_benchmark(cmd: list[str], outdir: str) -> dict:
    """Run the benchmark script, streaming its output, and return the parsed summary."""
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    with open(os.path.join(outdir, "summary.json")) as f:
        return json.load(f)


# --------------------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------------------


@dataclass
class Moments:
    """Count, mean and (population) std of a sample set, as reported by the runner."""

    n: int
    mean: float
    std: float



def t_half_width(std: float, n: int, confidence: float = CONFIDENCE) -> float:
    """Half-width of the two-sided t confidence interval for a mean; NaN when n < 2."""
    if n < 2 or not np.isfinite(std):
        return float("nan")
    t = stats.t.ppf(0.5 + confidence / 2.0, df=n - 1)
    return float(t * std / math.sqrt(n))


def mean_and_ci(samples: list[float]) -> tuple[float, float]:
    """Mean and 95% t half-width of a list of samples (sample std, ddof=1)."""
    arr = np.asarray(samples, dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    std = float(arr.std(ddof=1)) if arr.size > 1 else float("nan")
    return float(arr.mean()), t_half_width(std, arr.size)


@dataclass
class ScenarioSamples:
    """Everything collected for one (swept value, scenario) pair from the runner's JSON."""

    batch_size: int = 1
    throughput: list[float] = field(default_factory=list)  # events/s, one per runner repeat
    latency_batch: Moments | None = None  # per-batch ms over all repeats
    memory: list[float] = field(default_factory=list)  # MiB, one per runner repeat
    mae: float | None = None  # None for the baseline

    @classmethod
    def from_scenario(cls, scenario: dict) -> "ScenarioSamples":
        timing, memory, error = scenario["timing"], scenario["memory"], scenario.get("error")
        lat = timing["latency_ms_per_batch"]
        return cls(
            batch_size=scenario["batch_size"],
            throughput=[timing["num_events"] / t for t in timing["total_inference_time_s_per_repeat"]],
            latency_batch=Moments(timing["num_batches_per_repeat"] * timing["num_repeats"], lat["mean"], lat["std"]),
            memory=list(memory.get("process_gpu_memory_max_mib_nvml_per_repeat") or []),
            mae=error["mae"] if error is not None else None,
        )

    # --- aggregated metrics: each returns (mean, 95% half-width) -----------------------

    def throughput_stats(self) -> tuple[float, float]:
        return mean_and_ci(self.throughput)

    def latency_per_event_stats(self) -> tuple[float, float]:
        m = self.latency_batch
        return m.mean / self.batch_size, t_half_width(m.std, m.n) / self.batch_size

    def memory_stats(self) -> tuple[float, float]:
        return mean_and_ci(self.memory)

    def mae_mean(self) -> float | None:
        return self.mae


def collect(args: argparse.Namespace) -> tuple[dict[int, dict[str, ScenarioSamples]], list[str]]:
    """
    Run the sweep. Returns ``results[value][scenario]`` and the scenario names in the
    order the runner reported them (baseline first).
    """
    results: dict[int, dict[str, ScenarioSamples]] = {v: {} for v in args.values}
    scenario_order: list[str] = []

    with tempfile.TemporaryDirectory(prefix="mlpf_sweep_") as tmp:
        for value in args.values:
            outdir = os.path.join(tmp, f"{args.vary}_{value}")
            summary = run_benchmark(build_command(args, value, outdir), outdir)
            for name, scenario in summary["scenarios"].items():
                if name not in scenario_order:
                    scenario_order.append(name)
                results[value][name] = ScenarioSamples.from_scenario(scenario)

    return results, scenario_order


# --------------------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------------------


def plot_metric_lines(ax, results, scenarios, values, colors, stats_fn, ylabel, xlabel):
    """Line per scenario with a shaded 95% CI band (band omitted where the CI is undefined)."""
    x = np.asarray(values, dtype=float)
    for name in scenarios:
        means, halves = [], []
        for v in values:
            m, h = stats_fn(results[v][name]) if name in results[v] else (float("nan"), float("nan"))
            means.append(m)
            halves.append(h)
        means, halves = np.asarray(means), np.asarray(halves)
        ax.plot(x, means, marker="o", lw=1.8, color=colors[name], label=name)
        band = np.isfinite(halves)
        if band.any():
            ax.fill_between(x[band], (means - halves)[band], (means + halves)[band], color=colors[name], alpha=0.2, lw=0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.grid(True, alpha=0.3)


def plot_error_bars(ax, results, scenarios, values, xlabel_param):
    """Grouped bar chart: one group per scenario, one bar per swept value, gap between groups."""
    scenarios = [s for s in scenarios if any(results[v][s].mae_mean() is not None for v in values if s in results[v])]
    if not scenarios:
        ax.text(0.5, 0.5, "No error data (only the baseline ran)", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return

    n_vals, group_width = len(values), len(values) + 1  # +1 bar width of spacing between groups
    cmap = plt.get_cmap("viridis")
    value_colors = [cmap(i / max(n_vals - 1, 1)) for i in range(n_vals)]

    for gi, name in enumerate(scenarios):
        for vi, v in enumerate(values):
            mae = results[v][name].mae_mean() if name in results[v] else None
            if mae is None:
                continue
            ax.bar(gi * group_width + vi, mae, width=0.9, color=value_colors[vi], label=f"{xlabel_param} = {v}" if gi == 0 else None)

    centers = [gi * group_width + (n_vals - 1) / 2 for gi in range(len(scenarios))]
    ax.set_xticks(centers)
    ax.set_xticklabels(scenarios, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Mean absolute error vs. baseline")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8, title="swept value", title_fontsize=8)


def render_report(results, scenarios, args: argparse.Namespace) -> None:
    """Lay out the four panels on a single landscape page and save as vector PDF."""
    xlabel = VARIABLE_PARAMS[args.vary]
    cmap = plt.get_cmap("tab10")
    colors = {name: cmap(i % cmap.N) for i, name in enumerate(scenarios)}

    fig = plt.figure(figsize=(16, 11))
    gs = GridSpec(2, 2, figure=fig, left=0.06, right=0.98, top=0.86, bottom=0.08, hspace=0.35, wspace=0.22)

    fig.text(0.5, 0.955, args.title, ha="center", va="center", fontsize=20, fontweight="bold")
    subtitle = f"Varying: {xlabel.lower()}   |   shaded bands: {int(CONFIDENCE * 100)}% t-distribution confidence interval"
    fig.text(0.5, 0.915, subtitle, ha="center", va="center", fontsize=12, color="0.3")

    ax_thr = fig.add_subplot(gs[0, 0])
    plot_metric_lines(ax_thr, results, scenarios, args.values, colors, ScenarioSamples.throughput_stats, "Throughput [events / s]", xlabel)
    ax_thr.set_title("Throughput")
    ax_thr.legend(fontsize=8)

    ax_lat = fig.add_subplot(gs[0, 1])
    plot_metric_lines(ax_lat, results, scenarios, args.values, colors, ScenarioSamples.latency_per_event_stats, "Latency [ms / event]", xlabel)
    ax_lat.set_title("Latency (device inference time per event)")

    ax_mem = fig.add_subplot(gs[1, 0])
    plot_metric_lines(ax_mem, results, scenarios, args.values, colors, ScenarioSamples.memory_stats, "Process GPU memory, max [MiB]", xlabel)
    ax_mem.set_title("GPU memory (NVML, per process)")
    if not any(s.memory for v in results.values() for s in v.values()):
        ax_mem.text(0.5, 0.5, "No NVML memory data (pynvml not available in the runner)", ha="center", va="center", transform=ax_mem.transAxes)

    ax_err = fig.add_subplot(gs[1, 1])
    plot_error_bars(ax_err, results, scenarios, args.values, xlabel)
    ax_err.set_title("Error vs. unbatched FP32 baseline")

    fig.savefig(args.output, format="pdf", dpi=300)
    plt.close(fig)


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------


def print_table(results, scenarios, values, vary):
    print(f"\n{'scenario':28s} {vary:>13s} {'events/s':>12s} {'ms/event':>10s} {'mem MiB':>9s} {'MAE':>12s}")
    for name in scenarios:
        for v in values:
            if name not in results[v]:
                continue
            s = results[v][name]
            thr, lat, mem, mae = s.throughput_stats()[0], s.latency_per_event_stats()[0], s.memory_stats()[0], s.mae_mean()
            mem_str = f"{mem:9.0f}" if np.isfinite(mem) else f"{'n/a':>9s}"
            mae_str = f"{mae:12.3e}" if mae is not None else f"{'baseline':>12s}"
            print(f"{name:28s} {v:13d} {thr:12.1f} {lat:10.3f} {mem_str} {mae_str}")


def main() -> None:
    args = parse_args()
    warn_if_fixed_and_varied(args)

    results, scenarios = collect(args)
    print_table(results, scenarios, args.values, args.vary)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    render_report(results, scenarios, args)
    print(f"\nReport written to {args.output}")


if __name__ == "__main__":
    main()