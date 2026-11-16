#!/usr/bin/env python3
"""MLPF physics check: run --variants exactly the way benchmark_worker.py runs them and
compare jet-level physics against the FP32 reference variant (fp32_efficient). Run from the MLPF repo root.
fp32_efficient is a hardcoded reference: it is always run first (even if not listed in --variants)
and every other variant is compared against it. No TFDS ground truth, target or PF baseline is used.
All variants run sequentially in THIS process; Dynamo caches and CUDA memory are reset between them.
Physics only: no timing, NVML or memory statistics. Each variant first passes the worker's untimed
preflight (backend validation, FP32 weights/layers, packed-varlen event isolation, synthetic
dispatch check). Every real forward is then checked against the variant's expected attention kernel
(operator/kernel NAMES only); for external FlashAttention backends a real CUDA flash/cute kernel
must launch and no PyTorch SDPA op may appear. A mismatch aborts the run.
Outputs: <outdir>/physics/data/<variant>.npz and <outdir>/physics/physics.pdf.
Example:
  python /path/to/benchmark_physics.py --checkpoint model.pt --model-kwargs config.pkl \\
      --dataset colliderml_ttbar_nopu_pf --data-dir /path/to/tfds --split test \\
      --variants fp16_flash bf16_flash --batch-size 4 --num-events 512
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re
import sys
import types
import numpy as np
from benchmark_variants import REGISTRY

REQUIRED = ("cls_id", "pt", "eta", "phi", "energy")
N_BOOTSTRAP = 200  # Resamples for the median and IQR/median uncertainties on the binned-response page.
REFERENCE = "fp32_efficient"  # Hardcoded reference. The worker never compiles it, even with --compile.
# Same patterns benchmark_worker.verify_dispatch applies to external FlashAttention backends.
EXTERNAL_KERNEL_RX = r"flash|fmha|cute|cutlass.*attn|attention.*fwd"
PYTORCH_SDPA_RX = r"scaled_dot_product_flash_attention|efficient_attention_forward"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Same names, types and defaults as benchmark.py for every option that affects how a variant runs.
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--model-kwargs", type=Path, required=True,
                   help="Trusted pickled MLPFConfig/dict from the MLPF project")
    p.add_argument("--dataset", default="cms_pf_ttbar")
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--split", default="train")
    p.add_argument("--outdir", type=Path, default=Path("./pt_inference_benchmark"))
    p.add_argument("--batch-size", type=int, required=True, help="Batch size of the physics pass")
    p.add_argument("--num-events", type=int, default=64)
    p.add_argument("--pad-bin-size", type=int, default=0)
    p.add_argument("--batch-order", choices=["sequential", "length-sorted"],
                   default="sequential",
                   help="Length sorting reduces padding but assumes offline bucketing")
    p.add_argument("--input-clip", type=float, default=None,
                   help="Explicit optional input clipping. Default: never alter dataset values")
    p.add_argument("--fixed-seq-len", type=int, default=None)
    p.add_argument("--warmup-seq-len", type=int, default=256,
                   help="Sequence length of the synthetic preflight checks (as the worker's warmup_seq_len)")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--variants", nargs="+", choices=list(REGISTRY), default=["fp16_flash"],
                   help=f"Variants to compare against the {REFERENCE} reference "
                        f"(the reference is always run, first, even if not listed)")
    p.add_argument("--compile", action="store_true", help="Override and compile all variants")
    p.add_argument("--compile-mode", choices=["default", "reduce-overhead", "max-autotune"],
                   default="default")
    p.add_argument("--compile-dynamic", choices=["auto", "true", "false"],
                   default="auto", help="torch.compile(dynamic=...) strategy")
    p.add_argument("--cudnn-benchmark", action="store_true",
                   help="Enable cuDNN autotuning (usually not advantageous for many shapes)")
    p.add_argument("--no-pin-memory", action="store_true")
    p.add_argument("--allow-partial-state", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--plot-only", action="store_true", help="Skip running; regenerate plots from saved data")
    p.add_argument("--skip-plots", action="store_true", help="Run and save data only")
    args = p.parse_args()
    if (args.batch_size <= 0 or args.num_events < args.batch_size
            or args.pad_bin_size < 0
            or (args.fixed_seq_len is not None and args.fixed_seq_len <= 0)
            or args.warmup_seq_len <= 0
            or (args.input_clip is not None and (not 0 < args.input_clip < float("inf")))):
        p.error("Invalid batch size, event count, padding, fixed length or input clip")
    if not args.device.startswith("cuda:") or not args.device.split(":", 1)[1].isdigit():
        p.error("--device must be an explicit CUDA ordinal, e.g. cuda:0")
    if args.plot_only and args.skip_plots:
        p.error("--plot-only and --skip-plots are mutually exclusive")
    if not args.checkpoint.is_file() or not args.model_kwargs.is_file():
        p.error("--checkpoint and --model-kwargs must exist")
    others = [v for v in dict.fromkeys(args.variants) if v != REFERENCE]
    if not others:
        p.error(f"--variants must contain at least one variant other than the reference {REFERENCE}")
    args.variants = [REFERENCE, *others]  # Reference always first; every other variant is compared to it.
    return args


def make_settings(args):
    """Every key read by the imported worker functions and the mirrored run_job setup."""
    return {
        "checkpoint": str(args.checkpoint.resolve()), "model_kwargs": str(args.model_kwargs.resolve()),
        "dataset": args.dataset, "data_dir": str(args.data_dir.resolve()), "split": args.split,
        "num_events": args.num_events, "batch_size": args.batch_size,
        "pad_bin_size": args.pad_bin_size, "batch_order": args.batch_order,
        "input_clip": args.input_clip, "fixed_seq_len": args.fixed_seq_len, "device": args.device,
        "warmup_seq_len": args.warmup_seq_len,
        "compile": args.compile, "compile_mode": args.compile_mode,
        "compile_dynamic": args.compile_dynamic, "cudnn_benchmark": args.cudnn_benchmark,
        "no_pin_memory": args.no_pin_memory, "allow_partial_state": args.allow_partial_state,
        "seed": args.seed,
    }


def colliderml_dataset(name: str):
    from mlpf.conf import Dataset
    base = name.split("/")[0].split(":")[0].lower()
    return Dataset.COLLIDERML_HITS if "hits" in base else Dataset.COLLIDERML


# ----------------------------------------------------------------------------- storage
def pack(per_event: dict[int, dict[str, np.ndarray]], n_events: int) -> dict[str, np.ndarray]:
    """{event_id: {field: 1D}} -> flat fields + per-event counts, ordered by event_id."""
    if sorted(per_event) != list(range(n_events)):
        raise RuntimeError(f"Expected events 0..{n_events - 1}, got {len(per_event)} events")
    fields = list(per_event[0])
    out = {"event_id": np.arange(n_events, dtype=np.int64),
           "counts": np.array([len(per_event[i][fields[0]]) for i in range(n_events)], dtype=np.int64)}
    for f in fields:
        out[f] = np.concatenate([np.asarray(per_event[i][f]) for i in range(n_events)])
    return out


def save_npz(path: Path, collections: dict[str, dict[str, np.ndarray]], meta: dict) -> None:
    arrays = {f"{c}__{k}": v for c, d in collections.items() for k, v in d.items()}
    np.savez(path, meta=np.array(json.dumps(meta)), **arrays)
    print(f"Saved {path}", flush=True)


def load_npz(path: Path):
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["meta"]))
        colls: dict[str, dict[str, np.ndarray]] = {}
        for key in z.files:
            if key != "meta":
                c, k = key.split("__", 1)
                colls.setdefault(c, {})[k] = z[key]
    return meta, colls


# ----------------------------------------------------------------------------- decoding
class _OutputsProxy:
    """Stands in for the eager MLPF inside predict_particles. Class methods are re-bound to the
    proxy, so its model call returns the outputs ALREADY produced by benchmark_worker.forward
    (same compiled model, SDPA context and forward_fn as the worker), and only once."""

    def __init__(self, model, outputs):
        self._model, self._outputs, self.calls = model, outputs, 0

    def __getattr__(self, name):
        attr = getattr(type(self._model), name, None)
        return types.MethodType(attr, self) if isinstance(attr, types.FunctionType) else getattr(self._model, name)

    def __call__(self, *args, **kwargs):
        self.calls += 1
        if self.calls > 1:
            raise RuntimeError("predict_particles called the model more than once; refusing to decode")
        return self._outputs

    forward = __call__


def decode_outputs(torch, eager_model, outputs, x, mask, set_mode):
    """Model's OWN post-processing applied to precomputed outputs -> {field: [per-event arrays]}."""
    predict = getattr(type(eager_model), "predict_particles", None)
    if predict is None:
        raise TypeError(f"{type(eager_model).__name__} has no predict_particles (model_transform wrapper?)")
    proxy = _OutputsProxy(eager_model, outputs)

    def tripwire(module, _inputs):
        raise RuntimeError(f"predict_particles ran {type(module).__name__}: decoding must not re-run "
                           "network compute outside benchmark_worker.forward")
    handle = torch.nn.modules.module.register_module_forward_pre_hook(tripwire)
    try:
        parts = predict(proxy, x, mask)
    finally:
        handle.remove()
    if proxy.calls != 1:
        raise RuntimeError(f"predict_particles called the model {proxy.calls} times; expected exactly 1")
    if not isinstance(parts, dict) or not isinstance(parts.get("cls_id"), torch.Tensor):
        raise TypeError(f"predict_particles must return a dict with tensor 'cls_id'; got {type(parts)}")
    shape = tuple(parts["cls_id"].shape)
    if not set_mode and shape != tuple(mask.shape):
        raise ValueError(f"Decoded shape {shape} is not aligned with input mask {tuple(mask.shape)} "
                         "(forward_fn/input_transform changed the layout?)")
    # Elementwise: valid input elements. SET: every slot is a candidate; cls_id != 0 filters later.
    keep = (torch.ones(shape, dtype=torch.bool) if set_mode else mask).cpu()
    out = {}
    for k, v in parts.items():
        if isinstance(v, torch.Tensor) and tuple(v.shape) == shape:
            v = v.detach().float().cpu()
            out[k] = [v[j][keep[j]].numpy() for j in range(shape[0])]
    missing = [k for k in REQUIRED if k not in out]
    if missing:
        raise KeyError(f"predict_particles lacks per-particle fields {missing}; has {sorted(parts)}")
    return out


# ----------------------------------------------------------------------------- kernel check
def checked_forward(torch, W, model, variant, x, mask, device, patterns):
    """benchmark_worker.forward under the PyTorch profiler, used ONLY to read operator/kernel names.
    The registry's expected_ops regexes are written against profiler keys (as in verify_dispatch),
    which include CUDA kernel names; no other hook sees all of them. No timings are read or kept.
    External FlashAttention variants get the worker's stricter check on EVERY real batch: a CUDA
    flash/cute kernel must actually launch (not just the CPU record_function scope) and no PyTorch
    SDPA flash/efficient op may appear."""
    external = variant.attention.backend is not None
    if not patterns and not external:
        return W.forward(model, variant, x, mask), []
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], acc_events=True) as prof:
        out = W.forward(model, variant, x, mask)
        torch.cuda.synchronize(device)  # Flush CUDA activity so kernel names are recorded.
    keys = W.profiler_keys(prof)
    observed = [k for k in keys if any(re.search(rx, k, flags=re.I) for rx in patterns)]
    if patterns and not observed:
        nearby = [k for k in keys if any(t in k.lower() for t in ("attn", "flash", "attention", "sdpa"))]
        raise RuntimeError(f"{variant.name}: expected kernel {list(patterns)} did NOT run on real batch "
                           f"{tuple(x.shape)}. Attention-like ops seen: {nearby[:30]}")
    if external:
        cuda_names = sorted({e.name for e in prof.events()
                             if "CUDA" in str(getattr(e, "device_type", "")).upper()})
        kernels = [n for n in cuda_names if re.search(EXTERNAL_KERNEL_RX, n, re.I)]
        if not kernels:
            raise RuntimeError(f"{variant.name}: external {variant.attention.backend} scope ran on real batch "
                               f"{tuple(x.shape)} but no identifiable CUDA attention kernel launched. "
                               f"CUDA sample: {cuda_names[:25]}")
        if any(re.search(PYTORCH_SDPA_RX, k, re.I) for k in keys):
            raise RuntimeError(f"{variant.name}: unexpected PyTorch SDPA flash/efficient op with external "
                               f"backend {variant.attention.backend} on real batch {tuple(x.shape)}")
        observed = observed + [f"cuda:{k}" for k in kernels[:8]]
    return out, observed


# ----------------------------------------------------------------------------- running
def run_variant(W, torch, OutputMode, variant, settings, config, state, multiple, events, device):
    seed = settings["seed"]                                                        # run_job
    torch.manual_seed(seed)                                                        # run_job
    np.random.seed(seed)                                                           # run_job
    torch.cuda.set_device(device)                                                  # run_job
    torch.backends.cuda.matmul.allow_tf32 = variant.precision.allow_tf32           # run_job
    torch.backends.cudnn.allow_tf32 = variant.precision.allow_tf32                 # run_job
    torch.set_float32_matmul_precision("high" if variant.precision.allow_tf32 else "highest")  # run_job
    torch.backends.cudnn.benchmark = settings.get("cudnn_benchmark", False)        # run_job
    from attention_backends import validate_backend
    backend_validation = ({"status": "skipped (FULL_FP16)"} if W.FULL_FP16 and variant.attention.backend is not None
                          else validate_backend(variant.attention.backend,
                                                variant.attention.kernel_dtype, device,
                                                int(config.model.attention.head_dim))
                          if variant.attention.backend is not None else
                          {"status": REFERENCE, "native_varlen": False})                     # run_job
    model = W.build_model(config, state, variant, device, settings["allow_partial_state"])  # run_job
    parameter_validation = W.verify_fp32_parameters(model)                         # run_job
    event_isolation_validation = W.verify_external_kernel_event_isolation(         # run_job
        variant, device, int(config.model.attention.num_heads), int(config.model.attention.head_dim))
    warm_len = settings["warmup_seq_len"]                                          # run_job
    if multiple:
        warm_len = (warm_len + multiple - 1) // multiple * multiple
    # Untimed synthetic preflight on the EAGER model, exactly as run_job does before compiling.
    # (The worker's warmup() iterations are intentionally omitted: physics is one lazy pass.)
    dtype_validation = W.verify_fp32_layers(model, variant, config.input_dim, warm_len, device)  # run_job
    dispatch_verification = W.verify_dispatch(model, variant, config.input_dim, warm_len, device)  # run_job
    eager_model = model                                                            # run_job
    # run_job never compiles the FP32 reference, whatever --compile says.
    compiled = variant.name != REFERENCE and bool(variant.compile or settings["compile"])
    compile_mode = (settings["compile_mode"] if settings["compile"] else variant.compile_mode)  # run_job
    compile_dynamic = {"auto": None, "true": True, "false": False}[               # run_job
        settings.get("compile_dynamic", "auto")]
    if compiled:                                                                   # run_job
        if compile_dynamic is False:
            print("WARNING: dynamic=False compiles one static graph per shape; past Dynamo's per-function "
                  "cache limit later shapes can silently run eager.", flush=True)
        model = torch.compile(model, mode=compile_mode, dynamic=compile_dynamic)  # run_job
    pinned = not settings["no_pin_memory"]                                         # run_job
    batches = W.host_batches(W.make_batches(events, settings["batch_size"], multiple,  # run_job (per-B body)
                                            settings["fixed_seq_len"], settings.get("batch_order", "sequential")),
                             variant, pinned)
    set_mode = getattr(eager_model, "output_mode", None) == OutputMode.SET
    # Same pattern choice as run_job's compiled dispatch check; an empty tuple means "unchecked" per the registry.
    patterns = (variant.attention.compiled_expected_ops
                if compiled and variant.attention.compiled_expected_ops is not None
                else variant.attention.expected_ops)
    if not patterns:
        print(f"WARNING: {variant.name} defines no expected_ops; its kernel is NOT checked", flush=True)
    matched_ops: set[str] = set()
    per_event = {}
    graphs_before = W.compiled_graph_count()
    with torch.inference_mode():  # Single pass, no warmup of any kind: compilation happens lazily here.
        for batch in batches:
            x, mask = W.to_gpu(batch, device, pinned)
            out, ops = checked_forward(torch, W, model, variant, x, mask, device, patterns)
            matched_ops.update(ops)
            parts = decode_outputs(torch, eager_model, out, x, mask, set_mode)
            for j, event_id in enumerate(batch[2]):
                per_event[event_id] = {k: v[j] for k, v in parts.items()}
            del out, x, mask, parts
        torch.cuda.synchronize(device)
    new_graphs = W.compiled_graph_count() - graphs_before
    pred = pack(per_event, len(events))
    meta = {
        "variant": variant.name, "variant_spec": variant.describe(), "batch_size": settings["batch_size"],
        "compiled": compiled, "compile_mode": compile_mode if compiled else None,
        "compile_dynamic": settings["compile_dynamic"] if compiled else None,
        "dynamo_graph_count": new_graphs,
        "padded_shapes": sorted({tuple(int(s) for s in b[0].shape) for b in batches}),
        "pad_multiple": multiple, "output_mode": str(getattr(eager_model, "output_mode", None)),
        "backend_validation": backend_validation,
        "fp32_parameter_validation": parameter_validation,
        "fp32_execution_validation": dtype_validation,
        "event_isolation_validation": event_isolation_validation,
        "dispatch_verification": dispatch_verification,
        "kernel_check": {"status": "verified" if patterns else "unchecked", "patterns": list(patterns),
                         "batches_checked": len(batches) if patterns else 0, "matched_ops": sorted(matched_ops)},
        "nonfinite": {k: int((~np.isfinite(v)).sum()) for k, v in pred.items() if k not in ("event_id", "counts")},
    }
    check = meta["kernel_check"]
    print(f"  kernel check: {check['status']} on {check['batches_checked']}/{len(batches)} batches, "
          f"matched {check['matched_ops']}", flush=True)
    print(f"  {len(batches)} batches, {len(meta['padded_shapes'])} padded shapes, "
          f"{new_graphs} Dynamo graphs, non-finite: {meta['nonfinite']}", flush=True)
    return pred, meta


def run_all(args, settings, data_dir: Path):
    import torch
    # benchmark_worker falls back to a no-op nvtx if the package is missing; time_batch is never called here.
    import benchmark_worker as W
    from benchmark_variants import get_variant
    from mlpf.conf import OutputMode
    if not torch.cuda.is_available():                                              # run_job
        raise RuntimeError("A CUDA-enabled PyTorch build and an NVIDIA GPU are required")
    device = torch.device(settings["device"])
    config, state, multiple = W.load_config_and_state(settings)                    # run_job
    events = W.load_events(settings, config.input_dim)                             # run_job
    fingerprint = W.event_fingerprint(events)                                      # run_job
    common = {"dataset_fingerprint_sha256": fingerprint, "num_events": len(events),
              "dataset": settings["dataset"], "split": settings["split"]}
    for name in args.variants:  # Reference first (guaranteed by parse_args).
        output = data_dir / f"{name}.npz"
        output.unlink(missing_ok=True)  # Never leave a stale result behind a failed run.
        print(f"\n=== {name}{' (reference)' if name == REFERENCE else ''} ===", flush=True)
        try:
            pred, meta = run_variant(W, torch, OutputMode, get_variant(name), settings, config, state,
                                     multiple, events, device)
        finally:
            torch._dynamo.reset()        # Drop compiled graphs / guards (and cudagraph trees) of this variant.
            W.cuda_cleanup(device)       # gc.collect + synchronize + empty_cache
        save_npz(output, {"pred": pred}, {**common, **meta})


# ----------------------------------------------------------------------------- jets
def compute_jets(variants, data_dir: Path, ds):
    import awkward
    import fastjet
    import vector
    from mlpf.conf import JET_CONFIG
    from mlpf.jet_utils import get_jet_config, match_two_jet_collections
    from mlpf.plotting.plot_utils import jet_response_metrics

    paths = [data_dir / f"{v}.npz" for v in variants]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing saved data (run without --plot-only first): {missing}")
    metas, colls = {}, {}
    for v, path in zip(variants, paths):
        metas[v], loaded = load_npz(path)
        colls[v] = loaded["pred"]
        if metas[v]["dataset_fingerprint_sha256"] != metas[REFERENCE]["dataset_fingerprint_sha256"]:
            raise RuntimeError(f"Fingerprint mismatch: {v} was run on different events than {REFERENCE}")
    jetdef, ptcut, dr = get_jet_config(ds)
    rel_cut = JET_CONFIG[ds.value].match_rel_pt

    def p4(coll, keep):
        ev = np.repeat(np.arange(len(coll["counts"])), coll["counts"])
        counts = np.bincount(ev[keep], minlength=len(coll["counts"]))
        pt, eta, phi, e = (np.asarray(coll[k], np.float64)[keep] for k in ("pt", "eta", "phi", "energy"))
        flat = awkward.zip({"px": pt * np.cos(phi), "py": pt * np.sin(phi), "pz": pt * np.sinh(eta), "E": e})
        return awkward.unflatten(flat, counts), int(keep.sum())

    jets, dropped, n_particles = {}, {}, {}
    for name, coll in colls.items():
        finite = np.logical_and.reduce([np.isfinite(coll[k]) for k in REQUIRED])
        particles = coll["cls_id"] != 0
        n_particles[name] = int(particles.sum())
        dropped[name] = int((particles & ~finite).sum())  # NaN/inf particles cannot be clustered.
        vec, n = p4(coll, particles & finite)
        if n == 0:
            print(f"WARNING: {name}: no finite particles with cls_id != 0; no jets", flush=True)
            jets[name] = vector.awk(vec)
        else:
            cl = fastjet.ClusterSequence(vec, jetdef).inclusive_jets(min_pt=ptcut)
            jets[name] = vector.awk(awkward.zip({"px": cl.px, "py": cl.py, "pz": cl.pz, "E": cl.E}))
        if dropped[name]:
            print(f"WARNING: {name}: {dropped[name]} non-finite particles excluded from clustering", flush=True)

    def match(ref, cand):
        m = match_two_jet_collections(jets, ref, cand, dr)
        ref_pt = awkward.to_numpy(awkward.flatten(jets[ref].pt[m[ref]], axis=1))
        ratio = awkward.to_numpy(awkward.flatten(jets[cand].pt[m[cand]], axis=1)) / ref_pt
        metrics = jet_response_metrics({f"jet_ratio_{ref}_to_{cand}_pt": ratio, f"jets_{ref}_pt": jets[ref].pt,
                                        f"jets_{cand}_pt": jets[cand].pt}, ref, cand, rel_cut)
        if len(ratio) == 0:  # plot_utils.med_iqr reports 0, 0 for empty input; make emptiness visible.
            metrics["med"] = metrics["iqr"] = float("nan")
        return {"ratio": ratio, "ref_pt": ref_pt, **metrics}

    flat = {name: {k: awkward.to_numpy(awkward.flatten(getattr(j, k))) for k in ("pt", "eta")}
            for name, j in jets.items()}
    for name in flat:
        if len(flat[name]["pt"]) == 0:
            print(f"WARNING: {name}: zero jets above {ptcut} GeV", flush=True)
    vs_ref = {name: match(REFERENCE, name) for name in colls if name != REFERENCE}
    return dict(flat=flat, vs_ref=vs_ref, dropped=dropped, n_particles=n_particles, metas=metas,
                ref_meta=metas[REFERENCE], ptcut=ptcut, dr=dr, rel_cut=rel_cut, ds_name=ds.value)


# ----------------------------------------------------------------------------- plots
def plot_report(r, variants, path: Path):
    import matplotlib
    import matplotlib.ticker
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    import mplhep
    from mlpf.plotting.plot_utils import EVALUATION_DATASET_NAMES, GENJET_BINS_PT_DATASET

    order = list(variants)  # Reference first.
    compared = [n for n in order if n != REFERENCE]
    tab = plt.get_cmap("tab10")

    def style(name):  # Variants: tab10 in --variants order; earlier = thicker, so identical lines stay visible.
        if name == REFERENCE:
            return {"label": f"{REFERENCE} (reference)", "color": "black", "linestyle": "-", "linewidth": 2.0}
        i = compared.index(name)
        return {"label": name, "color": tab(i % 10), "linestyle": "-", "linewidth": max(1.0, 3.2 - 0.45 * i)}

    series = {name: k for k, name in enumerate(order)}

    def dodge(edges, name, logx):
        """Bin positions shifted slightly per collection so overlapping error bars stay distinguishable."""
        lo, hi = np.asarray(edges[:-1], float), np.asarray(edges[1:], float)
        f = 0.5 + 0.5 * (series[name] - (len(series) - 1) / 2) / len(series)  # within the central half of the bin
        return np.exp(np.log(lo) + f * np.log(hi / lo)) if logx else lo + f * (hi - lo)

    def poisson_bars(ax, counts, edges, name, logx=False):
        """sqrt(N) error bars on every non-empty bin of a counts histogram."""
        ok = counts > 0
        ax.errorbar(dodge(edges, name, logx)[ok], counts[ok], yerr=np.sqrt(counts[ok]), fmt="o",
                    color=style(name)["color"], ecolor=style(name)["color"], markersize=3,
                    elinewidth=1.6, capsize=2.5, capthick=1.2, zorder=5)

    rng = np.random.default_rng(0)  # Fixed seed: reproducible bootstrap uncertainties.

    def bootstrap_stats(values):
        """Median and IQR/median with bootstrap std-dev uncertainties (jets resampled independently)."""
        if len(values) == 0:
            return (np.nan,) * 4
        p25, p50, p75 = np.percentile(values, [25, 50, 75])
        if len(values) < 2:  # A single jet has no measurable spread: no error bar rather than a fake zero.
            return p50, (p75 - p25) / p50, np.nan, np.nan
        boot = np.array([np.percentile(rng.choice(values, len(values)), [25, 50, 75]) for _ in range(N_BOOTSTRAP)])
        return (p50, (p75 - p25) / p50, float(np.std(boot[:, 1])),
                float(np.std((boot[:, 2] - boot[:, 0]) / boot[:, 1])))

    def legend_label(name, m):
        return (f"{style(name)['label']}: {m['med']:.3f} ± {m['iqr']:.3f}" if m["num_angular_matches"]
                else f"{style(name)['label']}: no matched jets")

    meta0 = r["ref_meta"]
    base = meta0["dataset"].split("/")[0].split(":")[0]
    title = (f"{EVALUATION_DATASET_NAMES.get(base, base)} | {meta0['num_events']} events "
             f"({meta0['split']} split) | anti-kt R=0.4, pT>{r['ptcut']:g} GeV, ΔR<{r['dr']:g} | ref: {REFERENCE}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(mplhep.styles.CMS), PdfPages(path) as pdf:
        # (a) spectra with ratio-to-reference
        fig = plt.figure(figsize=(18, 9))
        grid = fig.add_gridspec(2, 2, height_ratios=[3, 1], hspace=0.05)
        for col, (key, bins, xlabel) in enumerate([
                ("pt", np.logspace(np.log10(r["ptcut"]), 3, 41), "Jet $p_T$ [GeV]"),
                ("eta", np.linspace(-5, 5, 41), r"Jet $\eta$")]):
            ax = fig.add_subplot(grid[0, col])
            rax = fig.add_subplot(grid[1, col], sharex=ax)
            ref_h = np.histogram(r["flat"][REFERENCE][key], bins)[0]
            for name in order:
                h = np.histogram(r["flat"][name][key], bins)[0]
                kw = style(name)
                ax.stairs(h, bins, **{**kw, "label": f"{kw['label']} ({len(r['flat'][name][key])} jets)"})
                poisson_bars(ax, h, bins, name, logx=key == "pt")
                with np.errstate(divide="ignore", invalid="ignore"):
                    ratio = np.where(ref_h > 0, h / ref_h, np.nan)
                    rax.stairs(ratio, bins, **{**kw, "label": None})
                    if name != REFERENCE:  # ref/ref is exactly 1; others: uncorrelated Poisson propagation.
                        ok = (h > 0) & (ref_h > 0)
                        rax.errorbar(dodge(bins, name, key == "pt")[ok], ratio[ok],
                            yerr=ratio[ok] * np.sqrt(1 / h[ok] + 1 / ref_h[ok]), fmt="o",
                            color=kw["color"], ecolor=kw["color"], markersize=3,
                            elinewidth=1.6, capsize=2.5, capthick=1.2, zorder=5)
            ax.set(yscale="log", ylabel="Jets / bin")
            ax.tick_params(labelbottom=False)
            ax.legend(fontsize=11, loc="upper right")
            rax.axhline(1, color="black", lw=0.8)
            rax.set(xlabel=xlabel, ylabel=f"/ {REFERENCE}", ylim=(0.5, 1.5),
                    xscale="log" if key == "pt" else "linear")
        fig.suptitle(f"Jet spectra (error bars: Poisson √N)  |  {title}", fontsize=14)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # (b) matched-jet response vs the reference
        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
        # Left: plot_utils.plot_jet_ratio binning exactly. Right: zoom, coarser so overlaid lines stay readable.
        for col, bins in enumerate([np.linspace(0, 5, 500), np.linspace(0.5, 1.5, 101)]):
            ax = axes[col]
            for name in compared:
                m = r["vs_ref"][name]
                counts = np.histogram(m["ratio"], bins)[0]
                ax.stairs(counts, bins, **{**style(name), "label": legend_label(name, m)})
                poisson_bars(ax, counts, bins, name)
            ax.axvline(1, color="black", lw=0.8, ls="--")
            ax.set(xlabel=f"Matched jet $p_T$ / {REFERENCE} jet $p_T$", ylabel="Matched jets / bin")
            ax.set_title(f"Response vs {REFERENCE}" + (" (zoom)" if col else "") +
                         "  |  legend: median ± IQR", fontsize=13)
            ax.legend(fontsize=10, loc="upper right")
        fig.suptitle(f"Jet response (error bars: Poisson √N)  |  {title}", fontsize=14)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # (c) response and resolution vs reference-jet pT (plot_utils GENJET_BINS_PT_DATASET, (lo, hi] bins)
        edges = GENJET_BINS_PT_DATASET[r["ds_name"]]
        fig, (ax_med, ax_res) = plt.subplots(1, 2, figsize=(18, 7.5))
        for name in compared:
            m = r["vs_ref"][name]
            med, res, med_err, res_err = np.array([
                bootstrap_stats(m["ratio"][(m["ref_pt"] > lo) & (m["ref_pt"] <= hi)])
                for lo, hi in zip(edges[:-1], edges[1:])]).T
            x = dodge(edges, name, logx=True)
            ax_med.errorbar(x, med, yerr=med_err, marker="o", capsize=2, **style(name))
            ax_res.errorbar(x, res, yerr=res_err, marker="o", capsize=2, **style(name))
        for ax in (ax_med, ax_res):
            ax.set_xscale("log")
            ax.set_xticks(edges, [str(e) for e in edges])
            ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax_med.axhline(1, color="black", lw=0.8, ls="--", label=f"{REFERENCE} (reference)")
        ax_res.axhline(0, color="black", lw=0.8, ls="--")
        ax_med.set(xlabel=f"{REFERENCE} jet $p_T$ [GeV]", ylabel="Response median")
        ax_res.set(xlabel=f"{REFERENCE} jet $p_T$ [GeV]", ylabel="Response IQR / median", ylim=(0, None))
        ax_med.legend(fontsize=11)
        fig.suptitle(f"Binned response (error bars: bootstrap std, {N_BOOTSTRAP} resamples; "
                     f"empty bins left blank)  |  {title}", fontsize=14)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # (d) summary table
        header = ["Collection", "Backend", "Kernel check", "Compile / mode / dynamic", "Graphs", "Shapes", "Events",
                  "Particles", "Non-finite decoded", "Dropped", "Jets", "Matched frac",
                  f"|r-1|<{r['rel_cut']:g} frac", "Median", "IQR", "IQR/med"]
        rows = []
        for name in order:
            g, meta = r["vs_ref"].get(name), r["metas"][name]
            nonfinite = {k: v for k, v in meta["nonfinite"].items() if v}
            bv = meta["backend_validation"]
            rows.append([
                style(name)["label"],
                (f"{bv['backend']} {bv['attention_dtype']} ({bv['distribution']} {bv['version']})"
                 if "backend" in bv else bv.get("status", "-")),
                f"{meta['kernel_check']['status']} {meta['kernel_check']['batches_checked']} batches",
                f"{meta['compiled']} / {meta['compile_mode']} / {meta['compile_dynamic']}",
                str(meta["dynamo_graph_count"]),
                str(len(meta["padded_shapes"])),
                str(meta0["num_events"]),
                str(r["n_particles"][name]),
                ", ".join(f"{k}:{v}" for k, v in nonfinite.items()) or "0",
                str(r["dropped"][name]), str(len(r["flat"][name]["pt"])),
                f"{g['angular_recall']:.3f}" if g else "-",
                f"{g['response_qualified_recall']:.3f}" if g else "-",
                f"{g['med']:.4f}" if g else "-",
                f"{g['iqr']:.4f}" if g else "-",
                f"{g['iqr'] / g['med']:.4f}" if g else "-",
            ])
        fig, ax = plt.subplots(figsize=(24, 1.6 + 0.55 * len(rows)))
        ax.axis("off")
        table = ax.table(cellText=rows, colLabels=header, loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.auto_set_column_width(list(range(len(header))))
        table.scale(1, 1.8)
        for i, name in enumerate(order, start=1):
            table[i, 0].get_text().set_color(style(name)["color"])
        ax.set_title(f"Summary  |  {title}\nMatched frac = one-to-one matches / {REFERENCE} jets; "
                     f"resolution = IQR (p75 - p25) of matched pT ratio; Dropped = non-finite particles "
                     "excluded from clustering", fontsize=12)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
    print(f"Report: {path}", flush=True)


def main():
    args = parse_args()
    ds = colliderml_dataset(args.dataset)
    root = args.outdir.resolve() / "physics"
    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    if not args.plot_only:
        run_all(args, make_settings(args), data_dir)
    if not args.skip_plots:
        report = compute_jets(args.variants, data_dir, ds)
        for name, m in report["vs_ref"].items():
            print(f"{name:>24}: jets={len(report['flat'][name]['pt'])} matched={m['angular_recall']:.3f} "
                  f"median={m['med']:.4f} IQR={m['iqr']:.4f}  (vs {REFERENCE})", flush=True)
        plot_report(report, args.variants, root / "physics.pdf")


if __name__ == "__main__":
    main()