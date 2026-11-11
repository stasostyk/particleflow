#!/usr/bin/env python3
"""MLPF physics check: run --variants exactly the way benchmark_worker.py runs them and
compare jet-level physics against TFDS ground truth (ColliderML). Run from the MLPF repo root.
All variants run sequentially in THIS process; Dynamo caches and CUDA memory are reset between them.
Physics only: no timing, NVML or memory statistics. Every real forward is checked against the
variant's expected attention kernel (operator/kernel NAMES only); a mismatch aborts the run.
Outputs: <outdir>/physics/data/{truth,<variant>}.npz and <outdir>/physics/physics.pdf.
Example:
  python /path/to/benchmark_physics.py --checkpoint model.pt --model-kwargs config.pkl \\
      --dataset colliderml_ttbar_nopu_pf --data-dir /path/to/tfds --split test \\
      --variants fp32_efficient fp16_flash bf16_flash --batch-size 4 --num-events 512
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
NEUTRAL = {"gen": ("Gen jets (truth)", "black", "-"), "target": ("Target", "dimgray", "--"),
           "cand": ("PF", "darkgray", ":")}


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
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--variants", nargs="+", choices=list(REGISTRY), default=["fp16_flash"],
                   help="Variants to run; each is compared against ground truth")
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
            or (args.input_clip is not None and (not 0 < args.input_clip < float("inf")))):
        p.error("Invalid batch size, event count, padding, fixed length or input clip")
    if not args.device.startswith("cuda:") or not args.device.split(":", 1)[1].isdigit():
        p.error("--device must be an explicit CUDA ordinal, e.g. cuda:0")
    if args.plot_only and args.skip_plots:
        p.error("--plot-only and --skip-plots are mutually exclusive")
    if not args.checkpoint.is_file() or not args.model_kwargs.is_file():
        p.error("--checkpoint and --model-kwargs must exist")
    args.variants = list(dict.fromkeys(args.variants))  # Every listed variant runs; no reference.
    return args


def make_settings(args):
    """Every key read by the imported worker functions and the mirrored run_job setup."""
    return {
        "checkpoint": str(args.checkpoint.resolve()), "model_kwargs": str(args.model_kwargs.resolve()),
        "dataset": args.dataset, "data_dir": str(args.data_dir.resolve()), "split": args.split,
        "num_events": args.num_events, "batch_size": args.batch_size,
        "pad_bin_size": args.pad_bin_size, "batch_order": args.batch_order,
        "input_clip": args.input_clip, "fixed_seq_len": args.fixed_seq_len, "device": args.device,
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


# ----------------------------------------------------------------------------- truth
def load_truth(settings):
    """Truth for the SAME event indices load_events reads (same builder, split and order)."""
    import tensorflow_datasets as tfds
    source = tfds.builder(settings["dataset"], data_dir=settings["data_dir"]).as_data_source(
        split=settings["split"])
    events = {"target": {}, "cand": {}, "gen": {}}
    pt_source = "ytarget_pt_orig/ytarget_e_orig" if "ytarget_pt_orig" in source[0] else "ytarget[:, 2]/[:, 6]"
    for i in range(settings["num_events"]):
        ev = source[i]
        missing = [k for k in ("ytarget", "ycand", "genjets") if k not in ev]
        if missing:
            raise KeyError(f"Event {i}: TFDS fields {missing} missing; available: {sorted(ev)}")
        for name, key in (("target", "ytarget"), ("cand", "ycand")):
            # Column layout as in inference.load_pf_baseline: 0 cls, 2 pt, 3 eta, 4 sin, 5 cos, 6 E.
            y = np.asarray(ev[key], dtype=np.float64)
            pt, energy = y[:, 2], y[:, 6]
            if name == "target" and "ytarget_pt_orig" in ev:
                pt, energy = np.asarray(ev["ytarget_pt_orig"], np.float64), np.asarray(ev["ytarget_e_orig"], np.float64)
            valid = y[:, 0] != 0
            if not (np.isfinite(pt[valid]).all() and (pt[valid] > 0).all() and np.isfinite(energy[valid]).all()):
                raise ValueError(f"Event {i}: {key} pT/E for cls_id != 0 are non-finite or <= 0 "
                                 f"(source {pt_source}); is pT stored transformed in this TFDS version?")
            events[name][i] = {"cls_id": y[:, 0], "pt": pt, "eta": y[:, 3],
                               "phi": np.arctan2(y[:, 4], y[:, 5]), "energy": energy}
        g = np.asarray(ev["genjets"], dtype=np.float64)
        g = g.reshape(0, 4) if g.size == 0 else g
        if g.ndim != 2 or g.shape[1] < 4 or not np.isfinite(g[:, :4]).all():
            raise ValueError(f"Event {i}: genjets must be finite (jets, >=4) [pt, eta, phi, E]; got {g.shape}")
        events["gen"][i] = {"pt": g[:, 0], "eta": g[:, 1], "phi": g[:, 2], "energy": g[:, 3]}
    print(f"Truth: {settings['num_events']} events, target pT from {pt_source}", flush=True)
    return {name: pack(ev, settings["num_events"]) for name, ev in events.items()}, pt_source


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
    which include CUDA kernel names; no other hook sees all of them. No timings are read or kept."""
    if not patterns:
        return W.forward(model, variant, x, mask), []
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], acc_events=True) as prof:
        out = W.forward(model, variant, x, mask)
        torch.cuda.synchronize(device)  # Flush CUDA activity so kernel names are recorded.
    keys = W.profiler_keys(prof)
    observed = [k for k in keys if any(re.search(rx, k, flags=re.I) for rx in patterns)]
    if not observed:
        nearby = [k for k in keys if any(t in k.lower() for t in ("attn", "flash", "attention", "sdpa"))]
        raise RuntimeError(f"{variant.name}: expected kernel {list(patterns)} did NOT run on real batch "
                           f"{tuple(x.shape)}. Attention-like ops seen: {nearby[:30]}")
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
    if any(dt == "bfloat16" for dt in (variant.precision.weights, variant.precision.input_dtype)):  # run_job
        if not torch.cuda.is_bf16_supported():                                     # run_job
            raise RuntimeError("BF16 variant requested, but GPU does not support BF16")  # run_job
    model = W.build_model(config, state, variant, device, settings["allow_partial_state"])  # run_job
    eager_model = model                                                            # run_job
    compiled = bool(variant.compile or settings["compile"])                        # run_job, no reference exception
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
    # benchmark_worker imports nvtx at module level, only for time_batch's annotation, which is
    # never called here. A bare placeholder keeps the real nvtx out of this process entirely.
    sys.modules.setdefault("nvtx", types.ModuleType("nvtx"))
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
    truth, pt_source = load_truth(settings)
    save_npz(data_dir / "truth.npz", truth, {**common, "target_pt_source": pt_source})
    for name in args.variants:
        output = data_dir / f"{name}.npz"
        output.unlink(missing_ok=True)  # Never leave a stale result behind a failed run.
        print(f"\n=== {name} ===", flush=True)
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

    paths = [data_dir / "truth.npz", *(data_dir / f"{v}.npz" for v in variants)]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing saved data (run without --plot-only first): {missing}")
    truth_meta, truth = load_npz(paths[0])
    metas, colls = {}, {"target": truth["target"], "cand": truth["cand"]}
    for v, path in zip(variants, paths[1:]):
        metas[v], loaded = load_npz(path)
        colls[v] = loaded["pred"]
        if metas[v]["dataset_fingerprint_sha256"] != truth_meta["dataset_fingerprint_sha256"]:
            raise RuntimeError(f"Fingerprint mismatch: {v} was run on different events than truth.npz")
    jetdef, ptcut, dr = get_jet_config(ds)
    rel_cut = JET_CONFIG[ds.value].match_rel_pt

    def p4(coll, keep):
        ev = np.repeat(np.arange(len(coll["counts"])), coll["counts"])
        counts = np.bincount(ev[keep], minlength=len(coll["counts"]))
        pt, eta, phi, e = (np.asarray(coll[k], np.float64)[keep] for k in ("pt", "eta", "phi", "energy"))
        flat = awkward.zip({"px": pt * np.cos(phi), "py": pt * np.sin(phi), "pz": pt * np.sinh(eta), "E": e})
        return awkward.unflatten(flat, counts), int(keep.sum())

    gen, _ = p4(truth["gen"], truth["gen"]["pt"] > ptcut)
    jets, dropped, n_particles = {"gen": vector.awk(gen)}, {}, {}
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
    vs_gen = {name: match("gen", name) for name in colls}
    vs_target = {name: match("target", name) for name in colls if name != "target"}
    return dict(flat=flat, vs_gen=vs_gen, vs_target=vs_target, dropped=dropped, n_particles=n_particles, metas=metas,
                truth_meta=truth_meta, ptcut=ptcut, dr=dr, rel_cut=rel_cut, ds_name=ds.value)


# ----------------------------------------------------------------------------- plots
def plot_report(r, variants, path: Path):
    import matplotlib
    import matplotlib.ticker
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    import mplhep
    from mlpf.plotting.plot_utils import EVALUATION_DATASET_NAMES, GENJET_BINS_PT_DATASET

    order = ["target", "cand", *variants]
    tab = plt.get_cmap("tab10")

    def style(name):  # Variants: tab10 in --variants order; earlier = thicker, so identical lines stay visible.
        if name in NEUTRAL:
            label, color, ls = NEUTRAL[name]
            return {"label": label, "color": color, "linestyle": ls, "linewidth": 2.0}
        i = variants.index(name)
        return {"label": name, "color": tab(i % 10), "linestyle": "-", "linewidth": max(1.0, 3.2 - 0.45 * i)}

    def legend_label(name, m):
        return (f"{style(name)['label']}: {m['med']:.3f} ± {m['iqr']:.3f}" if m["num_angular_matches"]
                else f"{style(name)['label']}: no matched jets")

    base = r["truth_meta"]["dataset"].split("/")[0].split(":")[0]
    title = (f"{EVALUATION_DATASET_NAMES.get(base, base)} | {r['truth_meta']['num_events']} events "
             f"({r['truth_meta']['split']} split) | anti-kt R=0.4, pT>{r['ptcut']:g} GeV, ΔR<{r['dr']:g}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(mplhep.styles.CMS), PdfPages(path) as pdf:
        # (a) spectra with ratio-to-gen
        fig = plt.figure(figsize=(18, 9))
        grid = fig.add_gridspec(2, 2, height_ratios=[3, 1], hspace=0.05)
        for col, (key, bins, xlabel) in enumerate([
                ("pt", np.logspace(np.log10(r["ptcut"]), 3, 41), "Jet $p_T$ [GeV]"),
                ("eta", np.linspace(-5, 5, 41), r"Jet $\eta$")]):
            ax = fig.add_subplot(grid[0, col])
            rax = fig.add_subplot(grid[1, col], sharex=ax)
            gen_h = np.histogram(r["flat"]["gen"][key], bins)[0]
            for name in ["gen", *order]:
                h = np.histogram(r["flat"][name][key], bins)[0]
                kw = style(name)
                ax.stairs(h, bins, **{**kw, "label": f"{kw['label']} ({len(r['flat'][name][key])} jets)"})
                with np.errstate(divide="ignore", invalid="ignore"):
                    rax.stairs(np.where(gen_h > 0, h / gen_h, np.nan), bins, **{**kw, "label": None})
            ax.set(yscale="log", ylabel="Jets / bin")
            ax.tick_params(labelbottom=False)
            ax.legend(fontsize=11, loc="upper right")
            rax.axhline(1, color="black", lw=0.8)
            rax.set(xlabel=xlabel, ylabel="/ gen", ylim=(0.5, 1.5), xscale="log" if key == "pt" else "linear")
        fig.suptitle(f"Jet spectra  |  {title}", fontsize=14)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # (b) matched-jet response (plot_utils binning), vs gen and vs target
        fig, axes = plt.subplots(2, 2, figsize=(18, 13))
        for row, (key, ref) in enumerate([("vs_gen", "gen"), ("vs_target", "target")]):
            # Left: plot_utils.plot_jet_ratio binning exactly. Right: zoom, coarser so overlaid lines stay readable.
            for col, bins in enumerate([np.linspace(0, 5, 500), np.linspace(0.5, 1.5, 101)]):
                ax = axes[row, col]
                for name in [n for n in order if n in r[key]]:
                    m = r[key][name]
                    ax.hist(m["ratio"], bins=bins, histtype="step", **{**style(name), "label": legend_label(name, m)})
                ax.axvline(1, color="black", lw=0.8, ls="--")
                ax.set(xlabel=f"Matched jet $p_T$ / {ref} jet $p_T$", ylabel="Matched jets / bin")
                ax.set_title(f"Response vs {ref}" + (" (zoom)" if col else "") +
                             "  |  legend: median ± IQR", fontsize=13)
                ax.legend(fontsize=10, loc="upper right")
        fig.suptitle(f"Jet response  |  {title}", fontsize=14)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # (c) response and resolution vs gen-jet pT (plot_utils GENJET_BINS_PT_DATASET, (lo, hi] bins)
        edges = GENJET_BINS_PT_DATASET[r["ds_name"]]
        centers = [0.5 * (lo + hi) for lo, hi in zip(edges[:-1], edges[1:])]
        fig, (ax_med, ax_res) = plt.subplots(1, 2, figsize=(18, 7.5))
        for name in order:
            m, med, res = r["vs_gen"][name], [], []
            for lo, hi in zip(edges[:-1], edges[1:]):
                sel = m["ratio"][(m["ref_pt"] > lo) & (m["ref_pt"] <= hi)]
                p25, p50, p75 = np.percentile(sel, [25, 50, 75]) if len(sel) else (np.nan,) * 3
                med.append(p50)
                res.append((p75 - p25) / p50)
            ax_med.plot(centers, med, marker="o", **style(name))
            ax_res.plot(centers, res, marker="o", **style(name))
        for ax in (ax_med, ax_res):
            ax.set_xscale("log")
            ax.set_xticks(edges, [str(e) for e in edges])
            ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax_med.axhline(1, color="black", lw=0.8, ls="--")
        ax_med.set(xlabel="Gen jet $p_T$ [GeV]", ylabel="Response median")
        ax_res.set(xlabel="Gen jet $p_T$ [GeV]", ylabel="Response IQR / median", ylim=(0, None))
        ax_med.legend(fontsize=11)
        fig.suptitle(f"Binned response (empty bins left blank)  |  {title}", fontsize=14)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # (d) summary table
        header = ["Collection", "Kernel check", "Compile / mode / dynamic", "Graphs", "Shapes", "Events", "Particles",
                  "Non-finite decoded", "Dropped", "Jets", "Matched frac", f"|r-1|<{r['rel_cut']:g} frac",
                  "Median (gen)", "IQR (gen)", "IQR/med (gen)", "Median (target)"]
        rows = []
        for name in order:
            g, t, meta = r["vs_gen"][name], r["vs_target"].get(name), r["metas"].get(name)
            nonfinite = ({k: v for k, v in meta["nonfinite"].items() if v} if meta else {})
            rows.append([
                style(name)["label"],
                (f"{meta['kernel_check']['status']} {meta['kernel_check']['batches_checked']} batches"
                 if meta else "-"),
                (f"{meta['compiled']} / {meta['compile_mode']} / {meta['compile_dynamic']}" if meta else "-"),
                str(meta["dynamo_graph_count"]) if meta else "-",
                str(len(meta["padded_shapes"])) if meta else "-",
                str(r["truth_meta"]["num_events"]),
                str(r["n_particles"][name]),
                (", ".join(f"{k}:{v}" for k, v in nonfinite.items()) or "0") if meta else "-",
                str(r["dropped"][name]), str(len(r["flat"][name]["pt"])),
                f"{g['angular_recall']:.3f}", f"{g['response_qualified_recall']:.3f}",
                f"{g['med']:.4f}", f"{g['iqr']:.4f}", f"{g['iqr'] / g['med']:.4f}",
                f"{t['med']:.4f}" if t else "-",
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
        ax.set_title(f"Summary  |  {title}\nMatched frac = one-to-one matches / gen jets; "
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
        for name, m in report["vs_gen"].items():
            print(f"{name:>24}: jets={len(report['flat'][name]['pt'])} matched={m['angular_recall']:.3f} "
                  f"median={m['med']:.4f} IQR={m['iqr']:.4f}", flush=True)
        plot_report(report, args.variants, root / "physics.pdf")


if __name__ == "__main__":
    main()