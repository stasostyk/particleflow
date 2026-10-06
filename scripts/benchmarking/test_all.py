#!/usr/bin/env python
"""
Batched GPU inference benchmark for MLPF attention models.

This script exports an MLPF attention model to ONNX (fused attention variants) and
benchmarks batched inference on CUDA with both PyTorch and ONNX Runtime. Every
batched scenario is compared against an unbatched FP32 PyTorch baseline run with the
exact "math" attention kernel, which is the most numerically conservative path.

Scenarios
---------
Baseline (never batched, batch size 1, same as the original validation script):
    PT_ATTN_MATH_FP32            PyTorch, FP32, SDPA math backend

Batched (batch size from ``--batch-size``):
    PT_ATTN_FLASH_FP16           PyTorch, autocast FP16, SDPA auto (flash/efficient)
    ONNX_ATTN_FLASH_FP32         ORT, FP32 model, com.microsoft.MultiHeadAttention in FP32
    ONNX_ATTN_FLASH_FP32_FP16    ORT, FP32 model, MultiHeadAttention cast to FP16
    ONNX_ATTN_FLASH_FP16         ORT, fully FP16 model, FP16 MultiHeadAttention

What is measured
----------------
Only device-side inference is timed. Host-to-device copies, output copies back to the
host, jet clustering and error computation are deliberately excluded:

* ONNX Runtime: inputs are uploaded to the GPU as ``OrtValue`` objects *before* the
  timer starts, bound with ``IOBinding``, outputs are bound to the GPU, and the session
  is run with ``run_with_iobinding``. ORT synchronises the CUDA stream at the end of a
  run by default, so a wall-clock timer around ``run_with_iobinding`` measures the
  kernel work plus launch overhead - exactly what a deployed server pays.
* PyTorch: inputs are moved to the GPU before the timer starts, and
  ``torch.cuda.synchronize()`` is called immediately before and after the forward pass
  so the wall-clock time covers the full device computation and nothing else.

Both measurements therefore include the same thing (synchronised device execution),
which makes the latencies directly comparable.

Choices and their justification
-------------------------------
* ``torch.inference_mode()`` instead of ``no_grad()``: it additionally disables
  version-counter and view tracking, is the documented recommendation for inference,
  and is a strict superset of ``no_grad`` for this use case.
* ``torch.autocast(dtype=float16)`` for the FP16 PyTorch run: autocast is the supported
  way to run mixed precision; it keeps numerically sensitive ops (softmax, layer norm,
  reductions) in FP32 automatically, unlike a blanket ``model.half()``.
* SDPA with the default backend selection for the "FLASH" run: ``F.scaled_dot_product_attention``
  picks the fastest available fused kernel (flash / memory-efficient) for the given
  dtype and shape. Forcing a specific backend is only used for the baseline, where we
  *want* the reference math implementation.
* ``torch.set_float32_matmul_precision("high")`` (TF32 matmuls) matches the setting
  used in the original validation script and in training. Note that TF32 is a
  reduced-precision matmul; set it to "highest" if a bit-exact FP32 baseline is required.
* ``torch.compile`` is optional (``--compile``). It is a standard tool, but with
  per-batch variable sequence lengths it recompiles / falls back to dynamic shapes,
  and for attention-dominated models the win over eager SDPA is usually modest.
  Leaving it off by default keeps the benchmark deterministic and easy to debug.
* ONNX export uses the TorchScript exporter (``dynamo=False``) because the custom
  ``aten::scaled_dot_product_attention`` symbolic that maps to
  ``com.microsoft.MultiHeadAttention`` is registered through that API, as in the
  original script. Batch and sequence axes are dynamic so a single export serves any
  batch size.
* ORT session: ``CUDAExecutionProvider`` only (we do not want silent CPU fallback to
  hide in the timings), ``ORT_ENABLE_ALL`` graph optimisations (the documented default
  for production), and ``IOBinding`` (the documented way to avoid host/device copies
  in the hot loop). No experimental options (CUDA graphs, TensorRT EP, etc.).
* Batching: events in a batch are padded to the longest event in the batch, rounded up
  to ``--pad-bin-size`` (and to the model's ``pad_to_multiple_elements``) so the number
  of distinct shapes seen by the runtimes stays small. Padded positions are masked and
  are excluded from error computation and jet clustering.
* Bucketing (``--sort-by-length``): events are sorted by element count before batching
  so each batch contains similarly sized events and padding waste is minimised. This is
  the standard "length bucketing" trick; it is optional because it is only realistic
  for offline processing where the input order can be chosen.
* Repeats (``--num-repeats``): each scenario is run N times over the full dataset after
  a single warm-up. Latency statistics are computed over all timed batches, and
  throughput is reported as mean +/- std across repeats. Error and jet metrics are taken
  from the first repeat only (inference is deterministic up to floating-point noise).
* Memory and GPU verification: PyTorch peak memory comes from
  ``torch.cuda.max_memory_allocated`` (the caching allocator's own peak counter). ORT
  allocates outside that allocator, so for every scenario the process' GPU memory is
  also sampled through NVML (``pynvml``, optional) after each batch and the maximum is
  reported. After warm-up each runner reports where its weights, inputs and outputs
  live (``verify_on_gpu``) and the script fails if anything is not on CUDA.

Usage
-----
    python benchmark_batched_inference.py \
        --checkpoint experiments/.../checkpoints/checkpoint-10-1.234.pth \
        --model-kwargs experiments/.../model_kwargs.pkl \
        --data-dir /path/to/tensorflow_datasets \
        --dataset cms_pf_ttbar --num-events 200 --batch-size 8 --outdir ./bench_out
"""

import argparse
import contextlib
import gc
import json
import os
import pickle as pkl
import platform
import subprocess
import sys
import time

# Make the local ``mlpf`` package importable when running from the repository root.
sys.path.insert(0, os.getcwd())

import awkward  # noqa: E402
import boost_histogram as bh  # noqa: E402
import fastjet  # noqa: E402
import matplotlib  # noqa: E402
import mplhep  # noqa: E402
import numpy as np  # noqa: E402
import onnx  # noqa: E402
import onnxruntime as rt  # noqa: E402
import onnxscript  # noqa: E402
import tensorflow_datasets as tfds  # noqa: E402
import torch  # noqa: E402
import vector  # noqa: E402
from onnxscript import opset20 as op  # noqa: E402
from tqdm import tqdm  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

try:  # NVML is optional; without it only PyTorch-side memory statistics are reported.
    import pynvml

    pynvml.nvmlInit()
    NVML_AVAILABLE = True
except Exception:  # pragma: no cover - depends on the environment
    NVML_AVAILABLE = False

from mlpf.conf import AttentionType, MLPFConfig, ModelType  # noqa: E402
from mlpf.model.mlpf import MLPF  # noqa: E402
from mlpf.model.utils import unpack_predictions  # noqa: E402

# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------

INPUT_NAMES = ["Xfeat_normed", "mask"]
OUTPUT_NAMES = ["bid", "id", "momentum", "pu"]
# Outputs compared against the baseline: (name, index in the prediction tuple).
COMPARED_OUTPUTS = [("id", 1), ("momentum", 2), ("pu", 3)]
ONNX_OPSET = 20
DYNAMIC_AXES = {name: {0: "num_batch", 1: "num_elements"} for name in INPUT_NAMES}

BASELINE = "PT_ATTN_MATH_FP32"
BATCHED_CONFIGS = [
    "PT_ATTN_FLASH_FP16",
    "ONNX_ATTN_FLASH_FP32",
    "ONNX_ATTN_FLASH_FP32_FP16",
    "ONNX_ATTN_FLASH_FP16",
]

# Declarative description of each scenario. To add a scenario: add an entry here and,
# if it needs a new ONNX file, make sure ``export_onnx_models`` produces it.
SCENARIOS = {
    "PT_ATTN_MATH_FP32": dict(
        backend="torch",
        attention_type=AttentionType.MATH,
        autocast=False,
        sdpa_backend=torch.nn.attention.SDPBackend.MATH,
    ),
    "PT_ATTN_FLASH_FP16": dict(
        backend="torch",
        attention_type=AttentionType.FLASH,
        autocast=True,
        sdpa_backend=None,  # let SDPA choose the fastest fused kernel
    ),
    "ONNX_ATTN_FLASH_FP32": dict(
        backend="onnx",
        filename="model_fused_fp32.onnx",
        sdpa_precision="fp32",
        half_model=False,
        input_dtype=np.float32,
    ),
    "ONNX_ATTN_FLASH_FP32_FP16": dict(
        backend="onnx",
        filename="model_fused_fp32_fp16.onnx",
        sdpa_precision="fp16",
        half_model=False,
        input_dtype=np.float32,
    ),
    "ONNX_ATTN_FLASH_FP16": dict(
        backend="onnx",
        filename="model_fused_fp16.onnx",
        sdpa_precision="fp16",
        half_model=True,
        input_dtype=np.float16,
    ),
}

# FP16 has a maximum magnitude of ~65504; clip features so FP16 scenarios do not overflow.
FEATURE_CLIP = 60000.0


# --------------------------------------------------------------------------------------
# Argument parsing
# --------------------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the checkpoint .pth file")
    parser.add_argument("--model-kwargs", type=str, required=True, help="Path to the model_kwargs.pkl file")
    parser.add_argument("--dataset", type=str, default="cms_pf_ttbar", help="TFDS dataset name")
    parser.add_argument("--data-dir", type=str, required=True, help="Directory for TFDS datasets")
    parser.add_argument("--outdir", type=str, default="./batched_benchmark", help="Output directory for ONNX files, plots and JSON")
    parser.add_argument("--num-events", type=int, default=100, help="Number of events to benchmark")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for the batched scenarios")
    parser.add_argument("--pad-bin-size", type=int, default=0, help="Round padded sequence lengths up to a multiple of this value (0 = disabled)")
    parser.add_argument("--num-warmup", type=int, default=3, help="Number of warm-up iterations per scenario (not timed)")
    parser.add_argument("--num-repeats", type=int, default=1, help="Run each scenario this many times over the dataset and average the timings")
    parser.add_argument("--sort-by-length", action="store_true", help="Sort events by element count before batching (length bucketing)")
    parser.add_argument("--num-threads", type=int, default=1, help="CPU threads for ORT/PyTorch host-side work")
    parser.add_argument("--compile", action="store_true", help="Wrap PyTorch models in torch.compile (see module docstring)")
    parser.add_argument(
        "--configs",
        type=str,
        nargs="+",
        choices=BATCHED_CONFIGS,
        default=BATCHED_CONFIGS,
        help="Batched scenarios to run (the baseline is always run)",
    )
    return parser.parse_args()


# --------------------------------------------------------------------------------------
# Model construction and ONNX export
# --------------------------------------------------------------------------------------


def load_model_config(path):
    """Load the pickled MLPFConfig, re-validating so enums are correct across versions."""
    with open(path, "rb") as f:
        raw = pkl.load(f)
    if isinstance(raw, dict):
        return MLPFConfig.model_validate(raw)
    return MLPFConfig.model_validate(raw.model_dump())


def make_mlpf_config(base_config, **overrides):
    """Deep-copy the config and apply attention / top-level / model-level overrides."""
    config = base_config.model_copy(deep=True)
    for k, v in overrides.items():
        if k in ["export_onnx_fused", "save_attention", "attention_type"]:
            if config.model.type == ModelType.ATTENTION:
                setattr(config.model.attention, k, v)
        elif hasattr(config, k):
            setattr(config, k, v)
        elif hasattr(config.model, k):
            setattr(config.model, k, v)
    return config


def build_model(base_config, state_dict, device, half=False, **overrides):
    """Instantiate MLPF with config overrides, load weights and move to ``device``."""
    model = MLPF(config=make_mlpf_config(base_config, **overrides))
    model.eval()
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    if half:
        model = model.half()
    return model


def register_sdpa_symbolic(num_heads, precision):
    """
    Map ``aten::scaled_dot_product_attention`` to ``com.microsoft.MultiHeadAttention``
    during ONNX export. ``precision="fp16"`` casts Q/K/V to FP16 inside the op and
    casts the output back to the query dtype, so it can be used in an FP32 graph.
    """
    custom_opset = onnxscript.values.Opset(domain="onnx-script", version=1)
    msft_op = onnxscript.values.Opset("com.microsoft", 1)

    if precision == "fp16":

        @onnxscript.script(custom_opset)
        def SDPA(query, key, value):
            q16 = op.Cast(query, to=onnx.TensorProto.FLOAT16)
            k16 = op.Cast(key, to=onnx.TensorProto.FLOAT16)
            v16 = op.Cast(value, to=onnx.TensorProto.FLOAT16)
            output, _, _ = msft_op.MultiHeadAttention(q16, k16, v16, num_heads=num_heads)
            return op.CastLike(output, query)

    elif precision == "fp32":

        @onnxscript.script(custom_opset)
        def SDPA(query, key, value):
            output, _, _ = msft_op.MultiHeadAttention(query, key, value, num_heads=num_heads)
            return output

    else:
        raise ValueError(f"Unknown SDPA precision {precision}")

    def symbolic(g, query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False):
        return g.onnxscript_op(SDPA, query, key, value).setType(query.type())

    torch.onnx.register_custom_op_symbolic(
        symbolic_name="aten::scaled_dot_product_attention",
        symbolic_fn=symbolic,
        opset_version=ONNX_OPSET,
    )


def export_onnx_models(configs, base_config, state_dict, num_heads, input_dim, pad_multiple, device, outdir):
    """Export one ONNX file per requested ONNX scenario. Returns {config: path}."""
    # Attention type SIMPLE + export_onnx_fused=True produces a graph in which attention
    # is a single SDPA call that our custom symbolic can intercept.
    export_config = make_mlpf_config(base_config, attention_type=AttentionType.SIMPLE, export_onnx_fused=True)

    dummy_len = 400
    if pad_multiple > 0:
        dummy_len = ((dummy_len + pad_multiple - 1) // pad_multiple) * pad_multiple
    dummy_x = torch.randn(1, dummy_len, input_dim, device=device)
    dummy_mask = torch.ones(1, dummy_len, device=device)

    paths = {}
    for cfg in configs:
        spec = SCENARIOS[cfg]
        if spec["backend"] != "onnx":
            continue
        path = os.path.join(outdir, spec["filename"])
        print(f"Exporting {cfg} -> {path}")

        model = build_model(export_config, state_dict, device, half=spec["half_model"])
        register_sdpa_symbolic(num_heads, spec["sdpa_precision"])
        inputs = (dummy_x.half(), dummy_mask.half()) if spec["half_model"] else (dummy_x, dummy_mask)
        torch.onnx.export(
            model,
            inputs,
            path,
            opset_version=ONNX_OPSET,
            input_names=INPUT_NAMES,
            output_names=OUTPUT_NAMES,
            dynamic_axes=DYNAMIC_AXES,
            dynamo=False,
            verbose=False,
        )
        paths[cfg] = path
        del model
        gc.collect()
        torch.cuda.empty_cache()
    return paths


# --------------------------------------------------------------------------------------
# GPU memory helpers
# --------------------------------------------------------------------------------------

MIB = 1024**2


def process_gpu_memory_mib(device_id=0):
    """
    GPU memory used by *this process* in MiB, read through NVML. Covers every allocator
    (PyTorch caching allocator, ORT arena, cuBLAS/cuDNN workspaces, CUDA context).
    Returns None when NVML or per-process accounting is unavailable.
    """
    if not NVML_AVAILABLE:
        return None
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
        for proc in pynvml.nvmlDeviceGetComputeRunningProcesses(handle):
            if proc.pid == os.getpid() and proc.usedGpuMemory is not None:
                return proc.usedGpuMemory / MIB
    except pynvml.NVMLError:
        pass
    return None


def torch_peak_memory_mib():
    """Peak memory held by tensors in PyTorch's caching allocator since the last reset."""
    return torch.cuda.max_memory_allocated() / MIB


# --------------------------------------------------------------------------------------
# Inference runners (one small class per backend; both expose ``run(X, mask)`` and
# ``verify_on_gpu()``)
# --------------------------------------------------------------------------------------


class TorchRunner:
    """Runs a PyTorch model and times only the synchronised device-side forward pass."""

    def __init__(self, model, device, autocast=False, sdpa_backend=None):
        self.model = model
        self.device = device
        self.autocast = autocast
        self.sdpa_backend = sdpa_backend

    def run(self, X, mask):
        """
        X: [B, L, F] float32 CPU tensor, mask: [B, L] bool CPU tensor.
        Returns (prediction tuple of float32 CPU tensors, inference seconds).
        """
        X_dev = X.to(self.device)
        mask_dev = mask.to(self.device)

        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.float16) if self.autocast else contextlib.nullcontext()
        )
        sdpa_ctx = (
            torch.nn.attention.sdpa_kernel(self.sdpa_backend) if self.sdpa_backend is not None else contextlib.nullcontext()
        )

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode(), autocast_ctx, sdpa_ctx:
            pred = self.model(X_dev, mask_dev)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0

        # Remember where the raw outputs live so verify_on_gpu() can report it.
        self._last_output_devices = sorted({str(p.device) for p in pred})
        pred = tuple(p.float().cpu() for p in pred)
        return pred, dt

    def verify_on_gpu(self):
        """Return evidence that weights and outputs are on CUDA; raise if they are not."""
        param_devices = sorted({str(p.device) for p in self.model.parameters()})
        info = {
            "backend": "torch",
            "parameter_devices": param_devices,
            "output_devices": getattr(self, "_last_output_devices", []),
            "autocast_fp16": self.autocast,
        }
        if not all(d.startswith("cuda") for d in param_devices + info["output_devices"]):
            raise RuntimeError(f"PyTorch model is not fully on CUDA: {info}")
        return info


class OrtRunner:
    """Runs an ONNX Runtime session on CUDA with IOBinding; times only ``run_with_iobinding``."""

    def __init__(self, path, input_dtype, num_threads, device_id=0):
        self.input_dtype = input_dtype
        self.device_id = device_id

        so = rt.SessionOptions()
        so.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.intra_op_num_threads = num_threads
        so.inter_op_num_threads = num_threads
        # CUDA EP only: no silent CPU fallback hiding in the measurements.
        providers = [("CUDAExecutionProvider", {"device_id": device_id})]
        self.sess = rt.InferenceSession(path, so, providers=providers)
        if "CUDAExecutionProvider" not in self.sess.get_providers():
            raise RuntimeError("CUDAExecutionProvider is not available in this onnxruntime build")
        self.binding = self.sess.io_binding()

    def run(self, X, mask):
        """
        X: [B, L, F] float32 CPU tensor, mask: [B, L] bool CPU tensor.
        Returns (prediction tuple of float32 CPU tensors, inference seconds).
        """
        X_np = np.ascontiguousarray(X.numpy().astype(self.input_dtype, copy=False))
        mask_np = np.ascontiguousarray(mask.numpy().astype(self.input_dtype))

        # Host -> device copies happen here, outside the timed region.
        x_ort = rt.OrtValue.ortvalue_from_numpy(X_np, "cuda", self.device_id)
        mask_ort = rt.OrtValue.ortvalue_from_numpy(mask_np, "cuda", self.device_id)

        self.binding.clear_binding_inputs()
        self.binding.clear_binding_outputs()
        self.binding.bind_ortvalue_input(INPUT_NAMES[0], x_ort)
        self.binding.bind_ortvalue_input(INPUT_NAMES[1], mask_ort)
        # Output shapes depend on the (dynamic) input shape, so let ORT allocate them on
        # the device from its arena; after warm-up this is a cheap arena lookup.
        for name in OUTPUT_NAMES:
            self.binding.bind_output(name, "cuda", self.device_id)

        # ORT synchronises the device at the end of run_with_iobinding by default.
        t0 = time.perf_counter()
        self.sess.run_with_iobinding(self.binding)
        dt = time.perf_counter() - t0

        # Remember input/output placement so verify_on_gpu() can report it.
        self._last_input_devices = sorted({x_ort.device_name(), mask_ort.device_name()})
        self._last_output_devices = sorted({o.device_name() for o in self.binding.get_outputs()})

        # Device -> host copies happen here, outside the timed region.
        outputs = self.binding.copy_outputs_to_cpu()
        pred = tuple(torch.from_numpy(np.asarray(o)).float() for o in outputs)
        return pred, dt

    def verify_on_gpu(self):
        """Return evidence that the session, inputs and outputs are on CUDA; raise if not."""
        info = {
            "backend": "onnxruntime",
            "session_providers": self.sess.get_providers(),
            "input_devices": getattr(self, "_last_input_devices", []),
            "output_devices": getattr(self, "_last_output_devices", []),
            "input_dtype": np.dtype(self.input_dtype).name,
        }
        ok = (
            info["session_providers"][0] == "CUDAExecutionProvider"
            and all(d == "cuda" for d in info["input_devices"] + info["output_devices"])
        )
        if not ok:
            raise RuntimeError(f"ONNX Runtime scenario is not fully on CUDA: {info}")
        return info


def make_runner(cfg, base_config, state_dict, onnx_paths, args, device):
    """Build the runner for a scenario name."""
    spec = SCENARIOS[cfg]
    if spec["backend"] == "torch":
        model = build_model(
            base_config,
            state_dict,
            device,
            attention_type=spec["attention_type"],
            export_onnx_fused=False,
            save_attention=False,
        )
        if args.compile:
            model = torch.compile(model)
        return TorchRunner(model, device, autocast=spec["autocast"], sdpa_backend=spec["sdpa_backend"])
    if spec["backend"] == "onnx":
        return OrtRunner(onnx_paths[cfg], spec["input_dtype"], args.num_threads)
    raise ValueError(f"Unknown backend {spec['backend']}")


# --------------------------------------------------------------------------------------
# Data batching
# --------------------------------------------------------------------------------------


def round_up(n, multiple):
    return n if multiple <= 0 else ((n + multiple - 1) // multiple) * multiple


def event_order(event_sizes, sort_by_length):
    """
    Order in which events are fed to the batched scenarios.

    With ``sort_by_length`` events are sorted by element count so neighbouring events,
    and therefore the events inside one batch, have similar lengths ("bucketing").
    Returned indices refer to positions in the dataset; the baseline predictions are keyed
    by the same indices, so the comparison is unaffected by the order.
    """
    indices = list(range(len(event_sizes)))
    if sort_by_length:
        indices.sort(key=lambda i: event_sizes[i])
    return indices


def make_batches(ds, order, batch_size, pad_multiple, input_dim):
    """
    Group consecutive events of ``order`` into batches padded to the longest event in each.

    Yields (event_indices, X [B, L, F] float32, mask [B, L] bool, lengths). Padded
    positions are zero in X and False in mask.
    """
    for start in range(0, len(order), batch_size):
        indices = order[start : start + batch_size]
        xs = [torch.as_tensor(np.asarray(ds[i]["X"]), dtype=torch.float32) for i in indices]
        lengths = [x.shape[0] for x in xs]
        padded_len = round_up(max(lengths), pad_multiple)

        X = torch.zeros(len(indices), padded_len, input_dim, dtype=torch.float32)
        mask = torch.zeros(len(indices), padded_len, dtype=torch.bool)
        for b, (x, n) in enumerate(zip(xs, lengths)):
            X[b, :n] = x
            mask[b, :n] = True
        X.clamp_(-FEATURE_CLIP, FEATURE_CLIP)
        yield indices, X, mask, lengths


# --------------------------------------------------------------------------------------
# Physics helpers (same jet definition as the original validation script)
# --------------------------------------------------------------------------------------


def particles_to_jets(pred, mask):
    """Cluster anti-kT R=0.4 jets (pT > 3 GeV) from the predicted particles of a batch."""
    jetdef = fastjet.JetDefinition(fastjet.antikt_algorithm, 0.4)
    ypred = unpack_predictions(pred)
    for k, v in ypred.items():
        ypred[k] = v[mask].detach().cpu().float().contiguous().numpy()

    counts = torch.sum(mask, axis=1).cpu().numpy()
    clsid = awkward.unflatten(ypred["cls_id"], counts)
    msk = clsid != 0
    p4 = awkward.unflatten(ypred["p4"], counts)

    vec = vector.awk(
        awkward.zip(
            {
                "pt": p4[msk][:, :, 0],
                "eta": p4[msk][:, :, 1],
                "phi": p4[msk][:, :, 2],
                "e": p4[msk][:, :, 3],
            }
        )
    )
    cluster = fastjet.ClusterSequence(vec.to_xyzt(), jetdef)
    jets = cluster.inclusive_jets(min_pt=3)
    return awkward.to_numpy(awkward.flatten(jets.pt))


def sum_overflow_into_last_bin(all_values):
    values = all_values[1:-1]
    values[-1] = values[-1] + all_values[-1]
    values[0] = values[0] + all_values[0]
    return values


def to_bh(data, bins):
    h = bh.Histogram(bh.axis.Variable(bins))
    h.fill(data)
    h[:] = sum_overflow_into_last_bin(h.values(flow=True)[:])
    return h


# --------------------------------------------------------------------------------------
# Benchmark loop
# --------------------------------------------------------------------------------------


class ErrorAccumulator:
    """Accumulates absolute differences vs. the baseline, per output and overall."""

    def __init__(self):
        self.sum = {name: 0.0 for name, _ in COMPARED_OUTPUTS}
        self.count = {name: 0 for name, _ in COMPARED_OUTPUTS}
        self.num_invalid = 0

    def add(self, pred, pred_base, n_valid):
        """``pred``/``pred_base`` are single-event [L, ...] tensors; only the first n_valid rows are compared."""
        for name, idx in COMPARED_OUTPUTS:
            diff = (pred_base[idx][:n_valid] - pred[idx][:n_valid]).abs().flatten().numpy()
            invalid = ~np.isfinite(diff)
            self.num_invalid += int(invalid.sum())
            diff = np.nan_to_num(diff, nan=1e6, posinf=1e6, neginf=1e6)
            self.sum[name] += float(diff.sum())
            self.count[name] += diff.size

    def summary(self):
        per_output = {name: (self.sum[name] / self.count[name] if self.count[name] else None) for name in self.sum}
        total_count = sum(self.count.values())
        return {
            "mae": sum(self.sum.values()) / total_count if total_count else None,
            "mae_per_output": per_output,
            "num_invalid": self.num_invalid,
        }


def timing_summary(batch_times, repeat_times, num_events, num_valid_elements, num_padded_elements):
    """
    Latency and throughput statistics.

    ``batch_times``: inference seconds of every timed batch across all repeats.
    ``repeat_times``: total inference seconds per repeat (len == num_repeats).
    The event/element counts refer to a single pass over the dataset.
    """
    times_ms = np.asarray(batch_times) * 1e3
    repeat_times = np.asarray(repeat_times)
    throughput_per_repeat = num_events / repeat_times
    total_s = float(repeat_times.mean())  # mean time for one pass over the dataset
    return {
        "num_repeats": len(repeat_times),
        "num_batches_per_repeat": len(batch_times) // len(repeat_times),
        "num_events": num_events,
        "padding_efficiency": num_valid_elements / num_padded_elements,
        "total_inference_time_s_per_repeat": repeat_times.tolist(),
        "total_inference_time_s_mean": total_s,
        "total_inference_time_s_std": float(repeat_times.std()),
        "throughput_events_per_s_std": float(throughput_per_repeat.std()),
        "latency_ms_per_batch": {
            "mean": float(times_ms.mean()),
            "std": float(times_ms.std()),
            "median": float(np.median(times_ms)),
            "p95": float(np.percentile(times_ms, 95)),
            "min": float(times_ms.min()),
            "max": float(times_ms.max()),
        },
        "latency_ms_per_event_mean": total_s / num_events * 1e3,
        "throughput_events_per_s": num_events / total_s,
        "throughput_valid_elements_per_s": num_valid_elements / total_s,
        "throughput_padded_elements_per_s": num_padded_elements / total_s,
    }


def run_scenario(cfg, runner, batches, num_warmup, num_repeats, baseline_preds=None):
    """
    Run all batches through ``runner`` ``num_repeats`` times.

    Returns a dict with timing, memory, GPU-placement evidence, error (if a baseline is
    given), jet pTs and - for the baseline - per-event predictions for later comparison.
    Error and jet metrics are collected on the first repeat only.
    """
    batches = list(batches)

    # Warm-up on the first batch: triggers lazy CUDA init, cuDNN/cuBLAS heuristics,
    # ORT arena allocation and (optionally) torch.compile. Not timed.
    _, X, mask, _ = batches[0]
    for _ in range(num_warmup):
        runner.run(X, mask)

    # Confirm placement once the runner has actually executed. Raises if not on CUDA.
    gpu_info = runner.verify_on_gpu()
    print(f"  GPU placement: {gpu_info}")

    # Reset PyTorch's peak counter after warm-up so it reflects steady-state inference.
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    nvml_max_per_repeat = []  # one high-water mark per repeat, so downstream tools get N samples

    batch_times, repeat_times, runs, jets_pt = [], [], [], []
    num_valid, num_padded, num_events = 0, 0, 0
    errors = ErrorAccumulator()
    per_event_preds = {}

    for repeat in range(num_repeats):
        repeat_time = 0.0
        repeat_mem_max = None
        for indices, X, mask, lengths in tqdm(batches, desc=f"{cfg} [repeat {repeat + 1}/{num_repeats}]"):
            pred, dt = runner.run(X, mask)
            batch_times.append(dt)
            repeat_time += dt

            # NVML sampling is outside the timed region; the arena does not shrink between
            # batches so sampling after each run captures the high-water mark well.
            mem = process_gpu_memory_mib()
            if mem is not None:
                repeat_mem_max = mem if repeat_mem_max is None else max(repeat_mem_max, mem)

            if repeat > 0:
                continue  # metrics below depend only on the predictions, taken from repeat 0

            num_events += len(indices)
            num_valid += sum(lengths)
            num_padded += X.shape[0] * X.shape[1]
            runs.append({"event_indices": indices, "padded_len": X.shape[1], "inference_ms": dt * 1e3})
            jets_pt.append(particles_to_jets(pred, mask))
            for b, (i, n) in enumerate(zip(indices, lengths)):
                event_pred = tuple(p[b] for p in pred)
                if baseline_preds is None:
                    per_event_preds[i] = event_pred
                else:
                    errors.add(event_pred, baseline_preds[i], n)
        repeat_times.append(repeat_time)
        if repeat_mem_max is not None:
            nvml_max_per_repeat.append(repeat_mem_max)

    result = {
        "timing": timing_summary(batch_times, repeat_times, num_events, num_valid, num_padded),
        "memory": {
            "torch_peak_allocated_mib": torch_peak_memory_mib(),
            "process_gpu_memory_max_mib_nvml": max(nvml_max_per_repeat) if nvml_max_per_repeat else None,
            "process_gpu_memory_max_mib_nvml_per_repeat": nvml_max_per_repeat,
        },
        "gpu_placement": gpu_info,
        "runs": runs,
        "jets_pt": np.concatenate(jets_pt) if jets_pt else np.array([]),
    }
    if baseline_preds is not None:
        result["error"] = errors.summary()
    else:
        result["per_event_preds"] = per_event_preds
    return result


# --------------------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------------------


def plot_jet_pt(results, configs, baseline, outdir):
    """Jet pT spectrum per scenario with a ratio panel vs. the baseline."""
    cmap = plt.get_cmap("tab10")
    bins = np.logspace(0, 2, 101)
    hists = {cfg: to_bh(results[cfg]["jets_pt"], bins) for cfg in configs}

    f, (a0, a1) = plt.subplots(2, 1, gridspec_kw={"height_ratios": [3, 1]}, sharex=True, figsize=(10, 10))
    for idx, cfg in enumerate(configs):
        mplhep.histplot(hists[cfg], label=cfg, lw=1.5, yerr=0, color=cmap(idx), ax=a0)
        mplhep.histplot(hists[cfg] / hists[baseline], lw=1.5, yerr=0, color=cmap(idx), ax=a1)
    a0.set_yscale("log")
    a0.set_xscale("log")
    a0.set_ylabel("Number of jets")
    a0.legend(fontsize=8)
    a1.set_ylim(0.5, 1.5)
    a1.set_ylabel(f"vs. {baseline}")
    a1.set_xlabel("jet $p_T$ [GeV]")
    plt.savefig(os.path.join(outdir, "jet_pt_distribution.pdf"), bbox_inches="tight")
    plt.close()


def print_report(results, configs, baseline, batch_size, num_repeats):
    def fmt_mem(v):
        return f"{v:9.0f}" if v is not None else f"{'n/a':>9s}"

    print(f"\nTimings are averaged over {num_repeats} repeat(s); +/- is the std across repeats.")
    if not NVML_AVAILABLE:
        print("pynvml not available: per-process GPU memory (which also covers ORT) is not reported; pip install nvidia-ml-py")
    header = (
        f"{'Scenario':28s} {'batch':>5s} {'lat/batch ms':>13s} {'lat/event ms':>13s} {'events/s':>18s} "
        f"{'pad eff':>8s} {'torch MiB':>9s} {'proc MiB':>9s} {'MAE vs baseline':>18s}"
    )
    print(header)
    print("-" * len(header))
    for cfg in configs:
        t = results[cfg]["timing"]
        m = results[cfg]["memory"]
        err = results[cfg].get("error")
        mae_str = "(baseline)" if err is None else f"{err['mae']:.3e}" + (f" inv={err['num_invalid']}" if err["num_invalid"] else "")
        bs = 1 if cfg == baseline else batch_size
        thr = f"{t['throughput_events_per_s']:.1f} +/- {t['throughput_events_per_s_std']:.1f}"
        print(
            f"{cfg:28s} {bs:5d} {t['latency_ms_per_batch']['mean']:13.3f} {t['latency_ms_per_event_mean']:13.3f} {thr:>18s} "
            f"{t['padding_efficiency']:8.2f} {fmt_mem(m['torch_peak_allocated_mib'])} {fmt_mem(m['process_gpu_memory_max_mib_nvml'])} {mae_str:>18s}"
        )
    print("\nGPU placement per scenario:")
    for cfg in configs:
        print(f"  {cfg:28s} {results[cfg]['gpu_placement']}")


def system_info(args):
    try:
        cpu = subprocess.check_output("lscpu | grep 'Model name'", shell=True).decode().split(":")[1].strip()
    except Exception:
        cpu = platform.processor() or "Unknown CPU"
    return {
        "cpu": cpu,
        "gpu": torch.cuda.get_device_name(0),
        "pytorch_version": torch.__version__,
        "onnxruntime_version": rt.__version__,
        "cuda_version": torch.version.cuda,
        "num_threads": args.num_threads,
        "torch_compile": args.compile,
        "nvml_available": NVML_AVAILABLE,
    }


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("This benchmark requires a CUDA device.")
    device = "cuda"

    torch.manual_seed(42)
    np.random.seed(42)
    torch.set_num_threads(args.num_threads)
    torch.set_float32_matmul_precision("high")  # TF32, as in the original script / training
    os.makedirs(args.outdir, exist_ok=True)
    mplhep.style.use("CMS")

    # ---- Model configuration and weights ---------------------------------------------
    base_config = load_model_config(args.model_kwargs)
    if base_config.model.type != ModelType.ATTENTION:
        raise SystemExit(f"This benchmark only supports attention models, got {base_config.model.type}")
    state_dict = torch.load(args.checkpoint, map_location="cpu", weights_only=True)["model_state_dict"]
    num_heads = base_config.model.attention.num_heads
    input_dim = base_config.input_dim

    pad_multiple = max(args.pad_bin_size, base_config.pad_to_multiple_elements or 0)
    configs = [BASELINE] + list(args.configs)

    # ---- ONNX export -----------------------------------------------------------------
    onnx_paths = export_onnx_models(configs, base_config, state_dict, num_heads, input_dim, pad_multiple, device, args.outdir)

    # ---- Data ------------------------------------------------------------------------
    print(f"Loading dataset {args.dataset} from {args.data_dir}")
    ds = tfds.builder(args.dataset, data_dir=args.data_dir).as_data_source(split="train")
    num_events = min(args.num_events, len(ds))
    event_sizes = [int(ds[i]["X"].shape[0]) for i in range(num_events)]
    # The baseline is unbatched (batch size 1) and always in dataset order; the batched
    # scenarios use --batch-size and, optionally, length-sorted order (bucketing).
    baseline_batches = list(make_batches(ds, list(range(num_events)), 1, pad_multiple, input_dim))
    batched_order = event_order(event_sizes, args.sort_by_length)
    batched_batches = list(make_batches(ds, batched_order, args.batch_size, pad_multiple, input_dim))
    print(f"{num_events} events, {len(batched_batches)} batches of size {args.batch_size}, " f"bucketing={'on' if args.sort_by_length else 'off'}")

    # ---- Baseline --------------------------------------------------------------------
    results = {}
    print(f"\nRunning baseline {BASELINE} (unbatched)")
    runner = make_runner(BASELINE, base_config, state_dict, onnx_paths, args, device)
    results[BASELINE] = run_scenario(BASELINE, runner, baseline_batches, args.num_warmup, args.num_repeats)
    baseline_preds = results[BASELINE].pop("per_event_preds")
    del runner
    gc.collect()
    torch.cuda.empty_cache()

    # ---- Batched scenarios -----------------------------------------------------------
    for cfg in args.configs:
        print(f"\nRunning {cfg} (batch size {args.batch_size})")
        runner = make_runner(cfg, base_config, state_dict, onnx_paths, args, device)
        results[cfg] = run_scenario(cfg, runner, batched_batches, args.num_warmup, args.num_repeats, baseline_preds)
        del runner
        gc.collect()
        torch.cuda.empty_cache()

    # ---- Report ----------------------------------------------------------------------
    plot_jet_pt(results, configs, BASELINE, args.outdir)
    print_report(results, configs, BASELINE, args.batch_size, args.num_repeats)

    base_jets = results[BASELINE]["jets_pt"]
    summary = {
        "args": vars(args),
        "baseline": BASELINE,
        "num_events": num_events,
        "event_sizes": event_sizes,
        "batched_event_order": batched_order,
        "pad_multiple": pad_multiple,
        "system": system_info(args),
        "scenarios": {
            cfg: {
                "batch_size": 1 if cfg == BASELINE else args.batch_size,
                "timing": results[cfg]["timing"],
                "memory": results[cfg]["memory"],
                "gpu_placement": results[cfg]["gpu_placement"],
                "error": results[cfg].get("error"),
                "jets": {
                    "num_jets": int(len(results[cfg]["jets_pt"])),
                    "num_jets_ratio_to_baseline": float(len(results[cfg]["jets_pt"]) / len(base_jets)) if len(base_jets) else None,
                    "mean_pt": float(np.mean(results[cfg]["jets_pt"])) if len(results[cfg]["jets_pt"]) else None,
                },
                "runs": results[cfg]["runs"],
            }
            for cfg in configs
        },
    }
    summary_path = os.path.join(args.outdir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)
    print(f"\nResults written to {summary_path}")


if __name__ == "__main__":
    main()