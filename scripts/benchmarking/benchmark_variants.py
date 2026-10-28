#!/usr/bin/env python3
"""MLPF benchmark variant registry (no benchmark machinery here).

Add one Variant to VARIANTS to make it available via --variants or --reference.
All custom functions live here and run INSIDE a newly spawned worker process.

Precision is *native* throughout: no torch.autocast and no autocast variants.
For mixed/module-wise precision, supply a model_transform that installs the
explicit dtype conversions required by your architecture. Merely converting
individual module weights is not sufficient when skip connections cross dtypes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

# No torch import in this module: the orchestrator stays CUDA-framework-free.
# Optional transform callbacks execute only in workers (where torch is loaded).
ModelTransform = Callable[[Any], Any]
InputTransform = Callable[[Any, Any], tuple[Any, Any]]
ForwardFunction = Callable[[Any, Any, Any], Any]


@dataclass(frozen=True)
class Precision:
    # For mixed precision, set the base dtype then use model_transform / input_transform.
    weights: str = "float32"
    inputs: str | None = None  # Defaults to weights
    allow_tf32: bool = False

    @property
    def input_dtype(self) -> str:
        return self.inputs if self.inputs is not None else self.weights


@dataclass(frozen=True)
class Attention:
    # MLPF AttentionType enum name: MATH, FLASH, SIMPLE, or a future/custom name.
    implementation: str
    # Forced PyTorch SDPA backend name. None leaves backend selection to MLPF.
    sdpa_backend: str | None = None  # math, flash, efficient, cudnn (PyTorch-dependent)
    jagged: bool = False
    varlen: bool = False
    # At least one of these regexes must occur in eager profiler op names.
    # Empty disables dispatch validation (useful for entirely custom kernels).
    expected_ops: tuple[str, ...] = ()
    # After torch.compile, None uses eager patterns. Use () to disable compiled
    # operator-name checks when fusion deliberately hides the original operator.
    compiled_expected_ops: tuple[str, ...] | None = None


@dataclass(frozen=True)
class ModulePrecisionRule:
    # fnmatch-style named_modules() path, e.g. "encoder.layers.0" or "*norm*".
    module_glob: str
    dtype: str
    # Explicit, native boundary casts (NO autocast). The wrapper casts floating
    # inputs into dtype and usually returns floating outputs to the input dtype.
    restore_output_dtype: bool = True


@dataclass(frozen=True)
class Variant:
    name: str
    attention: Attention
    precision: Precision = field(default_factory=Precision)
    # Override arbitrary Pydantic model config attributes, e.g.:
    # {"model.attention.some_new_option": True}
    config_overrides: dict[str, Any] = field(default_factory=dict)
    # Convenient native mixed precision: explicit casts around selected modules.
    module_precision_rules: tuple[ModulePrecisionRule, ...] = ()
    model_transform: ModelTransform | None = None
    input_transform: InputTransform | None = None
    forward_fn: ForwardFunction | None = None
    compile: bool = False
    compile_mode: str = "default"
    notes: str = ""

    def describe(self) -> dict[str, Any]:
        """Serializable specification for reproducibility in summary.json."""
        def function_name(fn: Callable[..., Any] | None) -> str | None:
            return None if fn is None else f"{fn.__module__}.{fn.__qualname__}"

        return {
            "name": self.name,
            "attention": {
                "implementation": self.attention.implementation,
                "sdpa_backend": self.attention.sdpa_backend,
                "jagged": self.attention.jagged,
                "varlen": self.attention.varlen,
                "expected_ops": list(self.attention.expected_ops),
                "compiled_expected_ops": (list(self.attention.compiled_expected_ops)
                                          if self.attention.compiled_expected_ops is not None else None),
            },
            "precision": {
                "weights": self.precision.weights,
                "inputs": self.precision.input_dtype,
                "allow_tf32": self.precision.allow_tf32,
            },
            "config_overrides": self.config_overrides,
            "module_precision_rules": [
                {"module_glob": r.module_glob, "dtype": r.dtype,
                 "restore_output_dtype": r.restore_output_dtype}
                for r in self.module_precision_rules
            ],
            "model_transform": function_name(self.model_transform),
            "input_transform": function_name(self.input_transform),
            "forward_fn": function_name(self.forward_fn),
            "compile": self.compile,
            "compile_mode": self.compile_mode,
            "notes": self.notes,
        }


# Accuracy reference is run ONCE at batch size 1, not included in timing plots.
# Keep fp32_efficient as the default to preserve the original script's naming.
# Select --reference fp32_math to compare against strict FP32 math SDPA instead;
# quadratic attention can require dramatically more GPU memory for long events.
REFERENCE_NAME = "fp32_efficient"

VARIANTS: tuple[Variant, ...] = (
    Variant(
        "fp32_efficient",
        Attention("SIMPLE", sdpa_backend="efficient", expected_ops=(r"_efficient_attention_forward",)),
        Precision("float32"),
        notes="FP32 memory-efficient PyTorch SDPA; validated without math fallback",
    ),
    Variant(
        "fp32_math",
        Attention("MATH", sdpa_backend="math", expected_ops=(r"_scaled_dot_product_attention_math",)),
        Precision("float32"),
        notes="Optional strict FP32 mathematical SDPA reference; can OOM at long sequence lengths",
    ),
    Variant(
        "fp16_flash",
        Attention("FLASH", sdpa_backend="flash", jagged=True, varlen=True,
                  expected_ops=(r"flash_attn_varlen",)),
        Precision("float16"),
        notes="Native FP16 weights/inputs; require actual packed flash-attn varlen op",
    ),
    Variant(
        "bf16_flash",
        Attention("FLASH", sdpa_backend="flash", jagged=True, varlen=True,
                  expected_ops=(r"flash_attn_varlen",)),
        Precision("bfloat16"),
        notes="Native BF16 weights/inputs; require actual packed flash-attn varlen op",
    ),
    Variant(
        "fp16_flash_compiled",
        Attention("FLASH", sdpa_backend="flash", jagged=True, varlen=True,
                  expected_ops=(r"flash_attn_varlen",)),
        Precision("float16"),
        compile=True,
        notes="Same FP16 varlen FlashAttention as fp16_flash, with torch.compile enabled",
    ),
    Variant(
        "bf16_flash_compiled",
        Attention("FLASH", sdpa_backend="flash", jagged=True, varlen=True,
                  expected_ops=(r"flash_attn_varlen",)),
        Precision("bfloat16"),
        compile=True,
        notes="Same BF16 varlen FlashAttention as bf16_flash, with torch.compile enabled",
    ),
    Variant(
        "fp32_efficient_tf32",
        Attention("SIMPLE", sdpa_backend="efficient",
                  expected_ops=(r"_efficient_attention_forward",)),
        Precision("float32", allow_tf32=True),
        notes="Explicit opt-in TF32 matmul; NOT a strict FP32 accuracy reference",
    ),
    Variant(
        "fp16_cudnn",
        Attention(
            "SIMPLE",
            sdpa_backend="cudnn",
            expected_ops=(r"cudnn.*attention|cudnn.*sdpa",),
        ),
        Precision("float16"),
        notes="Native FP16 cuDNN fused SDPA",
    ),

    Variant(
        "bf16_cudnn",
        Attention(
            "SIMPLE",
            sdpa_backend="cudnn",
            expected_ops=(r"cudnn.*attention|cudnn.*sdpa",),
        ),
        Precision("bfloat16"),
        notes="Native BF16 cuDNN fused SDPA",
    ),
)

# Example extension (NOT activated):
#
# def customize_mixed_precision(model):
#     # Install explicit casts at the boundaries of your architecture's modules,
#     # or wrap a submodule with a typed forward. Return the modified model.
#     # There is no implicit autocast; dtype edges are your responsibility.
#     return model
#
# VARIANTS += (
#     Variant(
#         "mixed_my_kernel",
#         Attention("SIMPLE", sdpa_backend=None, expected_ops=(r"my_cuda_kernel",)),
#         Precision(weights="bfloat16", inputs="bfloat16"),
#         config_overrides={"model.attention.some_kernel_switch": True},
#         module_precision_rules=(ModulePrecisionRule("my_submodule", "float32"),),
#         model_transform=customize_mixed_precision,
#         compile=True,
#         notes="Example of an architecture-specific mixed-precision variant",
#     ),
# )

REGISTRY = {v.name: v for v in VARIANTS}
if len(REGISTRY) != len(VARIANTS):
    raise ValueError("Variant names must be unique")


def get_variant(name: str) -> Variant:
    try:
        return REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"Unknown variant {name!r}; choices: {list(REGISTRY)}") from exc


