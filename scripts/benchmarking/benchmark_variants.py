#!/usr/bin/env python3
"""MLPF attention-only mixed-precision benchmark variant registry.

All inputs, weights, features, projections, FFNs and heads stay FP32. The chosen
external FlashAttention version casts only Q/K/V *at the packed kernel boundary*.
No autocast, model/inputs dtype overrides, SDPA Flash, NestedTensor, cuDNN or
SageAttention variants. This module remains importable without importing torch.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable

ModelTransform = Callable[[Any], Any]
InputTransform = Callable[[Any, Any], tuple[Any, Any]]
ForwardFunction = Callable[[Any, Any, Any], Any]


@dataclass(frozen=True)
class Precision:
    """TF32 controls FP32 linear algebra only; weights and inputs always FP32."""
    allow_tf32: bool = False


@dataclass(frozen=True)
class Attention:
    implementation: str = "SIMPLE"  # PyTorch reference / constructor configuration
    backend: str | None = None  # None for reference; fa2, fa3, fa4 for external kernels
    kernel_dtype: str = "float32"  # numerical format confined to the backbone kernel
    sdpa_backend: str | None = None  # 'efficient' only for the unchanged golden path
    jagged: bool = False
    varlen: bool = False
    # This explicit profiler annotation is emitted ONLY around the external
    # library call. Worker ALSO requires a nonzero CUDA kernel event.
    expected_ops: tuple[str, ...] = ()
    compiled_expected_ops: tuple[str, ...] | None = None


@dataclass(frozen=True)
class Variant:
    name: str
    attention: Attention
    precision: Precision = field(default_factory=Precision)
    config_overrides: dict[str, Any] = field(default_factory=dict)
    model_transform: ModelTransform | None = None
    input_transform: InputTransform | None = None
    forward_fn: ForwardFunction | None = None
    compile: bool = False
    compile_mode: str = "default"
    notes: str = ""

    def describe(self) -> dict[str, Any]:
        def function_name(fn):
            return None if fn is None else f"{fn.__module__}.{fn.__qualname__}"
        return {
            "name": self.name,
            "attention": {
                "implementation": self.attention.implementation,
                "backend": self.attention.backend,
                "kernel_dtype": self.attention.kernel_dtype,
                "sdpa_backend": self.attention.sdpa_backend,
                "jagged": self.attention.jagged,
                "varlen": self.attention.varlen,
                "expected_ops": list(self.attention.expected_ops),
                "compiled_expected_ops": (
                    list(self.attention.compiled_expected_ops)
                    if self.attention.compiled_expected_ops is not None else None
                ),
            },
            "precision": {"allow_tf32": self.precision.allow_tf32},
            "model_dtype": "float32",
            "input_dtype": "float32",
            "config_overrides": self.config_overrides,
            "model_transform": function_name(self.model_transform),
            "input_transform": function_name(self.input_transform),
            "forward_fn": function_name(self.forward_fn),
            "compile": self.compile,
            "compile_mode": self.compile_mode,
            "notes": self.notes,
        }


REFERENCE_NAME = "fp32_efficient"
_REFERENCE = Variant(
    "fp32_efficient",
    Attention(implementation="SIMPLE", sdpa_backend="efficient",
              expected_ops=(r"_efficient_attention_forward",)),
    notes="Uncompiled, unchanged FP32 dense padded/masked PyTorch memory-efficient SDPA reference",
)


def _flash_variant(generation: int, dtype_label: str, dtype: str) -> Variant:
    backend = f"fa{generation}"
    return Variant(
        name=f"flash_attention_{generation}_{dtype_label}",
        attention=Attention(implementation="SIMPLE", backend=backend,
                            kernel_dtype=dtype, jagged=True, varlen=True,
                            expected_ops=(f"mlpf_external_{backend}_native_varlen",)),
        notes=(f"External FA{generation} native packed-varlen {dtype_label.upper()} kernel only; "
               "full model and all other operations FP32; no backend substitutions")
    )


_FLASH = (
    _flash_variant(2, "fp16", "float16"),
    _flash_variant(2, "bf16", "bfloat16"),
    _flash_variant(3, "fp16", "float16"),
    _flash_variant(3, "bf16", "bfloat16"),
    _flash_variant(4, "fp16", "float16"),
    _flash_variant(4, "bf16", "bfloat16"),
)
VARIANTS: tuple[Variant, ...] = (
    _REFERENCE,
    *_FLASH,
    *(replace(v, name=v.name + "_compiled", compile=True,
               notes=v.notes + "; torch.compile with documented external-call graph break")
      for v in _FLASH),
)
REGISTRY = {v.name: v for v in VARIANTS}
if len(REGISTRY) != len(VARIANTS):
    raise ValueError("Variant names must be unique")


def get_variant(name: str) -> Variant:
    try:
        return REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"Unknown variant {name!r}; choices: {list(REGISTRY)}") from exc
