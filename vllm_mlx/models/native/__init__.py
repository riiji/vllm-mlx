# SPDX-License-Identifier: Apache-2.0
"""
Native model implementations for vllm-mlx.

These models implement fused kernel operations (such as fused QKV and fused Gate-Up projections)
natively without runtime monkey-patching, maximizing inference efficiency on Apple Silicon.
"""

from typing import Any

_NATIVE_MODEL_REGISTRY: dict[str, tuple[str, str, str]] = {
    "qwen2": ("vllm_mlx.models.native.qwen2", "NativeQwen2ForCausalLM", "ModelArgs"),
    "qwen2.5": ("vllm_mlx.models.native.qwen2", "NativeQwen2ForCausalLM", "ModelArgs"),
}


def has_native_model(model_type: str) -> bool:
    """Check if a native model implementation is available for the given model_type."""
    return model_type.lower() in _NATIVE_MODEL_REGISTRY


def get_native_model_class(model_type: str) -> tuple[Any, Any] | None:
    """
    Get the (model_class, model_args_class) tuple for the given model_type.

    Returns:
        (model_class, model_args_class) if registered, else None.
    """
    key = model_type.lower()
    if key not in _NATIVE_MODEL_REGISTRY:
        return None

    module_name, model_cls_name, args_cls_name = _NATIVE_MODEL_REGISTRY[key]
    import importlib

    mod = importlib.import_module(module_name)
    return getattr(mod, model_cls_name), getattr(mod, args_cls_name)
