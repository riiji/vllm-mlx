# SPDX-License-Identifier: Apache-2.0
"""
Qwen2 / Qwen2.5 implementation with fused QKV and Gate-Up projections.

Projections for Query, Key, and Value are fused into a single matrix multiplication,
and Gate and Up projections in the MLP are fused into a single matrix multiplication,
reducing kernel dispatches during inference.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import BaseModelArgs
from mlx_lm.models.qwen2 import (
    Model as UpstreamQwen2Model,
    create_attention_mask,
    initialize_rope,
    scaled_dot_product_attention,
    swiglu,
)


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    rms_norm_eps: float
    vocab_size: int
    num_key_value_heads: int
    max_position_embeddings: int = 32768
    rope_theta: float = 1000000.0
    rope_traditional: bool = False
    rope_scaling: Optional[dict[str, Union[float, str]]] = None
    tie_word_embeddings: bool = True


class FusedQwen2Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        dim = args.hidden_size
        self.n_heads = args.num_attention_heads
        assert args.num_key_value_heads is not None
        self.n_kv_heads = args.num_key_value_heads

        head_dim = args.hidden_size // self.n_heads
        self.head_dim = head_dim
        self.scale = head_dim**-0.5

        self.q_dim = self.n_heads * head_dim
        self.kv_dim = self.n_kv_heads * head_dim
        self.split_indices = [self.q_dim, self.q_dim + self.kv_dim]

        # Unified QKV projection (1 GEMM instead of 3)
        self.qkv_proj = nn.Linear(dim, self.q_dim + 2 * self.kv_dim, bias=True)
        self.o_proj = nn.Linear(self.n_heads * head_dim, dim, bias=False)

        self.rope = initialize_rope(
            head_dim,
            base=args.rope_theta,
            traditional=args.rope_traditional,
            scaling_config=args.rope_scaling,
            max_position_embeddings=args.max_position_embeddings,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, _ = x.shape
        qkv = self.qkv_proj(x)
        queries, keys, values = mx.split(qkv, self.split_indices, axis=-1)

        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class FusedQwen2MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        # Unified Gate-Up projection (1 GEMM instead of 2)
        self.gate_up_proj = nn.Linear(dim, 2 * hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        gu = self.gate_up_proj(x)
        gate, up = mx.split(gu, 2, axis=-1)
        return self.down_proj(swiglu(gate, up))


class FusedTransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_attention_heads = args.num_attention_heads
        self.hidden_size = args.hidden_size
        self.self_attn = FusedQwen2Attention(args)
        self.mlp = FusedQwen2MLP(args.hidden_size, args.intermediate_size)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )
        self.args = args

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        r = self.self_attn(self.input_layernorm(x), mask=mask, cache=cache)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r


class FusedQwen2Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers
        assert self.vocab_size > 0
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            FusedTransformerBlock(args=args) for _ in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
        input_embeddings: Optional[mx.array] = None,
    ) -> mx.array:
        if input_embeddings is not None:
            h = input_embeddings
        else:
            h = self.embed_tokens(inputs)

        mask = None
        if h.shape[1] > 1:
            mask = create_attention_mask(h, cache)

        if cache is None:
            cache = [None] * len(self.layers)

        for layer, c in zip(self.layers, cache):
            h = layer(h, mask=mask, cache=c)

        return self.norm(h)


class Qwen2Model(nn.Module):
    """
    Qwen2 causal language model with fused QKV and Gate-Up projections.
    Compatible with mlx_lm generation, KVCache, and engine batch generators.
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = FusedQwen2Model(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    @property
    def layers(self):
        """Expose layers property for cache construction and batch generators."""
        return self.model.layers

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
        input_embeddings: Optional[mx.array] = None,
    ) -> mx.array:
        out = self.model(inputs, cache=cache, input_embeddings=input_embeddings)
        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
        return out

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        """
        Transform checkpoint weights into fused QKV and Gate-Up representations.
        Supports both floating-point and quantized (scales/biases) formats.
        """
        weights = UpstreamQwen2Model.sanitize(self, weights)
        num_layers = getattr(self.args, "num_hidden_layers", 0)
        return self._fuse_weights(weights, num_layers=num_layers)

    @classmethod
    def _fuse_weights(
        cls, weights: dict[str, mx.array], num_layers: int
    ) -> dict[str, mx.array]:
        """Fuse separate Q, K, V and Gate, Up projections into unified tensors."""
        for i in range(num_layers):
            p_attn = f"model.layers.{i}.self_attn"
            for suffix in ("weight", "bias", "scales", "biases"):
                q = f"{p_attn}.q_proj.{suffix}"
                k = f"{p_attn}.k_proj.{suffix}"
                v = f"{p_attn}.v_proj.{suffix}"
                if q in weights and k in weights and v in weights:
                    weights[f"{p_attn}.qkv_proj.{suffix}"] = mx.concatenate(
                        [weights.pop(q), weights.pop(k), weights.pop(v)], axis=0
                    )

            p_mlp = f"model.layers.{i}.mlp"
            for suffix in ("weight", "bias", "scales", "biases"):
                g = f"{p_mlp}.gate_proj.{suffix}"
                u = f"{p_mlp}.up_proj.{suffix}"
                if g in weights and u in weights:
                    weights[f"{p_mlp}.gate_up_proj.{suffix}"] = mx.concatenate(
                        [weights.pop(g), weights.pop(u)], axis=0
                    )

        return weights

    @classmethod
    def sanitize_weights(
        cls, weights: dict[str, mx.array], num_layers: int = 1
    ) -> dict[str, mx.array]:
        """Transform checkpoint weights for testing and offline conversion."""
        return cls._fuse_weights(weights, num_layers=num_layers)

    @classmethod
    def from_pretrained(
        cls,
        model_path: Union[str, Path],
        lazy: bool = False,
        strict: bool = True,
    ) -> "Qwen2Model":
        """Load and initialize a model from a local directory or HF snapshot."""
        from mlx_lm.utils import load_model

        model, _ = load_model(
            Path(model_path),
            lazy=lazy,
            strict=strict,
            get_model_classes=lambda *args, **kwargs: (cls, ModelArgs),
        )
        return model
