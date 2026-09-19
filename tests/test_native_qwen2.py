import os
from pathlib import Path
import mlx.core as mx
import pytest

from vllm_mlx.native_models import get_native_model_class, has_native_model
from vllm_mlx.native_models.qwen2 import (
    FusedQwen2Attention,
    FusedQwen2MLP,
    ModelArgs,
    Qwen2Model,
)
from vllm_mlx.utils.tokenizer import load_model_with_fallback


def test_native_model_registry():
    """Verify registry lookup for native models."""
    assert has_native_model("qwen2") is True
    assert has_native_model("Qwen2") is True
    assert has_native_model("nonexistent_arch") is False

    model_cls, args_cls = get_native_model_class("qwen2")
    assert model_cls is Qwen2Model
    assert args_cls is ModelArgs

    assert get_native_model_class("unknown") is None


def test_native_models_no_collisions():
    """Verify registry builds cleanly without duplicate architecture keys."""
    from vllm_mlx.native_models import build_native_model_registry

    registry = build_native_model_registry()
    assert "qwen2" in registry
    assert "qwen2.5" in registry

    # Verify duplicate detection triggers ValueError when two modules register the same arch
    from unittest.mock import MagicMock, patch

    mock_mod = MagicMock()
    mock_mod.name = "duplicate_qwen"

    mock_imported = MagicMock()
    mock_imported.SUPPORTED_ARCHITECTURES = ("qwen2",)
    mock_imported.Model = Qwen2Model
    mock_imported.ModelArgs = ModelArgs

    with patch("pkgutil.iter_modules") as mock_iter:
        qwen_mod_info = MagicMock()
        qwen_mod_info.name = "qwen2"
        mock_iter.return_value = [qwen_mod_info, mock_mod]

        with patch("importlib.import_module") as mock_import:
            from vllm_mlx.native_models import qwen2

            def side_effect(name):
                if "duplicate" in name:
                    return mock_imported
                return qwen2

            mock_import.side_effect = side_effect
            with pytest.raises(ValueError, match="Architecture collision"):
                build_native_model_registry()


def test_fused_qwen2_synthetic_forward():
    """Verify forward pass of Qwen2Model with synthetic weights."""
    args = ModelArgs(
        model_type="qwen2",
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=128,
        num_hidden_layers=2,
        vocab_size=100,
        rms_norm_eps=1e-6,
    )
    model = Qwen2Model(args)

    assert hasattr(model, "layers")
    assert len(model.layers) == 2
    assert isinstance(model.layers[0].self_attn, FusedQwen2Attention)
    assert isinstance(model.layers[0].mlp, FusedQwen2MLP)

    tokens = mx.array([[1, 5, 20, 42]], dtype=mx.int32)
    out = model(tokens)
    assert out.shape == (1, 4, 100)


def test_fused_qwen2_sanitize_weights():
    """Verify sanitize_weights fuses separate Q, K, V and Gate, Up weights correctly."""
    q = mx.ones((64, 64))
    k = mx.full((32, 64), 2.0)
    v = mx.full((32, 64), 3.0)
    gate = mx.full((128, 64), 4.0)
    up = mx.full((128, 64), 5.0)

    raw_weights = {
        "model.layers.0.self_attn.q_proj.weight": q,
        "model.layers.0.self_attn.k_proj.weight": k,
        "model.layers.0.self_attn.v_proj.weight": v,
        "model.layers.0.mlp.gate_proj.weight": gate,
        "model.layers.0.mlp.up_proj.weight": up,
    }

    sanitized = Qwen2Model.sanitize_weights(raw_weights)

    assert "model.layers.0.self_attn.qkv_proj.weight" in sanitized
    assert "model.layers.0.mlp.gate_up_proj.weight" in sanitized
    assert "model.layers.0.self_attn.q_proj.weight" not in sanitized
    assert "model.layers.0.mlp.gate_proj.weight" not in sanitized

    # Check concatenated values
    qkv = sanitized["model.layers.0.self_attn.qkv_proj.weight"]
    assert qkv.shape == (128, 64)
    assert mx.all(qkv[:64] == 1.0)
    assert mx.all(qkv[64:96] == 2.0)
    assert mx.all(qkv[96:] == 3.0)

    gate_up = sanitized["model.layers.0.mlp.gate_up_proj.weight"]
    assert gate_up.shape == (256, 64)
    assert mx.all(gate_up[:128] == 4.0)
    assert mx.all(gate_up[128:] == 5.0)


def test_fused_qwen2_sanitize_quantized_weights():
    """Verify sanitize_weights fuses quantized scales and biases."""
    raw_weights = {
        "model.layers.0.self_attn.q_proj.weight": mx.zeros((64, 8), dtype=mx.uint32),
        "model.layers.0.self_attn.q_proj.scales": mx.ones((64, 2), dtype=mx.float16),
        "model.layers.0.self_attn.q_proj.biases": mx.zeros((64, 2), dtype=mx.float16),
        "model.layers.0.self_attn.k_proj.weight": mx.zeros((32, 8), dtype=mx.uint32),
        "model.layers.0.self_attn.k_proj.scales": mx.full((32, 2), 2.0, dtype=mx.float16),
        "model.layers.0.self_attn.k_proj.biases": mx.zeros((32, 2), dtype=mx.float16),
        "model.layers.0.self_attn.v_proj.weight": mx.zeros((32, 8), dtype=mx.uint32),
        "model.layers.0.self_attn.v_proj.scales": mx.full((32, 2), 3.0, dtype=mx.float16),
        "model.layers.0.self_attn.v_proj.biases": mx.zeros((32, 2), dtype=mx.float16),
    }
    sanitized = Qwen2Model.sanitize_weights(raw_weights)

    assert "model.layers.0.self_attn.qkv_proj.weight" in sanitized
    assert "model.layers.0.self_attn.qkv_proj.scales" in sanitized
    assert "model.layers.0.self_attn.qkv_proj.biases" in sanitized
    assert sanitized["model.layers.0.self_attn.qkv_proj.weight"].shape == (128, 8)
    assert sanitized["model.layers.0.self_attn.qkv_proj.scales"].shape == (128, 2)


LOCAL_QWEN_PATH = Path(
    os.path.expanduser(
        "~/.cache/huggingface/hub/models--mlx-community--Qwen2.5-0.5B-Instruct-4bit/snapshots/a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3"
    )
)


@pytest.mark.skipif(
    not LOCAL_QWEN_PATH.exists(),
    reason="Local Qwen 2.5 0.5B model weights not cached",
)
def test_qwen2_numerical_parity():
    """Verify numerical parity between standard mlx_lm model and native fused model."""
    orig_model, tokenizer = load_model_with_fallback(
        str(LOCAL_QWEN_PATH), enable_native_models=False
    )
    native_model, _ = load_model_with_fallback(
        str(LOCAL_QWEN_PATH), enable_native_models=True
    )

    assert isinstance(native_model, Qwen2Model)
    assert not isinstance(orig_model, Qwen2Model)

    prompt = "Apple Silicon MLX is"
    input_ids = mx.array(tokenizer.encode(prompt))[None]

    out_orig = orig_model(input_ids)
    out_native = native_model(input_ids)

    diff = mx.max(mx.abs(out_orig - out_native)).item()
    assert diff < 1e-4, f"Max difference {diff} exceeds tolerance 1e-4"

    # Verify greedy token prediction matches
    assert int(mx.argmax(out_orig[:, -1, :], axis=-1)[0]) == int(
        mx.argmax(out_native[:, -1, :], axis=-1)[0]
    )


@pytest.mark.skipif(
    not LOCAL_QWEN_PATH.exists(),
    reason="Local Qwen 2.5 0.5B model weights not cached",
)
def test_qwen2_batch_generator_compatibility():
    """Verify native model compatibility with mlx_lm BatchGenerator."""
    from mlx_lm.generate import BatchGenerator

    native_model, tokenizer = load_model_with_fallback(
        str(LOCAL_QWEN_PATH), enable_native_models=True
    )

    bg = BatchGenerator(native_model)
    tokens = tokenizer.encode("Hello, world!")
    bg.insert([tokens])

    # Advance steps until at least 3 tokens are generated
    generated = []
    for _ in range(20):
        prompt_resps, gen_resps = bg.next()
        for r in gen_resps:
            generated.append(r.token)
        if len(generated) >= 3:
            break

    assert len(generated) >= 3


def test_cli_enable_native_models_flag():
    """Verify CLI parser options for --enable-native-models."""
    from vllm_mlx.cli import create_parser

    parser = create_parser()
    args = parser.parse_args(["serve", "dummy-model", "--enable-native-models"])
    assert args.enable_native_models is True

    default_args = parser.parse_args(["serve", "dummy-model"])
    assert default_args.enable_native_models is False

    bench_args = parser.parse_args(["bench", "dummy-model", "--enable-native-models"])
    assert bench_args.enable_native_models is True


@pytest.mark.skipif(
    not LOCAL_QWEN_PATH.exists(),
    reason="Local Qwen 2.5 0.5B model weights not cached",
)
def test_simple_engine_loads_native_model():
    """Verify SimpleEngine loads Qwen2Model when enable_native_models=True."""
    from vllm_mlx.engine.simple import SimpleEngine

    engine = SimpleEngine(str(LOCAL_QWEN_PATH), enable_native_models=True)
    engine.prepare_for_start()
    assert isinstance(engine._model.model, Qwen2Model)


@pytest.mark.skipif(
    not LOCAL_QWEN_PATH.exists(),
    reason="Local Qwen 2.5 0.5B model weights not cached",
)
def test_batched_engine_loads_native_model():
    """Verify BatchedEngine loads Qwen2Model when enable_native_models=True."""
    from vllm_mlx.engine.batched import BatchedEngine

    engine = BatchedEngine(str(LOCAL_QWEN_PATH), enable_native_models=True)
    engine._prepare_llm_model()
    assert isinstance(engine._model, Qwen2Model)
