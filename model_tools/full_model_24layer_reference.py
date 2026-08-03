#!/usr/bin/env python3
"""阶段 H3：真实 Qwen2.5-0.5B 24 层单 token 连贯软件金标准。

该模块只负责 Transformer layer0..23，不包含 Embedding、最终 RMSNorm、LM Head、
logits 或采样。第一条冻结基线使用 position=0/count=1：

initial hidden Q6.10
→ layer0 完整 G2 等价 Block
→ layer1 ...
→ layer23 output Q6.10

每层的 input RMSNorm、Q/K/V、O_proj、post RMSNorm、gate/up/down 参数均从
真实 P50 的对应 ``model.layers.N`` 张量读取。所有数值运算复用已经验证的
RMSNorm、Linear、RoPE、Attention、Softmax、SiLU 和残差函数，不另定义近似。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

try:
    from .attention_oproj_reference import (
        case_from_attention_q28,
        load_oproj_model,
    )
    from .attention_output_reference import attention_output_q28, flatten_attention_heads
    from .attention_residual_reference import (
        DEFAULT_HIDDEN_SEED_BASE,
        DEFAULT_IMAGE,
        AttentionSublayerContext,
        attention_residual_q10,
    )
    from .attention_score_reference import attention_scores_q28
    from .full_model_layer_sequence import load_reference as load_layer_sequence_reference
    from .mlp_down_proj_reference import case_from_source_q28, load_down_projection_model
    from .mlp_gate_up_reference import (
        case_from_post_attention_q10,
        load_projection_model as load_mlp_projection_model,
    )
    from .mlp_residual_reference import mlp_residual_q10
    from .mlp_silu_reference import case_from_gate_q28
    from .mlp_silu_up_mul_reference import case_from_inputs
    from .p50_format import P50Image
    from .post_attention_layernorm_reference import case_from_input_q10, load_gamma
    from .qkv_linear_reference import (
        HEAD_DIM,
        KV_HEADS,
        Q_HEADS,
        ProjectionSpec,
        case_from_model,
        load_projection_model as load_qkv_projection_model,
        reshape_heads,
    )
    from .rmsnorm_fixed_reference import (
        DEFAULT_EPSILON,
        compute_rmsnorm_reference,
        make_deterministic_input,
    )
    from .rope_fixed_reference import apply_rope_fixed_q28, generate_trig_row
    from .softmax_fixed_reference import softmax_scores_q31
    from .transformer_block_reference import (
        HIDDEN_SIZE,
        BlockContext,
        TransformerBlockCase,
        sha256_array,
    )
except ImportError:
    from attention_oproj_reference import case_from_attention_q28, load_oproj_model
    from attention_output_reference import attention_output_q28, flatten_attention_heads
    from attention_residual_reference import (
        DEFAULT_HIDDEN_SEED_BASE,
        DEFAULT_IMAGE,
        AttentionSublayerContext,
        attention_residual_q10,
    )
    from attention_score_reference import attention_scores_q28
    from full_model_layer_sequence import load_reference as load_layer_sequence_reference
    from mlp_down_proj_reference import case_from_source_q28, load_down_projection_model
    from mlp_gate_up_reference import (
        case_from_post_attention_q10,
        load_projection_model as load_mlp_projection_model,
    )
    from mlp_residual_reference import mlp_residual_q10
    from mlp_silu_reference import case_from_gate_q28
    from mlp_silu_up_mul_reference import case_from_inputs
    from p50_format import P50Image
    from post_attention_layernorm_reference import case_from_input_q10, load_gamma
    from qkv_linear_reference import (
        HEAD_DIM,
        KV_HEADS,
        Q_HEADS,
        ProjectionSpec,
        case_from_model,
        load_projection_model as load_qkv_projection_model,
        reshape_heads,
    )
    from rmsnorm_fixed_reference import (
        DEFAULT_EPSILON,
        compute_rmsnorm_reference,
        make_deterministic_input,
    )
    from rope_fixed_reference import apply_rope_fixed_q28, generate_trig_row
    from softmax_fixed_reference import softmax_scores_q31
    from transformer_block_reference import (
        HIDDEN_SIZE,
        BlockContext,
        TransformerBlockCase,
        sha256_array,
    )

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

DEFAULT_REFERENCE = Path(__file__).with_name("full_model_24layer_reference.json")
SCHEMA = "pangu50k.full_model_24layer_reference.v1"
ACTIVE_LAYER_COUNT = 24
Q10_FACTOR = 1 << 10
POSITION = 0
COUNT = 1


class FullModel24LayerReferenceError(ValueError):
    """表示真实 24 层参数、输入、定点结果或冻结清单不合法。"""


@dataclass(frozen=True)
class LayerContext:
    layer_index: int
    input_gamma_name: str
    post_gamma_name: str
    block: BlockContext


@dataclass(frozen=True)
class LayerSummary:
    layer_index: int
    label: str
    input_sha256: str
    output_sha256: str
    tensor_sha256: dict[str, str]
    saturation_counts: dict[str, int]
    parameter_upload_set_sha256: str


@dataclass(frozen=True)
class FullModelSequence:
    initial_hidden_q10: np.ndarray
    final_hidden_q10: np.ndarray
    layers: tuple[LayerSummary, ...]
    elapsed_seconds: float


def validate_layer_index(layer_index: int) -> int:
    resolved = int(layer_index)
    if not 0 <= resolved < ACTIVE_LAYER_COUNT:
        raise FullModel24LayerReferenceError("真实模型 layer 必须为 0..23")
    return resolved


def _layer_tensor(layer_index: int, suffix: str) -> str:
    return f"model.layers.{validate_layer_index(layer_index)}.{suffix}"


def layer_projection_specs(layer_index: int) -> dict[str, ProjectionSpec]:
    layer = validate_layer_index(layer_index)
    return {
        "q": ProjectionSpec(
            key="q",
            selector=0,
            command=b"Q",
            rows=Q_HEADS * HEAD_DIM,
            heads=Q_HEADS,
            weight_name=_layer_tensor(layer, "self_attn.q_proj.weight"),
            bias_name=_layer_tensor(layer, "self_attn.q_proj.bias"),
        ),
        "k": ProjectionSpec(
            key="k",
            selector=1,
            command=b"K",
            rows=KV_HEADS * HEAD_DIM,
            heads=KV_HEADS,
            weight_name=_layer_tensor(layer, "self_attn.k_proj.weight"),
            bias_name=_layer_tensor(layer, "self_attn.k_proj.bias"),
        ),
        "v": ProjectionSpec(
            key="v",
            selector=2,
            command=b"V",
            rows=KV_HEADS * HEAD_DIM,
            heads=KV_HEADS,
            weight_name=_layer_tensor(layer, "self_attn.v_proj.weight"),
            bias_name=_layer_tensor(layer, "self_attn.v_proj.bias"),
        ),
    }


def load_layer_context(image: P50Image, layer_index: int) -> LayerContext:
    layer = validate_layer_index(layer_index)
    input_gamma_name = _layer_tensor(layer, "input_layernorm.weight")
    post_gamma_name = _layer_tensor(layer, "post_attention_layernorm.weight")
    input_gamma = image.read_float16_tensor(input_gamma_name).astype(np.float32).reshape(-1)
    if input_gamma.shape != (HIDDEN_SIZE,):
        raise FullModel24LayerReferenceError(
            f"layer{layer} input RMS gamma shape={input_gamma.shape}"
        )
    specs = layer_projection_specs(layer)
    qkv_models = {
        key: load_qkv_projection_model(image, spec)
        for key, spec in specs.items()
    }
    oproj_name = _layer_tensor(layer, "self_attn.o_proj.weight")
    attention = AttentionSublayerContext(
        image=image,
        gamma=input_gamma,
        qkv_models=qkv_models,
        oproj_model=load_oproj_model(image, oproj_name),
    )
    post_gamma = load_gamma(image, post_gamma_name)
    gate_name = _layer_tensor(layer, "mlp.gate_proj.weight")
    up_name = _layer_tensor(layer, "mlp.up_proj.weight")
    down_name = _layer_tensor(layer, "mlp.down_proj.weight")
    names = set(image.tensor_names())
    for forbidden in (
        _layer_tensor(layer, "mlp.gate_proj.bias"),
        _layer_tensor(layer, "mlp.up_proj.bias"),
        _layer_tensor(layer, "mlp.down_proj.bias"),
    ):
        if forbidden in names:
            raise FullModel24LayerReferenceError(
                f"当前 Qwen2 MLP 要求无 bias，但存在 {forbidden}"
            )
    return LayerContext(
        layer_index=layer,
        input_gamma_name=input_gamma_name,
        post_gamma_name=post_gamma_name,
        block=BlockContext(
            attention=attention,
            post_attention_gamma=post_gamma,
            gate_model=load_mlp_projection_model(image, gate_name),
            up_model=load_mlp_projection_model(image, up_name),
            down_model=load_down_projection_model(image, down_name),
        ),
    )


def _require_hidden_q10(values: np.ndarray | Sequence[int]) -> np.ndarray:
    source = np.asarray(values)
    if source.ndim != 1:
        source = source.reshape(-1)
    if source.shape != (HIDDEN_SIZE,):
        raise FullModel24LayerReferenceError(
            f"hidden shape={source.shape}, expected=({HIDDEN_SIZE},)"
        )
    if not np.issubdtype(source.dtype, np.integer):
        raise FullModel24LayerReferenceError("hidden 必须为 signed Q6.10 整数")
    wide = source.astype(np.int64)
    if np.any(wide < -32768) or np.any(wide > 32767):
        raise FullModel24LayerReferenceError("hidden 超出 signed int16")
    return wide.astype(np.int16)


def build_initial_hidden_q10(
    layer0_context: LayerContext,
    *,
    hidden_seed: int = DEFAULT_HIDDEN_SEED_BASE,
) -> np.ndarray:
    """构造与 G2 query0 完全相同的初始 hidden Q6.10。"""

    source = make_deterministic_input(HIDDEN_SIZE, int(hidden_seed))
    result = compute_rmsnorm_reference(
        activation_values=source,
        gamma_values=layer0_context.block.attention.gamma,
        epsilon=DEFAULT_EPSILON,
        gamma_name=layer0_context.input_gamma_name,
    )
    return result.activation.quantized.astype(np.int16).copy()


def build_single_token_layer_case(
    context: LayerContext,
    hidden_q10: np.ndarray | Sequence[int],
    *,
    position: int = POSITION,
) -> TransformerBlockCase:
    """对一个显式 hidden 执行一个真实层的 count=1 完整 Block。"""

    layer = context.layer_index
    source = _require_hidden_q10(hidden_q10)
    if not 0 <= int(position) < 16_384:
        raise FullModel24LayerReferenceError("position 必须为 0..16383")
    input_float = source.astype(np.float32) / np.float32(Q10_FACTOR)
    rms = compute_rmsnorm_reference(
        activation_values=input_float,
        gamma_values=context.block.attention.gamma,
        epsilon=DEFAULT_EPSILON,
        gamma_name=context.input_gamma_name,
    )
    if not np.array_equal(rms.activation.quantized, source):
        raise FullModel24LayerReferenceError(
            f"layer{layer} 输入 Q6.10 经 RMS 入口未逐位还原"
        )
    norm_q10 = rms.output_lut_q10.astype(np.int16)
    norm_float = norm_q10.astype(np.float32) / np.float32(Q10_FACTOR)

    q_case = case_from_model(
        context.block.attention.qkv_models["q"],
        activation_values=norm_float,
        label=f"layer{layer} q_proj position={position}",
    )
    k_case = case_from_model(
        context.block.attention.qkv_models["k"],
        activation_values=norm_float,
        label=f"layer{layer} k_proj position={position}",
    )
    v_case = case_from_model(
        context.block.attention.qkv_models["v"],
        activation_values=norm_float,
        label=f"layer{layer} v_proj position={position}",
    )
    q_before = reshape_heads(q_case.expected_q28, q_case.spec).astype(np.int64)
    k_before = reshape_heads(k_case.expected_q28, k_case.spec).astype(np.int64)
    v_heads = reshape_heads(v_case.expected_q28, v_case.spec).astype(np.int64)
    trig = generate_trig_row(int(position))
    q_rope = apply_rope_fixed_q28(q_before, trig, heads=Q_HEADS).astype(np.int64)
    k_rope = apply_rope_fixed_q28(k_before, trig, heads=KV_HEADS).astype(np.int64)

    k_window = k_rope[np.newaxis, :, :]
    v_window = v_heads[np.newaxis, :, :]
    scores = attention_scores_q28(
        q_rope,
        k_window,
        query_position=int(position),
        window_start=int(position),
        count=COUNT,
    )
    probabilities, _ = softmax_scores_q31(scores)
    attention_heads, _ = attention_output_q28(
        probabilities,
        v_window,
        count=COUNT,
    )
    attention_concat = flatten_attention_heads(attention_heads)
    oproj = case_from_attention_q28(
        context.block.attention.oproj_model,
        attention_concat,
        label=f"layer{layer} o_proj position={position}",
    )
    first_residual, _oproj_q10, first_rescale_sat, first_residual_sat = (
        attention_residual_q10(source, oproj.expected_q28)
    )
    post_norm = case_from_input_q10(
        input_q10=first_residual,
        gamma_values=context.block.post_attention_gamma,
        gamma_name=context.post_gamma_name,
        label=f"layer{layer} post RMS position={position}",
        query_position=int(position),
        count=COUNT,
    )
    gate_up = case_from_post_attention_q10(
        context.block.gate_model,
        context.block.up_model,
        post_norm.output_lut_q10,
        label=f"layer{layer} gate/up position={position}",
        query_position=int(position),
        count=COUNT,
    )
    silu = case_from_gate_q28(
        gate_up.gate.expected_q28,
        label=f"layer{layer} SiLU position={position}",
        query_position=int(position),
        count=COUNT,
    )
    silu_up = case_from_inputs(
        silu.output_pwl_q10,
        gate_up.up.expected_q28,
        label=f"layer{layer} SiLU*up position={position}",
        query_position=int(position),
        count=COUNT,
    )
    down = case_from_source_q28(
        context.block.down_model,
        silu_up.output_q28,
        label=f"layer{layer} down_proj position={position}",
        query_position=int(position),
        count=COUNT,
    )
    output, down_q10, down_rescale_sat, second_residual_sat = mlp_residual_q10(
        first_residual,
        down.expected_q28,
    )
    return TransformerBlockCase(
        label=f"layer{layer} complete Transformer Block position={position}, count=1",
        query_position=int(position),
        window_start=int(position),
        count=COUNT,
        hidden_seed_base=0,
        block_input_q10=source.copy(),
        input_norm_q10=norm_q10.copy(),
        current_q_q28=q_before.copy(),
        current_k_q28=k_before.copy(),
        current_q_rope_q28=q_rope.copy(),
        current_k_rope_q28=k_rope.copy(),
        current_v_q28=v_heads.copy(),
        history_k_q28=np.empty((0, KV_HEADS, HEAD_DIM), dtype=np.int64),
        history_v_q28=np.empty((0, KV_HEADS, HEAD_DIM), dtype=np.int64),
        scores_q28=scores.astype(np.int64).copy(),
        probabilities_q31=probabilities.astype(np.uint32).copy(),
        attention_concat_q28=attention_concat.astype(np.int64).copy(),
        oproj_q28=oproj.expected_q28.astype(np.int64).copy(),
        first_residual_q10=first_residual.astype(np.int16).copy(),
        post_attention_norm_q10=post_norm.output_lut_q10.astype(np.int16).copy(),
        gate_q28=gate_up.gate.expected_q28.astype(np.int64).copy(),
        up_q28=gate_up.up.expected_q28.astype(np.int64).copy(),
        silu_gate_q10=silu.output_pwl_q10.astype(np.int16).copy(),
        silu_up_q28=silu_up.output_q28.astype(np.int64).copy(),
        down_proj_q28=down.expected_q28.astype(np.int64).copy(),
        down_proj_q10=down_q10.astype(np.int16).copy(),
        output_q10=output.astype(np.int16).copy(),
        first_rescale_saturated_count=int(first_rescale_sat),
        first_residual_saturated_count=int(first_residual_sat),
        silu_rescale_saturated_count=int(silu.rescale_saturated_count),
        silu_up_saturated_count=int(silu_up.saturated_count),
        down_rescale_saturated_count=int(down_rescale_sat),
        second_residual_saturated_count=int(second_residual_sat),
    )


TENSOR_DTYPES: tuple[tuple[str, str], ...] = (
    ("input_norm_q10", "<i2"),
    ("current_q_q28", "<i8"),
    ("current_k_q28", "<i8"),
    ("current_q_rope_q28", "<i8"),
    ("current_k_rope_q28", "<i8"),
    ("current_v_q28", "<i8"),
    ("scores_q28", "<i8"),
    ("probabilities_q31", "<u4"),
    ("attention_concat_q28", "<i8"),
    ("oproj_q28", "<i8"),
    ("first_residual_q10", "<i2"),
    ("post_attention_norm_q10", "<i2"),
    ("gate_q28", "<i8"),
    ("up_q28", "<i8"),
    ("silu_gate_q10", "<i2"),
    ("silu_up_q28", "<i8"),
    ("down_proj_q28", "<i8"),
    ("output_q10", "<i2"),
)


def case_tensor_hashes(case: TransformerBlockCase) -> dict[str, str]:
    return {
        name: sha256_array(getattr(case, name), dtype)
        for name, dtype in TENSOR_DTYPES
    }


def tensor_hash_set_sha256(values: dict[str, str]) -> str:
    payload = json.dumps(values, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def case_saturation_counts(case: TransformerBlockCase) -> dict[str, int]:
    return {
        "first_rescale": case.first_rescale_saturated_count,
        "first_residual": case.first_residual_saturated_count,
        "silu_rescale": case.silu_rescale_saturated_count,
        "silu_up": case.silu_up_saturated_count,
        "down_rescale": case.down_rescale_saturated_count,
        "second_residual": case.second_residual_saturated_count,
    }


def build_24layer_sequence(
    *,
    image_path: str | Path = DEFAULT_IMAGE,
    hidden_seed: int = DEFAULT_HIDDEN_SEED_BASE,
    position: int = POSITION,
    start_layer: int = 0,
    end_layer: int = ACTIVE_LAYER_COUNT - 1,
) -> FullModelSequence:
    if int(start_layer) != 0:
        raise FullModel24LayerReferenceError("连贯序列必须从 layer0 开始")
    if not 0 <= int(end_layer) < ACTIVE_LAYER_COUNT:
        raise FullModel24LayerReferenceError("end_layer 必须位于 0..23")
    image = P50Image(image_path)
    image.validate()
    model_layers = int(image.metadata.get("model", {}).get("num_hidden_layers", -1))
    if model_layers != ACTIVE_LAYER_COUNT:
        raise FullModel24LayerReferenceError(
            f"P50 真实层数 {model_layers} != {ACTIVE_LAYER_COUNT}"
        )
    sequence_reference = load_layer_sequence_reference()
    parameter_hashes = sequence_reference["layer_upload_set_sha256"]
    if len(parameter_hashes) != ACTIVE_LAYER_COUNT:
        raise FullModel24LayerReferenceError("H3 参数事务 SHA 数量不是 24")

    started = time.perf_counter()
    layer0_context = load_layer_context(image, 0)
    initial = build_initial_hidden_q10(layer0_context, hidden_seed=hidden_seed)
    hidden = initial.copy()
    summaries: list[LayerSummary] = []
    for layer_index in range(int(start_layer), int(end_layer) + 1):
        context = layer0_context if layer_index == 0 else load_layer_context(image, layer_index)
        case = build_single_token_layer_case(context, hidden, position=position)
        summaries.append(
            LayerSummary(
                layer_index=layer_index,
                label=case.label,
                input_sha256=sha256_array(case.block_input_q10, "<i2"),
                output_sha256=sha256_array(case.output_q10, "<i2"),
                tensor_sha256=case_tensor_hashes(case),
                saturation_counts=case_saturation_counts(case),
                parameter_upload_set_sha256=str(parameter_hashes[layer_index]),
            )
        )
        hidden = case.output_q10.copy()
    return FullModelSequence(
        initial_hidden_q10=initial,
        final_hidden_q10=hidden,
        layers=tuple(summaries),
        elapsed_seconds=time.perf_counter() - started,
    )


def sequence_manifest(sequence: FullModelSequence) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "definition": {
            "model": "Qwen2.5-0.5B real P50 Transformer layers",
            "layers": ACTIVE_LAYER_COUNT,
            "layer_range": [0, ACTIVE_LAYER_COUNT - 1],
            "position": POSITION,
            "count": COUNT,
            "input_format": "signed int16 Q6.10 [896]",
            "output_format": "signed int16 Q6.10 [896]",
            "initial_hidden_seed": DEFAULT_HIDDEN_SEED_BASE,
            "parameter_contract": "pangu50k.full_model_layer_sequence.v1",
            "excluded": ["embedding", "final_rmsnorm", "lm_head", "logits", "sampling"],
        },
        "initial_hidden_sha256": sha256_array(sequence.initial_hidden_q10, "<i2"),
        "final_hidden_sha256": sha256_array(sequence.final_hidden_q10, "<i2"),
        "layer_output_sha256": [layer.output_sha256 for layer in sequence.layers],
        "layer_tensor_count": len(TENSOR_DTYPES),
        "layer_tensor_set_sha256": [
            tensor_hash_set_sha256(layer.tensor_sha256) for layer in sequence.layers
        ],
        "layer_parameter_upload_set_sha256": [
            layer.parameter_upload_set_sha256 for layer in sequence.layers
        ],
        "saturation_events": [
            {
                "layer_index": layer.layer_index,
                "stage": stage,
                "count": count,
            }
            for layer in sequence.layers
            for stage, count in layer.saturation_counts.items()
            if count
        ],
    }


def load_reference(path: str | Path = DEFAULT_REFERENCE) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify_reference(
    *,
    image_path: str | Path = DEFAULT_IMAGE,
    reference_path: str | Path = DEFAULT_REFERENCE,
) -> FullModelSequence:
    sequence = build_24layer_sequence(image_path=image_path)
    actual = sequence_manifest(sequence)
    expected = load_reference(reference_path)
    if actual != expected:
        raise FullModel24LayerReferenceError("24 层冻结清单不一致")
    return sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="真实 Qwen2.5-0.5B 24 层单 token 软件参考")
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("summary")
    sub.add_parser("verify")
    dump = sub.add_parser("dump")
    dump.add_argument("--output", type=Path)
    layer = sub.add_parser("layer")
    layer.add_argument("layer_index", type=int)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "layer":
        image = P50Image(args.image)
        image.validate()
        layer0 = load_layer_context(image, 0)
        initial = build_initial_hidden_q10(layer0)
        context = layer0 if args.layer_index == 0 else load_layer_context(image, args.layer_index)
        case = build_single_token_layer_case(context, initial)
        print(f"layer={args.layer_index}")
        print(f"input_sha256={sha256_array(case.block_input_q10, '<i2')}")
        print(f"output_sha256={sha256_array(case.output_q10, '<i2')}")
        print(json.dumps(case_saturation_counts(case), ensure_ascii=False, sort_keys=True))
        return 0

    sequence = build_24layer_sequence(image_path=args.image)
    manifest = sequence_manifest(sequence)
    if args.command == "summary":
        print(f"schema={manifest['schema']}")
        print(f"layers={len(manifest['layer_output_sha256'])}")
        print(f"initial_hidden_sha256={manifest['initial_hidden_sha256']}")
        print(f"final_hidden_sha256={manifest['final_hidden_sha256']}")
        print(f"elapsed={sequence.elapsed_seconds:.3f}s")
        previous = manifest["initial_hidden_sha256"]
        for layer_index, output_hash in enumerate(manifest["layer_output_sha256"]):
            print(
                f"layer{layer_index:02d}: {previous[:12]} -> {output_hash[:12]}"
            )
            previous = output_hash
        return 0
    if args.command == "dump":
        payload = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
        if args.output is None:
            print(payload, end="")
        else:
            args.output.write_text(payload, encoding="utf-8")
        return 0
    if args.command == "verify":
        expected = load_reference(args.reference)
        if manifest != expected:
            raise FullModel24LayerReferenceError("24 层冻结清单不一致")
        print(
            "H3 24 层单 token 软件参考验证通过："
            f"24/24 layers, final={manifest['final_hidden_sha256']}"
        )
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
