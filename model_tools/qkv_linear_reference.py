#!/usr/bin/env python3
"""layer0 Q/K/V 真实 INT4 Linear 的统一软件参考与载荷定义。

本模块在已验证 ``linear_quant_reference`` 基础上统一描述 Qwen2.5-0.5B
layer0 的三个 Attention 投影：

- q_proj: [896, 896] -> 14 个 Q heads × head_dim 64；
- k_proj: [128, 896] -> 2 个 KV heads × head_dim 64；
- v_proj: [128, 896] -> 2 个 KV heads × head_dim 64。

三者共用同一份逐向量对称 INT8 hidden state、真实 P50 分组 INT4 权重、
UQ4.28 combined scale 和 signed int64 Q28 输出定义。输出行按 head-major、
head 内维度连续排列，即 ``flat.reshape(num_heads, 64)``。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

try:
    from .linear_quant_reference import (
        DEFAULT_ACTIVATION_SEED,
        LinearReferenceResult,
        compute_groupwise_linear_reference,
        make_deterministic_activation,
        pack_int4_low_nibble_first,
    )
    from .p50_format import P50Image
except ImportError:
    from linear_quant_reference import (
        DEFAULT_ACTIVATION_SEED,
        LinearReferenceResult,
        compute_groupwise_linear_reference,
        make_deterministic_activation,
        pack_int4_low_nibble_first,
    )
    from p50_format import P50Image

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE = PROJECT_ROOT / "model_output/yanbo_qwen25_0.5b_int4.p50"
DEFAULT_MANIFEST = Path(__file__).with_name("qkv_layer0_reference.json")

K = 896
GROUP_SIZE = 64
GROUPS = K // GROUP_SIZE
HEAD_DIM = 64
Q_HEADS = 14
KV_HEADS = 2
ACTIVATION_BYTES = K
WEIGHT_ROW_BYTES = K // 2
SCALE_ROW_BYTES = 64
BIAS_ROW_BYTES = 32


class QKVReferenceError(ValueError):
    """表示 Q/K/V 参考、布局或载荷不合法。"""


@dataclass(frozen=True)
class ProjectionSpec:
    """一个真实 Attention 投影的固定结构。"""

    key: str
    selector: int
    command: bytes
    rows: int
    heads: int
    weight_name: str
    bias_name: str

    @property
    def groups(self) -> int:
        return GROUPS

    @property
    def result_bytes(self) -> int:
        return self.rows * 8

    @property
    def result_beats(self) -> int:
        return self.rows // 4

    @property
    def weight_bytes(self) -> int:
        return self.rows * WEIGHT_ROW_BYTES

    @property
    def scale_bytes(self) -> int:
        return self.rows * SCALE_ROW_BYTES

    @property
    def bias_bytes(self) -> int:
        return self.rows * BIAS_ROW_BYTES

    @property
    def upload_bytes(self) -> int:
        return (
            ACTIVATION_BYTES
            + self.weight_bytes
            + self.scale_bytes
            + self.bias_bytes
        )


PROJECTION_SPECS: Mapping[str, ProjectionSpec] = {
    "q": ProjectionSpec(
        key="q",
        selector=0,
        command=b"Q",
        rows=Q_HEADS * HEAD_DIM,
        heads=Q_HEADS,
        weight_name="model.layers.0.self_attn.q_proj.weight",
        bias_name="model.layers.0.self_attn.q_proj.bias",
    ),
    "k": ProjectionSpec(
        key="k",
        selector=1,
        command=b"K",
        rows=KV_HEADS * HEAD_DIM,
        heads=KV_HEADS,
        weight_name="model.layers.0.self_attn.k_proj.weight",
        bias_name="model.layers.0.self_attn.k_proj.bias",
    ),
    "v": ProjectionSpec(
        key="v",
        selector=2,
        command=b"V",
        rows=KV_HEADS * HEAD_DIM,
        heads=KV_HEADS,
        weight_name="model.layers.0.self_attn.v_proj.weight",
        bias_name="model.layers.0.self_attn.v_proj.bias",
    ),
}


@dataclass(frozen=True)
class ProjectionModel:
    spec: ProjectionSpec
    weights: np.ndarray
    weight_scales: np.ndarray
    bias: np.ndarray


@dataclass(frozen=True)
class ProjectionCase:
    spec: ProjectionSpec
    activation: np.ndarray
    weights: np.ndarray
    scales_q28: np.ndarray
    bias_q28: np.ndarray
    expected_q28: np.ndarray
    activation_scale: float
    label: str

    @property
    def heads_q28(self) -> np.ndarray:
        return reshape_heads(self.expected_q28, self.spec)


def projection_spec(value: str | ProjectionSpec) -> ProjectionSpec:
    if isinstance(value, ProjectionSpec):
        return value
    key = value.lower()
    try:
        return PROJECTION_SPECS[key]
    except KeyError as error:
        raise QKVReferenceError(f"未知投影类型：{value!r}，应为 q/k/v") from error


def projection_sequence(value: str) -> tuple[ProjectionSpec, ...]:
    key = value.lower()
    if key == "all":
        return tuple(PROJECTION_SPECS[item] for item in ("q", "k", "v"))
    return (projection_spec(key),)


def _require_shape(array: np.ndarray, shape: tuple[int, ...], label: str) -> None:
    if array.shape != shape:
        raise QKVReferenceError(f"{label} 形状错误：{array.shape}，预期 {shape}")


def sha256_array(array: np.ndarray, dtype: str | np.dtype) -> str:
    normalized = np.asarray(array, dtype=dtype)
    return hashlib.sha256(normalized.tobytes(order="C")).hexdigest()


def reshape_heads(values: np.ndarray | Sequence[int], spec: ProjectionSpec) -> np.ndarray:
    """按 GQA 的 head-major 布局把平坦输出重排为 ``[heads, 64]``。"""

    resolved = projection_spec(spec)
    array = np.asarray(values)
    if array.ndim != 1 or array.size != resolved.rows:
        raise QKVReferenceError(
            f"{resolved.key}_proj 输出形状错误：{array.shape}，预期 ({resolved.rows},)"
        )
    heads = array.reshape(resolved.heads, HEAD_DIM)
    if not np.array_equal(heads.reshape(-1), array):
        raise QKVReferenceError("head-major 重排后无法无损还原平坦输出")
    return heads


def validate_gqa_layout(cases: Mapping[str, ProjectionCase]) -> None:
    """检查 14Q/2KV、head_dim=64 的输出布局及共享 hidden state。"""

    required = {"q", "k", "v"}
    if set(cases) != required:
        raise QKVReferenceError(f"GQA 布局检查要求同时提供 {sorted(required)}")
    q_case = cases["q"]
    k_case = cases["k"]
    v_case = cases["v"]
    if not np.array_equal(q_case.activation, k_case.activation) or not np.array_equal(
        q_case.activation, v_case.activation
    ):
        raise QKVReferenceError("Q/K/V 必须使用同一个量化 hidden state")
    if q_case.spec.heads != Q_HEADS or k_case.spec.heads != KV_HEADS or v_case.spec.heads != KV_HEADS:
        raise QKVReferenceError("GQA head 数与模型配置不一致")
    for case in (q_case, k_case, v_case):
        heads = reshape_heads(case.expected_q28, case.spec)
        _require_shape(heads, (case.spec.heads, HEAD_DIM), f"{case.spec.key}_heads")


def load_projection_model(image: P50Image, spec: ProjectionSpec | str) -> ProjectionModel:
    resolved = projection_spec(spec)
    entry = image.tensor(resolved.weight_name)
    if list(entry.get("shape", [])) != [resolved.rows, K]:
        raise QKVReferenceError(
            f"{resolved.weight_name} 形状错误：{entry.get('shape')}，预期 {[resolved.rows, K]}"
        )
    if int(entry.get("groups_per_row", -1)) != GROUPS:
        raise QKVReferenceError(
            f"{resolved.weight_name} groups_per_row 错误：{entry.get('groups_per_row')}"
        )
    block = image.extract_block(resolved.weight_name, 0, resolved.rows, 0, K)
    if block.quantized is None or block.scales is None:
        raise QKVReferenceError(f"{resolved.weight_name} 不是分组 INT4 张量")
    bias = image.read_float16_tensor(resolved.bias_name).astype(np.float32).reshape(-1)
    _require_shape(block.quantized, (resolved.rows, K), "weight_int4")
    _require_shape(block.scales, (resolved.rows, GROUPS), "weight_scales")
    _require_shape(bias, (resolved.rows,), "bias")
    return ProjectionModel(
        spec=resolved,
        weights=block.quantized.astype(np.int8),
        weight_scales=block.scales.astype(np.float32),
        bias=bias,
    )


def load_qkv_models(image: P50Image) -> dict[str, ProjectionModel]:
    return {
        key: load_projection_model(image, spec)
        for key, spec in PROJECTION_SPECS.items()
    }


def compute_q28_reference(
    activation: Sequence[int],
    weights: np.ndarray,
    scales_q28: np.ndarray,
    bias_q28: Sequence[int],
    spec: ProjectionSpec | str,
) -> np.ndarray:
    """按 FPGA 精确定义独立重算一个投影的 signed int64 Q28 输出。"""

    resolved = projection_spec(spec)
    act = np.asarray(activation, dtype=np.int8)
    weight_values = np.asarray(weights, dtype=np.int8)
    scales = np.asarray(scales_q28, dtype=np.uint32)
    bias = np.asarray(bias_q28, dtype=np.int64)
    _require_shape(act, (K,), "activation")
    _require_shape(weight_values, (resolved.rows, K), "weights")
    _require_shape(scales, (resolved.rows, GROUPS), "scales_q28")
    _require_shape(bias, (resolved.rows,), "bias_q28")

    grouped_weights = weight_values.astype(np.int32).reshape(
        resolved.rows, GROUPS, GROUP_SIZE
    )
    grouped_activation = act.astype(np.int32).reshape(GROUPS, GROUP_SIZE)
    accumulators = np.sum(
        grouped_weights * grouped_activation[np.newaxis, :, :],
        axis=2,
        dtype=np.int64,
    )
    if np.any(accumulators < np.iinfo(np.int32).min) or np.any(
        accumulators > np.iinfo(np.int32).max
    ):
        raise OverflowError("分组点积超出 signed int32")

    outputs: list[int] = []
    for row in range(resolved.rows):
        total = int(bias[row])
        for group in range(GROUPS):
            total += int(accumulators[row, group]) * int(scales[row, group])
        if not -(1 << 63) <= total <= (1 << 63) - 1:
            raise OverflowError(f"第 {row} 行 Q28 累加超出 signed int64")
        outputs.append(total)
    return np.asarray(outputs, dtype=np.int64)


def validate_case(case: ProjectionCase) -> None:
    spec = case.spec
    _require_shape(np.asarray(case.activation), (K,), "activation")
    _require_shape(np.asarray(case.weights), (spec.rows, K), "weights")
    _require_shape(np.asarray(case.scales_q28), (spec.rows, GROUPS), "scales_q28")
    _require_shape(np.asarray(case.bias_q28), (spec.rows,), "bias_q28")
    _require_shape(np.asarray(case.expected_q28), (spec.rows,), "expected_q28")
    if np.any(case.activation.astype(np.int16) < -127) or np.any(
        case.activation.astype(np.int16) > 127
    ):
        raise QKVReferenceError("activation 必须位于 [-127,127]")
    if np.any(case.weights.astype(np.int16) < -7) or np.any(
        case.weights.astype(np.int16) > 7
    ):
        raise QKVReferenceError("真实 P50 INT4 权重必须位于 [-7,7]")
    reshape_heads(case.expected_q28, spec)


def case_from_model(
    model: ProjectionModel,
    *,
    activation_values: np.ndarray | None = None,
    activation_seed: int = DEFAULT_ACTIVATION_SEED,
    label: str | None = None,
) -> ProjectionCase:
    source = (
        make_deterministic_activation(K, seed=activation_seed)
        if activation_values is None
        else np.asarray(activation_values, dtype=np.float32)
    )
    result: LinearReferenceResult = compute_groupwise_linear_reference(
        weight_quantized=model.weights,
        weight_scales=model.weight_scales,
        activation_values=source,
        bias=model.bias,
        group_size=GROUP_SIZE,
        weight_name=model.spec.weight_name,
        bias_name=model.spec.bias_name,
    )
    if result.combined_scale_saturated_count:
        raise QKVReferenceError(
            f"{model.spec.key}_proj combined scale 出现 UQ4.28 饱和"
        )
    case = ProjectionCase(
        spec=model.spec,
        activation=result.activation.quantized.astype(np.int8),
        weights=model.weights,
        scales_q28=result.combined_scale_q28.astype(np.uint32),
        bias_q28=result.bias_q28.astype(np.int64),
        expected_q28=result.output_fixed_q28.astype(np.int64),
        activation_scale=float(result.activation.scale),
        label=label or f"layer0 {model.spec.key}_proj seed={activation_seed}",
    )
    validate_case(case)
    independent = compute_q28_reference(
        case.activation,
        case.weights,
        case.scales_q28,
        case.bias_q28,
        case.spec,
    )
    if not np.array_equal(independent, case.expected_q28):
        raise QKVReferenceError(
            f"{model.spec.key}_proj 独立 Q28 重算与统一参考不一致"
        )
    return case


def build_qkv_cases(
    models: Mapping[str, ProjectionModel],
    *,
    activation_seed: int = DEFAULT_ACTIVATION_SEED,
) -> dict[str, ProjectionCase]:
    activation = make_deterministic_activation(K, seed=activation_seed)
    cases = {
        key: case_from_model(
            models[key],
            activation_values=activation,
            activation_seed=activation_seed,
        )
        for key in ("q", "k", "v")
    }
    validate_gqa_layout(cases)
    return cases


def build_upload_payload(case: ProjectionCase) -> bytes:
    """生成硬件 V1 载荷：activation + weight + padded scale + padded bias。"""

    validate_case(case)
    spec = case.spec
    activation = np.asarray(case.activation, dtype=np.int8).tobytes(order="C")
    packed_weights = pack_int4_low_nibble_first(case.weights).astype(np.uint8)
    scale_rows = np.zeros((spec.rows, SCALE_ROW_BYTES // 4), dtype="<u4")
    scale_rows[:, :GROUPS] = np.asarray(case.scales_q28, dtype="<u4")
    bias_rows = np.zeros((spec.rows, BIAS_ROW_BYTES // 8), dtype="<i8")
    bias_rows[:, 0] = np.asarray(case.bias_q28, dtype="<i8")
    payload = (
        activation
        + packed_weights.tobytes(order="C")
        + scale_rows.tobytes(order="C")
        + bias_rows.tobytes(order="C")
    )
    if len(payload) != spec.upload_bytes:
        raise QKVReferenceError(
            f"{spec.key}_proj 上传载荷长度错误：{len(payload)} != {spec.upload_bytes}"
        )
    return payload


def unpack_int4_matrix(payload: bytes, spec: ProjectionSpec | str) -> np.ndarray:
    resolved = projection_spec(spec)
    if len(payload) != resolved.weight_bytes:
        raise QKVReferenceError(
            f"{resolved.key}_proj packed 权重长度错误：{len(payload)}"
        )
    packed = np.frombuffer(payload, dtype=np.uint8).reshape(
        resolved.rows, WEIGHT_ROW_BYTES
    )
    output = np.empty((resolved.rows, K), dtype=np.int8)
    low = np.bitwise_and(packed, 0x0F).astype(np.int8)
    high = np.right_shift(packed, 4).astype(np.int8)
    low[low >= 8] -= 16
    high[high >= 8] -= 16
    output[:, 0::2] = low
    output[:, 1::2] = high
    return output


def verify_payload_roundtrip(case: ProjectionCase) -> str:
    spec = case.spec
    payload = build_upload_payload(case)
    activation_end = ACTIVATION_BYTES
    weight_end = activation_end + spec.weight_bytes
    scale_end = weight_end + spec.scale_bytes

    activation = np.frombuffer(payload[:activation_end], dtype=np.int8).copy()
    weights = unpack_int4_matrix(payload[activation_end:weight_end], spec)
    scales = np.frombuffer(payload[weight_end:scale_end], dtype="<u4").reshape(
        spec.rows, SCALE_ROW_BYTES // 4
    )
    bias = np.frombuffer(payload[scale_end:], dtype="<i8").reshape(
        spec.rows, BIAS_ROW_BYTES // 8
    )
    if not np.array_equal(activation, case.activation.astype(np.int8)):
        raise QKVReferenceError("activation 上传往返不一致")
    if not np.array_equal(weights, case.weights.astype(np.int8)):
        raise QKVReferenceError("packed INT4 上传往返不一致")
    if not np.array_equal(scales[:, :GROUPS], case.scales_q28.astype(np.uint32)):
        raise QKVReferenceError("UQ4.28 scale 上传往返不一致")
    if np.any(scales[:, GROUPS:] != 0):
        raise QKVReferenceError("scale 行补齐区域必须为 0")
    if not np.array_equal(bias[:, 0], case.bias_q28.astype(np.int64)):
        raise QKVReferenceError("bias_q28 上传往返不一致")
    if np.any(bias[:, 1:] != 0):
        raise QKVReferenceError("bias 行补齐区域必须为 0")
    return hashlib.sha256(payload).hexdigest()


def case_hashes(case: ProjectionCase) -> dict[str, str]:
    return {
        "activation_int8": sha256_array(case.activation, np.int8),
        "packed_weight_int4": sha256_array(
            pack_int4_low_nibble_first(case.weights), np.uint8
        ),
        "combined_scale_uq4_28": sha256_array(case.scales_q28, "<u4"),
        "bias_q28": sha256_array(case.bias_q28, "<i8"),
        "output_fixed_q28": sha256_array(case.expected_q28, "<i8"),
        "upload_payload": verify_payload_roundtrip(case),
    }


def case_manifest(case: ProjectionCase) -> dict[str, object]:
    heads = reshape_heads(case.expected_q28, case.spec)
    return {
        "projection": case.spec.key,
        "weight_tensor": case.spec.weight_name,
        "bias_tensor": case.spec.bias_name,
        "shape": {
            "M": case.spec.rows,
            "K": K,
            "group_size": GROUP_SIZE,
            "groups_per_row": GROUPS,
            "heads": case.spec.heads,
            "head_dim": HEAD_DIM,
            "layout": "head_major_contiguous",
        },
        "activation": {
            "format": "symmetric_per_vector_int8",
            "scale": case.activation_scale,
        },
        "upload_layout_bytes": {
            "activation_int8": ACTIVATION_BYTES,
            "packed_weight_int4": case.spec.weight_bytes,
            "combined_scale_uq4_28_padded": case.spec.scale_bytes,
            "bias_q28_padded": case.spec.bias_bytes,
            "total": case.spec.upload_bytes,
        },
        "expected": {
            "first_8_output_fixed_q28": case.expected_q28[:8].tolist(),
            "last_8_output_fixed_q28": case.expected_q28[-8:].tolist(),
            "head0_first_8_fixed_q28": heads[0, :8].tolist(),
            "last_head_last_8_fixed_q28": heads[-1, -8:].tolist(),
        },
        "sha256": case_hashes(case),
    }


def qkv_manifest(
    cases: Mapping[str, ProjectionCase], activation_seed: int
) -> dict[str, object]:
    validate_gqa_layout(cases)
    return {
        "format_version": 1,
        "layer": 0,
        "model_layout": {
            "hidden_size": K,
            "num_attention_heads": Q_HEADS,
            "num_key_value_heads": KV_HEADS,
            "head_dim": HEAD_DIM,
            "q_shape": [Q_HEADS, HEAD_DIM],
            "k_shape": [KV_HEADS, HEAD_DIM],
            "v_shape": [KV_HEADS, HEAD_DIM],
            "flat_order": "head_major_contiguous",
        },
        "activation": {
            "generator": "32-bit LCG",
            "seed": activation_seed,
            "shared_by_qkv": True,
            "int8_sha256": sha256_array(cases["q"].activation, np.int8),
        },
        "projections": {
            key: case_manifest(cases[key]) for key in ("q", "k", "v")
        },
    }


def validate_manifest(
    cases: Mapping[str, ProjectionCase], manifest_path: Path, activation_seed: int
) -> dict[str, object]:
    generated = qkv_manifest(cases, activation_seed)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"固定清单不存在：{manifest_path}")
    committed = json.loads(manifest_path.read_text(encoding="utf-8"))
    if generated != committed:
        raise QKVReferenceError("Q/K/V 固定向量与已提交 JSON 清单不一致")
    return generated


def _print_summary(cases: Mapping[str, ProjectionCase]) -> None:
    validate_gqa_layout(cases)
    print("=== layer0 Q/K/V 真实 Linear 统一参考 ===")
    print("GQA：14 Q heads，2 KV heads，head_dim=64，输出按 head-major 连续排列")
    for key in ("q", "k", "v"):
        case = cases[key]
        hashes = case_hashes(case)
        print(
            f"{key}_proj: shape=[{case.spec.rows},{K}]，heads={case.spec.heads}，"
            f"upload={case.spec.upload_bytes} B，output_sha256={hashes['output_fixed_q28']}"
        )
        print(f"  first8={case.expected_q28[:8].tolist()}")
        print(f"  last8={case.expected_q28[-8:].tolist()}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="layer0 Q/K/V 真实 Linear 统一软件参考")
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--activation-seed", type=int, default=DEFAULT_ACTIVATION_SEED)
    parser.add_argument(
        "--json", action="store_true", help="输出可提交的完整固定清单 JSON"
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        image = P50Image(args.image)
        image.validate()
        models = load_qkv_models(image)
        cases = build_qkv_cases(models, activation_seed=args.activation_seed)
        if args.json:
            print(
                json.dumps(
                    qkv_manifest(cases, args.activation_seed),
                    ensure_ascii=False,
                    indent=2,
                )
            )
        else:
            _print_summary(cases)
        return 0
    except (FileNotFoundError, KeyError, IndexError, ValueError, OverflowError) as error:
        print(f"错误：{error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
