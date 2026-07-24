#!/usr/bin/env python3
"""Qwen2.5-0.5B layer0 MLP gate_proj/up_proj 双投影软件金标准。

输入直接来自已经真实上板逐位通过的 ``post_attention_layernorm``
``[896]`` signed Q6.10 输出。两路真实权重均为 ``[4864,896]``、
group size 64 的对称 groupwise INT4，且模型中均不存在 bias。

统一硬件定义：

1. Q6.10 输入精确转换为 float32；
2. 逐向量对称量化为 INT8 ``[-127,127]``，zero point=0，RNE；
3. 主机预计算 ``activation_scale * weight_scale``，编码为 UQ4.28；
4. 每 64 元素执行 signed INT32 点积；
5. 点积乘 unsigned UQ4.28，并在 signed int64 Q28 中跨 14 组累加；
6. gate/up 均无 bias，``bias_q28`` 固定全零。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

try:
    from .linear_quant_reference import (
        LinearReferenceResult,
        compute_groupwise_linear_reference,
        pack_int4_low_nibble_first,
    )
    from .p50_format import P50Image
    from .post_attention_layernorm_reference import (
        build_fixed_real_cases as build_post_attention_fixed_cases,
        make_random_input_q10,
    )
except ImportError:
    from linear_quant_reference import (
        LinearReferenceResult,
        compute_groupwise_linear_reference,
        pack_int4_low_nibble_first,
    )
    from p50_format import P50Image
    from post_attention_layernorm_reference import (
        build_fixed_real_cases as build_post_attention_fixed_cases,
        make_random_input_q10,
    )

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE = PROJECT_ROOT / "model_output/yanbo_qwen25_0.5b_int4.p50"
DEFAULT_MANIFEST = Path(__file__).with_name("mlp_gate_up_g1_reference.json")
DEFAULT_GATE_WEIGHT = "model.layers.0.mlp.gate_proj.weight"
DEFAULT_UP_WEIGHT = "model.layers.0.mlp.up_proj.weight"
DEFAULT_STRESS_SEED = 20260808

M = 4864
K = 896
GROUP_SIZE = 64
GROUPS = K // GROUP_SIZE
Q10_FACTOR = 1 << 10
WEIGHT_ROW_BYTES = K // 2
SCALE_ROW_BYTES = 64
BIAS_ROW_BYTES = 32
ACTIVATION_BYTES = K
WEIGHT_BYTES = M * WEIGHT_ROW_BYTES
SCALE_BYTES = M * SCALE_ROW_BYTES
BIAS_BYTES = M * BIAS_ROW_BYTES
UPLOAD_BYTES = ACTIVATION_BYTES + WEIGHT_BYTES + SCALE_BYTES + BIAS_BYTES
RESULT_BYTES = M * 8


class MLPGateUpReferenceError(ValueError):
    """表示 MLP gate/up 输入、张量、载荷或定点结果不合法。"""


@dataclass(frozen=True)
class ProjectionModel:
    name: str
    weights: np.ndarray
    weight_scales: np.ndarray


@dataclass(frozen=True)
class ProjectionCase:
    name: str
    activation_int8: np.ndarray
    activation_scale: float
    weights: np.ndarray
    scales_q28: np.ndarray
    bias_q28: np.ndarray
    expected_q28: np.ndarray
    linear_result: LinearReferenceResult


@dataclass(frozen=True)
class GateUpCase:
    label: str
    query_position: int | None
    count: int | None
    source_post_attention_q10: np.ndarray
    gate: ProjectionCase
    up: ProjectionCase


def _require_shape(array: np.ndarray, shape: tuple[int, ...], label: str) -> None:
    if array.shape != shape:
        raise MLPGateUpReferenceError(f"{label} 形状错误：{array.shape}，预期 {shape}")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_array(array: np.ndarray, dtype: str | np.dtype) -> str:
    return sha256_bytes(np.asarray(array, dtype=dtype).tobytes(order="C"))


def q10_to_float32(values: np.ndarray | Sequence[int]) -> np.ndarray:
    q10 = np.asarray(values)
    if q10.ndim != 1:
        q10 = q10.reshape(-1)
    _require_shape(q10, (K,), "post_attention_layernorm Q6.10 输入")
    if not np.issubdtype(q10.dtype, np.integer):
        raise MLPGateUpReferenceError("Q6.10 输入必须为整数")
    wide = q10.astype(np.int64)
    if np.any(wide < -32768) or np.any(wide > 32767):
        raise MLPGateUpReferenceError("Q6.10 输入超出 signed int16")
    result = (wide.astype(np.float32) / np.float32(Q10_FACTOR)).astype(np.float32)
    if not np.all(np.isfinite(result)):
        raise MLPGateUpReferenceError("Q6.10 输入转换后出现非有限值")
    return result


def load_projection_model(image: P50Image, name: str) -> ProjectionModel:
    block = image.extract_block(name, 0, M, 0, K)
    if block.quantized is None or block.scales is None:
        raise MLPGateUpReferenceError(f"{name} 不是分组 INT4 张量")
    weights = block.quantized.astype(np.int8)
    scales = block.scales.astype(np.float32)
    _require_shape(weights, (M, K), f"{name} weights")
    _require_shape(scales, (M, GROUPS), f"{name} scales")
    return ProjectionModel(name=name, weights=weights, weight_scales=scales)


def load_gate_up_models(image: P50Image) -> tuple[ProjectionModel, ProjectionModel]:
    names = set(image.tensor_names())
    for bias_name in (
        "model.layers.0.mlp.gate_proj.bias",
        "model.layers.0.mlp.up_proj.bias",
    ):
        if bias_name in names:
            raise MLPGateUpReferenceError(f"当前参考要求无 bias，但镜像中存在 {bias_name}")
    return (
        load_projection_model(image, DEFAULT_GATE_WEIGHT),
        load_projection_model(image, DEFAULT_UP_WEIGHT),
    )


def compute_q28_reference(
    activation_int8: np.ndarray,
    weights: np.ndarray,
    scales_q28: np.ndarray,
) -> np.ndarray:
    activation = np.asarray(activation_int8, dtype=np.int8)
    weight_values = np.asarray(weights, dtype=np.int8)
    scales = np.asarray(scales_q28, dtype=np.uint32)
    _require_shape(activation, (K,), "activation_int8")
    _require_shape(weight_values, (M, K), "weights")
    _require_shape(scales, (M, GROUPS), "scales_q28")

    grouped_weights = weight_values.astype(np.int32).reshape(M, GROUPS, GROUP_SIZE)
    grouped_activation = activation.astype(np.int32).reshape(GROUPS, GROUP_SIZE)
    group_acc = np.sum(
        grouped_weights * grouped_activation[np.newaxis, :, :],
        axis=2,
        dtype=np.int64,
    )
    if np.any(group_acc < np.iinfo(np.int32).min) or np.any(
        group_acc > np.iinfo(np.int32).max
    ):
        raise MLPGateUpReferenceError("分组点积超出 signed int32")

    outputs = np.empty(M, dtype=np.int64)
    for row in range(M):
        total = 0
        for group in range(GROUPS):
            total += int(group_acc[row, group]) * int(scales[row, group])
        if not -(1 << 63) <= total <= (1 << 63) - 1:
            raise MLPGateUpReferenceError(f"第 {row} 行 Q28 累加超出 signed int64")
        outputs[row] = total
    return outputs


def projection_case(
    model: ProjectionModel,
    input_float: np.ndarray,
) -> ProjectionCase:
    result = compute_groupwise_linear_reference(
        weight_quantized=model.weights,
        weight_scales=model.weight_scales,
        activation_values=input_float,
        bias=None,
        group_size=GROUP_SIZE,
        weight_name=model.name,
        bias_name=None,
    )
    if result.combined_scale_saturated_count:
        raise MLPGateUpReferenceError(f"{model.name} combined scale 出现 UQ4.28 饱和")
    if np.any(result.bias_q28):
        raise MLPGateUpReferenceError(f"{model.name} 无 bias 却产生非零 bias_q28")
    independent = compute_q28_reference(
        result.activation.quantized,
        model.weights,
        result.combined_scale_q28,
    )
    if not np.array_equal(independent, result.output_fixed_q28):
        raise MLPGateUpReferenceError(f"{model.name} 独立 Q28 重算不一致")
    return ProjectionCase(
        name=model.name,
        activation_int8=result.activation.quantized.astype(np.int8),
        activation_scale=float(result.activation.scale),
        weights=model.weights,
        scales_q28=result.combined_scale_q28.astype(np.uint32),
        bias_q28=np.zeros(M, dtype=np.int64),
        expected_q28=result.output_fixed_q28.astype(np.int64),
        linear_result=result,
    )


def case_from_post_attention_q10(
    gate_model: ProjectionModel,
    up_model: ProjectionModel,
    input_q10: np.ndarray | Sequence[int],
    *,
    label: str,
    query_position: int | None = None,
    count: int | None = None,
) -> GateUpCase:
    source = np.asarray(input_q10, dtype=np.int16).reshape(-1)
    _require_shape(source, (K,), "source_post_attention_q10")
    input_float = q10_to_float32(source)
    gate = projection_case(gate_model, input_float)
    up = projection_case(up_model, input_float)
    if not np.array_equal(gate.activation_int8, up.activation_int8):
        raise MLPGateUpReferenceError("gate/up 未共享完全一致的 INT8 激活")
    if gate.activation_scale != up.activation_scale:
        raise MLPGateUpReferenceError("gate/up 未共享完全一致的 activation scale")
    return GateUpCase(
        label=label,
        query_position=query_position,
        count=count,
        source_post_attention_q10=source.copy(),
        gate=gate,
        up=up,
    )


def build_fixed_real_cases(*, image_path: Path = DEFAULT_IMAGE) -> list[GateUpCase]:
    image = P50Image(image_path)
    image.validate()
    gate_model, up_model = load_gate_up_models(image)
    source_cases = build_post_attention_fixed_cases(image_path=image_path)
    return [
        case_from_post_attention_q10(
            gate_model,
            up_model,
            source.output_lut_q10,
            label=(
                f"layer0 MLP gate/up query={source.query_position}, count={source.count}"
            ),
            query_position=source.query_position,
            count=source.count,
        )
        for source in source_cases
    ]


def build_projection_payload(case: ProjectionCase) -> bytes:
    packed_weight = pack_int4_low_nibble_first(case.weights).astype(np.uint8)
    scale_rows = np.zeros((M, SCALE_ROW_BYTES // 4), dtype="<u4")
    scale_rows[:, :GROUPS] = np.asarray(case.scales_q28, dtype="<u4")
    bias_rows = np.zeros((M, BIAS_ROW_BYTES // 8), dtype="<i8")
    payload = (
        np.asarray(case.activation_int8, dtype=np.int8).tobytes(order="C")
        + packed_weight.tobytes(order="C")
        + scale_rows.tobytes(order="C")
        + bias_rows.tobytes(order="C")
    )
    if len(payload) != UPLOAD_BYTES:
        raise MLPGateUpReferenceError(
            f"{case.name} 上传载荷长度错误：{len(payload)} != {UPLOAD_BYTES}"
        )
    return payload


def verify_projection_payload(case: ProjectionCase) -> str:
    payload = build_projection_payload(case)
    act_end = ACTIVATION_BYTES
    weight_end = act_end + WEIGHT_BYTES
    scale_end = weight_end + SCALE_BYTES
    decoded_act = np.frombuffer(payload[:act_end], dtype=np.int8).copy()
    decoded_weight = np.frombuffer(payload[act_end:weight_end], dtype=np.uint8).copy()
    decoded_scales = np.frombuffer(payload[weight_end:scale_end], dtype="<u4").reshape(
        M, SCALE_ROW_BYTES // 4
    )
    decoded_bias = np.frombuffer(payload[scale_end:], dtype="<i8").reshape(
        M, BIAS_ROW_BYTES // 8
    )
    if not np.array_equal(decoded_act, case.activation_int8):
        raise MLPGateUpReferenceError("激活载荷往返不一致")
    expected_weight = pack_int4_low_nibble_first(case.weights).reshape(-1)
    if not np.array_equal(decoded_weight, expected_weight):
        raise MLPGateUpReferenceError("权重载荷往返不一致")
    if not np.array_equal(decoded_scales[:, :GROUPS], case.scales_q28):
        raise MLPGateUpReferenceError("scale 载荷往返不一致")
    if np.any(decoded_scales[:, GROUPS:]):
        raise MLPGateUpReferenceError("scale padding 非零")
    if np.any(decoded_bias):
        raise MLPGateUpReferenceError("无 bias 投影的 bias padding 非零")
    return sha256_bytes(payload)


def _projection_manifest(case: ProjectionCase) -> dict[str, object]:
    return {
        "weight_tensor": case.name,
        "activation_scale": case.activation_scale,
        "activation_clipped_count": case.linear_result.activation.clipped_count,
        "combined_scale_saturated_count": case.linear_result.combined_scale_saturated_count,
        "max_activation_quantization_error": float(
            np.max(np.abs(case.linear_result.activation_error))
        ),
        "max_fixed_scale_error": float(np.max(np.abs(case.linear_result.fixed_error))),
        "max_fixed_error_bound": float(np.max(case.linear_result.fixed_error_bound)),
        "sha256": {
            "activation_int8": sha256_array(case.activation_int8, np.int8),
            "packed_weight_int4": sha256_array(
                pack_int4_low_nibble_first(case.weights), np.uint8
            ),
            "combined_scale_uq4_28": sha256_array(case.scales_q28, "<u4"),
            "bias_q28": sha256_array(case.bias_q28, "<i8"),
            "output_fixed_q28": sha256_array(case.expected_q28, "<i8"),
            "upload_payload": verify_projection_payload(case),
        },
        "preview": {
            "activation_first16_int8": case.activation_int8[:16].tolist(),
            "output_first8_q28": case.expected_q28[:8].tolist(),
            "output_last8_q28": case.expected_q28[-8:].tolist(),
        },
    }


def fixed_manifest(cases: Sequence[GateUpCase]) -> dict[str, object]:
    return {
        "format_version": 1,
        "operator": "qwen2_layer0_mlp_gate_up_projection",
        "definition": {
            "input_source": "verified layer0 post_attention_layernorm output",
            "input_format": "signed Q6.10 int16 [896]",
            "activation_quantization": (
                "exact Q6.10 to float32, symmetric per-vector INT8 [-127,127], "
                "RNE, zero_point=0"
            ),
            "weight_shape": [M, K],
            "weight_storage": "signed INT4 groupwise symmetric",
            "group_size": GROUP_SIZE,
            "groups_per_row": GROUPS,
            "combined_scale": "unsigned UQ4.28",
            "bias": "gate_proj/up_proj both absent; padded bias_q28 all zero",
            "output_format": "signed int64 Q28 [4864] per projection",
            "upload_bytes_per_projection": UPLOAD_BYTES,
            "result_bytes_per_projection": RESULT_BYTES,
        },
        "cases": [
            {
                "label": case.label,
                "query_position": case.query_position,
                "count": case.count,
                "sha256": {
                    "source_post_attention_q10": sha256_array(
                        case.source_post_attention_q10, "<i2"
                    )
                },
                "gate": _projection_manifest(case.gate),
                "up": _projection_manifest(case.up),
            }
            for case in cases
        ],
    }


def validate_manifest(
    cases: Sequence[GateUpCase], manifest_path: Path = DEFAULT_MANIFEST
) -> dict[str, object]:
    actual = fixed_manifest(cases)
    expected = json.loads(manifest_path.read_text(encoding="utf-8"))
    if actual != expected:
        raise MLPGateUpReferenceError(f"MLP gate/up 固定清单不一致：{manifest_path}")
    return expected


def software_stress(
    *, image_path: Path = DEFAULT_IMAGE, rounds: int = 1000, seed: int = DEFAULT_STRESS_SEED
) -> None:
    if rounds <= 0:
        raise MLPGateUpReferenceError("rounds 必须大于 0")
    image = P50Image(image_path)
    image.validate()
    gate_model, up_model = load_gate_up_models(image)
    rng = np.random.default_rng(seed)
    for index in range(rounds):
        source = make_random_input_q10(rng, index)
        case = case_from_post_attention_q10(
            gate_model,
            up_model,
            source,
            label=f"MLP gate/up stress {index + 1}/{rounds}",
        )
        if index < 8:
            verify_projection_payload(case.gate)
            verify_projection_payload(case.up)
        if not np.array_equal(case.gate.activation_int8, case.up.activation_int8):
            raise MLPGateUpReferenceError("压力测试中 gate/up 激活不一致")
        if not np.any(source):
            if np.any(case.gate.expected_q28) or np.any(case.up.expected_q28):
                raise MLPGateUpReferenceError("全零输入没有严格输出全零")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="layer0 MLP gate_proj/up_proj 双投影参考")
    sub = parser.add_subparsers(dest="command", required=True)
    manifest = sub.add_parser("manifest", help="输出四组真实固定清单")
    manifest.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    check = sub.add_parser("check", help="校验已提交固定清单")
    check.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    check.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    stress = sub.add_parser("stress", help="运行随机/边界软件压力")
    stress.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    stress.add_argument("--rounds", type=int, default=1000)
    stress.add_argument("--seed", type=int, default=DEFAULT_STRESS_SEED)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.command == "manifest":
            cases = build_fixed_real_cases(image_path=args.image)
            print(json.dumps(fixed_manifest(cases), ensure_ascii=False, indent=2))
        elif args.command == "check":
            cases = build_fixed_real_cases(image_path=args.image)
            manifest = validate_manifest(cases, args.manifest)
            print("layer0 MLP gate_proj/up_proj 固定清单：PASS")
            for item in manifest["cases"]:
                print(
                    f"query={item['query_position']} count={item['count']} "
                    f"gate={item['gate']['sha256']['output_fixed_q28']} "
                    f"up={item['up']['sha256']['output_fixed_q28']}"
                )
        elif args.command == "stress":
            software_stress(image_path=args.image, rounds=args.rounds, seed=args.seed)
            print(f"MLP gate/up 软件压力 PASS：{args.rounds}/{args.rounds}，seed={args.seed}")
        else:  # pragma: no cover
            raise AssertionError(args.command)
        return 0
    except (
        FileNotFoundError,
        KeyError,
        ValueError,
        OverflowError,
        RuntimeError,
    ) as error:
        print(f"错误：{error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
