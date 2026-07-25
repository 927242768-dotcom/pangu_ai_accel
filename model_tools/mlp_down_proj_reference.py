#!/usr/bin/env python3
"""Qwen2.5-0.5B layer0 MLP ``down_proj`` 硬件等价软件参考。

输入直接来自已经真实上板逐位通过的 ``SiLU(gate) × up``：
``[4864]`` signed int64 Q28。真实权重为
``model.layers.0.mlp.down_proj.weight``，shape ``[896,4864]``，group size 64
的对称 signed INT4；模型中不存在 bias。

冻结的硬件定义：

1. Q28 输入按实数解释并转换为 float32；
2. 逐向量对称量化为 INT8 ``[-127,127]``，RNE，zero point=0；
3. ``activation_scale * weight_scale`` 量化为 unsigned UQ4.28，RNE 后饱和；
4. 每 64 元素执行 signed INT32 点积；
5. 76 个 ``group_acc * combined_scale_uq4_28`` 在 signed int64 Q28 中精确累加；
6. down_proj 无 bias，输出为 ``[896]`` signed int64 Q28。

按 INT8/INT4 和 UQ4.28 的全范围计算，76 组最坏绝对累加仍小于 signed
INT64 上限，因此硬件不需要隐含截断或回绕；软件会对该边界进行显式证明与检查。
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
    from .mlp_silu_up_mul_reference import (
        DEFAULT_IMAGE,
        M as INPUT_SIZE,
        build_fixed_real_cases as build_silu_up_fixed_cases,
        make_random_inputs as make_silu_up_random_inputs,
        multiply_q10_q28_to_q28,
    )
    from .p50_format import P50Image
except ImportError:
    from linear_quant_reference import (
        LinearReferenceResult,
        compute_groupwise_linear_reference,
        pack_int4_low_nibble_first,
    )
    from mlp_silu_up_mul_reference import (
        DEFAULT_IMAGE,
        M as INPUT_SIZE,
        build_fixed_real_cases as build_silu_up_fixed_cases,
        make_random_inputs as make_silu_up_random_inputs,
        multiply_q10_q28_to_q28,
    )
    from p50_format import P50Image

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

DEFAULT_MANIFEST = Path(__file__).with_name("mlp_down_proj_g1_reference.json")
DEFAULT_WEIGHT = "model.layers.0.mlp.down_proj.weight"
DEFAULT_STRESS_SEED = 20260816

M = 896
K = INPUT_SIZE
GROUP_SIZE = 64
GROUPS = K // GROUP_SIZE
Q28_FACTOR = 1 << 28
WEIGHT_ROW_BYTES = K // 2
SCALE_ROW_BYTES = ((GROUPS * 4 + 31) // 32) * 32
BIAS_ROW_BYTES = 32
ACTIVATION_BYTES = K
WEIGHT_BYTES = M * WEIGHT_ROW_BYTES
SCALE_BYTES = M * SCALE_ROW_BYTES
BIAS_BYTES = M * BIAS_ROW_BYTES
UPLOAD_BYTES = ACTIVATION_BYTES + WEIGHT_BYTES + SCALE_BYTES + BIAS_BYTES
RESULT_BYTES = M * 8
INT64_MIN = -(1 << 63)
INT64_MAX = (1 << 63) - 1
MAX_GROUP_ACC = GROUP_SIZE * 127 * 7
MAX_UQ4_28 = (1 << 32) - 1
MAX_OUTPUT_MAGNITUDE = GROUPS * MAX_GROUP_ACC * MAX_UQ4_28


class MLPDownProjReferenceError(ValueError):
    """表示 down_proj 输入、张量、载荷或定点结果不合法。"""


@dataclass(frozen=True)
class DownProjectionModel:
    weights: np.ndarray
    weight_scales: np.ndarray


@dataclass(frozen=True)
class DownProjectionCase:
    label: str
    query_position: int | None
    count: int | None
    source_q28: np.ndarray
    activation_int8: np.ndarray
    activation_scale: float
    weights: np.ndarray
    scales_q28: np.ndarray
    bias_q28: np.ndarray
    expected_q28: np.ndarray
    linear_result: LinearReferenceResult


def _require_shape(array: np.ndarray, shape: tuple[int, ...], label: str) -> None:
    if array.shape != shape:
        raise MLPDownProjReferenceError(f"{label} 形状错误：{array.shape}，预期 {shape}")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_array(array: np.ndarray, dtype: str | np.dtype) -> str:
    return sha256_bytes(np.asarray(array, dtype=dtype).tobytes(order="C"))


def q28_to_float32(values: np.ndarray | Sequence[int]) -> np.ndarray:
    source = np.asarray(values)
    if source.ndim != 1:
        source = source.reshape(-1)
    if not np.issubdtype(source.dtype, np.integer):
        raise MLPDownProjReferenceError("Q28 输入必须为整数")
    source_i64 = source.astype(np.int64, copy=False)
    _require_shape(source_i64, (K,), "SiLU(gate) × up Q28 输入")
    result = (source_i64.astype(np.float32) / np.float32(Q28_FACTOR)).astype(np.float32)
    if not np.all(np.isfinite(result)):
        raise MLPDownProjReferenceError("Q28 输入转换后出现非有限值")
    return result


def load_down_projection_model(image: P50Image) -> DownProjectionModel:
    names = set(image.tensor_names())
    bias_name = "model.layers.0.mlp.down_proj.bias"
    if bias_name in names:
        raise MLPDownProjReferenceError(f"当前参考要求无 bias，但镜像中存在 {bias_name}")
    block = image.extract_block(DEFAULT_WEIGHT, 0, M, 0, K)
    if block.quantized is None or block.scales is None:
        raise MLPDownProjReferenceError(f"{DEFAULT_WEIGHT} 不是分组 INT4 张量")
    weights = block.quantized.astype(np.int8)
    scales = block.scales.astype(np.float32)
    _require_shape(weights, (M, K), "down_proj weights")
    _require_shape(scales, (M, GROUPS), "down_proj scales")
    return DownProjectionModel(weights=weights, weight_scales=scales)


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
        raise MLPDownProjReferenceError("分组点积超出 signed int32")

    outputs = np.empty(M, dtype=np.int64)
    for row in range(M):
        total = 0
        for group in range(GROUPS):
            total += int(group_acc[row, group]) * int(scales[row, group])
        if not INT64_MIN <= total <= INT64_MAX:
            raise MLPDownProjReferenceError(f"第 {row} 行 Q28 累加超出 signed int64")
        outputs[row] = total
    return outputs


def case_from_source_q28(
    model: DownProjectionModel,
    source_q28: np.ndarray | Sequence[int],
    *,
    label: str,
    query_position: int | None = None,
    count: int | None = None,
) -> DownProjectionCase:
    source = np.asarray(source_q28, dtype=np.int64).reshape(-1)
    _require_shape(source, (K,), "source_q28")
    input_float = q28_to_float32(source)
    result = compute_groupwise_linear_reference(
        weight_quantized=model.weights,
        weight_scales=model.weight_scales,
        activation_values=input_float,
        bias=None,
        group_size=GROUP_SIZE,
        weight_name=DEFAULT_WEIGHT,
        bias_name=None,
    )
    independent = compute_q28_reference(
        result.activation.quantized,
        model.weights,
        result.combined_scale_q28,
    )
    if not np.array_equal(independent, result.output_fixed_q28):
        raise MLPDownProjReferenceError("down_proj 独立 Q28 重算不一致")
    if np.any(result.bias_q28):
        raise MLPDownProjReferenceError("down_proj 无 bias 却产生非零 bias_q28")
    return DownProjectionCase(
        label=label,
        query_position=query_position,
        count=count,
        source_q28=source.copy(),
        activation_int8=result.activation.quantized.astype(np.int8),
        activation_scale=float(result.activation.scale),
        weights=model.weights,
        scales_q28=result.combined_scale_q28.astype(np.uint32),
        bias_q28=np.zeros(M, dtype=np.int64),
        expected_q28=result.output_fixed_q28.astype(np.int64),
        linear_result=result,
    )


def build_fixed_real_cases(*, image_path: Path = DEFAULT_IMAGE) -> list[DownProjectionCase]:
    image = P50Image(image_path)
    image.validate()
    model = load_down_projection_model(image)
    source_cases = build_silu_up_fixed_cases(image_path=image_path)
    return [
        case_from_source_q28(
            model,
            source.output_q28,
            label=f"layer0 MLP down_proj query={source.query_position}, count={source.count}",
            query_position=source.query_position,
            count=source.count,
        )
        for source in source_cases
    ]


def build_upload_payload(case: DownProjectionCase) -> bytes:
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
        raise MLPDownProjReferenceError(
            f"down_proj 上传载荷长度错误：{len(payload)} != {UPLOAD_BYTES}"
        )
    return payload


def verify_upload_payload(case: DownProjectionCase) -> str:
    payload = build_upload_payload(case)
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
        raise MLPDownProjReferenceError("激活载荷往返不一致")
    expected_weight = pack_int4_low_nibble_first(case.weights).reshape(-1)
    if not np.array_equal(decoded_weight, expected_weight):
        raise MLPDownProjReferenceError("权重载荷往返不一致")
    if not np.array_equal(decoded_scales[:, :GROUPS], case.scales_q28):
        raise MLPDownProjReferenceError("scale 载荷往返不一致")
    if np.any(decoded_scales[:, GROUPS:]):
        raise MLPDownProjReferenceError("scale padding 非零")
    if np.any(decoded_bias):
        raise MLPDownProjReferenceError("无 bias 投影的 bias padding 非零")
    return sha256_bytes(payload)


def fixed_manifest(cases: Sequence[DownProjectionCase]) -> dict[str, object]:
    return {
        "format_version": 1,
        "operator": "qwen2_layer0_mlp_down_projection",
        "definition": {
            "input_source": "verified layer0 SiLU(gate) times up output",
            "input_format": "signed int64 Q28 [4864]",
            "activation_quantization": (
                "Q28 to float32, symmetric per-vector INT8 [-127,127], RNE, zero_point=0"
            ),
            "weight_tensor": DEFAULT_WEIGHT,
            "weight_shape": [M, K],
            "weight_storage": "signed INT4 groupwise symmetric",
            "group_size": GROUP_SIZE,
            "groups_per_row": GROUPS,
            "combined_scale": "unsigned UQ4.28 with RNE and saturation",
            "bias": "absent; padded bias_q28 all zero",
            "output_format": "signed int64 Q28 [896]",
            "scale_row_bytes": SCALE_ROW_BYTES,
            "upload_bytes": UPLOAD_BYTES,
            "result_bytes": RESULT_BYTES,
            "max_group_acc": MAX_GROUP_ACC,
            "max_output_magnitude_full_uq4_28": MAX_OUTPUT_MAGNITUDE,
            "int64_safe": MAX_OUTPUT_MAGNITUDE <= INT64_MAX,
            "forbidden_next_operation": "MLP residual is not part of this stage",
        },
        "cases": [
            {
                "label": case.label,
                "query_position": case.query_position,
                "count": case.count,
                "activation_scale": case.activation_scale,
                "activation_clipped_count": case.linear_result.activation.clipped_count,
                "combined_scale_saturated_count": (
                    case.linear_result.combined_scale_saturated_count
                ),
                "max_activation_quantization_error": float(
                    np.max(np.abs(case.linear_result.activation_error))
                ),
                "max_fixed_scale_error": float(
                    np.max(np.abs(case.linear_result.fixed_error))
                ),
                "max_fixed_error_bound": float(
                    np.max(case.linear_result.fixed_error_bound)
                ),
                "range": {
                    "source_q28_min": int(np.min(case.source_q28)),
                    "source_q28_max": int(np.max(case.source_q28)),
                    "output_q28_min": int(np.min(case.expected_q28)),
                    "output_q28_max": int(np.max(case.expected_q28)),
                },
                "sha256": {
                    "source_q28": sha256_array(case.source_q28, "<i8"),
                    "activation_int8": sha256_array(case.activation_int8, np.int8),
                    "packed_weight_int4": sha256_array(
                        pack_int4_low_nibble_first(case.weights), np.uint8
                    ),
                    "combined_scale_uq4_28": sha256_array(case.scales_q28, "<u4"),
                    "bias_q28": sha256_array(case.bias_q28, "<i8"),
                    "output_fixed_q28": sha256_array(case.expected_q28, "<i8"),
                    "upload_payload": verify_upload_payload(case),
                },
                "preview": {
                    "activation_first16_int8": case.activation_int8[:16].tolist(),
                    "output_first8_q28": case.expected_q28[:8].tolist(),
                    "output_last8_q28": case.expected_q28[-8:].tolist(),
                },
            }
            for case in cases
        ],
    }


def validate_manifest(
    cases: Sequence[DownProjectionCase], manifest_path: Path = DEFAULT_MANIFEST
) -> dict[str, object]:
    actual = fixed_manifest(cases)
    expected = json.loads(manifest_path.read_text(encoding="utf-8"))
    if actual != expected:
        raise MLPDownProjReferenceError(f"MLP down_proj 固定清单不一致：{manifest_path}")
    return expected


def make_random_source_q28(rng: np.random.Generator, index: int) -> np.ndarray:
    """生成真实来源、稀疏、RNE、全位宽和 scale 饱和边界输入。"""

    mode = index % 8
    if mode <= 6:
        silu_q10, up_q28 = make_silu_up_random_inputs(rng, mode)
        output, _ = multiply_q10_q28_to_q28(silu_q10, up_q28)
        return output

    raw = rng.integers(0, np.iinfo(np.uint64).max, size=K, dtype=np.uint64)
    source = raw.view(np.int64).copy()
    source[0] = np.iinfo(np.int64).min
    source[1] = np.iinfo(np.int64).max
    return source


def software_stress(
    *, image_path: Path = DEFAULT_IMAGE, rounds: int = 1000, seed: int = DEFAULT_STRESS_SEED
) -> None:
    if rounds <= 0:
        raise MLPDownProjReferenceError("rounds 必须大于 0")
    image = P50Image(image_path)
    image.validate()
    model = load_down_projection_model(image)
    rng = np.random.default_rng(seed)
    for index in range(rounds):
        source = make_random_source_q28(rng, index)
        case = case_from_source_q28(
            model,
            source,
            label=f"MLP down_proj stress {index + 1}/{rounds} mode={index % 8}",
        )
        if index < 8:
            verify_upload_payload(case)
        if not np.any(source) and np.any(case.expected_q28):
            raise MLPDownProjReferenceError("全零输入没有严格输出全零")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="layer0 MLP down_proj 软件参考")
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
            print("layer0 MLP down_proj 固定清单：PASS")
            for item in manifest["cases"]:
                print(
                    f"query={item['query_position']} count={item['count']} "
                    f"output={item['sha256']['output_fixed_q28']}"
                )
        elif args.command == "stress":
            software_stress(image_path=args.image, rounds=args.rounds, seed=args.seed)
            print(f"MLP down_proj 软件压力 PASS：{args.rounds}/{args.rounds}，seed={args.seed}")
        else:  # pragma: no cover
            raise AssertionError(args.command)
        return 0
    except (FileNotFoundError, KeyError, ValueError, OverflowError, RuntimeError) as error:
        print(f"错误：{error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
