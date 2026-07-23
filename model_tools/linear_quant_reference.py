#!/usr/bin/env python3
"""真实 P50 INT4 线性层的量化与定点软件参考。

本模块定义 D2 阶段统一的数据格式：

- 激活：逐向量对称 INT8，范围 ``[-127, 127]``，zero point 为 0；
- 权重：直接使用 ``.p50`` 中的分组 INT4 与 FP16 scale；
- 组合 scale：``activation_scale * weight_scale``，编码为 32 位 UQ4.28；
- 分组累加：每组先完成 INT4×INT8 的 INT32 点积；
- 最终输出：各组乘 UQ4.28 后在带 28 位小数的有符号 INT64 中累加。

所有浮点到整数的转换统一使用 round-to-nearest-even（RNE），随后饱和。
该定义可直接作为后续 FPGA GEMV 分组缩放数据通路的金标准。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .p50_format import P50FormatError, P50Image
except ImportError:
    from p50_format import P50FormatError, P50Image

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

DEFAULT_IMAGE = Path("model_output/yanbo_qwen25_0.5b_int4.p50")
DEFAULT_WEIGHT = "model.layers.0.self_attn.q_proj.weight"
DEFAULT_BIAS = "model.layers.0.self_attn.q_proj.bias"
DEFAULT_ROW_START = 0
DEFAULT_ROW_COUNT = 4
DEFAULT_ACTIVATION_SEED = 20260723

ACTIVATION_QMIN = -127
ACTIVATION_QMAX = 127
COMBINED_SCALE_FRACTION_BITS = 28
COMBINED_SCALE_INTEGER_BITS = 4
COMBINED_SCALE_TOTAL_BITS = 32
COMBINED_SCALE_FACTOR = 1 << COMBINED_SCALE_FRACTION_BITS
COMBINED_SCALE_QMAX = (1 << COMBINED_SCALE_TOTAL_BITS) - 1
OUTPUT_FRACTION_BITS = COMBINED_SCALE_FRACTION_BITS
OUTPUT_FACTOR = 1 << OUTPUT_FRACTION_BITS


class LinearReferenceError(ValueError):
    """表示线性层量化参考的参数或数值不合法。"""


@dataclass(frozen=True)
class ActivationQuantization:
    """一个逐向量对称 INT8 激活量化结果。"""

    source: np.ndarray
    quantized: np.ndarray
    scale: float
    dequantized: np.ndarray
    clipped_count: int


@dataclass(frozen=True)
class LinearReferenceResult:
    """真实 INT4 线性层切片的完整软件参考结果。"""

    weight_name: str
    bias_name: str | None
    row_start: int
    row_count: int
    column_start: int
    column_count: int
    group_size: int
    activation: ActivationQuantization
    weight_quantized: np.ndarray
    weight_scales: np.ndarray
    bias: np.ndarray
    group_accumulators: np.ndarray
    combined_scales: np.ndarray
    combined_scale_q28: np.ndarray
    combined_scale_saturated_count: int
    bias_q28: np.ndarray
    output_p50_float: np.ndarray
    output_quantized_float: np.ndarray
    output_fixed_q28: np.ndarray
    output_fixed_float: np.ndarray
    fixed_error_bound: np.ndarray

    @property
    def activation_error(self) -> np.ndarray:
        return self.output_quantized_float - self.output_p50_float

    @property
    def fixed_error(self) -> np.ndarray:
        return self.output_fixed_float - self.output_quantized_float


def _as_finite_float32(values: np.ndarray | list[float], label: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.size == 0:
        raise LinearReferenceError(f"{label} 不能为空")
    if not np.all(np.isfinite(array)):
        raise LinearReferenceError(f"{label} 包含 NaN 或无穷大")
    return array


def quantize_activation_int8(values: np.ndarray | list[float]) -> ActivationQuantization:
    """按逐向量对称方式量化为 INT8。

    ``scale = max(abs(x)) / 127``；全零向量使用 scale=1，仍可精确表示。
    舍入采用 NumPy ``rint``，即 round-to-nearest-even，随后饱和到
    ``[-127, 127]``。
    """

    source = _as_finite_float32(values, "激活")
    maximum = float(np.max(np.abs(source.astype(np.float64))))
    scale = maximum / ACTIVATION_QMAX if maximum > 0.0 else 1.0
    rounded = np.rint(source.astype(np.float64) / scale)
    clipped = np.clip(rounded, ACTIVATION_QMIN, ACTIVATION_QMAX)
    clipped_count = int(np.count_nonzero(rounded != clipped))
    quantized = clipped.astype(np.int8)
    dequantized = quantized.astype(np.float32) * np.float32(scale)
    return ActivationQuantization(
        source=source.copy(),
        quantized=quantized,
        scale=scale,
        dequantized=dequantized,
        clipped_count=clipped_count,
    )


def quantize_uq4_28(values: np.ndarray) -> tuple[np.ndarray, int]:
    """把非负 scale 编码为 32 位 UQ4.28，返回量化值与饱和元素数。"""

    array = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise LinearReferenceError("组合 scale 包含 NaN 或无穷大")
    if np.any(array < 0.0):
        raise LinearReferenceError("组合 scale 必须非负")
    rounded = np.rint(array * COMBINED_SCALE_FACTOR)
    clipped = np.clip(rounded, 0.0, float(COMBINED_SCALE_QMAX))
    saturated_count = int(np.count_nonzero(rounded != clipped))
    return clipped.astype(np.uint32), saturated_count


def quantize_signed_q28(values: np.ndarray) -> tuple[np.ndarray, int]:
    """把有符号浮点值编码为带 28 位小数的 int64，采用 RNE 与饱和。"""

    array = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise LinearReferenceError("有符号 Q28 输入包含 NaN 或无穷大")
    rounded = np.rint(array * OUTPUT_FACTOR)
    minimum = np.iinfo(np.int64).min
    maximum = np.iinfo(np.int64).max
    output = np.empty(rounded.shape, dtype=np.int64)
    saturated_count = 0
    for index, item in np.ndenumerate(rounded):
        if item <= minimum:
            output[index] = minimum
            saturated_count += 1
        elif item >= maximum:
            output[index] = maximum
            saturated_count += 1
        else:
            output[index] = int(item)
    return output, saturated_count


def make_deterministic_activation(
    length: int, seed: int = DEFAULT_ACTIVATION_SEED
) -> np.ndarray:
    """生成跨平台可复现、仅含二进制精确小数的固定激活向量。

    使用 32 位 LCG 产生高 16 位，再映射到约 ``[-4, 4)``。该向量不依赖
    NumPy 随机数实现，适合作为长期 FPGA 固定测试向量。
    """

    if length <= 0:
        raise LinearReferenceError(f"激活长度必须大于 0：{length}")
    state = int(seed) & 0xFFFFFFFF
    output = np.empty(length, dtype=np.float32)
    for index in range(length):
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        signed = ((state >> 8) & 0xFFFF) - 32768
        output[index] = np.float32(signed / 8192.0)
    return output


def pack_int4_low_nibble_first(values: np.ndarray) -> np.ndarray:
    """把二维 INT4 数组按低半字节在前的规则打包为 uint8。"""

    array = np.asarray(values)
    if array.ndim != 2:
        raise LinearReferenceError("INT4 打包输入必须是二维数组")
    if array.shape[1] % 2:
        raise LinearReferenceError("INT4 打包列数必须为偶数")
    if np.any(array < -8) or np.any(array > 7):
        raise LinearReferenceError("INT4 打包输入超出 [-8, 7]")
    nibble = np.bitwise_and(array.astype(np.int16), 0x0F)
    low = nibble[:, 0::2]
    high = np.left_shift(nibble[:, 1::2], 4)
    return np.bitwise_or(low, high).astype(np.uint8)


def compute_groupwise_linear_reference(
    *,
    weight_quantized: np.ndarray,
    weight_scales: np.ndarray,
    activation_values: np.ndarray,
    bias: np.ndarray | None,
    group_size: int,
    weight_name: str = "linear.weight",
    bias_name: str | None = "linear.bias",
    row_start: int = 0,
    column_start: int = 0,
) -> LinearReferenceResult:
    """根据已提取的 INT4、FP16 scale 与浮点激活计算三条参考路径。"""

    weights = np.asarray(weight_quantized, dtype=np.int8)
    scales = np.asarray(weight_scales, dtype=np.float32)
    if weights.ndim != 2:
        raise LinearReferenceError("weight_quantized 必须是二维数组")
    rows, columns = weights.shape
    if rows <= 0 or columns <= 0:
        raise LinearReferenceError("权重切片不能为空")
    if group_size <= 0 or columns % group_size:
        raise LinearReferenceError(
            f"列数必须是 group_size 的整数倍：columns={columns}, group={group_size}"
        )
    groups = columns // group_size
    if scales.shape != (rows, groups):
        raise LinearReferenceError(
            f"weight_scales 形状错误：{scales.shape}，预期 {(rows, groups)}"
        )
    if np.any(weights < -7) or np.any(weights > 7):
        raise LinearReferenceError("真实 P50 权重必须位于 [-7, 7]")
    if not np.all(np.isfinite(scales)) or np.any(scales <= 0.0):
        raise LinearReferenceError("weight_scales 必须是有限正数")

    activation = quantize_activation_int8(activation_values)
    if activation.source.ndim != 1 or activation.source.size != columns:
        raise LinearReferenceError(
            f"激活形状错误：{activation.source.shape}，预期 ({columns},)"
        )

    if bias is None:
        bias_values = np.zeros(rows, dtype=np.float32)
        resolved_bias_name = None
    else:
        bias_values = _as_finite_float32(bias, "bias").reshape(-1)
        if bias_values.shape != (rows,):
            raise LinearReferenceError(
                f"bias 形状错误：{bias_values.shape}，预期 ({rows},)"
            )
        resolved_bias_name = bias_name

    weights_grouped = weights.astype(np.int32).reshape(rows, groups, group_size)
    activation_grouped = activation.quantized.astype(np.int32).reshape(groups, group_size)
    accumulators64 = np.sum(
        weights_grouped * activation_grouped[np.newaxis, :, :],
        axis=2,
        dtype=np.int64,
    )
    if np.any(accumulators64 < np.iinfo(np.int32).min) or np.any(
        accumulators64 > np.iinfo(np.int32).max
    ):
        raise LinearReferenceError("分组点积超出 INT32")
    accumulators = accumulators64.astype(np.int32)

    expanded_scales = np.repeat(scales, group_size, axis=1)
    dequantized_weights = weights.astype(np.float32) * expanded_scales
    output_p50_float = (
        dequantized_weights.astype(np.float64) @ activation.source.astype(np.float64)
        + bias_values.astype(np.float64)
    )

    combined_scales = scales.astype(np.float64) * activation.scale
    combined_scale_q28, scale_saturated_count = quantize_uq4_28(combined_scales)
    output_quantized_float = (
        np.sum(
            accumulators.astype(np.float64) * combined_scales,
            axis=1,
            dtype=np.float64,
        )
        + bias_values.astype(np.float64)
    )

    bias_q28, bias_saturated_count = quantize_signed_q28(bias_values)
    if bias_saturated_count:
        raise LinearReferenceError("bias 转为有符号 Q28 时发生饱和")
    int64_max = np.iinfo(np.int64).max
    for row in range(rows):
        worst_case_magnitude = abs(int(bias_q28[row]))
        for group in range(groups):
            worst_case_magnitude += abs(int(accumulators[row, group])) * int(
                combined_scale_q28[row, group]
            )
        if worst_case_magnitude > int64_max:
            raise LinearReferenceError(
                f"第 {row} 行 Q28 最坏情况超过有符号 INT64：{worst_case_magnitude}"
            )

    products_q28 = accumulators.astype(np.int64) * combined_scale_q28.astype(np.int64)
    output_fixed_q28 = np.sum(products_q28, axis=1, dtype=np.int64) + bias_q28
    output_fixed_float = output_fixed_q28.astype(np.float64) / OUTPUT_FACTOR

    half_lsb = 0.5 / COMBINED_SCALE_FACTOR
    fixed_error_bound = (
        np.sum(np.abs(accumulators.astype(np.int64)), axis=1).astype(np.float64)
        * half_lsb
        + 0.5 / OUTPUT_FACTOR
    )

    return LinearReferenceResult(
        weight_name=weight_name,
        bias_name=resolved_bias_name,
        row_start=row_start,
        row_count=rows,
        column_start=column_start,
        column_count=columns,
        group_size=group_size,
        activation=activation,
        weight_quantized=weights,
        weight_scales=scales,
        bias=bias_values,
        group_accumulators=accumulators,
        combined_scales=combined_scales,
        combined_scale_q28=combined_scale_q28,
        combined_scale_saturated_count=scale_saturated_count,
        bias_q28=bias_q28,
        output_p50_float=output_p50_float,
        output_quantized_float=output_quantized_float,
        output_fixed_q28=output_fixed_q28,
        output_fixed_float=output_fixed_float,
        fixed_error_bound=fixed_error_bound,
    )


def reference_from_p50(
    image: P50Image,
    *,
    weight_name: str,
    bias_name: str | None,
    row_start: int,
    row_count: int,
    column_start: int,
    column_count: int,
    activation_values: np.ndarray,
) -> LinearReferenceResult:
    """从 P50 镜像提取真实线性层切片并计算参考结果。"""

    if column_start % image.header.group_size:
        raise LinearReferenceError("column_start 必须按量化 group 对齐")
    if column_count % image.header.group_size:
        raise LinearReferenceError("column_count 必须是量化 group 的整数倍")
    block = image.extract_block(
        weight_name,
        row_start,
        row_count,
        column_start,
        column_count,
    )
    if block.quantized is None or block.scales is None:
        raise LinearReferenceError(f"权重不是 INT4 张量：{weight_name}")

    bias_values: np.ndarray | None = None
    if bias_name is not None:
        full_bias = image.read_float16_tensor(bias_name).astype(np.float32).reshape(-1)
        if row_start + row_count > full_bias.size:
            raise LinearReferenceError("bias 切片越界")
        bias_values = full_bias[row_start : row_start + row_count]

    return compute_groupwise_linear_reference(
        weight_quantized=block.quantized,
        weight_scales=block.scales,
        activation_values=activation_values,
        bias=bias_values,
        group_size=image.header.group_size,
        weight_name=weight_name,
        bias_name=bias_name,
        row_start=row_start,
        column_start=column_start,
    )


def _sha256_array(array: np.ndarray, dtype: str | np.dtype | None = None) -> str:
    normalized = np.asarray(array, dtype=dtype) if dtype is not None else np.asarray(array)
    return hashlib.sha256(normalized.tobytes(order="C")).hexdigest()


def result_manifest(result: LinearReferenceResult) -> dict[str, Any]:
    """生成可提交到 Git 的小型固定向量清单。"""

    packed_weights = pack_int4_low_nibble_first(result.weight_quantized)
    max_activation_error = float(np.max(np.abs(result.activation_error)))
    max_fixed_error = float(np.max(np.abs(result.fixed_error)))
    max_fixed_bound = float(np.max(result.fixed_error_bound))
    return {
        "format_version": 1,
        "weight_tensor": result.weight_name,
        "bias_tensor": result.bias_name,
        "slice": {
            "row_start": result.row_start,
            "row_count": result.row_count,
            "column_start": result.column_start,
            "column_count": result.column_count,
            "group_size": result.group_size,
        },
        "activation_format": {
            "scheme": "symmetric_per_vector",
            "dtype": "int8",
            "range": [ACTIVATION_QMIN, ACTIVATION_QMAX],
            "zero_point": 0,
            "rounding": "round_to_nearest_even",
            "scale_formula": "max(abs(x)) / 127; all-zero vector uses 1.0",
            "scale": result.activation.scale,
            "clipped_count": result.activation.clipped_count,
        },
        "fixed_scale_format": {
            "name": "UQ4.28",
            "storage_bits": COMBINED_SCALE_TOTAL_BITS,
            "fraction_bits": COMBINED_SCALE_FRACTION_BITS,
            "rounding": "round_to_nearest_even",
            "saturation_range": [0, COMBINED_SCALE_QMAX],
            "saturated_count": result.combined_scale_saturated_count,
            "formula": "round_rne(activation_scale * weight_scale * 2^28)",
        },
        "output_format": {
            "storage": "signed_int64",
            "fraction_bits": OUTPUT_FRACTION_BITS,
            "formula": "bias_q28 + sum(group_acc_int32 * combined_scale_uq4_28)",
        },
        "expected": {
            "output_p50_float": result.output_p50_float.tolist(),
            "output_quantized_float": result.output_quantized_float.tolist(),
            "output_fixed_q28": result.output_fixed_q28.tolist(),
            "output_fixed_float": result.output_fixed_float.tolist(),
            "fixed_error_bound": result.fixed_error_bound.tolist(),
            "max_activation_quantization_error": max_activation_error,
            "max_fixed_scale_error": max_fixed_error,
            "max_fixed_error_bound": max_fixed_bound,
        },
        "sha256": {
            "activation_float32": _sha256_array(result.activation.source, "<f4"),
            "activation_int8": _sha256_array(result.activation.quantized, np.int8),
            "packed_weight_int4": _sha256_array(packed_weights, np.uint8),
            "weight_scale_float16": _sha256_array(result.weight_scales, "<f2"),
            "group_accumulator_int32": _sha256_array(result.group_accumulators, "<i4"),
            "combined_scale_uq4_28": _sha256_array(result.combined_scale_q28, "<u4"),
            "bias_q28": _sha256_array(result.bias_q28, "<i8"),
            "output_fixed_q28": _sha256_array(result.output_fixed_q28, "<i8"),
        },
    }


def save_npz(result: LinearReferenceResult, path: Path) -> None:
    """保存 FPGA 后续可直接消费的完整固定测试向量。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        weight_tensor=np.asarray(result.weight_name),
        bias_tensor=np.asarray(result.bias_name or ""),
        row_start=np.asarray(result.row_start, dtype=np.int64),
        row_count=np.asarray(result.row_count, dtype=np.int64),
        column_start=np.asarray(result.column_start, dtype=np.int64),
        column_count=np.asarray(result.column_count, dtype=np.int64),
        group_size=np.asarray(result.group_size, dtype=np.int64),
        activation_float32=result.activation.source.astype("<f4"),
        activation_int8=result.activation.quantized.astype(np.int8),
        activation_scale=np.asarray(result.activation.scale, dtype="<f4"),
        packed_weight_int4=pack_int4_low_nibble_first(result.weight_quantized),
        weight_int4=result.weight_quantized.astype(np.int8),
        weight_scale_float16=result.weight_scales.astype("<f2"),
        group_accumulator_int32=result.group_accumulators.astype("<i4"),
        combined_scale_uq4_28=result.combined_scale_q28.astype("<u4"),
        bias_float16=result.bias.astype("<f2"),
        bias_q28=result.bias_q28.astype("<i8"),
        output_p50_float=result.output_p50_float.astype("<f4"),
        output_quantized_float=result.output_quantized_float.astype("<f4"),
        output_fixed_q28=result.output_fixed_q28.astype("<i8"),
        output_fixed_float=result.output_fixed_float.astype("<f4"),
        fixed_error_bound=result.fixed_error_bound.astype("<f4"),
    )


def _print_result(result: LinearReferenceResult) -> None:
    print("=== 真实 q_proj 量化软件参考 ===")
    print(f"权重：{result.weight_name}")
    print(f"bias：{result.bias_name}")
    print(
        f"切片：rows [{result.row_start}, {result.row_start + result.row_count})，"
        f"columns [{result.column_start}, {result.column_start + result.column_count})"
    )
    print(
        "激活：对称逐向量 INT8，范围 [-127,127]，zero_point=0，"
        f"scale={result.activation.scale:.10g}，clipped={result.activation.clipped_count}"
    )
    print(
        "组合 scale：UQ4.28，"
        f"范围=[{result.combined_scales.min():.10g}, {result.combined_scales.max():.10g}]，"
        f"饱和={result.combined_scale_saturated_count}"
    )
    print(f"P50 浮点基线：{result.output_p50_float.tolist()}")
    print(f"量化激活浮点参考：{result.output_quantized_float.tolist()}")
    print(f"定点输出 Q28：{result.output_fixed_q28.tolist()}")
    print(f"定点反量化：{result.output_fixed_float.tolist()}")
    print(
        "激活量化最大绝对误差："
        f"{np.max(np.abs(result.activation_error)):.9g}"
    )
    print(
        "UQ4.28 最大绝对误差："
        f"{np.max(np.abs(result.fixed_error)):.9g}，"
        f"理论上界={np.max(result.fixed_error_bound):.9g}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="真实 P50 INT4 线性层的激活量化、分组 scale 与定点参考"
    )
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--weight", default=DEFAULT_WEIGHT)
    parser.add_argument("--bias", default=DEFAULT_BIAS)
    parser.add_argument("--no-bias", action="store_true")
    parser.add_argument("--row-start", type=int, default=DEFAULT_ROW_START)
    parser.add_argument("--row-count", type=int, default=DEFAULT_ROW_COUNT)
    parser.add_argument("--column-start", type=int, default=0)
    parser.add_argument("--column-count", type=int)
    parser.add_argument("--activation-seed", type=int, default=DEFAULT_ACTIVATION_SEED)
    parser.add_argument("--output", type=Path, help="可选：保存完整 FPGA 测试向量 NPZ")
    parser.add_argument("--manifest", type=Path, help="可选：保存小型 JSON 清单")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        image = P50Image(args.image)
        image.validate()
        entry = image.tensor(args.weight)
        columns = int(entry["shape"][1])
        column_count = args.column_count
        if column_count is None:
            column_count = columns - args.column_start
        activation = make_deterministic_activation(
            column_count, seed=args.activation_seed
        )
        result = reference_from_p50(
            image,
            weight_name=args.weight,
            bias_name=None if args.no_bias else args.bias,
            row_start=args.row_start,
            row_count=args.row_count,
            column_start=args.column_start,
            column_count=column_count,
            activation_values=activation,
        )
        _print_result(result)
        if result.combined_scale_saturated_count:
            raise LinearReferenceError("UQ4.28 组合 scale 发生饱和")
        if np.any(np.abs(result.fixed_error) > result.fixed_error_bound + 1e-12):
            raise LinearReferenceError("定点误差超过理论上界")
        if args.output is not None:
            save_npz(result, args.output)
            print(f"完整测试向量已保存：{args.output}")
        if args.manifest is not None:
            args.manifest.parent.mkdir(parents=True, exist_ok=True)
            args.manifest.write_text(
                json.dumps(result_manifest(result), ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print(f"测试向量清单已保存：{args.manifest}")
        return 0
    except (
        FileNotFoundError,
        KeyError,
        IndexError,
        P50FormatError,
        LinearReferenceError,
    ) as error:
        print(f"错误：{error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
