#!/usr/bin/env python3
"""Qwen2 layer0 RMSNorm 的定点软件金标准与 rsqrt 方案比较。

第一版统一格式：

- 输入 hidden state：signed Q6.10，16 位，范围 [-32, 31.9990234375]；
- gamma：signed Q6.10，16 位；
- 平方和：无符号 40 位足够覆盖 K=896 的最坏情况；
- 均值与 epsilon：Q12.20（由 Q6.10 平方自然得到 20 位小数）；
- rsqrt：unsigned UQ12.20，32 位；
- 输出：signed Q6.10，16 位，RNE 后饱和。

Qwen2 RMSNorm 定义为：

    y_i = gamma_i * x_i * rsqrt(mean(x^2) + epsilon)

本模块同时实现：

1. 精确浮点 rsqrt 量化后的硬件等价定点参考；
2. 256 项归一化尾数中点 LUT；
3. 32 项种子 LUT + 一次 Newton-Raphson。

所有右移、除法和浮点转整数均使用 round-to-nearest-even（RNE）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

try:
    from .p50_format import P50FormatError, P50Image
except ImportError:
    from p50_format import P50FormatError, P50Image

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

DEFAULT_IMAGE = Path("model_output/yanbo_qwen25_0.5b_int4.p50")
DEFAULT_GAMMA = "model.layers.0.input_layernorm.weight"
DEFAULT_LENGTH = 896
DEFAULT_EPSILON = 1e-6
DEFAULT_INPUT_SEED = 20260726

ACTIVATION_TOTAL_BITS = 16
ACTIVATION_FRACTION_BITS = 10
ACTIVATION_FACTOR = 1 << ACTIVATION_FRACTION_BITS
ACTIVATION_QMIN = -(1 << (ACTIVATION_TOTAL_BITS - 1))
ACTIVATION_QMAX = (1 << (ACTIVATION_TOTAL_BITS - 1)) - 1

GAMMA_TOTAL_BITS = 16
GAMMA_FRACTION_BITS = 10
GAMMA_FACTOR = 1 << GAMMA_FRACTION_BITS
GAMMA_QMIN = -(1 << (GAMMA_TOTAL_BITS - 1))
GAMMA_QMAX = (1 << (GAMMA_TOTAL_BITS - 1)) - 1

OUTPUT_TOTAL_BITS = 16
OUTPUT_FRACTION_BITS = 10
OUTPUT_FACTOR = 1 << OUTPUT_FRACTION_BITS
OUTPUT_QMIN = -(1 << (OUTPUT_TOTAL_BITS - 1))
OUTPUT_QMAX = (1 << (OUTPUT_TOTAL_BITS - 1)) - 1

VARIANCE_FRACTION_BITS = 2 * ACTIVATION_FRACTION_BITS
VARIANCE_FACTOR = 1 << VARIANCE_FRACTION_BITS

RSQRT_TOTAL_BITS = 32
RSQRT_FRACTION_BITS = 20
RSQRT_FACTOR = 1 << RSQRT_FRACTION_BITS
RSQRT_QMAX = (1 << RSQRT_TOTAL_BITS) - 1

MANTISSA_FRACTION_BITS = 30
MANTISSA_ONE = 1 << MANTISSA_FRACTION_BITS
INV_SQRT2_Q20 = int(np.rint((1.0 / math.sqrt(2.0)) * RSQRT_FACTOR))

LUT_ONLY_INDEX_BITS = 8
NR_SEED_INDEX_BITS = 5
SELECTED_RSQRT_SCHEME = "lut256_midpoint"


class RMSNormReferenceError(ValueError):
    """表示 RMSNorm 定点参考的参数或数值不合法。"""


@dataclass(frozen=True)
class QuantizedVector:
    """浮点向量及其有符号定点量化结果。"""

    source: np.ndarray
    quantized: np.ndarray
    dequantized: np.ndarray
    clipped_count: int
    fraction_bits: int


@dataclass(frozen=True)
class RMSNormReferenceResult:
    """一个完整 RMSNorm 固定向量的多路径参考结果。"""

    gamma_name: str
    length: int
    epsilon: float
    epsilon_q20: int
    activation: QuantizedVector
    gamma: QuantizedVector
    sum_squares: int
    mean_square_q20: int
    variance_q20: int
    exact_rsqrt_q20: int
    lut_rsqrt_q20: int
    nr_rsqrt_q20: int
    output_float: np.ndarray
    output_quantized_float: np.ndarray
    output_exact_q10: np.ndarray
    output_lut_q10: np.ndarray
    output_nr_q10: np.ndarray
    exact_output_saturated_count: int
    lut_output_saturated_count: int
    nr_output_saturated_count: int

    @property
    def output_exact_float(self) -> np.ndarray:
        return self.output_exact_q10.astype(np.float64) / OUTPUT_FACTOR

    @property
    def output_lut_float(self) -> np.ndarray:
        return self.output_lut_q10.astype(np.float64) / OUTPUT_FACTOR

    @property
    def output_nr_float(self) -> np.ndarray:
        return self.output_nr_q10.astype(np.float64) / OUTPUT_FACTOR


@dataclass(frozen=True)
class SchemeMetrics:
    """一种 rsqrt 方案相对精确定点路径的误差统计。"""

    name: str
    rsqrt_q20: int
    rsqrt_absolute_error: float
    rsqrt_relative_error: float
    output_max_abs_error: float
    output_mean_abs_error: float
    output_mismatch_count: int
    output_saturated_count: int
    lut_entries: int
    lut_bits_per_entry: int
    estimated_lut_bits: int
    multiplier_note: str
    normalized_latency_note: str


def _as_finite_1d(values: np.ndarray | Iterable[float], label: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32).reshape(-1)
    if array.size == 0:
        raise RMSNormReferenceError(f"{label} 不能为空")
    if not np.all(np.isfinite(array)):
        raise RMSNormReferenceError(f"{label} 包含 NaN 或无穷大")
    return array


def _round_shift_unsigned(value: int, shift: int) -> int:
    """对非负整数右移并执行 RNE；shift<=0 时左移。"""

    if value < 0:
        raise RMSNormReferenceError("无符号 RNE 输入不能为负")
    if shift <= 0:
        return value << (-shift)
    quotient, remainder = divmod(value, 1 << shift)
    half = 1 << (shift - 1)
    if remainder > half or (remainder == half and (quotient & 1)):
        quotient += 1
    return quotient


def _round_shift_signed_scalar(value: int, shift: int) -> int:
    """对有符号整数右移并执行对称 RNE；shift<=0 时左移。"""

    if shift <= 0:
        return value << (-shift)
    magnitude = _round_shift_unsigned(abs(value), shift)
    return -magnitude if value < 0 else magnitude


def _round_shift_signed_array(values: np.ndarray, shift: int) -> np.ndarray:
    """对 int64 数组执行有符号 RNE 右移。"""

    array = np.asarray(values, dtype=np.int64)
    if shift <= 0:
        return np.left_shift(array, -shift)
    magnitude = np.abs(array)
    divisor = np.int64(1 << shift)
    quotient = magnitude // divisor
    remainder = magnitude % divisor
    half = np.int64(1 << (shift - 1))
    increment = (remainder > half) | ((remainder == half) & ((quotient & 1) != 0))
    rounded = quotient + increment.astype(np.int64)
    return np.where(array < 0, -rounded, rounded)


def _round_div_unsigned(value: int, divisor: int) -> int:
    """对非负整数除法执行 RNE。"""

    if value < 0 or divisor <= 0:
        raise RMSNormReferenceError("RNE 除法要求非负被除数和正除数")
    quotient, remainder = divmod(value, divisor)
    doubled = remainder * 2
    if doubled > divisor or (doubled == divisor and (quotient & 1)):
        quotient += 1
    return quotient


def _quantize_signed(
    values: np.ndarray | Iterable[float],
    *,
    fraction_bits: int,
    qmin: int,
    qmax: int,
    dtype: np.dtype,
    label: str,
) -> QuantizedVector:
    source = _as_finite_1d(values, label)
    factor = 1 << fraction_bits
    rounded = np.rint(source.astype(np.float64) * factor)
    clipped = np.clip(rounded, qmin, qmax)
    clipped_count = int(np.count_nonzero(rounded != clipped))
    quantized = clipped.astype(dtype)
    dequantized = quantized.astype(np.float64) / factor
    return QuantizedVector(
        source=source.copy(),
        quantized=quantized,
        dequantized=dequantized,
        clipped_count=clipped_count,
        fraction_bits=fraction_bits,
    )


def quantize_activation_q6_10(
    values: np.ndarray | Iterable[float],
) -> QuantizedVector:
    """把 hidden state 量化为 signed Q6.10。"""

    return _quantize_signed(
        values,
        fraction_bits=ACTIVATION_FRACTION_BITS,
        qmin=ACTIVATION_QMIN,
        qmax=ACTIVATION_QMAX,
        dtype=np.int16,
        label="RMSNorm 输入",
    )


def quantize_gamma_q6_10(values: np.ndarray | Iterable[float]) -> QuantizedVector:
    """把真实 FP16 gamma 量化为 signed Q6.10。"""

    return _quantize_signed(
        values,
        fraction_bits=GAMMA_FRACTION_BITS,
        qmin=GAMMA_QMIN,
        qmax=GAMMA_QMAX,
        dtype=np.int16,
        label="RMSNorm gamma",
    )


def quantize_epsilon_q20(epsilon: float) -> int:
    """把 epsilon 量化为非负 Q12.20。"""

    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise RMSNormReferenceError("epsilon 必须是有限正数")
    quantized = int(np.rint(epsilon * VARIANCE_FACTOR))
    if quantized <= 0:
        raise RMSNormReferenceError("epsilon 在 Q12.20 中量化为 0")
    return quantized


def make_deterministic_input(
    length: int = DEFAULT_LENGTH, seed: int = DEFAULT_INPUT_SEED
) -> np.ndarray:
    """生成跨平台可复现、范围约为 [-4,4) 的固定输入。"""

    if length <= 0:
        raise RMSNormReferenceError("输入长度必须大于 0")
    state = int(seed) & 0xFFFFFFFF
    output = np.empty(length, dtype=np.float32)
    for index in range(length):
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        signed = ((state >> 8) & 0xFFFF) - 32768
        output[index] = np.float32(signed / 8192.0)
    return output


def build_rsqrt_lut(index_bits: int) -> np.ndarray:
    """生成 m∈[1,2) 的中点采样 ``1/sqrt(m)`` UQ12.20 LUT。"""

    if index_bits <= 0 or index_bits > 16:
        raise RMSNormReferenceError("LUT 索引位数必须位于 [1,16]")
    entries = 1 << index_bits
    indices = np.arange(entries, dtype=np.float64)
    mantissas = 1.0 + (indices + 0.5) / entries
    values = np.rint((1.0 / np.sqrt(mantissas)) * RSQRT_FACTOR)
    return np.clip(values, 0, RSQRT_QMAX).astype(np.uint32)


_LUT_CACHE: dict[int, np.ndarray] = {}


def _cached_lut(index_bits: int) -> np.ndarray:
    lut = _LUT_CACHE.get(index_bits)
    if lut is None:
        lut = build_rsqrt_lut(index_bits)
        _LUT_CACHE[index_bits] = lut
    return lut


def _normalized_mantissa_q30(variance_q20: int) -> tuple[int, int]:
    """返回 Q1.30 尾数 m∈[1,2) 和二进制指数 p，使 variance=m*2^p。"""

    if variance_q20 <= 0:
        raise RMSNormReferenceError("variance_q20 必须大于 0")
    leading_bit = variance_q20.bit_length() - 1
    exponent = leading_bit - VARIANCE_FRACTION_BITS
    shift = MANTISSA_FRACTION_BITS - leading_bit
    if shift >= 0:
        mantissa_q30 = variance_q20 << shift
    else:
        mantissa_q30 = _round_shift_unsigned(variance_q20, -shift)
    if mantissa_q30 < MANTISSA_ONE:
        mantissa_q30 = MANTISSA_ONE
    if mantissa_q30 >= (MANTISSA_ONE << 1):
        mantissa_q30 = (MANTISSA_ONE << 1) - 1
    return mantissa_q30, exponent


def _lut_seed_for_mantissa(mantissa_q30: int, index_bits: int) -> int:
    fractional = mantissa_q30 - MANTISSA_ONE
    shift = MANTISSA_FRACTION_BITS - index_bits
    index = fractional >> shift
    entries = 1 << index_bits
    if index >= entries:
        index = entries - 1
    return int(_cached_lut(index_bits)[index])


def _apply_binary_exponent(normalized_rsqrt_q20: int, exponent: int) -> int:
    """把 ``1/sqrt(m)`` 扩展为 ``1/sqrt(m*2^exponent)``。"""

    value = normalized_rsqrt_q20
    half_exponent = exponent // 2
    odd = exponent - 2 * half_exponent
    if odd:
        value = _round_shift_unsigned(value * INV_SQRT2_Q20, RSQRT_FRACTION_BITS)
    if half_exponent > 0:
        value = _round_shift_unsigned(value, half_exponent)
    elif half_exponent < 0:
        value <<= -half_exponent
    return min(value, RSQRT_QMAX)


def rsqrt_lut_q20(variance_q20: int, index_bits: int = LUT_ONLY_INDEX_BITS) -> int:
    """使用归一化尾数中点 LUT 近似 rsqrt。"""

    mantissa_q30, exponent = _normalized_mantissa_q30(variance_q20)
    seed = _lut_seed_for_mantissa(mantissa_q30, index_bits)
    return _apply_binary_exponent(seed, exponent)


def rsqrt_newton_q20(
    variance_q20: int,
    seed_index_bits: int = NR_SEED_INDEX_BITS,
    iterations: int = 1,
) -> int:
    """使用小 LUT 种子和定点 Newton-Raphson 近似 rsqrt。"""

    if iterations <= 0 or iterations > 3:
        raise RMSNormReferenceError("Newton-Raphson 迭代次数必须位于 [1,3]")
    mantissa_q30, exponent = _normalized_mantissa_q30(variance_q20)
    value = _lut_seed_for_mantissa(mantissa_q30, seed_index_bits)
    one_and_half_q20 = 3 << (RSQRT_FRACTION_BITS - 1)
    for _ in range(iterations):
        product = mantissa_q30 * value * value
        m_y_squared_q20 = _round_shift_unsigned(
            product, MANTISSA_FRACTION_BITS + RSQRT_FRACTION_BITS
        )
        half_m_y_squared_q20 = _round_shift_unsigned(m_y_squared_q20, 1)
        correction_q20 = one_and_half_q20 - half_m_y_squared_q20
        if correction_q20 <= 0:
            raise RMSNormReferenceError("Newton-Raphson 修正项非正")
        value = _round_shift_unsigned(
            value * correction_q20, RSQRT_FRACTION_BITS
        )
    return _apply_binary_exponent(value, exponent)


def rsqrt_exact_q20(variance_q20: int) -> int:
    """用浮点精确 rsqrt 生成 UQ12.20 参考值。"""

    if variance_q20 <= 0:
        raise RMSNormReferenceError("variance_q20 必须大于 0")
    variance = variance_q20 / VARIANCE_FACTOR
    quantized = int(np.rint((1.0 / math.sqrt(variance)) * RSQRT_FACTOR))
    return min(max(quantized, 0), RSQRT_QMAX)


def _apply_output_path(
    activation_q10: np.ndarray,
    gamma_q10: np.ndarray,
    rsqrt_q20_value: int,
) -> tuple[np.ndarray, int]:
    """执行 x*rsqrt*gamma，返回 signed Q6.10 输出和饱和数。"""

    activation_i64 = np.asarray(activation_q10, dtype=np.int64)
    gamma_i64 = np.asarray(gamma_q10, dtype=np.int64)
    normalized_q10 = _round_shift_signed_array(
        activation_i64 * np.int64(rsqrt_q20_value), RSQRT_FRACTION_BITS
    )
    output_unclipped = _round_shift_signed_array(
        normalized_q10 * gamma_i64, GAMMA_FRACTION_BITS
    )
    clipped = np.clip(output_unclipped, OUTPUT_QMIN, OUTPUT_QMAX)
    saturated_count = int(np.count_nonzero(output_unclipped != clipped))
    return clipped.astype(np.int16), saturated_count


def compute_rmsnorm_reference(
    *,
    activation_values: np.ndarray | Iterable[float],
    gamma_values: np.ndarray | Iterable[float],
    epsilon: float = DEFAULT_EPSILON,
    gamma_name: str = DEFAULT_GAMMA,
) -> RMSNormReferenceResult:
    """计算 Qwen2 RMSNorm 的浮点、精确定点、LUT 和 NR 四条路径。"""

    activation = quantize_activation_q6_10(activation_values)
    gamma = quantize_gamma_q6_10(gamma_values)
    if activation.source.shape != gamma.source.shape:
        raise RMSNormReferenceError(
            f"输入与 gamma 长度不同：{activation.source.size} != {gamma.source.size}"
        )
    length = int(activation.source.size)
    epsilon_q20 = quantize_epsilon_q20(epsilon)

    activation_i64 = activation.quantized.astype(np.int64)
    sum_squares = int(np.sum(activation_i64 * activation_i64, dtype=np.int64))
    mean_square_q20 = _round_div_unsigned(sum_squares, length)
    variance_q20 = mean_square_q20 + epsilon_q20

    exact_rsqrt = rsqrt_exact_q20(variance_q20)
    lut_rsqrt = rsqrt_lut_q20(variance_q20, LUT_ONLY_INDEX_BITS)
    nr_rsqrt = rsqrt_newton_q20(variance_q20, NR_SEED_INDEX_BITS, 1)

    exact_output, exact_saturated = _apply_output_path(
        activation.quantized, gamma.quantized, exact_rsqrt
    )
    lut_output, lut_saturated = _apply_output_path(
        activation.quantized, gamma.quantized, lut_rsqrt
    )
    nr_output, nr_saturated = _apply_output_path(
        activation.quantized, gamma.quantized, nr_rsqrt
    )

    source64 = activation.source.astype(np.float64)
    gamma64 = gamma.source.astype(np.float64)
    variance_float = float(np.mean(source64 * source64, dtype=np.float64) + epsilon)
    output_float = source64 * (1.0 / math.sqrt(variance_float)) * gamma64

    activation_dequantized = activation.dequantized.astype(np.float64)
    gamma_dequantized = gamma.dequantized.astype(np.float64)
    fixed_epsilon = epsilon_q20 / VARIANCE_FACTOR
    quantized_variance = float(
        np.mean(activation_dequantized * activation_dequantized, dtype=np.float64)
        + fixed_epsilon
    )
    output_quantized_float = (
        activation_dequantized
        * (1.0 / math.sqrt(quantized_variance))
        * gamma_dequantized
    )

    return RMSNormReferenceResult(
        gamma_name=gamma_name,
        length=length,
        epsilon=epsilon,
        epsilon_q20=epsilon_q20,
        activation=activation,
        gamma=gamma,
        sum_squares=sum_squares,
        mean_square_q20=mean_square_q20,
        variance_q20=variance_q20,
        exact_rsqrt_q20=exact_rsqrt,
        lut_rsqrt_q20=lut_rsqrt,
        nr_rsqrt_q20=nr_rsqrt,
        output_float=output_float,
        output_quantized_float=output_quantized_float,
        output_exact_q10=exact_output,
        output_lut_q10=lut_output,
        output_nr_q10=nr_output,
        exact_output_saturated_count=exact_saturated,
        lut_output_saturated_count=lut_saturated,
        nr_output_saturated_count=nr_saturated,
    )


def reference_from_p50(
    image: P50Image,
    *,
    activation_values: np.ndarray | Iterable[float],
    gamma_name: str = DEFAULT_GAMMA,
    epsilon: float | None = None,
) -> RMSNormReferenceResult:
    """从真实 P50 镜像提取 gamma 和 epsilon 并计算参考。"""

    gamma = image.read_float16_tensor(gamma_name).astype(np.float32).reshape(-1)
    resolved_epsilon = epsilon
    if resolved_epsilon is None:
        resolved_epsilon = float(image.metadata["model"]["rms_norm_eps"])
    return compute_rmsnorm_reference(
        activation_values=activation_values,
        gamma_values=gamma,
        epsilon=resolved_epsilon,
        gamma_name=gamma_name,
    )


def scheme_metrics(result: RMSNormReferenceResult) -> list[SchemeMetrics]:
    """返回 LUT-only 与 NR 两种方案的固定向量误差和资源估算。"""

    exact_rsqrt_float = result.exact_rsqrt_q20 / RSQRT_FACTOR
    exact_output = result.output_exact_float
    schemes = [
        (
            "lut256_midpoint",
            result.lut_rsqrt_q20,
            result.output_lut_float,
            result.output_lut_q10,
            result.lut_output_saturated_count,
            1 << LUT_ONLY_INDEX_BITS,
            "rsqrt 本体无需乘法器；仅指数奇偶校正需要常数乘法",
            "归一化 + 1 次 ROM + 指数校正",
        ),
        (
            "lut32_newton1",
            result.nr_rsqrt_q20,
            result.output_nr_float,
            result.output_nr_q10,
            result.nr_output_saturated_count,
            1 << NR_SEED_INDEX_BITS,
            "一次 NR 需要 y²、m·y²、y·修正项，可流水复用乘法器",
            "归一化 + 1 次小 ROM + 1 次 NR + 指数校正",
        ),
    ]
    metrics: list[SchemeMetrics] = []
    for (
        name,
        rsqrt_q20_value,
        output_float,
        output_q10,
        saturated_count,
        entries,
        multiplier_note,
        latency_note,
    ) in schemes:
        rsqrt_float = rsqrt_q20_value / RSQRT_FACTOR
        absolute = abs(rsqrt_float - exact_rsqrt_float)
        relative = absolute / exact_rsqrt_float if exact_rsqrt_float else 0.0
        output_error = np.abs(output_float - exact_output)
        metrics.append(
            SchemeMetrics(
                name=name,
                rsqrt_q20=rsqrt_q20_value,
                rsqrt_absolute_error=float(absolute),
                rsqrt_relative_error=float(relative),
                output_max_abs_error=float(np.max(output_error)),
                output_mean_abs_error=float(np.mean(output_error)),
                output_mismatch_count=int(
                    np.count_nonzero(output_q10 != result.output_exact_q10)
                ),
                output_saturated_count=saturated_count,
                lut_entries=entries,
                lut_bits_per_entry=RSQRT_TOTAL_BITS,
                estimated_lut_bits=entries * RSQRT_TOTAL_BITS,
                multiplier_note=multiplier_note,
                normalized_latency_note=latency_note,
            )
        )
    return metrics


def _sha256_array(array: np.ndarray, dtype: str | np.dtype | None = None) -> str:
    normalized = np.asarray(array, dtype=dtype) if dtype is not None else np.asarray(array)
    return hashlib.sha256(normalized.tobytes(order="C")).hexdigest()


def result_manifest(result: RMSNormReferenceResult) -> dict[str, Any]:
    """生成可提交到 Git 的固定向量 JSON 清单。"""

    metrics = scheme_metrics(result)
    return {
        "format_version": 1,
        "operator": "qwen2_rmsnorm",
        "gamma_tensor": result.gamma_name,
        "length": result.length,
        "formula": "gamma * x * rsqrt(mean(x^2) + epsilon)",
        "epsilon": {
            "float": result.epsilon,
            "q12_20": result.epsilon_q20,
            "quantized_float": result.epsilon_q20 / VARIANCE_FACTOR,
        },
        "fixed_formats": {
            "activation": "signed Q6.10 int16",
            "gamma": "signed Q6.10 int16",
            "sum_squares": "unsigned 40-bit integer with 20 fractional bits",
            "mean_square": "unsigned Q12.20",
            "rsqrt": "unsigned UQ12.20 uint32",
            "output": "signed Q6.10 int16",
            "rounding": "round_to_nearest_even",
            "saturation": "explicit signed/unsigned storage saturation",
        },
        "selected_rsqrt_scheme": SELECTED_RSQRT_SCHEME,
        "fixed_scalar_values": {
            "sum_squares": result.sum_squares,
            "mean_square_q20": result.mean_square_q20,
            "variance_q20": result.variance_q20,
            "exact_rsqrt_q20": result.exact_rsqrt_q20,
            "lut_rsqrt_q20": result.lut_rsqrt_q20,
            "nr_rsqrt_q20": result.nr_rsqrt_q20,
        },
        "quantization": {
            "activation_clipped_count": result.activation.clipped_count,
            "gamma_clipped_count": result.gamma.clipped_count,
            "exact_output_saturated_count": result.exact_output_saturated_count,
            "lut_output_saturated_count": result.lut_output_saturated_count,
            "nr_output_saturated_count": result.nr_output_saturated_count,
        },
        "errors": {
            "input_and_gamma_quantization_max_abs": float(
                np.max(np.abs(result.output_quantized_float - result.output_float))
            ),
            "exact_fixed_vs_quantized_max_abs": float(
                np.max(np.abs(result.output_exact_float - result.output_quantized_float))
            ),
            "schemes": [metric.__dict__ for metric in metrics],
        },
        "preview": {
            "output_float_first16": result.output_float[:16].tolist(),
            "output_exact_q10_first16": result.output_exact_q10[:16].tolist(),
            "output_lut_q10_first16": result.output_lut_q10[:16].tolist(),
            "output_nr_q10_first16": result.output_nr_q10[:16].tolist(),
        },
        "sha256": {
            "activation_float32": _sha256_array(result.activation.source, "<f4"),
            "activation_q6_10": _sha256_array(result.activation.quantized, "<i2"),
            "gamma_float16": _sha256_array(result.gamma.source, "<f2"),
            "gamma_q6_10": _sha256_array(result.gamma.quantized, "<i2"),
            "output_exact_q6_10": _sha256_array(result.output_exact_q10, "<i2"),
            "output_lut_q6_10": _sha256_array(result.output_lut_q10, "<i2"),
            "output_nr_q6_10": _sha256_array(result.output_nr_q10, "<i2"),
            "lut256_uq12_20": _sha256_array(
                build_rsqrt_lut(LUT_ONLY_INDEX_BITS), "<u4"
            ),
            "lut32_uq12_20": _sha256_array(
                build_rsqrt_lut(NR_SEED_INDEX_BITS), "<u4"
            ),
        },
    }


def save_npz(result: RMSNormReferenceResult, path: Path) -> None:
    """保存后续 FPGA 上位机可直接消费的完整固定向量。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        gamma_tensor=np.asarray(result.gamma_name),
        length=np.asarray(result.length, dtype="<u4"),
        epsilon_float=np.asarray(result.epsilon, dtype="<f4"),
        epsilon_q20=np.asarray(result.epsilon_q20, dtype="<u4"),
        activation_float32=result.activation.source.astype("<f4"),
        activation_q6_10=result.activation.quantized.astype("<i2"),
        gamma_float16=result.gamma.source.astype("<f2"),
        gamma_q6_10=result.gamma.quantized.astype("<i2"),
        sum_squares=np.asarray(result.sum_squares, dtype="<u8"),
        mean_square_q20=np.asarray(result.mean_square_q20, dtype="<u8"),
        variance_q20=np.asarray(result.variance_q20, dtype="<u8"),
        exact_rsqrt_q20=np.asarray(result.exact_rsqrt_q20, dtype="<u4"),
        lut_rsqrt_q20=np.asarray(result.lut_rsqrt_q20, dtype="<u4"),
        nr_rsqrt_q20=np.asarray(result.nr_rsqrt_q20, dtype="<u4"),
        output_float=result.output_float.astype("<f4"),
        output_quantized_float=result.output_quantized_float.astype("<f4"),
        output_exact_q6_10=result.output_exact_q10.astype("<i2"),
        output_lut_q6_10=result.output_lut_q10.astype("<i2"),
        output_nr_q6_10=result.output_nr_q10.astype("<i2"),
        lut256_uq12_20=build_rsqrt_lut(LUT_ONLY_INDEX_BITS).astype("<u4"),
        lut32_uq12_20=build_rsqrt_lut(NR_SEED_INDEX_BITS).astype("<u4"),
    )


def _print_result(result: RMSNormReferenceResult) -> None:
    print("=== layer0 input_layernorm 定点软件参考 ===")
    print(f"gamma：{result.gamma_name}")
    print(f"K={result.length}，epsilon={result.epsilon:g} -> Q20={result.epsilon_q20}")
    print(
        "格式：input/gamma/output=signed Q6.10 int16，"
        "mean/epsilon=Q12.20，rsqrt=UQ12.20"
    )
    print(
        f"输入饱和={result.activation.clipped_count}，gamma饱和={result.gamma.clipped_count}，"
        f"sum_sq={result.sum_squares}，mean_q20={result.mean_square_q20}，"
        f"variance_q20={result.variance_q20}"
    )
    print(
        f"rsqrt Q20：exact={result.exact_rsqrt_q20}，"
        f"LUT256={result.lut_rsqrt_q20}，NR1={result.nr_rsqrt_q20}"
    )
    print(f"精确定点输出前16项：{result.output_exact_q10[:16].tolist()}")
    for metric in scheme_metrics(result):
        print(
            f"{metric.name}: rsqrt相对误差={metric.rsqrt_relative_error:.9g}，"
            f"输出最大绝对误差={metric.output_max_abs_error:.9g}，"
            f"Q10不一致={metric.output_mismatch_count}/{result.length}，"
            f"LUT={metric.lut_entries}×{metric.lut_bits_per_entry}="
            f"{metric.estimated_lut_bits} bit"
        )
    print(
        "结论：第一版选择 LUT256 中点方案，避免 NR 迭代乘法链；"
        "NR1 保留为后续精度/资源折中选项。"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Qwen2 layer0 RMSNorm 定点金标准与 LUT/NR rsqrt 比较"
    )
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--gamma", default=DEFAULT_GAMMA)
    parser.add_argument("--input-seed", type=int, default=DEFAULT_INPUT_SEED)
    parser.add_argument("--output", type=Path, help="可选：保存完整 NPZ")
    parser.add_argument("--manifest", type=Path, help="可选：保存 JSON 清单")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        image = P50Image(args.image)
        image.validate()
        gamma = image.read_float16_tensor(args.gamma).reshape(-1)
        activation = make_deterministic_input(gamma.size, args.input_seed)
        result = reference_from_p50(
            image,
            activation_values=activation,
            gamma_name=args.gamma,
        )
        _print_result(result)
        if result.activation.clipped_count or result.gamma.clipped_count:
            raise RMSNormReferenceError("固定向量输入或 gamma 发生饱和")
        if result.exact_output_saturated_count:
            raise RMSNormReferenceError("精确定点输出发生饱和")
        if args.output is not None:
            save_npz(result, args.output)
            print(f"完整测试向量已保存：{args.output}")
        if args.manifest is not None:
            args.manifest.parent.mkdir(parents=True, exist_ok=True)
            args.manifest.write_text(
                json.dumps(result_manifest(result), ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print(f"固定向量清单已保存：{args.manifest}")
        return 0
    except (
        FileNotFoundError,
        KeyError,
        IndexError,
        P50FormatError,
        RMSNormReferenceError,
    ) as error:
        print(f"错误：{error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
