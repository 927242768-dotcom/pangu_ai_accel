#!/usr/bin/env python3
"""G2 运行时 Linear 激活量化与 combined scale 的精确整数参考。

此前独立 Linear 工程由主机完成：

1. 中间激活转为 float32；
2. 逐向量对称量化为 INT8 ``[-127, 127]``；
3. ``activation_scale * FP16 weight_scale`` 量化为 unsigned UQ4.28。

完整 Transformer Block 的中间激活由 FPGA 自身产生，不能继续依赖主机预先
生成这些数据。本模块把既有 NumPy/float32 定义改写为可由 RTL 实现的整数/二进制
有理数规格，同时逐位检查结果仍与已经真实上板验证的 G1/F6 软件定义一致。

两类输入：

- Q6.10：input RMSNorm 和 post-attention RMSNorm 的 signed int16；
- Q28：Attention 拼接和 SiLU(gate)*up 的 signed int64，先按原定义舍入为
  IEEE-754 binary32，再进行对称 INT8 量化。

weight scale 来自 P50 的 FP16，读取后为 float32；它本身是精确的二进制有理数。
本模块不在核心计算中使用近似除法，而是用整数商、余数和 ties-to-even 实现 RNE。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

ACTIVATION_QMAX = 127
UQ4_28_MAX = (1 << 32) - 1
Q28_FACTOR = 1 << 28
Q10_FACTOR = 1 << 10


class RuntimeLinearQuantError(ValueError):
    """表示运行时激活量化或 combined scale 构建输入不合法。"""


@dataclass(frozen=True)
class Binary32Components:
    """IEEE binary32 的精确整数表示：``sign * mantissa * 2**exponent``。"""

    sign: np.ndarray
    mantissa: np.ndarray
    exponent: np.ndarray


@dataclass(frozen=True)
class RuntimeLinearQuantization:
    source_kind: str
    activation_int8: np.ndarray
    combined_scale_q28: np.ndarray
    max_abs_float32_bits: int
    activation_scale: float
    saturated_scale_count: int


def round_div_rne_nonnegative(numerator: int, denominator: int) -> int:
    """对非负整数分数执行 round-to-nearest-even。"""

    if numerator < 0:
        raise RuntimeLinearQuantError("RNE numerator 必须非负")
    if denominator <= 0:
        raise RuntimeLinearQuantError("RNE denominator 必须为正")
    quotient, remainder = divmod(int(numerator), int(denominator))
    doubled = remainder << 1
    if doubled > denominator or (doubled == denominator and (quotient & 1)):
        quotient += 1
    return quotient


def round_binary_ratio_rne(numerator: int, denominator: int, shift: int) -> int:
    """计算 ``round_rne(numerator * 2**shift / denominator)``。"""

    if shift >= 0:
        return round_div_rne_nonnegative(int(numerator) << shift, denominator)
    return round_div_rne_nonnegative(numerator, int(denominator) << (-shift))


def binary32_components(values: np.ndarray | Sequence[float]) -> Binary32Components:
    """把有限 float32 数组拆为精确 mantissa/exponent，不改变任何数值。"""

    array = np.asarray(values, dtype=np.float32)
    if not np.all(np.isfinite(array)):
        raise RuntimeLinearQuantError("binary32 输入包含 NaN 或无穷大")
    bits = array.view(np.uint32)
    signs = np.where((bits >> 31) != 0, -1, 1).astype(np.int8)
    exponent_bits = ((bits >> 23) & 0xFF).astype(np.int32)
    fraction_bits = (bits & 0x7FFFFF).astype(np.uint32)
    normal = exponent_bits != 0
    mantissa = np.where(normal, fraction_bits | 0x800000, fraction_bits).astype(np.uint32)
    exponent = np.where(normal, exponent_bits - 127 - 23, -126 - 23).astype(np.int16)
    zero = mantissa == 0
    signs = signs.copy()
    exponent = exponent.copy()
    signs[zero] = 1
    exponent[zero] = 0
    return Binary32Components(sign=signs, mantissa=mantissa, exponent=exponent)


def _float32_bits(value: np.float32 | float) -> int:
    return int(np.asarray(np.float32(value)).view(np.uint32))


def _positive_component(value: np.float32 | float) -> tuple[int, int]:
    array = np.asarray([value], dtype=np.float32)
    if not np.isfinite(array[0]) or array[0] < 0:
        raise RuntimeLinearQuantError("要求有限非负 binary32")
    components = binary32_components(array)
    return int(components.mantissa[0]), int(components.exponent[0])


def q28_int64_to_binary32_bits_exact(
    values: np.ndarray | Sequence[int],
) -> np.ndarray:
    """逐位复现 ``int64 -> binary64 -> /2^28 -> binary32``。

    NumPy 旧路径先把 signed int64 转为 IEEE binary64，再除以 2^28，最后转换为
    binary32。对超过 53 bit 的整数，不能把这两次 RNE 合并，否则极少数双重舍入
    边界会变化。本函数先生成 53 bit binary64 mantissa，再固定舍入到 24 bit。
    Q28 的整个 int64 动态范围在 binary32 中都是 normal finite。
    """

    source = np.asarray(values, dtype=np.int64)
    flat = source.reshape(-1)
    output = np.empty(flat.shape, dtype=np.uint32)
    for index, item in enumerate(flat):
        signed_value = int(item)
        if signed_value == 0:
            output[index] = np.uint32(0)
            continue
        sign_bit = 1 if signed_value < 0 else 0
        magnitude = abs(signed_value)
        msb = magnitude.bit_length() - 1

        # 第一舍入：signed int64 -> binary64（53 bit significand）。
        if msb <= 52:
            mantissa53 = magnitude << (52 - msb)
            exponent_msb = msb
        else:
            mantissa53 = round_div_rne_nonnegative(magnitude, 1 << (msb - 52))
            exponent_msb = msb
            if mantissa53 == (1 << 53):
                mantissa53 >>= 1
                exponent_msb += 1

        # /2^28 只改变指数；第二舍入是 binary64 -> binary32（24 bit）。
        mantissa24 = round_div_rne_nonnegative(mantissa53, 1 << 29)
        if mantissa24 == (1 << 24):
            mantissa24 >>= 1
            exponent_msb += 1
        exponent_bits = exponent_msb + 99  # exponent_msb - 28 + binary32 bias 127
        if not 1 <= exponent_bits <= 254:
            raise RuntimeLinearQuantError("Q28 转 binary32 指数超出 normal finite 范围")
        output[index] = np.uint32(
            (sign_bit << 31)
            | (exponent_bits << 23)
            | (mantissa24 & 0x7FFFFF)
        )
    return output.reshape(source.shape)


def _q28_to_float32(values: np.ndarray) -> np.ndarray:
    """严格复用并验证 F6/G1 的 Q28→float32 双重舍入。"""

    source = np.asarray(values, dtype=np.int64)
    numpy_converted = (source.astype(np.float64) / float(Q28_FACTOR)).astype(np.float32)
    exact_bits = q28_int64_to_binary32_bits_exact(source)
    converted = exact_bits.view(np.float32)
    if not np.array_equal(exact_bits, numpy_converted.view(np.uint32)):
        raise RuntimeLinearQuantError("Q28 精确双重舍入与 NumPy 旧路径不一致")
    if not np.all(np.isfinite(converted)):
        raise RuntimeLinearQuantError("Q28 转 float32 后出现非有限值")
    return converted


def _quantize_binary32_vector_exact(values: np.ndarray) -> tuple[np.ndarray, np.float32]:
    """按精确 binary32 比值生成 symmetric INT8。"""

    source = np.asarray(values, dtype=np.float32).reshape(-1)
    if source.size == 0:
        raise RuntimeLinearQuantError("激活不能为空")
    if not np.all(np.isfinite(source)):
        raise RuntimeLinearQuantError("激活包含 NaN 或无穷大")
    absolute = np.abs(source)
    maximum = np.max(absolute)
    if maximum == np.float32(0.0):
        return np.zeros(source.shape, dtype=np.int8), np.float32(0.0)

    max_mantissa, max_exponent = _positive_component(maximum)
    components = binary32_components(source)
    output = np.empty(source.shape, dtype=np.int8)
    for index in range(source.size):
        mantissa = int(components.mantissa[index])
        if mantissa == 0:
            output[index] = 0
            continue
        magnitude = round_binary_ratio_rne(
            mantissa * ACTIVATION_QMAX,
            max_mantissa,
            int(components.exponent[index]) - max_exponent,
        )
        magnitude = min(ACTIVATION_QMAX, magnitude)
        output[index] = np.int8(magnitude if components.sign[index] > 0 else -magnitude)
    return output, np.float32(maximum)


def quantize_q10_activation_exact(values_q10: np.ndarray | Sequence[int]) -> tuple[np.ndarray, int]:
    """对 signed int16 Q6.10 激活执行精确 symmetric INT8。"""

    source = np.asarray(values_q10, dtype=np.int16).reshape(-1)
    if source.size == 0:
        raise RuntimeLinearQuantError("Q10 激活不能为空")
    absolute = np.abs(source.astype(np.int32))
    maximum = int(np.max(absolute))
    if maximum == 0:
        return np.zeros(source.shape, dtype=np.int8), 0
    output = np.empty(source.shape, dtype=np.int8)
    for index, item in enumerate(source.astype(np.int32)):
        magnitude = round_div_rne_nonnegative(abs(int(item)) * ACTIVATION_QMAX, maximum)
        magnitude = min(ACTIVATION_QMAX, magnitude)
        output[index] = np.int8(magnitude if item >= 0 else -magnitude)
    return output, maximum


def quantize_q28_activation_exact(
    values_q28: np.ndarray | Sequence[int],
) -> tuple[np.ndarray, np.float32]:
    """复用原 Q28→binary32 语义，再用整数比值生成 INT8。"""

    converted = _q28_to_float32(np.asarray(values_q28, dtype=np.int64))
    return _quantize_binary32_vector_exact(converted)


def _combined_scales_from_binary_max(
    weight_scales: np.ndarray,
    *,
    max_mantissa: int,
    max_exponent: int,
) -> tuple[np.ndarray, int]:
    """精确计算 ``weight_scale * max_abs / 127 * 2^28`` 的 RNE。"""

    scales = np.asarray(weight_scales, dtype=np.float32)
    if scales.ndim != 2 or scales.size == 0:
        raise RuntimeLinearQuantError("weight_scales 必须是非空二维数组")
    if not np.all(np.isfinite(scales)) or np.any(scales <= 0.0):
        raise RuntimeLinearQuantError("weight_scales 必须为有限正数")
    components = binary32_components(scales)
    if np.any(components.sign < 0):
        raise RuntimeLinearQuantError("weight_scales 不得为负")

    flat_mantissa = components.mantissa.reshape(-1)
    flat_exponent = components.exponent.reshape(-1)
    output = np.empty(flat_mantissa.shape, dtype=np.uint32)
    saturated = 0
    for index in range(flat_mantissa.size):
        value = round_binary_ratio_rne(
            int(flat_mantissa[index]) * int(max_mantissa),
            ACTIVATION_QMAX,
            int(flat_exponent[index]) + int(max_exponent) + 28,
        )
        if value > UQ4_28_MAX:
            value = UQ4_28_MAX
            saturated += 1
        output[index] = np.uint32(value)
    return output.reshape(scales.shape), saturated


def _numpy_reference(
    source_float32: np.ndarray,
    weight_scales: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    """原 ``linear_quant_reference`` 的最小独立复刻，用于等价性断言。"""

    source = np.asarray(source_float32, dtype=np.float32).reshape(-1)
    maximum = float(np.max(np.abs(source.astype(np.float64))))
    scale = maximum / ACTIVATION_QMAX if maximum > 0.0 else 1.0
    rounded = np.rint(source.astype(np.float64) / scale)
    activation = np.clip(rounded, -ACTIVATION_QMAX, ACTIVATION_QMAX).astype(np.int8)
    combined_float = np.asarray(weight_scales, dtype=np.float32).astype(np.float64) * scale
    combined_rounded = np.rint(combined_float * float(Q28_FACTOR))
    combined_clipped = np.clip(combined_rounded, 0.0, float(UQ4_28_MAX))
    saturated = int(np.count_nonzero(combined_rounded != combined_clipped))
    return activation, combined_clipped.astype(np.uint32), scale, saturated


def quantize_q10_and_build_scales(
    values_q10: np.ndarray | Sequence[int],
    weight_scales: np.ndarray,
    *,
    verify_numpy: bool = True,
) -> RuntimeLinearQuantization:
    """生成 Q6.10 Linear 所需 INT8 激活与 UQ4.28 combined scales。"""

    source = np.asarray(values_q10, dtype=np.int16).reshape(-1)
    activation, maximum_q10 = quantize_q10_activation_exact(source)
    if maximum_q10 == 0:
        # 原定义全零向量固定 activation_scale=1.0。
        max_mantissa, max_exponent = _positive_component(np.float32(127.0))
        activation_scale = 1.0
    else:
        max_float = np.float32(maximum_q10 / Q10_FACTOR)
        max_mantissa, max_exponent = _positive_component(max_float)
        activation_scale = float(maximum_q10 / Q10_FACTOR) / ACTIVATION_QMAX
    combined, saturated = _combined_scales_from_binary_max(
        weight_scales,
        max_mantissa=max_mantissa,
        max_exponent=max_exponent,
    )
    source_float32 = source.astype(np.float32) / np.float32(Q10_FACTOR)
    if verify_numpy:
        expected_act, expected_scale, expected_activation_scale, expected_sat = _numpy_reference(
            source_float32,
            weight_scales,
        )
        if not np.array_equal(activation, expected_act):
            raise RuntimeLinearQuantError("Q10 精确 INT8 与原 NumPy 定义不一致")
        if not np.array_equal(combined, expected_scale):
            raise RuntimeLinearQuantError("Q10 精确 combined scale 与原 NumPy 定义不一致")
        if saturated != expected_sat or activation_scale != expected_activation_scale:
            raise RuntimeLinearQuantError("Q10 scale 元数据与原 NumPy 定义不一致")
    max_bits = _float32_bits(np.float32(maximum_q10 / Q10_FACTOR)) if maximum_q10 else 0
    return RuntimeLinearQuantization(
        source_kind="q6.10",
        activation_int8=activation,
        combined_scale_q28=combined,
        max_abs_float32_bits=max_bits,
        activation_scale=activation_scale,
        saturated_scale_count=saturated,
    )


def quantize_q28_and_build_scales(
    values_q28: np.ndarray | Sequence[int],
    weight_scales: np.ndarray,
    *,
    verify_numpy: bool = True,
) -> RuntimeLinearQuantization:
    """生成 Q28 Linear 所需 INT8 激活与 UQ4.28 combined scales。"""

    source_float32 = _q28_to_float32(np.asarray(values_q28, dtype=np.int64).reshape(-1))
    activation, maximum = _quantize_binary32_vector_exact(source_float32)
    if maximum == np.float32(0.0):
        max_mantissa, max_exponent = _positive_component(np.float32(127.0))
        activation_scale = 1.0
    else:
        max_mantissa, max_exponent = _positive_component(maximum)
        activation_scale = float(maximum) / ACTIVATION_QMAX
    combined, saturated = _combined_scales_from_binary_max(
        weight_scales,
        max_mantissa=max_mantissa,
        max_exponent=max_exponent,
    )
    if verify_numpy:
        expected_act, expected_scale, expected_activation_scale, expected_sat = _numpy_reference(
            source_float32,
            weight_scales,
        )
        if not np.array_equal(activation, expected_act):
            raise RuntimeLinearQuantError("Q28/binary32 精确 INT8 与原 NumPy 定义不一致")
        if not np.array_equal(combined, expected_scale):
            raise RuntimeLinearQuantError("Q28/binary32 combined scale 与原 NumPy 定义不一致")
        if saturated != expected_sat or activation_scale != expected_activation_scale:
            raise RuntimeLinearQuantError("Q28 scale 元数据与原 NumPy 定义不一致")
    return RuntimeLinearQuantization(
        source_kind="q28-via-binary32",
        activation_int8=activation,
        combined_scale_q28=combined,
        max_abs_float32_bits=_float32_bits(maximum),
        activation_scale=activation_scale,
        saturated_scale_count=saturated,
    )
