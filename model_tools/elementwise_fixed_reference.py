#!/usr/bin/env python3
"""Qwen2 元素级算子的 signed Q6.10 硬件等价软件参考。

E2 第一版统一格式：

- 输入 A/B、标量 scale 和输出：signed Q6.10 int16；
- 残差加法：扩展到 int32 后相加，再显式饱和到 int16；
- 定点缩放：A * scale，Q12.20 经 RNE 右移 10 位，再饱和；
- 元素级乘法：A * B，Q12.20 经 RNE 右移 10 位，再饱和；
- SiLU：x * sigmoid(x)，比较直接 LUT 与分段线性近似。

SiLU 候选方案：

1. 2048 项中点直接 LUT，覆盖 [-8, 8)，区间外分别输出 0 和 x；
2. 64 段分段线性，覆盖 [-8, 8)，保存 65 个 Q6.10 端点，段内乘法后 RNE。

所有浮点转整数和定点右移均采用 round-to-nearest-even（RNE）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

DEFAULT_LENGTH = 896
DEFAULT_SEED = 20260727
DEFAULT_SCALE_Q10 = -768  # -0.75

Q_TOTAL_BITS = 16
Q_FRACTION_BITS = 10
Q_FACTOR = 1 << Q_FRACTION_BITS
Q_MIN = -(1 << (Q_TOTAL_BITS - 1))
Q_MAX = (1 << (Q_TOTAL_BITS - 1)) - 1

SILU_CLIP_Q10 = 8 * Q_FACTOR
SILU_LUT_INDEX_BITS = 11
SILU_LUT_ENTRIES = 1 << SILU_LUT_INDEX_BITS
SILU_LUT_STEP_Q10 = (2 * SILU_CLIP_Q10) // SILU_LUT_ENTRIES
SILU_PWL_SEGMENTS = 64
SILU_PWL_STEP_Q10 = (2 * SILU_CLIP_Q10) // SILU_PWL_SEGMENTS
SILU_PWL_SHIFT = 8
SELECTED_SILU_SCHEME = "pwl64_endpoints"

OP_RESIDUAL_ADD = 0
OP_FIXED_SCALE = 1
OP_ELEMENTWISE_MUL = 2
OP_SILU = 3


class ElementwiseReferenceError(ValueError):
    """表示元素级定点参考的参数或数值不合法。"""


@dataclass(frozen=True)
class QuantizedVector:
    """浮点向量及其 Q6.10 量化结果。"""

    source: np.ndarray
    quantized: np.ndarray
    dequantized: np.ndarray
    clipped_count: int


@dataclass(frozen=True)
class SiLUSchemeMetrics:
    """SiLU 近似方案在完整 int16 输入域上的误差和资源估算。"""

    name: str
    max_abs_error_lsb: int
    mean_abs_error_lsb: float
    mismatch_count: int
    exact_match_count: int
    table_entries: int
    bits_per_entry: int
    estimated_table_bits: int
    multiplier_note: str
    normalized_latency_note: str


@dataclass(frozen=True)
class ElementwiseReferenceResult:
    """一个固定 K=896 向量的全部 E2 软件金标准。"""

    length: int
    seed: int
    vector_a_q10: np.ndarray
    vector_b_q10: np.ndarray
    scale_q10: int
    residual_q10: np.ndarray
    fixed_scale_q10: np.ndarray
    elementwise_mul_q10: np.ndarray
    silu_exact_q10: np.ndarray
    silu_lut_q10: np.ndarray
    silu_pwl_q10: np.ndarray
    residual_saturated_count: int
    fixed_scale_saturated_count: int
    elementwise_mul_saturated_count: int


def _as_finite_1d(values: np.ndarray | Iterable[float], label: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        raise ElementwiseReferenceError(f"{label} 不能为空")
    if not np.all(np.isfinite(array)):
        raise ElementwiseReferenceError(f"{label} 包含 NaN 或无穷大")
    return array


def quantize_q6_10(values: np.ndarray | Iterable[float], label: str = "输入") -> QuantizedVector:
    """把浮点向量使用 RNE 量化为 signed Q6.10。"""

    source = _as_finite_1d(values, label)
    rounded = np.rint(source * Q_FACTOR)
    clipped = np.clip(rounded, Q_MIN, Q_MAX)
    quantized = clipped.astype(np.int16)
    return QuantizedVector(
        source=source.astype(np.float32),
        quantized=quantized,
        dequantized=quantized.astype(np.float64) / Q_FACTOR,
        clipped_count=int(np.count_nonzero(rounded != clipped)),
    )


def _as_q10_int16(values: np.ndarray | Iterable[int], label: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        array = array.reshape(-1)
    if array.size == 0:
        raise ElementwiseReferenceError(f"{label} 不能为空")
    if not np.issubdtype(array.dtype, np.integer):
        raise ElementwiseReferenceError(f"{label} 必须是整数 Q6.10")
    wide = array.astype(np.int64)
    if np.any(wide < Q_MIN) or np.any(wide > Q_MAX):
        raise ElementwiseReferenceError(f"{label} 超出 signed int16 范围")
    return wide.astype(np.int16)


def _round_shift_signed_scalar(value: int, shift: int) -> int:
    """对有符号整数执行对称 RNE 右移；shift<=0 时左移。"""

    if shift <= 0:
        return int(value) << (-shift)
    magnitude = abs(int(value))
    quotient, remainder = divmod(magnitude, 1 << shift)
    half = 1 << (shift - 1)
    if remainder > half or (remainder == half and (quotient & 1)):
        quotient += 1
    return -quotient if value < 0 else quotient


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


def _saturate_int16(values: np.ndarray) -> tuple[np.ndarray, int]:
    """显式饱和到 signed int16，并返回饱和元素数量。"""

    wide = np.asarray(values, dtype=np.int64)
    clipped = np.clip(wide, Q_MIN, Q_MAX)
    return clipped.astype(np.int16), int(np.count_nonzero(wide != clipped))


def residual_add_q10(
    vector_a_q10: np.ndarray | Iterable[int],
    vector_b_q10: np.ndarray | Iterable[int],
) -> tuple[np.ndarray, int]:
    """执行 Q6.10 残差加法和 signed int16 饱和。"""

    a = _as_q10_int16(vector_a_q10, "残差输入 A")
    b = _as_q10_int16(vector_b_q10, "残差输入 B")
    if a.shape != b.shape:
        raise ElementwiseReferenceError("残差输入 A/B 长度不同")
    return _saturate_int16(a.astype(np.int32) + b.astype(np.int32))


def fixed_scale_q10(
    vector_q10: np.ndarray | Iterable[int], scale_q10: int
) -> tuple[np.ndarray, int]:
    """执行 Q6.10 向量乘 Q6.10 标量，RNE 后饱和回 Q6.10。"""

    vector = _as_q10_int16(vector_q10, "缩放输入")
    if scale_q10 < Q_MIN or scale_q10 > Q_MAX:
        raise ElementwiseReferenceError("scale_q10 超出 signed int16 范围")
    product_q20 = vector.astype(np.int64) * np.int64(scale_q10)
    rounded_q10 = _round_shift_signed_array(product_q20, Q_FRACTION_BITS)
    return _saturate_int16(rounded_q10)


def elementwise_mul_q10(
    vector_a_q10: np.ndarray | Iterable[int],
    vector_b_q10: np.ndarray | Iterable[int],
) -> tuple[np.ndarray, int]:
    """执行两个 Q6.10 向量逐元素相乘，RNE 后饱和回 Q6.10。"""

    a = _as_q10_int16(vector_a_q10, "乘法输入 A")
    b = _as_q10_int16(vector_b_q10, "乘法输入 B")
    if a.shape != b.shape:
        raise ElementwiseReferenceError("乘法输入 A/B 长度不同")
    product_q20 = a.astype(np.int64) * b.astype(np.int64)
    rounded_q10 = _round_shift_signed_array(product_q20, Q_FRACTION_BITS)
    return _saturate_int16(rounded_q10)


def _silu_float(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return array / (1.0 + np.exp(-array))


def silu_exact_q10(vector_q10: np.ndarray | Iterable[int]) -> np.ndarray:
    """对离散 Q6.10 输入计算浮点精确 SiLU，再 RNE 到 Q6.10。"""

    vector = _as_q10_int16(vector_q10, "SiLU 输入")
    output = np.rint(_silu_float(vector.astype(np.float64) / Q_FACTOR) * Q_FACTOR)
    return np.clip(output, Q_MIN, Q_MAX).astype(np.int16)


def build_silu_lut_midpoint(index_bits: int = SILU_LUT_INDEX_BITS) -> np.ndarray:
    """生成覆盖 [-8,8) 的中点直接 SiLU LUT。"""

    if index_bits <= 0 or index_bits > 14:
        raise ElementwiseReferenceError("SiLU LUT 索引位数必须位于 [1,14]")
    entries = 1 << index_bits
    width_q10 = 2 * SILU_CLIP_Q10
    if width_q10 % entries:
        raise ElementwiseReferenceError("SiLU LUT 项数必须整除 Q10 输入区间")
    step_q10 = width_q10 // entries
    sample_q10 = (
        -SILU_CLIP_Q10
        + np.arange(entries, dtype=np.float64) * step_q10
        + step_q10 / 2.0
    )
    output = np.rint(_silu_float(sample_q10 / Q_FACTOR) * Q_FACTOR)
    return np.clip(output, Q_MIN, Q_MAX).astype(np.int16)


def silu_lut_q10(
    vector_q10: np.ndarray | Iterable[int], index_bits: int = SILU_LUT_INDEX_BITS
) -> np.ndarray:
    """执行中点直接 LUT SiLU；[-8,8) 外使用 0/x 尾部规则。"""

    vector = _as_q10_int16(vector_q10, "SiLU LUT 输入")
    entries = 1 << index_bits
    width_q10 = 2 * SILU_CLIP_Q10
    if width_q10 % entries:
        raise ElementwiseReferenceError("SiLU LUT 项数必须整除 Q10 输入区间")
    step_q10 = width_q10 // entries
    lut = build_silu_lut_midpoint(index_bits)
    wide = vector.astype(np.int32)
    index = np.clip((wide + SILU_CLIP_Q10) // step_q10, 0, entries - 1)
    output = lut[index].astype(np.int32)
    output = np.where(wide < -SILU_CLIP_Q10, 0, output)
    output = np.where(wide >= SILU_CLIP_Q10, wide, output)
    return output.astype(np.int16)


def build_silu_pwl_endpoints(segments: int = SILU_PWL_SEGMENTS) -> np.ndarray:
    """生成覆盖 [-8,8] 的分段线性端点 Q6.10 表。"""

    width_q10 = 2 * SILU_CLIP_Q10
    if segments <= 0 or width_q10 % segments:
        raise ElementwiseReferenceError("SiLU PWL 段数必须为正且整除 Q10 输入区间")
    step_q10 = width_q10 // segments
    if step_q10 & (step_q10 - 1):
        raise ElementwiseReferenceError("SiLU PWL 步长必须是 2 的幂")
    endpoint_q10 = -SILU_CLIP_Q10 + np.arange(segments + 1) * step_q10
    output = np.rint(_silu_float(endpoint_q10.astype(np.float64) / Q_FACTOR) * Q_FACTOR)
    return np.clip(output, Q_MIN, Q_MAX).astype(np.int16)


def silu_pwl_q10(
    vector_q10: np.ndarray | Iterable[int], segments: int = SILU_PWL_SEGMENTS
) -> np.ndarray:
    """执行端点线性插值 SiLU，段内乘法结果使用 RNE。"""

    vector = _as_q10_int16(vector_q10, "SiLU PWL 输入")
    width_q10 = 2 * SILU_CLIP_Q10
    if segments <= 0 or width_q10 % segments:
        raise ElementwiseReferenceError("SiLU PWL 段数必须为正且整除 Q10 输入区间")
    step_q10 = width_q10 // segments
    if step_q10 & (step_q10 - 1):
        raise ElementwiseReferenceError("SiLU PWL 步长必须是 2 的幂")
    shift = step_q10.bit_length() - 1
    endpoints = build_silu_pwl_endpoints(segments).astype(np.int64)
    wide = vector.astype(np.int64)
    index = np.clip((wide + SILU_CLIP_Q10) // step_q10, 0, segments - 1)
    fraction = (wide + SILU_CLIP_Q10) - index * step_q10
    delta = endpoints[1:] - endpoints[:-1]
    interpolation = _round_shift_signed_array(delta[index] * fraction, shift)
    output = endpoints[index] + interpolation
    output = np.where(wide < -SILU_CLIP_Q10, 0, output)
    output = np.where(wide >= SILU_CLIP_Q10, wide, output)
    return np.clip(output, Q_MIN, Q_MAX).astype(np.int16)


def silu_scheme_metrics() -> list[SiLUSchemeMetrics]:
    """在完整 65536 个 signed int16 输入上比较两种 SiLU 方案。"""

    inputs = np.arange(Q_MIN, Q_MAX + 1, dtype=np.int32).astype(np.int16)
    exact = silu_exact_q10(inputs).astype(np.int32)
    candidates = [
        (
            "lut2048_midpoint",
            silu_lut_q10(inputs).astype(np.int32),
            SILU_LUT_ENTRIES,
            "SiLU 本体无需乘法器",
            "边界判断 + 1 次 ROM 读取",
        ),
        (
            "pwl64_endpoints",
            silu_pwl_q10(inputs).astype(np.int32),
            SILU_PWL_SEGMENTS + 1,
            "每个元素需要一次小位宽 delta×fraction 乘法，可流水复用",
            "段索引 + 端点读取 + 乘法/RNE + 加法",
        ),
    ]
    metrics: list[SiLUSchemeMetrics] = []
    for name, output, entries, multiplier_note, latency_note in candidates:
        error = np.abs(output - exact)
        mismatch = int(np.count_nonzero(error))
        metrics.append(
            SiLUSchemeMetrics(
                name=name,
                max_abs_error_lsb=int(np.max(error)),
                mean_abs_error_lsb=float(np.mean(error)),
                mismatch_count=mismatch,
                exact_match_count=int(error.size - mismatch),
                table_entries=entries,
                bits_per_entry=16,
                estimated_table_bits=entries * 16,
                multiplier_note=multiplier_note,
                normalized_latency_note=latency_note,
            )
        )
    return metrics


def make_deterministic_q10_vectors(
    length: int = DEFAULT_LENGTH, seed: int = DEFAULT_SEED
) -> tuple[np.ndarray, np.ndarray]:
    """生成含边界、RNE tie 和全范围随机值的固定 int16 向量。"""

    if length <= 0:
        raise ElementwiseReferenceError("向量长度必须大于 0")
    boundary_a = np.asarray(
        [
            32767, 32767, -32768, -32768,
            1, 3, 5, -1, -3, -5,
            8192, -8192, 1024, -1024, 0, 512,
            16384, -16384, 24576, -24576, 32760, -32760,
            7, -7, 511, -511, 513, -513, 2047, -2047, 2049, -2049,
        ],
        dtype=np.int16,
    )
    boundary_b = np.asarray(
        [
            1, 32767, -1, -32768,
            512, 512, 512, 512, 512, 512,
            8192, -8192, -1024, 1024, 0, -512,
            3072, 3072, -2048, -2048, 32760, 32760,
            -7, 7, 513, -513, 511, -511, -2049, 2049, -2047, 2047,
        ],
        dtype=np.int16,
    )
    output_a = np.empty(length, dtype=np.int16)
    output_b = np.empty(length, dtype=np.int16)
    prefix = min(length, boundary_a.size)
    output_a[:prefix] = boundary_a[:prefix]
    output_b[:prefix] = boundary_b[:prefix]
    state_a = int(seed) & 0xFFFFFFFF
    state_b = (int(seed) ^ 0xA5A5A5A5) & 0xFFFFFFFF
    for index in range(prefix, length):
        state_a = (1664525 * state_a + 1013904223) & 0xFFFFFFFF
        state_b = (22695477 * state_b + 1) & 0xFFFFFFFF
        output_a[index] = np.int16(((state_a >> 16) & 0xFFFF) - 32768)
        output_b[index] = np.int16(((state_b >> 16) & 0xFFFF) - 32768)
    return output_a, output_b


def compute_elementwise_reference(
    *,
    vector_a_q10: np.ndarray | Iterable[int],
    vector_b_q10: np.ndarray | Iterable[int],
    scale_q10: int = DEFAULT_SCALE_Q10,
    seed: int = DEFAULT_SEED,
) -> ElementwiseReferenceResult:
    """计算固定向量的残差、缩放、逐元素乘法和三条 SiLU 路径。"""

    a = _as_q10_int16(vector_a_q10, "固定向量 A")
    b = _as_q10_int16(vector_b_q10, "固定向量 B")
    if a.shape != b.shape:
        raise ElementwiseReferenceError("固定向量 A/B 长度不同")
    residual, residual_saturated = residual_add_q10(a, b)
    scaled, scaled_saturated = fixed_scale_q10(a, scale_q10)
    multiplied, multiplied_saturated = elementwise_mul_q10(a, b)
    return ElementwiseReferenceResult(
        length=int(a.size),
        seed=int(seed),
        vector_a_q10=a,
        vector_b_q10=b,
        scale_q10=int(scale_q10),
        residual_q10=residual,
        fixed_scale_q10=scaled,
        elementwise_mul_q10=multiplied,
        silu_exact_q10=silu_exact_q10(a),
        silu_lut_q10=silu_lut_q10(a),
        silu_pwl_q10=silu_pwl_q10(a),
        residual_saturated_count=residual_saturated,
        fixed_scale_saturated_count=scaled_saturated,
        elementwise_mul_saturated_count=multiplied_saturated,
    )


def _sha256_array(array: np.ndarray, dtype: str | np.dtype | None = None) -> str:
    normalized = np.asarray(array, dtype=dtype) if dtype is not None else np.asarray(array)
    return hashlib.sha256(normalized.tobytes(order="C")).hexdigest()


def result_manifest(result: ElementwiseReferenceResult) -> dict[str, Any]:
    """生成可提交到 Git 的 E2 固定向量和方案比较清单。"""

    metrics = silu_scheme_metrics()
    return {
        "format_version": 1,
        "operator_group": "qwen2_elementwise_k896",
        "length": result.length,
        "seed": result.seed,
        "fixed_formats": {
            "input_a": "signed Q6.10 int16",
            "input_b": "signed Q6.10 int16",
            "scale": "signed Q6.10 int16",
            "output": "signed Q6.10 int16",
            "multiply_intermediate": "signed Q12.20 int32",
            "rounding": "round_to_nearest_even",
            "saturation": "explicit signed int16 saturation",
        },
        "operations": {
            "0": "residual_add_saturating",
            "1": "fixed_scale_rne_saturating",
            "2": "elementwise_mul_rne_saturating",
            "3": "silu_pwl64_selected",
        },
        "scale": {
            "q6_10": result.scale_q10,
            "float": result.scale_q10 / Q_FACTOR,
        },
        "saturation_counts": {
            "residual_add": result.residual_saturated_count,
            "fixed_scale": result.fixed_scale_saturated_count,
            "elementwise_mul": result.elementwise_mul_saturated_count,
        },
        "silu": {
            "formula": "x * sigmoid(x)",
            "clip_range_q6_10": [-SILU_CLIP_Q10, SILU_CLIP_Q10],
            "tail_rules": "x < -8 -> 0; x >= 8 -> x",
            "selected_scheme": SELECTED_SILU_SCHEME,
            "pwl_segments": SILU_PWL_SEGMENTS,
            "pwl_step_q10": SILU_PWL_STEP_Q10,
            "lut_entries": SILU_LUT_ENTRIES,
            "lut_step_q10": SILU_LUT_STEP_Q10,
            "full_int16_domain_metrics": [metric.__dict__ for metric in metrics],
        },
        "preview": {
            "input_a_first16": result.vector_a_q10[:16].tolist(),
            "input_b_first16": result.vector_b_q10[:16].tolist(),
            "residual_first16": result.residual_q10[:16].tolist(),
            "fixed_scale_first16": result.fixed_scale_q10[:16].tolist(),
            "elementwise_mul_first16": result.elementwise_mul_q10[:16].tolist(),
            "silu_exact_first16": result.silu_exact_q10[:16].tolist(),
            "silu_lut_first16": result.silu_lut_q10[:16].tolist(),
            "silu_pwl_first16": result.silu_pwl_q10[:16].tolist(),
        },
        "sha256": {
            "input_a_q6_10": _sha256_array(result.vector_a_q10, "<i2"),
            "input_b_q6_10": _sha256_array(result.vector_b_q10, "<i2"),
            "residual_q6_10": _sha256_array(result.residual_q10, "<i2"),
            "fixed_scale_q6_10": _sha256_array(result.fixed_scale_q10, "<i2"),
            "elementwise_mul_q6_10": _sha256_array(result.elementwise_mul_q10, "<i2"),
            "silu_exact_q6_10": _sha256_array(result.silu_exact_q10, "<i2"),
            "silu_lut_q6_10": _sha256_array(result.silu_lut_q10, "<i2"),
            "silu_pwl_q6_10": _sha256_array(result.silu_pwl_q10, "<i2"),
            "silu_lut2048_q6_10": _sha256_array(build_silu_lut_midpoint(), "<i2"),
            "silu_pwl65_endpoints_q6_10": _sha256_array(
                build_silu_pwl_endpoints(), "<i2"
            ),
        },
    }


def save_npz(result: ElementwiseReferenceResult, path: Path) -> None:
    """保存后续 FPGA 上位机可直接消费的完整固定向量。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        length=np.asarray(result.length, dtype="<u4"),
        seed=np.asarray(result.seed, dtype="<u4"),
        scale_q6_10=np.asarray(result.scale_q10, dtype="<i2"),
        input_a_q6_10=result.vector_a_q10.astype("<i2"),
        input_b_q6_10=result.vector_b_q10.astype("<i2"),
        residual_q6_10=result.residual_q10.astype("<i2"),
        fixed_scale_q6_10=result.fixed_scale_q10.astype("<i2"),
        elementwise_mul_q6_10=result.elementwise_mul_q10.astype("<i2"),
        silu_exact_q6_10=result.silu_exact_q10.astype("<i2"),
        silu_lut_q6_10=result.silu_lut_q10.astype("<i2"),
        silu_pwl_q6_10=result.silu_pwl_q10.astype("<i2"),
        silu_pwl65_endpoints_q6_10=build_silu_pwl_endpoints().astype("<i2"),
    )


def _print_result(result: ElementwiseReferenceResult) -> None:
    print("=== E2 K=896 元素级定点软件参考 ===")
    print(
        f"格式：A/B/scale/output=signed Q6.10 int16，K={result.length}，"
        f"scale={result.scale_q10} ({result.scale_q10 / Q_FACTOR:g})"
    )
    print(
        "饱和数："
        f"residual={result.residual_saturated_count}，"
        f"scale={result.fixed_scale_saturated_count}，"
        f"mul={result.elementwise_mul_saturated_count}"
    )
    print(f"残差前16项：{result.residual_q10[:16].tolist()}")
    print(f"缩放前16项：{result.fixed_scale_q10[:16].tolist()}")
    print(f"逐元素乘法前16项：{result.elementwise_mul_q10[:16].tolist()}")
    print(f"SiLU PWL64前16项：{result.silu_pwl_q10[:16].tolist()}")
    for metric in silu_scheme_metrics():
        print(
            f"{metric.name}: 最大误差={metric.max_abs_error_lsb} LSB，"
            f"平均误差={metric.mean_abs_error_lsb:.6f} LSB，"
            f"不一致={metric.mismatch_count}/65536，"
            f"表规模={metric.table_entries}×{metric.bits_per_entry}="
            f"{metric.estimated_table_bits} bit"
        )
    print(
        "结论：第一版选择 64 段端点分段线性。它在完整 int16 输入域最大误差不超过 "
        "4 个 Q10 LSB，端点表仅 1040 bit；代价是一个可流水复用的小乘法器。"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="E2 signed Q6.10 元素级算子金标准")
    parser.add_argument("--length", type=int, default=DEFAULT_LENGTH)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--scale-q10", type=int, default=DEFAULT_SCALE_Q10)
    parser.add_argument("--output", type=Path, help="可选：保存完整 NPZ")
    parser.add_argument("--manifest", type=Path, help="可选：保存 JSON 清单")
    parser.add_argument(
        "--print-manifest", action="store_true", help="把 JSON 清单打印到标准输出"
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        vector_a, vector_b = make_deterministic_q10_vectors(args.length, args.seed)
        result = compute_elementwise_reference(
            vector_a_q10=vector_a,
            vector_b_q10=vector_b,
            scale_q10=args.scale_q10,
            seed=args.seed,
        )
        _print_result(result)
        if args.output is not None:
            save_npz(result, args.output)
            print(f"完整测试向量已保存：{args.output}")
        manifest = result_manifest(result)
        if args.manifest is not None:
            args.manifest.parent.mkdir(parents=True, exist_ok=True)
            args.manifest.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print(f"固定向量清单已保存：{args.manifest}")
        if args.print_manifest:
            print("---BEGIN MANIFEST---")
            print(json.dumps(manifest, ensure_ascii=False, indent=2))
            print("---END MANIFEST---")
        return 0
    except (ElementwiseReferenceError, ValueError) as error:
        print(f"错误：{error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
