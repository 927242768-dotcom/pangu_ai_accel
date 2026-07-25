#!/usr/bin/env python3
"""G2 运行时量化 DDR3 controller 的固定清单、载荷与访问契约。

本模块把完整 layer0 Block 中七个真实 Linear 调用统一为可由独立 FPGA
验证工程消费的事务：

- Q/K/V、gate/up 使用 signed int16 Q6.10 源；
- O_proj、down_proj 使用 signed int64 Q28 源；
- 原始 weight scale 保持 P50 FP16 bit pattern；
- 期望输出包含 packed INT8、逐行 8-word 对齐的 UQ4.28、完整 max metadata；
- 同时冻结 source/raw-scale 读取和 activation/combined-scale 写入的 AXI burst、
  beat 数量、首尾地址，防止“数值碰巧正确但访问范围错误”。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

try:
    from .runtime_linear_quant_reference import (
        RuntimeLinearQuantization,
        binary32_components,
        quantize_q10_and_build_scales,
        quantize_q28_and_build_scales,
    )
    from .transformer_block_reference import (
        DEFAULT_IMAGE,
        BlockContext,
        LinearInvocation,
        TransformerBlockCase,
        build_case,
        linear_invocations,
        load_context,
        parameter_regions,
        scratch_regions,
    )
except ImportError:
    from runtime_linear_quant_reference import (
        RuntimeLinearQuantization,
        binary32_components,
        quantize_q10_and_build_scales,
        quantize_q28_and_build_scales,
    )
    from transformer_block_reference import (
        DEFAULT_IMAGE,
        BlockContext,
        LinearInvocation,
        TransformerBlockCase,
        build_case,
        linear_invocations,
        load_context,
        parameter_regions,
        scratch_regions,
    )

DEFAULT_MANIFEST = Path(__file__).with_name("runtime_quantizer_g2_reference.json")
DEFAULT_QUERY_POSITION = 0
DEFAULT_WINDOW_START = 0
DEFAULT_STRESS_SEED = 20260819
CONFIG_STRUCT = struct.Struct("<4H4I")
RESULT_MAGIC = b"P50QTV1\0"
RESULT_HEADER_STRUCT = struct.Struct("<8s22I")
RESULT_VERSION = 1
MATRIX_NAMES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


class RuntimeQuantizerValidationError(ValueError):
    """表示量化验证事务、载荷、布局或固定清单不合法。"""


@dataclass(frozen=True)
class AxiTraceExpectation:
    source_read_commands: int
    source_read_beats: int
    raw_scale_read_commands: int
    raw_scale_read_beats: int
    activation_write_commands: int
    activation_write_beats: int
    combined_write_commands: int
    combined_write_beats: int
    source_end_ctrl_address: int
    raw_scale_end_ctrl_address: int
    activation_end_ctrl_address: int
    combined_end_ctrl_address: int

    @property
    def total_read_commands(self) -> int:
        return self.source_read_commands + self.raw_scale_read_commands

    @property
    def total_read_beats(self) -> int:
        return self.source_read_beats + self.raw_scale_read_beats

    @property
    def total_write_commands(self) -> int:
        return self.activation_write_commands + self.combined_write_commands

    @property
    def total_write_beats(self) -> int:
        return self.activation_write_beats + self.combined_write_beats


@dataclass(frozen=True)
class RuntimeQuantizerValidationCase:
    matrix_id: int
    name: str
    source_q28: bool
    vector_length: int
    rows: int
    groups: int
    source_ctrl_address: int
    activation_ctrl_address: int
    raw_scale_ctrl_address: int
    combined_scale_ctrl_address: int
    source_values: np.ndarray
    raw_scale_fp16_bits: np.ndarray
    activation_int8: np.ndarray
    combined_scale_q28_padded: np.ndarray
    all_zero: bool
    max_abs_q10: int
    max_mantissa_binary32: int
    max_exponent_binary32: int
    max_abs_binary32_bits: int
    saturated_count: int
    trace: AxiTraceExpectation

    @property
    def padded_groups(self) -> int:
        return ((self.groups + 7) // 8) * 8

    @property
    def source_bytes(self) -> int:
        return self.vector_length * (8 if self.source_q28 else 2)

    @property
    def raw_scale_bytes(self) -> int:
        return self.rows * self.groups * 2

    @property
    def activation_bytes(self) -> int:
        return self.vector_length

    @property
    def combined_scale_bytes(self) -> int:
        return self.rows * self.padded_groups * 4

    @property
    def upload_bytes(self) -> int:
        return self.source_bytes + self.raw_scale_bytes

    @property
    def result_bytes(self) -> int:
        return RESULT_HEADER_STRUCT.size + self.activation_bytes + self.combined_scale_bytes


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _array_bytes(values: np.ndarray, dtype: str | np.dtype) -> bytes:
    return np.asarray(values, dtype=dtype).tobytes(order="C")


def _region_ctrl_addresses() -> dict[str, int]:
    return {
        region.name: region.controller_address
        for region in (*scratch_regions(), *parameter_regions())
    }


def _matrix_model(context: BlockContext, name: str):
    if name in {"q_proj", "k_proj", "v_proj"}:
        return context.attention.qkv_models[name[0]]
    if name == "o_proj":
        return context.attention.oproj_model
    if name == "gate_proj":
        return context.gate_model
    if name == "up_proj":
        return context.up_model
    if name == "down_proj":
        return context.down_model
    raise RuntimeQuantizerValidationError(f"未知矩阵：{name}")


def _source_values(case: TransformerBlockCase, name: str) -> np.ndarray:
    if name in {"q_proj", "k_proj", "v_proj"}:
        return np.asarray(case.input_norm_q10, dtype=np.int16).reshape(-1)
    if name == "o_proj":
        return np.asarray(case.attention_concat_q28, dtype=np.int64).reshape(-1)
    if name in {"gate_proj", "up_proj"}:
        return np.asarray(case.post_attention_norm_q10, dtype=np.int16).reshape(-1)
    if name == "down_proj":
        return np.asarray(case.silu_up_q28, dtype=np.int64).reshape(-1)
    raise RuntimeQuantizerValidationError(f"未知矩阵：{name}")


def _source_region_name(name: str) -> str:
    if name in {"q_proj", "k_proj", "v_proj"}:
        return "input_norm_q10"
    if name == "o_proj":
        return "attention_concat_q28"
    if name in {"gate_proj", "up_proj"}:
        return "post_attention_norm_q10"
    if name == "down_proj":
        return "silu_up_q28"
    raise RuntimeQuantizerValidationError(f"未知矩阵：{name}")


def _positive_binary32_metadata(value_bits: int) -> tuple[int, int]:
    if value_bits == 0:
        return 0, 0
    value = np.asarray([np.uint32(value_bits)], dtype=np.uint32).view(np.float32)
    components = binary32_components(value)
    return int(components.mantissa[0]), int(components.exponent[0])


def _expected_trace(
    *,
    source_q28: bool,
    vector_length: int,
    rows: int,
    groups: int,
    source_ctrl_address: int,
    activation_ctrl_address: int,
    raw_scale_ctrl_address: int,
    combined_scale_ctrl_address: int,
) -> AxiTraceExpectation:
    source_elements_per_beat = 4 if source_q28 else 16
    source_beats = vector_length // source_elements_per_beat
    raw_scale_beats = (rows * groups) // 16
    activation_beats = vector_length // 32
    padded_groups = ((groups + 7) // 8) * 8
    combined_beats = rows * (padded_groups // 8)
    return AxiTraceExpectation(
        source_read_commands=(source_beats + 15) // 16,
        source_read_beats=source_beats,
        raw_scale_read_commands=raw_scale_beats,
        raw_scale_read_beats=raw_scale_beats,
        activation_write_commands=activation_beats,
        activation_write_beats=activation_beats,
        combined_write_commands=combined_beats,
        combined_write_beats=combined_beats,
        source_end_ctrl_address=source_ctrl_address + source_beats * 8,
        raw_scale_end_ctrl_address=raw_scale_ctrl_address + raw_scale_beats * 8,
        activation_end_ctrl_address=activation_ctrl_address + activation_beats * 8,
        combined_end_ctrl_address=combined_scale_ctrl_address + combined_beats * 8,
    )


def _validate_raw_fp16_roundtrip(weight_scales: np.ndarray) -> np.ndarray:
    source = np.asarray(weight_scales, dtype=np.float32)
    fp16 = source.astype(np.float16)
    restored = fp16.astype(np.float32)
    if not np.array_equal(restored.view(np.uint32), source.view(np.uint32)):
        raise RuntimeQuantizerValidationError("weight_scales 不是由 FP16 精确扩展得到")
    bits = fp16.view(np.uint16).copy()
    if np.any((bits & np.uint16(0x7FFF)) == 0) or np.any((bits & np.uint16(0x8000)) != 0):
        raise RuntimeQuantizerValidationError("原始 FP16 scale 必须全部为有限正数")
    if np.any((bits & np.uint16(0x7C00)) == np.uint16(0x7C00)):
        raise RuntimeQuantizerValidationError("原始 FP16 scale 包含 Inf/NaN")
    return bits


def build_validation_case(
    context: BlockContext,
    block_case: TransformerBlockCase,
    invocation: LinearInvocation,
    matrix_id: int,
) -> RuntimeQuantizerValidationCase:
    if invocation.name != MATRIX_NAMES[matrix_id]:
        raise RuntimeQuantizerValidationError("matrix_id 与 invocation 不一致")
    source_q28 = invocation.source_format.startswith("Q28")
    source = _source_values(block_case, invocation.name)
    model = _matrix_model(context, invocation.name)
    scales = np.asarray(model.weight_scales, dtype=np.float32)
    if source.size != invocation.columns:
        raise RuntimeQuantizerValidationError(f"{invocation.name} source 长度错误")
    if scales.shape != (invocation.rows, invocation.groups):
        raise RuntimeQuantizerValidationError(f"{invocation.name} weight scale 形状错误")

    result: RuntimeLinearQuantization
    if source_q28:
        result = quantize_q28_and_build_scales(source, scales)
    else:
        result = quantize_q10_and_build_scales(source, scales)
    padded_groups = ((invocation.groups + 7) // 8) * 8
    padded = np.zeros((invocation.rows, padded_groups), dtype=np.uint32)
    padded[:, : invocation.groups] = result.combined_scale_q28
    addresses = _region_ctrl_addresses()
    source_address = addresses[_source_region_name(invocation.name)]
    max_mantissa, max_exponent = _positive_binary32_metadata(
        result.max_abs_float32_bits
    )
    if not source_q28:
        source_i32 = np.asarray(source, dtype=np.int16).astype(np.int32)
        max_abs_q10 = int(np.max(np.abs(source_i32)))
    else:
        max_abs_q10 = 0
    raw_bits = _validate_raw_fp16_roundtrip(scales)
    trace = _expected_trace(
        source_q28=source_q28,
        vector_length=invocation.columns,
        rows=invocation.rows,
        groups=invocation.groups,
        source_ctrl_address=source_address,
        activation_ctrl_address=invocation.activation_ctrl_address,
        raw_scale_ctrl_address=invocation.raw_weight_scale_ctrl_address,
        combined_scale_ctrl_address=invocation.combined_scale_ctrl_address,
    )
    return RuntimeQuantizerValidationCase(
        matrix_id=matrix_id,
        name=invocation.name,
        source_q28=source_q28,
        vector_length=invocation.columns,
        rows=invocation.rows,
        groups=invocation.groups,
        source_ctrl_address=source_address,
        activation_ctrl_address=invocation.activation_ctrl_address,
        raw_scale_ctrl_address=invocation.raw_weight_scale_ctrl_address,
        combined_scale_ctrl_address=invocation.combined_scale_ctrl_address,
        source_values=source.copy(),
        raw_scale_fp16_bits=raw_bits,
        activation_int8=result.activation_int8.copy(),
        combined_scale_q28_padded=padded,
        all_zero=bool(np.all(source == 0)),
        max_abs_q10=max_abs_q10,
        max_mantissa_binary32=max_mantissa,
        max_exponent_binary32=max_exponent,
        max_abs_binary32_bits=int(result.max_abs_float32_bits),
        saturated_count=int(result.saturated_scale_count),
        trace=trace,
    )


def build_fixed_validation_cases(
    image_path: Path = DEFAULT_IMAGE,
    *,
    query_position: int = DEFAULT_QUERY_POSITION,
    window_start: int = DEFAULT_WINDOW_START,
) -> list[RuntimeQuantizerValidationCase]:
    context = load_context(image_path)
    block_case = build_case(
        context,
        query_position=query_position,
        window_start=window_start,
    )
    return [
        build_validation_case(context, block_case, invocation, matrix_id)
        for matrix_id, invocation in enumerate(linear_invocations())
    ]


def with_source_values(
    case: RuntimeQuantizerValidationCase,
    source_values: np.ndarray | Sequence[int],
    *,
    verify_numpy: bool = True,
) -> RuntimeQuantizerValidationCase:
    """保留真实矩阵 scale/地址，仅替换源向量并重建逐位期望。

    固定真实矩阵默认继续断言与旧 NumPy/G1/F6 路径逐位一致；随机与边界
    压力以 RTL 对应的精确整数/二进制有理数规格为权威，避免二进制浮点在
    数学半整数附近产生 ``63.5 -> 63.49999999999999`` 一类非 RNE 偏差。
    """

    source_dtype = np.int64 if case.source_q28 else np.int16
    source = np.asarray(source_values, dtype=source_dtype).reshape(-1)
    if source.size != case.vector_length:
        raise RuntimeQuantizerValidationError(
            f"{case.name} source 长度应为 {case.vector_length}，实际 {source.size}"
        )
    scales = case.raw_scale_fp16_bits.view(np.float16).astype(np.float32)
    if case.source_q28:
        result = quantize_q28_and_build_scales(
            source,
            scales,
            verify_numpy=verify_numpy,
        )
        max_abs_q10 = 0
    else:
        result = quantize_q10_and_build_scales(
            source,
            scales,
            verify_numpy=verify_numpy,
        )
        source_i32 = source.astype(np.int32)
        max_abs_q10 = int(np.max(np.abs(source_i32)))
    padded = np.zeros((case.rows, case.padded_groups), dtype=np.uint32)
    padded[:, : case.groups] = result.combined_scale_q28
    max_mantissa, max_exponent = _positive_binary32_metadata(
        result.max_abs_float32_bits
    )
    rebuilt = replace(
        case,
        source_values=source.copy(),
        activation_int8=result.activation_int8.copy(),
        combined_scale_q28_padded=padded,
        all_zero=bool(np.all(source == 0)),
        max_abs_q10=max_abs_q10,
        max_mantissa_binary32=max_mantissa,
        max_exponent_binary32=max_exponent,
        max_abs_binary32_bits=int(result.max_abs_float32_bits),
        saturated_count=int(result.saturated_scale_count),
    )
    verify_upload_roundtrip(rebuilt)
    return rebuilt


def random_source_case(
    case: RuntimeQuantizerValidationCase,
    rng: np.random.Generator,
    iteration: int,
) -> RuntimeQuantizerValidationCase:
    """生成覆盖全零、极值、稀疏、幂次边界与全范围随机值的事务。"""

    mode = iteration % 6
    length = case.vector_length
    if case.source_q28:
        if mode == 0:
            source = np.zeros(length, dtype=np.int64)
        elif mode == 1:
            boundaries = np.asarray(
                [
                    np.iinfo(np.int64).min,
                    np.iinfo(np.int64).max,
                    -1,
                    0,
                    1,
                    -(1 << 28),
                    1 << 28,
                    -(1 << 52),
                    1 << 52,
                ],
                dtype=np.int64,
            )
            source = np.resize(boundaries, length).copy()
        elif mode == 2:
            source = rng.bit_generator.random_raw(length).view(np.int64)
        elif mode == 3:
            source = np.zeros(length, dtype=np.int64)
            indices = rng.choice(length, size=max(1, length // 32), replace=False)
            source[indices] = rng.bit_generator.random_raw(indices.size).view(np.int64)
        elif mode == 4:
            powers = rng.integers(0, 63, size=length, dtype=np.int64)
            signs = rng.choice(np.asarray([-1, 1], dtype=np.int64), size=length)
            source = (np.left_shift(np.int64(1), powers) * signs).astype(np.int64)
        else:
            source = rng.integers(
                -(1 << 40), 1 << 40, size=length, dtype=np.int64
            )
    else:
        if mode == 0:
            source = np.zeros(length, dtype=np.int16)
        elif mode == 1:
            boundaries = np.asarray(
                [-32768, 32767, -1024, -1, 0, 1, 1024, 16384], dtype=np.int16
            )
            source = np.resize(boundaries, length).copy()
        elif mode == 2:
            source = rng.integers(-32768, 32768, size=length, dtype=np.int16)
        elif mode == 3:
            source = np.zeros(length, dtype=np.int16)
            indices = rng.choice(length, size=max(1, length // 32), replace=False)
            source[indices] = rng.integers(
                -32768, 32768, size=indices.size, dtype=np.int16
            )
        elif mode == 4:
            powers = rng.integers(0, 15, size=length, dtype=np.int16)
            magnitudes = np.left_shift(np.int32(1), powers.astype(np.int32))
            signs = rng.choice(np.asarray([-1, 1], dtype=np.int32), size=length)
            source = np.clip(magnitudes * signs, -32768, 32767).astype(np.int16)
        else:
            source = rng.integers(-4096, 4097, size=length, dtype=np.int16)
    return with_source_values(case, source, verify_numpy=False)


def build_config_payload(case: RuntimeQuantizerValidationCase) -> bytes:
    return CONFIG_STRUCT.pack(
        case.vector_length,
        case.rows,
        case.groups,
        case.matrix_id,
        case.source_ctrl_address,
        case.activation_ctrl_address,
        case.raw_scale_ctrl_address,
        case.combined_scale_ctrl_address,
    )


def build_upload_payload(case: RuntimeQuantizerValidationCase) -> bytes:
    source_dtype = "<i8" if case.source_q28 else "<i2"
    payload = _array_bytes(case.source_values, source_dtype) + _array_bytes(
        case.raw_scale_fp16_bits, "<u2"
    )
    if len(payload) != case.upload_bytes:
        raise RuntimeQuantizerValidationError("上传载荷长度错误")
    return payload


def expected_result_payload(case: RuntimeQuantizerValidationCase) -> bytes:
    header = RESULT_HEADER_STRUCT.pack(
        RESULT_MAGIC,
        RESULT_VERSION,
        case.matrix_id,
        int(case.source_q28),
        case.vector_length,
        case.rows,
        case.groups,
        case.padded_groups,
        int(case.all_zero),
        case.max_abs_q10,
        case.max_mantissa_binary32,
        case.max_exponent_binary32 & 0xFFFFFFFF,
        case.max_abs_binary32_bits,
        case.saturated_count,
        case.trace.source_read_commands,
        case.trace.source_read_beats,
        case.trace.raw_scale_read_commands,
        case.trace.raw_scale_read_beats,
        case.trace.activation_write_commands,
        case.trace.activation_write_beats,
        case.trace.combined_write_commands,
        case.trace.combined_write_beats,
        0,
    )
    payload = (
        header
        + _array_bytes(case.activation_int8, "i1")
        + _array_bytes(case.combined_scale_q28_padded, "<u4")
    )
    if len(payload) != case.result_bytes:
        raise RuntimeQuantizerValidationError("期望结果载荷长度错误")
    return payload


def verify_upload_roundtrip(case: RuntimeQuantizerValidationCase) -> None:
    config = CONFIG_STRUCT.unpack(build_config_payload(case))
    if config != (
        case.vector_length,
        case.rows,
        case.groups,
        case.matrix_id,
        case.source_ctrl_address,
        case.activation_ctrl_address,
        case.raw_scale_ctrl_address,
        case.combined_scale_ctrl_address,
    ):
        raise RuntimeQuantizerValidationError("配置载荷往返错误")
    upload = build_upload_payload(case)
    source_dtype = "<i8" if case.source_q28 else "<i2"
    source = np.frombuffer(upload[: case.source_bytes], dtype=source_dtype).copy()
    scales = np.frombuffer(upload[case.source_bytes :], dtype="<u2").reshape(
        case.rows, case.groups
    ).copy()
    if not np.array_equal(source, case.source_values):
        raise RuntimeQuantizerValidationError("source 上传往返错误")
    if not np.array_equal(scales, case.raw_scale_fp16_bits):
        raise RuntimeQuantizerValidationError("raw FP16 scale 上传往返错误")
    expected = expected_result_payload(case)
    unpacked = RESULT_HEADER_STRUCT.unpack_from(expected, 0)
    if unpacked[0] != RESULT_MAGIC or unpacked[1] != RESULT_VERSION:
        raise RuntimeQuantizerValidationError("结果 header 往返错误")
    offset = RESULT_HEADER_STRUCT.size
    activation = np.frombuffer(
        expected[offset : offset + case.activation_bytes], dtype=np.int8
    ).copy()
    offset += case.activation_bytes
    combined = np.frombuffer(expected[offset:], dtype="<u4").reshape(
        case.rows, case.padded_groups
    ).copy()
    if not np.array_equal(activation, case.activation_int8):
        raise RuntimeQuantizerValidationError("activation 结果往返错误")
    if not np.array_equal(combined, case.combined_scale_q28_padded):
        raise RuntimeQuantizerValidationError("combined scale 结果往返错误")
    if np.any(combined[:, case.groups :] != 0):
        raise RuntimeQuantizerValidationError("combined scale padding 非零")


def fixed_manifest(cases: Sequence[RuntimeQuantizerValidationCase]) -> dict[str, object]:
    return {
        "format_version": 1,
        "operator": "g2_runtime_quantizer_ddr3_validation",
        "query_position": DEFAULT_QUERY_POSITION,
        "window_start": DEFAULT_WINDOW_START,
        "result_header_bytes": RESULT_HEADER_STRUCT.size,
        "config_bytes": CONFIG_STRUCT.size,
        "cases": [
            {
                "matrix_id": case.matrix_id,
                "name": case.name,
                "source_q28": case.source_q28,
                "vector_length": case.vector_length,
                "rows": case.rows,
                "groups": case.groups,
                "padded_groups": case.padded_groups,
                "addresses": {
                    "source_ctrl": f"0x{case.source_ctrl_address:07x}",
                    "activation_ctrl": f"0x{case.activation_ctrl_address:07x}",
                    "raw_scale_ctrl": f"0x{case.raw_scale_ctrl_address:07x}",
                    "combined_scale_ctrl": f"0x{case.combined_scale_ctrl_address:07x}",
                    "source_end_ctrl": f"0x{case.trace.source_end_ctrl_address:07x}",
                    "activation_end_ctrl": f"0x{case.trace.activation_end_ctrl_address:07x}",
                    "raw_scale_end_ctrl": f"0x{case.trace.raw_scale_end_ctrl_address:07x}",
                    "combined_end_ctrl": f"0x{case.trace.combined_end_ctrl_address:07x}",
                },
                "bytes": {
                    "upload": case.upload_bytes,
                    "source": case.source_bytes,
                    "raw_scale": case.raw_scale_bytes,
                    "activation": case.activation_bytes,
                    "combined_scale": case.combined_scale_bytes,
                    "result": case.result_bytes,
                },
                "metadata": {
                    "all_zero": case.all_zero,
                    "max_abs_q10": case.max_abs_q10,
                    "max_mantissa_binary32": case.max_mantissa_binary32,
                    "max_exponent_binary32": case.max_exponent_binary32,
                    "max_abs_binary32_bits": f"0x{case.max_abs_binary32_bits:08x}",
                    "saturated_count": case.saturated_count,
                },
                "trace": {
                    "source_read_commands": case.trace.source_read_commands,
                    "source_read_beats": case.trace.source_read_beats,
                    "raw_scale_read_commands": case.trace.raw_scale_read_commands,
                    "raw_scale_read_beats": case.trace.raw_scale_read_beats,
                    "activation_write_commands": case.trace.activation_write_commands,
                    "activation_write_beats": case.trace.activation_write_beats,
                    "combined_write_commands": case.trace.combined_write_commands,
                    "combined_write_beats": case.trace.combined_write_beats,
                },
                "sha256": {
                    "source": _sha256(_array_bytes(case.source_values, "<i8" if case.source_q28 else "<i2")),
                    "raw_scale_fp16": _sha256(_array_bytes(case.raw_scale_fp16_bits, "<u2")),
                    "activation_int8": _sha256(_array_bytes(case.activation_int8, "i1")),
                    "combined_scale_padded": _sha256(_array_bytes(case.combined_scale_q28_padded, "<u4")),
                    "config": _sha256(build_config_payload(case)),
                    "upload": _sha256(build_upload_payload(case)),
                    "result": _sha256(expected_result_payload(case)),
                },
            }
            for case in cases
        ],
    }


def validate_manifest(
    cases: Sequence[RuntimeQuantizerValidationCase],
    manifest_path: Path = DEFAULT_MANIFEST,
) -> dict[str, object]:
    actual = fixed_manifest(cases)
    expected = json.loads(manifest_path.read_text(encoding="utf-8"))
    if actual != expected:
        raise RuntimeQuantizerValidationError(f"运行时量化固定清单不一致：{manifest_path}")
    return expected


def transaction_stress(rounds: int = 1000, seed: int = DEFAULT_STRESS_SEED) -> None:
    """随机检查尺寸、padding、burst 分段和地址末端公式。"""

    if rounds <= 0:
        raise RuntimeQuantizerValidationError("rounds 必须大于 0")
    rng = np.random.default_rng(seed)
    supported = (
        (False, 896, 128, 14),
        (False, 896, 896, 14),
        (False, 896, 4864, 14),
        (True, 896, 896, 14),
        (True, 4864, 896, 76),
    )
    for _ in range(rounds):
        source_q28, length, rows, groups = supported[int(rng.integers(0, len(supported)))]
        addresses = rng.integers(0, 1 << 24, size=4, dtype=np.uint32)
        addresses = (addresses & np.uint32(0xFFFF_FFF8)).astype(np.uint32)
        trace = _expected_trace(
            source_q28=source_q28,
            vector_length=length,
            rows=rows,
            groups=groups,
            source_ctrl_address=int(addresses[0]),
            activation_ctrl_address=int(addresses[1]),
            raw_scale_ctrl_address=int(addresses[2]),
            combined_scale_ctrl_address=int(addresses[3]),
        )
        source_elements_per_beat = 4 if source_q28 else 16
        assert trace.source_end_ctrl_address == int(addresses[0]) + (length // source_elements_per_beat) * 8
        assert trace.raw_scale_end_ctrl_address == int(addresses[2]) + ((rows * groups) // 16) * 8
        assert trace.activation_end_ctrl_address == int(addresses[1]) + (length // 32) * 8
        assert trace.combined_end_ctrl_address == int(addresses[3]) + rows * (((groups + 7) // 8)) * 8
        assert trace.total_read_beats >= trace.total_read_commands
        assert trace.total_write_beats == trace.total_write_commands


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="G2 运行时量化 DDR3 验证清单")
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--rounds", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=DEFAULT_STRESS_SEED)
    parser.add_argument("action", choices=("summary", "verify", "print-manifest", "stress"))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.action == "stress":
        transaction_stress(args.rounds, args.seed)
        print(f"运行时量化事务压力通过：{args.rounds}/{args.rounds} PASS, seed={args.seed}")
        return 0
    cases = build_fixed_validation_cases(args.image)
    for case in cases:
        verify_upload_roundtrip(case)
    if args.action == "verify":
        validate_manifest(cases, args.manifest)
        print(f"运行时量化固定清单：{len(cases)}/{len(cases)} PASS")
    elif args.action == "print-manifest":
        print(json.dumps(fixed_manifest(cases), ensure_ascii=False, indent=2))
    else:
        for case in cases:
            print(
                f"{case.matrix_id}: {case.name:9s} source={'Q28' if case.source_q28 else 'Q6.10'} "
                f"shape=({case.rows},{case.vector_length}) groups={case.groups}->{case.padded_groups} "
                f"upload={case.upload_bytes} result={case.result_bytes}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
