#!/usr/bin/env python3
"""Qwen2.5-0.5B layer0 完整 Transformer Block 硬件等价软件参考。

本模块是 G2 集成阶段的唯一软件入口。它不再把 Attention 和 MLP 当成彼此
独立的固定向量，而是从同一组 block hidden state 出发，严格按已验证定义执行：

hidden Q6.10
-> input RMSNorm
-> Q/K/V
-> RoPE
-> KV history / Attention Score / Softmax / probability*V
-> O_proj
-> 第一处残差
-> post_attention_layernorm
-> gate_proj / up_proj
-> SiLU(gate)
-> SiLU(gate) * up
-> down_proj
-> 第二处残差
-> block output Q6.10。

同时冻结 G2 独立集成工程使用的 DDR3 低端地址、阶段 ID 和动态执行载荷。
KV Cache 地址继续严格复用 F3 已验证布局，不在本阶段重新定义。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

try:
    from .attention_residual_reference import (
        DEFAULT_FIXED_QUERIES,
        DEFAULT_HIDDEN_SEED_BASE,
        DEFAULT_IMAGE,
        AttentionResidualCase,
        AttentionSublayerContext,
        TokenAttentionState,
        build_coherent_case,
        load_context as load_attention_context,
    )
    from .mlp_down_proj_reference import (
        DownProjectionModel,
        case_from_source_q28,
        load_down_projection_model,
    )
    from .mlp_gate_up_reference import (
        ProjectionModel,
        case_from_post_attention_q10,
        load_gate_up_models,
    )
    from .mlp_residual_reference import mlp_residual_q10
    from .mlp_silu_reference import case_from_gate_q28
    from .mlp_silu_up_mul_reference import case_from_inputs
    from .post_attention_layernorm_reference import case_from_input_q10, load_gamma
    from .rope_fixed_reference import generate_trig_row
except ImportError:
    from attention_residual_reference import (
        DEFAULT_FIXED_QUERIES,
        DEFAULT_HIDDEN_SEED_BASE,
        DEFAULT_IMAGE,
        AttentionResidualCase,
        AttentionSublayerContext,
        TokenAttentionState,
        build_coherent_case,
        load_context as load_attention_context,
    )
    from mlp_down_proj_reference import (
        DownProjectionModel,
        case_from_source_q28,
        load_down_projection_model,
    )
    from mlp_gate_up_reference import (
        ProjectionModel,
        case_from_post_attention_q10,
        load_gate_up_models,
    )
    from mlp_residual_reference import mlp_residual_q10
    from mlp_silu_reference import case_from_gate_q28
    from mlp_silu_up_mul_reference import case_from_inputs
    from post_attention_layernorm_reference import case_from_input_q10, load_gamma
    from rope_fixed_reference import generate_trig_row

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = Path(__file__).with_name("transformer_block_g2_reference.json")
DEFAULT_STRESS_SEED = 20260818

HIDDEN_SIZE = 896
INTERMEDIATE_SIZE = 4864
Q_HEADS = 14
KV_HEADS = 2
HEAD_DIM = 64
MAX_WINDOW = 16
LOW_DDR_LIMIT_BYTES = 0x08000000
KV_CACHE_BASE_BYTES = 0x08000000
KV_LAYER_STRIDE_BYTES = 0x02000000
KV_TOKEN_STRIDE_BYTES = 0x00000800
KV_VALUE_OFFSET_BYTES = 0x00000400
EXECUTION_MAGIC = b"P50G2B1\0"
EXECUTION_VERSION = 1
EXECUTION_HEADER = struct.Struct("<8s6I32s")


class TransformerBlockReferenceError(ValueError):
    """表示 G2 完整 Block 输入、布局、定点结果或固定清单不合法。"""


@dataclass(frozen=True)
class MemoryRegion:
    name: str
    byte_address: int
    size_bytes: int
    data_format: str
    producer: str
    consumer: str
    lifetime: str

    @property
    def controller_address(self) -> int:
        if self.byte_address & 0x3:
            raise TransformerBlockReferenceError(f"{self.name} 字节地址未按 4 字节对齐")
        return self.byte_address >> 2

    @property
    def end_byte_address(self) -> int:
        return self.byte_address + self.size_bytes


@dataclass(frozen=True)
class BlockContext:
    attention: AttentionSublayerContext
    post_attention_gamma: np.ndarray
    gate_model: ProjectionModel
    up_model: ProjectionModel
    down_model: DownProjectionModel


@dataclass(frozen=True)
class LinearInvocation:
    name: str
    stage_name: str
    source_tensor: str
    source_format: str
    rows: int
    columns: int
    groups: int
    act_beats: int
    weight_beats_per_row: int
    scale_beats_per_row: int
    has_bias: bool
    activation_ctrl_address: int
    weight_ctrl_address: int
    raw_weight_scale_ctrl_address: int
    combined_scale_ctrl_address: int
    bias_ctrl_address: int
    result_ctrl_address: int


@dataclass(frozen=True)
class TransformerBlockCase:
    label: str
    query_position: int
    window_start: int
    count: int
    hidden_seed_base: int
    block_input_q10: np.ndarray
    input_norm_q10: np.ndarray
    current_q_rope_q28: np.ndarray
    current_k_rope_q28: np.ndarray
    current_v_q28: np.ndarray
    history_k_q28: np.ndarray
    history_v_q28: np.ndarray
    scores_q28: np.ndarray
    probabilities_q31: np.ndarray
    attention_concat_q28: np.ndarray
    oproj_q28: np.ndarray
    first_residual_q10: np.ndarray
    post_attention_norm_q10: np.ndarray
    gate_q28: np.ndarray
    up_q28: np.ndarray
    silu_gate_q10: np.ndarray
    silu_up_q28: np.ndarray
    down_proj_q28: np.ndarray
    down_proj_q10: np.ndarray
    output_q10: np.ndarray
    first_rescale_saturated_count: int
    first_residual_saturated_count: int
    silu_rescale_saturated_count: int
    silu_up_saturated_count: int
    down_rescale_saturated_count: int
    second_residual_saturated_count: int


def _region(
    name: str,
    address: int,
    size: int,
    data_format: str,
    producer: str,
    consumer: str,
    lifetime: str = "one block invocation",
) -> MemoryRegion:
    return MemoryRegion(name, address, size, data_format, producer, consumer, lifetime)


def scratch_regions() -> tuple[MemoryRegion, ...]:
    """返回 G2 低端 DDR3 scratch 地址表，所有区域至少 4 KiB 起始对齐。"""

    return (
        _region("block_hidden_q10", 0x00000000, 1792, "int16 Q6.10 [896]", "host/previous layer", "input RMSNorm and residual1"),
        _region("input_norm_q10", 0x00001000, 1792, "int16 Q6.10 [896]", "input RMSNorm", "Q/K/V linear"),
        _region("q_q28", 0x00002000, 7168, "int64 Q28 [14,64]", "Q linear", "RoPE"),
        _region("k_q28", 0x00004000, 1024, "int64 Q28 [2,64]", "K linear", "RoPE"),
        _region("v_q28", 0x00005000, 1024, "int64 Q28 [2,64]", "V linear", "KV Cache"),
        _region("q_rope_q28", 0x00006000, 7168, "int64 Q28 [14,64]", "RoPE", "Attention Score"),
        _region("k_rope_q28", 0x00008000, 1024, "int64 Q28 [2,64]", "RoPE", "KV Cache"),
        _region("scores_q28", 0x00009000, 1792, "int64 Q28 [14,16]", "Attention Score", "Softmax"),
        _region("probabilities_q31", 0x0000A000, 896, "uint32 UQ1.31 [14,16]", "Softmax", "Attention output"),
        _region("attention_concat_q28", 0x0000B000, 7168, "int64 Q28 [896]", "Attention output", "O_proj"),
        _region("oproj_q28", 0x0000D000, 7168, "int64 Q28 [896]", "O_proj", "residual1"),
        _region("attention_residual_q10", 0x0000F000, 1792, "int16 Q6.10 [896]", "residual1", "post RMSNorm and residual2"),
        _region("post_attention_norm_q10", 0x00010000, 1792, "int16 Q6.10 [896]", "post RMSNorm", "gate/up linear"),
        _region("gate_q28", 0x00011000, 38912, "int64 Q28 [4864]", "gate_proj", "SiLU"),
        _region("up_q28", 0x0001B000, 38912, "int64 Q28 [4864]", "up_proj", "SiLU*up"),
        _region("silu_gate_q10", 0x00025000, 9728, "int16 Q6.10 [4864]", "SiLU", "SiLU*up"),
        _region("silu_up_q28", 0x00028000, 38912, "int64 Q28 [4864]", "SiLU*up", "down_proj"),
        _region("down_proj_q28", 0x00032000, 7168, "int64 Q28 [896]", "down_proj", "residual2"),
        _region("block_output_q10", 0x00034000, 1792, "int16 Q6.10 [896]", "residual2", "host/next layer", "persistent until consumed"),
        _region("linear_activation_int8", 0x00035000, 4864, "int8 symmetric [max 4864]", "runtime quantizer", "shared Linear engine"),
        _region("linear_quant_metadata", 0x00037000, 256, "max-abs binary32 / activation-scale metadata", "runtime quantizer", "scale builder/debug"),
        _region("execution_payload", 0x00100000, 0x00010000, "G2 execution header/hidden/trig/history staging", "host", "G2 loader"),
        _region("input_rms_gamma_q10", 0x00200000, 1792, "int16 Q6.10 [896]", "model loader", "input RMSNorm", "layer0 resident"),
        _region("post_rms_gamma_q10", 0x00201000, 1792, "int16 Q6.10 [896]", "model loader", "post RMSNorm", "layer0 resident"),
        _region("rms_lut_uq12_20", 0x00202000, 1024, "uint32 UQ12.20 [256]", "model loader", "both RMSNorm", "layer0 resident"),
        _region("softmax_exp_lut_q31", 0x00203000, 2052, "uint32 UQ1.31 [513]", "model loader", "Softmax", "layer0 resident"),
        _region("silu_pwl_q10", 0x00204000, 160, "int16 Q6.10 [80 padded]", "model loader", "SiLU", "layer0 resident"),
        _region("rope_trig_q30", 0x00205000, 256, "cos[32]+sin[32] int32 Q1.30", "host/table loader", "RoPE"),
    )


def parameter_regions() -> tuple[MemoryRegion, ...]:
    """返回 layer0 Linear 参数区。scale 为每次执行按激活 scale 生成的 UQ4.28。"""

    return (
        _region("q_weight_int4", 0x01000000, 401408, "packed int4 [896,896]", "model loader", "shared linear", "layer0 resident"),
        _region("q_scale_uq4_28", 0x01062000, 57344, "uint32 UQ4.28 padded [896,16]", "scale loader", "shared linear"),
        _region("q_bias_q28", 0x01070000, 28672, "int64 Q28 padded [896,4]", "model loader", "shared linear", "layer0 resident"),
        _region("k_weight_int4", 0x01077000, 57344, "packed int4 [128,896]", "model loader", "shared linear", "layer0 resident"),
        _region("k_scale_uq4_28", 0x01085000, 8192, "uint32 UQ4.28 padded [128,16]", "scale loader", "shared linear"),
        _region("k_bias_q28", 0x01087000, 4096, "int64 Q28 padded [128,4]", "model loader", "shared linear", "layer0 resident"),
        _region("v_weight_int4", 0x01088000, 57344, "packed int4 [128,896]", "model loader", "shared linear", "layer0 resident"),
        _region("v_scale_uq4_28", 0x01096000, 8192, "uint32 UQ4.28 padded [128,16]", "scale loader", "shared linear"),
        _region("v_bias_q28", 0x01098000, 4096, "int64 Q28 padded [128,4]", "model loader", "shared linear", "layer0 resident"),
        _region("oproj_weight_int4", 0x01099000, 401408, "packed int4 [896,896]", "model loader", "shared linear", "layer0 resident"),
        _region("oproj_scale_uq4_28", 0x010FB000, 57344, "uint32 UQ4.28 padded [896,16]", "scale loader", "shared linear"),
        _region("gate_weight_int4", 0x01109000, 2179072, "packed int4 [4864,896]", "model loader", "shared linear", "layer0 resident"),
        _region("gate_scale_uq4_28", 0x0131D000, 311296, "uint32 UQ4.28 padded [4864,16]", "scale loader", "shared linear"),
        _region("up_weight_int4", 0x01369000, 2179072, "packed int4 [4864,896]", "model loader", "shared linear", "layer0 resident"),
        _region("up_scale_uq4_28", 0x0157D000, 311296, "uint32 UQ4.28 padded [4864,16]", "scale loader", "shared linear"),
        _region("down_weight_int4", 0x015C9000, 2179072, "packed int4 [896,4864]", "model loader", "shared linear", "layer0 resident"),
        _region("down_scale_uq4_28", 0x017DD000, 286720, "uint32 UQ4.28 padded [896,80]", "runtime scale builder", "shared linear"),
        _region("q_weight_scale_fp16", 0x01830000, 25088, "FP16 [896,14]", "model loader", "runtime scale builder", "layer0 resident"),
        _region("k_weight_scale_fp16", 0x01837000, 3584, "FP16 [128,14]", "model loader", "runtime scale builder", "layer0 resident"),
        _region("v_weight_scale_fp16", 0x01838000, 3584, "FP16 [128,14]", "model loader", "runtime scale builder", "layer0 resident"),
        _region("oproj_weight_scale_fp16", 0x01839000, 25088, "FP16 [896,14]", "model loader", "runtime scale builder", "layer0 resident"),
        _region("gate_weight_scale_fp16", 0x01840000, 136192, "FP16 [4864,14]", "model loader", "runtime scale builder", "layer0 resident"),
        _region("up_weight_scale_fp16", 0x01862000, 136192, "FP16 [4864,14]", "model loader", "runtime scale builder", "layer0 resident"),
        _region("down_weight_scale_fp16", 0x01884000, 136192, "FP16 [896,76]", "model loader", "runtime scale builder", "layer0 resident"),
    )


def linear_invocations() -> tuple[LinearInvocation, ...]:
    """冻结七个 layer0 Linear 的运行时矩阵形状、量化来源和 DDR3 地址。"""

    regions = {region.name: region for region in (*scratch_regions(), *parameter_regions())}

    def build(
        name: str,
        stage_name: str,
        source_tensor: str,
        source_format: str,
        rows: int,
        columns: int,
        weight: str,
        raw_scale: str,
        combined_scale: str,
        bias: str | None,
        result: str,
    ) -> LinearInvocation:
        groups = columns // 64
        return LinearInvocation(
            name=name,
            stage_name=stage_name,
            source_tensor=source_tensor,
            source_format=source_format,
            rows=rows,
            columns=columns,
            groups=groups,
            act_beats=columns // 32,
            weight_beats_per_row=columns // 64,
            scale_beats_per_row=(groups + 7) // 8,
            has_bias=bias is not None,
            activation_ctrl_address=regions["linear_activation_int8"].controller_address,
            weight_ctrl_address=regions[weight].controller_address,
            raw_weight_scale_ctrl_address=regions[raw_scale].controller_address,
            combined_scale_ctrl_address=regions[combined_scale].controller_address,
            bias_ctrl_address=regions[bias].controller_address if bias else 0,
            result_ctrl_address=regions[result].controller_address,
        )

    return (
        build("q_proj", "Q_LINEAR", "input_norm_q10", "Q6.10", 896, 896, "q_weight_int4", "q_weight_scale_fp16", "q_scale_uq4_28", "q_bias_q28", "q_q28"),
        build("k_proj", "K_LINEAR", "input_norm_q10", "Q6.10", 128, 896, "k_weight_int4", "k_weight_scale_fp16", "k_scale_uq4_28", "k_bias_q28", "k_q28"),
        build("v_proj", "V_LINEAR", "input_norm_q10", "Q6.10", 128, 896, "v_weight_int4", "v_weight_scale_fp16", "v_scale_uq4_28", "v_bias_q28", "v_q28"),
        build("o_proj", "OPROJ_LINEAR", "attention_concat_q28", "Q28 via binary32", 896, 896, "oproj_weight_int4", "oproj_weight_scale_fp16", "oproj_scale_uq4_28", None, "oproj_q28"),
        build("gate_proj", "GATE_LINEAR", "post_attention_norm_q10", "Q6.10", 4864, 896, "gate_weight_int4", "gate_weight_scale_fp16", "gate_scale_uq4_28", None, "gate_q28"),
        build("up_proj", "UP_LINEAR", "post_attention_norm_q10", "Q6.10", 4864, 896, "up_weight_int4", "up_weight_scale_fp16", "up_scale_uq4_28", None, "up_q28"),
        build("down_proj", "DOWN_LINEAR", "silu_up_q28", "Q28 via binary32", 896, 4864, "down_weight_int4", "down_weight_scale_fp16", "down_scale_uq4_28", None, "down_proj_q28"),
    )


STAGES: tuple[tuple[int, str, str, str], ...] = (
    (0x00, "IDLE", "configuration valid", "LOAD_INPUT_RMS"),
    (0x01, "INPUT_RMS", "block_hidden_q10", "input_norm_q10"),
    (0x02, "QKV_QUANT", "input_norm_q10+q/k/v FP16 scales", "INT8 activation+q/k/v UQ4.28 scales"),
    (0x03, "Q_LINEAR", "QKV quantized input", "q_q28"),
    (0x04, "K_LINEAR", "QKV quantized input", "k_q28"),
    (0x05, "V_LINEAR", "QKV quantized input", "v_q28"),
    (0x06, "ROPE", "q_q28+k_q28+trig", "q_rope_q28+k_rope_q28"),
    (0x07, "KV_WRITE", "k_rope_q28+v_q28", "F3 KV slot"),
    (0x08, "ATTENTION_SCORE", "q_rope_q28+K history", "scores_q28"),
    (0x09, "SOFTMAX", "scores_q28", "probabilities_q31"),
    (0x0A, "ATTENTION_OUTPUT", "probabilities_q31+V history", "attention_concat_q28"),
    (0x0B, "OPROJ_QUANT", "attention_concat_q28+O FP16 scales", "INT8 activation+O UQ4.28 scales"),
    (0x0C, "OPROJ_LINEAR", "O quantized input", "oproj_q28"),
    (0x0D, "RESIDUAL1", "block_hidden_q10+oproj_q28", "attention_residual_q10"),
    (0x0E, "POST_RMS", "attention_residual_q10", "post_attention_norm_q10"),
    (0x0F, "GATE_UP_QUANT", "post_attention_norm_q10+gate/up FP16 scales", "INT8 activation+gate/up UQ4.28 scales"),
    (0x10, "GATE_LINEAR", "gate/up quantized input", "gate_q28"),
    (0x11, "UP_LINEAR", "gate/up quantized input", "up_q28"),
    (0x12, "SILU", "gate_q28", "silu_gate_q10"),
    (0x13, "SILU_UP_MUL", "silu_gate_q10+up_q28", "silu_up_q28"),
    (0x14, "DOWN_QUANT", "silu_up_q28+down FP16 scales", "INT8 activation+down UQ4.28 scales"),
    (0x15, "DOWN_LINEAR", "down quantized input", "down_proj_q28"),
    (0x16, "RESIDUAL2", "attention_residual_q10+down_proj_q28", "block_output_q10"),
    (0x17, "DONE", "block_output_q10 valid", "host/next layer"),
    (0x1F, "ERROR", "error_code valid", "reset required"),
)


def validate_memory_layout() -> None:
    regions = sorted((*scratch_regions(), *parameter_regions()), key=lambda item: item.byte_address)
    for region in regions:
        if region.byte_address < 0 or region.size_bytes <= 0:
            raise TransformerBlockReferenceError(f"非法内存区域：{region.name}")
        if region.byte_address & 0x3:
            raise TransformerBlockReferenceError(f"{region.name} 地址未按 4 字节对齐")
        if region.end_byte_address > LOW_DDR_LIMIT_BYTES:
            raise TransformerBlockReferenceError(f"{region.name} 越过低端 128 MiB")
    for previous, current in zip(regions, regions[1:]):
        if previous.end_byte_address > current.byte_address:
            raise TransformerBlockReferenceError(
                f"DDR3 区域重叠：{previous.name} 与 {current.name}"
            )
    if KV_CACHE_BASE_BYTES != LOW_DDR_LIMIT_BYTES:
        raise TransformerBlockReferenceError("KV Cache 必须从低端 128 MiB 之后开始")


def kv_slot_byte_addresses(layer: int, position: int) -> tuple[int, int]:
    if not 0 <= layer < 28:
        raise TransformerBlockReferenceError("layer 必须在 0..27")
    if not 0 <= position < 16384:
        raise TransformerBlockReferenceError("position 必须在 0..16383")
    k_address = KV_CACHE_BASE_BYTES + layer * KV_LAYER_STRIDE_BYTES + position * KV_TOKEN_STRIDE_BYTES
    v_address = k_address + KV_VALUE_OFFSET_BYTES
    return k_address, v_address


def load_context(image_path: Path = DEFAULT_IMAGE) -> BlockContext:
    attention = load_attention_context(image_path)
    post_gamma = load_gamma(attention.image)
    gate_model, up_model = load_gate_up_models(attention.image)
    down_model = load_down_projection_model(attention.image)
    return BlockContext(
        attention=attention,
        post_attention_gamma=post_gamma,
        gate_model=gate_model,
        up_model=up_model,
        down_model=down_model,
    )


def _history_arrays(
    token_cache: dict[int, TokenAttentionState],
    *,
    window_start: int,
    query_position: int,
) -> tuple[np.ndarray, np.ndarray]:
    positions = range(window_start, query_position)
    states = [token_cache[position] for position in positions]
    if not states:
        return (
            np.empty((0, KV_HEADS, HEAD_DIM), dtype=np.int64),
            np.empty((0, KV_HEADS, HEAD_DIM), dtype=np.int64),
        )
    return (
        np.stack([state.k_rope_q28 for state in states], axis=0).astype(np.int64),
        np.stack([state.v_q28 for state in states], axis=0).astype(np.int64),
    )


def build_case(
    context: BlockContext,
    *,
    query_position: int,
    window_start: int,
    hidden_seed_base: int = DEFAULT_HIDDEN_SEED_BASE,
    token_cache: dict[int, TokenAttentionState] | None = None,
) -> TransformerBlockCase:
    cache = {} if token_cache is None else token_cache
    attention: AttentionResidualCase = build_coherent_case(
        context.attention,
        query_position=query_position,
        window_start=window_start,
        hidden_seed_base=hidden_seed_base,
        token_cache=cache,
    )
    current = cache[int(query_position)]
    history_k, history_v = _history_arrays(
        cache,
        window_start=int(window_start),
        query_position=int(query_position),
    )
    post_norm = case_from_input_q10(
        input_q10=attention.output_q10,
        gamma_values=context.post_attention_gamma,
        label=f"G2 post RMS query={query_position}",
        query_position=query_position,
        count=attention.count,
    )
    gate_up = case_from_post_attention_q10(
        context.gate_model,
        context.up_model,
        post_norm.output_lut_q10,
        label=f"G2 gate/up query={query_position}",
        query_position=query_position,
        count=attention.count,
    )
    silu = case_from_gate_q28(
        gate_up.gate.expected_q28,
        label=f"G2 SiLU query={query_position}",
        query_position=query_position,
        count=attention.count,
    )
    silu_up = case_from_inputs(
        silu.output_pwl_q10,
        gate_up.up.expected_q28,
        label=f"G2 SiLU*up query={query_position}",
        query_position=query_position,
        count=attention.count,
    )
    down = case_from_source_q28(
        context.down_model,
        silu_up.output_q28,
        label=f"G2 down_proj query={query_position}",
        query_position=query_position,
        count=attention.count,
    )
    output, down_q10, down_rescale_sat, residual2_sat = mlp_residual_q10(
        attention.output_q10,
        down.expected_q28,
    )
    return TransformerBlockCase(
        label=f"layer0 complete Transformer Block query={query_position}, window={window_start}..{query_position}",
        query_position=int(query_position),
        window_start=int(window_start),
        count=int(attention.count),
        hidden_seed_base=int(hidden_seed_base),
        block_input_q10=current.hidden_q10.astype(np.int16).copy(),
        input_norm_q10=current.norm_q10.astype(np.int16).copy(),
        current_q_rope_q28=current.q_rope_q28.astype(np.int64).copy(),
        current_k_rope_q28=current.k_rope_q28.astype(np.int64).copy(),
        current_v_q28=current.v_q28.astype(np.int64).copy(),
        history_k_q28=history_k,
        history_v_q28=history_v,
        scores_q28=attention.scores_q28.astype(np.int64).copy(),
        probabilities_q31=attention.probabilities_q31.astype(np.uint32).copy(),
        attention_concat_q28=attention.attention_concat_q28.astype(np.int64).copy(),
        oproj_q28=attention.oproj_q28.astype(np.int64).copy(),
        first_residual_q10=attention.output_q10.astype(np.int16).copy(),
        post_attention_norm_q10=post_norm.output_lut_q10.astype(np.int16).copy(),
        gate_q28=gate_up.gate.expected_q28.astype(np.int64).copy(),
        up_q28=gate_up.up.expected_q28.astype(np.int64).copy(),
        silu_gate_q10=silu.output_pwl_q10.astype(np.int16).copy(),
        silu_up_q28=silu_up.output_q28.astype(np.int64).copy(),
        down_proj_q28=down.expected_q28.astype(np.int64).copy(),
        down_proj_q10=down_q10.astype(np.int16).copy(),
        output_q10=output.astype(np.int16).copy(),
        first_rescale_saturated_count=int(attention.oproj_rescale_saturated_count),
        first_residual_saturated_count=int(attention.residual_saturated_count),
        silu_rescale_saturated_count=int(silu.rescale_saturated_count),
        silu_up_saturated_count=int(silu_up.saturated_count),
        down_rescale_saturated_count=int(down_rescale_sat),
        second_residual_saturated_count=int(residual2_sat),
    )


def build_fixed_real_cases(
    *,
    image_path: Path = DEFAULT_IMAGE,
    hidden_seed_base: int = DEFAULT_HIDDEN_SEED_BASE,
    queries: Iterable[int] = DEFAULT_FIXED_QUERIES,
) -> list[TransformerBlockCase]:
    context = load_context(image_path)
    cache: dict[int, TokenAttentionState] = {}
    cases: list[TransformerBlockCase] = []
    for query in queries:
        resolved = int(query)
        start = max(0, resolved - (MAX_WINDOW - 1))
        cases.append(
            build_case(
                context,
                query_position=resolved,
                window_start=start,
                hidden_seed_base=hidden_seed_base,
                token_cache=cache,
            )
        )
    return cases


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_array(array: np.ndarray, dtype: str | np.dtype) -> str:
    return sha256_bytes(np.asarray(array, dtype=dtype).tobytes(order="C"))


def sha256_json(value: object) -> str:
    """对 JSON 可序列化对象执行稳定、紧凑的 UTF-8 SHA256。"""

    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256_bytes(payload)


def build_execution_payload(case: TransformerBlockCase) -> bytes:
    """构造 G2 动态载荷：64 B header + hidden + 当前 trig + 历史 K/V。"""

    history_count = case.count - 1
    if case.history_k_q28.shape != (history_count, KV_HEADS, HEAD_DIM):
        raise TransformerBlockReferenceError("history K 形状错误")
    if case.history_v_q28.shape != (history_count, KV_HEADS, HEAD_DIM):
        raise TransformerBlockReferenceError("history V 形状错误")
    trig = generate_trig_row(case.query_position)
    header = EXECUTION_HEADER.pack(
        EXECUTION_MAGIC,
        EXECUTION_VERSION,
        case.query_position,
        case.window_start,
        case.count,
        history_count,
        case.hidden_seed_base,
        bytes(32),
    )
    records = bytearray()
    for index in range(history_count):
        records.extend(np.asarray(case.history_k_q28[index], dtype="<i8").tobytes(order="C"))
        records.extend(np.asarray(case.history_v_q28[index], dtype="<i8").tobytes(order="C"))
    payload = (
        header
        + np.asarray(case.block_input_q10, dtype="<i2").tobytes(order="C")
        + np.asarray(trig.cos_q30, dtype="<i4").tobytes(order="C")
        + np.asarray(trig.sin_q30, dtype="<i4").tobytes(order="C")
        + bytes(records)
    )
    if len(payload) > 0x10000:
        raise TransformerBlockReferenceError("G2 动态载荷超过 execution_payload 区域")
    return payload


def verify_execution_payload(case: TransformerBlockCase) -> str:
    payload = build_execution_payload(case)
    header = EXECUTION_HEADER.unpack_from(payload, 0)
    magic, version, query, start, count, history_count, seed_base, reserved = header
    if magic != EXECUTION_MAGIC or version != EXECUTION_VERSION or any(reserved):
        raise TransformerBlockReferenceError("G2 动态载荷 header 错误")
    if (query, start, count, history_count, seed_base) != (
        case.query_position,
        case.window_start,
        case.count,
        case.count - 1,
        case.hidden_seed_base,
    ):
        raise TransformerBlockReferenceError("G2 动态载荷配置字段往返不一致")
    offset = EXECUTION_HEADER.size
    hidden_bytes = HIDDEN_SIZE * 2
    hidden = np.frombuffer(payload[offset : offset + hidden_bytes], dtype="<i2").copy()
    if not np.array_equal(hidden, case.block_input_q10):
        raise TransformerBlockReferenceError("G2 hidden 载荷往返不一致")
    offset += hidden_bytes + 32 * 4 * 2
    for index in range(history_count):
        k = np.frombuffer(payload[offset : offset + 1024], dtype="<i8").reshape(KV_HEADS, HEAD_DIM).copy()
        offset += 1024
        v = np.frombuffer(payload[offset : offset + 1024], dtype="<i8").reshape(KV_HEADS, HEAD_DIM).copy()
        offset += 1024
        if not np.array_equal(k, case.history_k_q28[index]):
            raise TransformerBlockReferenceError(f"history K[{index}] 往返不一致")
        if not np.array_equal(v, case.history_v_q28[index]):
            raise TransformerBlockReferenceError(f"history V[{index}] 往返不一致")
    if offset != len(payload):
        raise TransformerBlockReferenceError("G2 动态载荷尾部长度错误")
    return sha256_bytes(payload)


def _region_dict(region: MemoryRegion) -> dict[str, object]:
    return {
        "name": region.name,
        "byte_address": f"0x{region.byte_address:08x}",
        "controller_address": f"0x{region.controller_address:08x}",
        "size_bytes": region.size_bytes,
        "end_byte_address": f"0x{region.end_byte_address:08x}",
        "format": region.data_format,
        "producer": region.producer,
        "consumer": region.consumer,
        "lifetime": region.lifetime,
    }


def integration_contract() -> dict[str, object]:
    validate_memory_layout()
    return {
        "controller_address_unit_bytes": 4,
        "low_ddr_limit_bytes": LOW_DDR_LIMIT_BYTES,
        "kv_cache": {
            "base_byte_address": f"0x{KV_CACHE_BASE_BYTES:08x}",
            "layer_stride_bytes": KV_LAYER_STRIDE_BYTES,
            "token_stride_bytes": KV_TOKEN_STRIDE_BYTES,
            "v_offset_bytes": KV_VALUE_OFFSET_BYTES,
            "layers": 28,
            "positions": 16384,
        },
        "scratch_and_tables": [_region_dict(region) for region in scratch_regions()],
        "linear_parameters": [_region_dict(region) for region in parameter_regions()],
        "linear_invocations": [
            {
                "name": item.name,
                "stage_name": item.stage_name,
                "source_tensor": item.source_tensor,
                "source_format": item.source_format,
                "rows": item.rows,
                "columns": item.columns,
                "groups": item.groups,
                "act_beats": item.act_beats,
                "weight_beats_per_row": item.weight_beats_per_row,
                "scale_beats_per_row": item.scale_beats_per_row,
                "has_bias": item.has_bias,
                "activation_ctrl_address": f"0x{item.activation_ctrl_address:08x}",
                "weight_ctrl_address": f"0x{item.weight_ctrl_address:08x}",
                "raw_weight_scale_ctrl_address": f"0x{item.raw_weight_scale_ctrl_address:08x}",
                "combined_scale_ctrl_address": f"0x{item.combined_scale_ctrl_address:08x}",
                "bias_ctrl_address": f"0x{item.bias_ctrl_address:08x}",
                "result_ctrl_address": f"0x{item.result_ctrl_address:08x}",
            }
            for item in linear_invocations()
        ],
        "stages": [
            {"id": stage_id, "name": name, "input": input_name, "output": output_name}
            for stage_id, name, input_name, output_name in STAGES
        ],
        "handshake": {
            "start": "one-cycle pulse accepted only while engine idle",
            "busy": "asserted from accepted start until done/error",
            "done": "one-cycle pulse; output DDR write must already be committed",
            "error": "sticky error_code; transition to stage 0x1f until reset",
            "timeout": "controller owns a per-stage watchdog and records failing stage id",
        },
    }


def fixed_manifest(cases: Sequence[TransformerBlockCase]) -> dict[str, object]:
    return {
        "format_version": 1,
        "operator": "qwen2_layer0_complete_transformer_block_g2",
        "definition": {
            "input": "signed int16 Q6.10 [896]",
            "output": "signed int16 Q6.10 [896]",
            "hidden_seed_base": cases[0].hidden_seed_base if cases else DEFAULT_HIDDEN_SEED_BASE,
            "fixed_queries": [case.query_position for case in cases],
            "max_attention_window": MAX_WINDOW,
            "complete_path": [stage[1] for stage in STAGES if stage[1] not in {"IDLE", "DONE", "ERROR"}],
            "execution_payload_header_bytes": EXECUTION_HEADER.size,
            "hardware_status": "software reference and integration contract only; RTL/PDS/board pending",
        },
        "integration_contract": {
            "sha256": sha256_json(integration_contract()),
            "scratch_region_count": len(scratch_regions()),
            "linear_parameter_region_count": len(parameter_regions()),
            "linear_invocation_count": len(linear_invocations()),
            "stage_count": len(STAGES),
            "low_ddr_limit_bytes": LOW_DDR_LIMIT_BYTES,
            "kv_cache_base_bytes": KV_CACHE_BASE_BYTES,
        },
        "cases": [
            {
                "label": case.label,
                "query_position": case.query_position,
                "window_start": case.window_start,
                "count": case.count,
                "history_count": case.count - 1,
                "saturation_counts": {
                    "oproj_q28_to_q10": case.first_rescale_saturated_count,
                    "attention_residual": case.first_residual_saturated_count,
                    "gate_q28_to_q10": case.silu_rescale_saturated_count,
                    "silu_times_up_to_int64": case.silu_up_saturated_count,
                    "down_q28_to_q10": case.down_rescale_saturated_count,
                    "mlp_residual": case.second_residual_saturated_count,
                },
                "sha256": {
                    "block_input_q10": sha256_array(case.block_input_q10, "<i2"),
                    "input_norm_q10": sha256_array(case.input_norm_q10, "<i2"),
                    "current_q_rope_q28": sha256_array(case.current_q_rope_q28, "<i8"),
                    "current_k_rope_q28": sha256_array(case.current_k_rope_q28, "<i8"),
                    "current_v_q28": sha256_array(case.current_v_q28, "<i8"),
                    "scores_q28": sha256_array(case.scores_q28, "<i8"),
                    "probabilities_q31": sha256_array(case.probabilities_q31, "<u4"),
                    "attention_concat_q28": sha256_array(case.attention_concat_q28, "<i8"),
                    "oproj_q28": sha256_array(case.oproj_q28, "<i8"),
                    "first_residual_q10": sha256_array(case.first_residual_q10, "<i2"),
                    "post_attention_norm_q10": sha256_array(case.post_attention_norm_q10, "<i2"),
                    "gate_q28": sha256_array(case.gate_q28, "<i8"),
                    "up_q28": sha256_array(case.up_q28, "<i8"),
                    "silu_gate_q10": sha256_array(case.silu_gate_q10, "<i2"),
                    "silu_up_q28": sha256_array(case.silu_up_q28, "<i8"),
                    "down_proj_q28": sha256_array(case.down_proj_q28, "<i8"),
                    "block_output_q10": sha256_array(case.output_q10, "<i2"),
                    "execution_payload": verify_execution_payload(case),
                },
            }
            for case in cases
        ],
    }


def validate_manifest(
    cases: Sequence[TransformerBlockCase],
    manifest_path: Path = DEFAULT_MANIFEST,
) -> dict[str, object]:
    actual = fixed_manifest(cases)
    expected = json.loads(manifest_path.read_text(encoding="utf-8"))
    if actual != expected:
        raise TransformerBlockReferenceError(f"G2 固定清单不一致：{manifest_path}")
    return expected


def software_stress(
    *,
    rounds: int = 4,
    seed: int = DEFAULT_STRESS_SEED,
    image_path: Path = DEFAULT_IMAGE,
) -> None:
    """对不同 hidden seed 和 query/window 执行完整软件链确定性压力。

    完整 Block 计算量较大；该函数用于 G2 软件集成回归，不替代最终要求的
    1000 轮软件压力和真实 FPGA 压力。
    """

    if rounds <= 0:
        raise TransformerBlockReferenceError("rounds 必须大于 0")
    context = load_context(image_path)
    rng = np.random.default_rng(seed)
    for index in range(rounds):
        query = int(rng.integers(0, MAX_WINDOW))
        start = int(rng.integers(max(0, query - (MAX_WINDOW - 1)), query + 1))
        seed_base = int(seed + 1000 + index * 37)
        first = build_case(
            context,
            query_position=query,
            window_start=start,
            hidden_seed_base=seed_base,
        )
        second = build_case(
            context,
            query_position=query,
            window_start=start,
            hidden_seed_base=seed_base,
        )
        if not np.array_equal(first.output_q10, second.output_q10):
            raise TransformerBlockReferenceError(f"第 {index} 轮完整 Block 非确定性")
        if verify_execution_payload(first) != verify_execution_payload(second):
            raise TransformerBlockReferenceError(f"第 {index} 轮动态载荷非确定性")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="layer0 完整 Transformer Block G2 软件参考")
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--seed", type=int, default=DEFAULT_STRESS_SEED)
    parser.add_argument("action", choices=("summary", "verify", "stress", "print-manifest"))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    validate_memory_layout()
    if args.action == "stress":
        software_stress(rounds=args.rounds, seed=args.seed, image_path=args.image)
        print(f"G2 完整软件链压力通过：{args.rounds}/{args.rounds} PASS, seed={args.seed}")
        return 0
    cases = build_fixed_real_cases(image_path=args.image)
    if args.action == "verify":
        validate_manifest(cases, args.manifest)
        print(f"G2 固定清单验证通过：{len(cases)}/{len(cases)} PASS")
    elif args.action == "print-manifest":
        print(json.dumps(fixed_manifest(cases), ensure_ascii=False, indent=2))
    else:
        print("G2 layer0 完整 Transformer Block 软件参考")
        for case in cases:
            print(
                f"query={case.query_position:2d} count={case.count:2d} "
                f"output_sha256={sha256_array(case.output_q10, '<i2')} "
                f"payload={len(build_execution_payload(case))} B"
            )
        print("硬件状态：尚未完成 RTL/PDS/JTAG/真实板卡验收")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
