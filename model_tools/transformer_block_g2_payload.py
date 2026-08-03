#!/usr/bin/env python3
"""G2 完整 layer0 Transformer Block 的 DDR3 常驻参数与动态用例载荷。

本模块只组合已经验证的软件权威定义，不重新定义任何量化规则：

- 地址来自 :mod:`transformer_block_reference`；
- INT4 nibble 顺序来自 :mod:`linear_quant_reference`；
- gamma、bias、RMS/Softmax/SiLU 查表来自既有定点参考；
- 动态 hidden、RoPE trig 和历史 K/V 来自同一个
  :class:`TransformerBlockCase`。

所有上传事务均按 DDR3 Controller 的 32-bit 地址表示，且载荷长度按 256-bit
beat（32 B）对齐，可直接交给 ``transformer_block_host_ctrl`` 的 ``W`` 命令。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

try:
    from .elementwise_fixed_reference import build_silu_pwl_endpoints
    from .linear_quant_reference import (
        pack_int4_low_nibble_first,
        quantize_signed_q28,
    )
    from .rmsnorm_fixed_reference import (
        LUT_ONLY_INDEX_BITS,
        build_rsqrt_lut,
        quantize_gamma_q6_10,
    )
    from .rope_fixed_reference import generate_trig_row
    from .softmax_fixed_reference import build_exp_lut_payload
    from .transformer_block_reference import (
        DEFAULT_IMAGE,
        BlockContext,
        MemoryRegion,
        TransformerBlockCase,
        build_case,
        build_fixed_real_cases,
        kv_slot_byte_addresses,
        load_context,
        parameter_regions,
        scratch_regions,
        sha256_array,
        validate_memory_layout,
    )
except ImportError:
    from elementwise_fixed_reference import build_silu_pwl_endpoints
    from linear_quant_reference import pack_int4_low_nibble_first, quantize_signed_q28
    from rmsnorm_fixed_reference import (
        LUT_ONLY_INDEX_BITS,
        build_rsqrt_lut,
        quantize_gamma_q6_10,
    )
    from rope_fixed_reference import generate_trig_row
    from softmax_fixed_reference import build_exp_lut_payload
    from transformer_block_reference import (
        DEFAULT_IMAGE,
        BlockContext,
        MemoryRegion,
        TransformerBlockCase,
        build_case,
        build_fixed_real_cases,
        kv_slot_byte_addresses,
        load_context,
        parameter_regions,
        scratch_regions,
        sha256_array,
        validate_memory_layout,
    )

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

BEAT_BYTES = 32
CONTROLLER_WORD_BYTES = 4
KV_VECTOR_BYTES = 2 * 64 * 8


class TransformerBlockPayloadError(ValueError):
    """表示 G2 DDR3 上传事务、参数形状或固定数据不合法。"""


@dataclass(frozen=True)
class DDRUpload:
    """一段可由 host ``W`` 命令直接上传的 DDR3 数据。"""

    name: str
    controller_address: int
    payload: bytes
    persistent: bool

    @property
    def byte_address(self) -> int:
        return self.controller_address * CONTROLLER_WORD_BYTES

    @property
    def end_byte_address(self) -> int:
        return self.byte_address + len(self.payload)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.payload).hexdigest()


def _regions_by_name() -> dict[str, MemoryRegion]:
    return {region.name: region for region in (*scratch_regions(), *parameter_regions())}


def _require_beat_aligned(payload: bytes, name: str) -> bytes:
    if not payload:
        raise TransformerBlockPayloadError(f"{name} 载荷不能为空")
    if len(payload) % BEAT_BYTES:
        raise TransformerBlockPayloadError(
            f"{name} 长度 {len(payload)} 未按 {BEAT_BYTES} B 对齐"
        )
    return payload


def _pad_to_beat(payload: bytes) -> bytes:
    padding = (-len(payload)) % BEAT_BYTES
    return payload + bytes(padding)


def _upload(
    regions: dict[str, MemoryRegion],
    region_name: str,
    payload: bytes,
    *,
    persistent: bool,
    allow_padding: bool = False,
) -> DDRUpload:
    try:
        region = regions[region_name]
    except KeyError as error:
        raise TransformerBlockPayloadError(f"未知 DDR3 区域：{region_name}") from error
    resolved = _pad_to_beat(payload) if allow_padding else payload
    _require_beat_aligned(resolved, region_name)
    if len(payload) != region.size_bytes:
        raise TransformerBlockPayloadError(
            f"{region_name} 原始长度 {len(payload)} != 契约 {region.size_bytes}"
        )
    # 允许查表区域尾部仅为满足 256-bit 写事务补 0，但不得越过下一地址区。
    if any(resolved[len(payload) :]):
        raise TransformerBlockPayloadError(f"{region_name} padding 必须全 0")
    return DDRUpload(
        name=region_name,
        controller_address=region.controller_address,
        payload=resolved,
        persistent=persistent,
    )


def _packed_int4_bytes(values: np.ndarray, expected_bytes: int, name: str) -> bytes:
    packed = pack_int4_low_nibble_first(np.asarray(values, dtype=np.int8))
    payload = np.asarray(packed, dtype=np.uint8).tobytes(order="C")
    if len(payload) != expected_bytes:
        raise TransformerBlockPayloadError(
            f"{name} packed INT4 长度 {len(payload)} != {expected_bytes}"
        )
    return payload


def _raw_fp16_scale_bytes(
    values: np.ndarray,
    expected_bytes: int,
    name: str,
) -> bytes:
    source = np.asarray(values, dtype=np.float32)
    fp16 = source.astype("<f2")
    # P50 scale 的源存储即 FP16；若回转不一致，说明数据不再是原始 FP16 值，
    # 此时不能静默生成不同的运行时 combined scale。
    if not np.array_equal(fp16.astype(np.float32), source):
        mismatch = int(np.flatnonzero(fp16.astype(np.float32) != source)[0])
        raise TransformerBlockPayloadError(
            f"{name} 第 {mismatch} 项无法无损回转为 FP16"
        )
    payload = fp16.tobytes(order="C")
    if len(payload) != expected_bytes:
        raise TransformerBlockPayloadError(
            f"{name} FP16 scale 长度 {len(payload)} != {expected_bytes}"
        )
    return payload


def _padded_bias_q28(values: np.ndarray, rows: int, name: str) -> bytes:
    bias_q28, saturated = quantize_signed_q28(np.asarray(values, dtype=np.float32))
    if saturated:
        raise TransformerBlockPayloadError(f"{name} bias Q28 出现 {saturated} 项饱和")
    if bias_q28.shape != (rows,):
        raise TransformerBlockPayloadError(
            f"{name} bias shape {bias_q28.shape} != ({rows},)"
        )
    padded = np.zeros((rows, 4), dtype="<i8")
    padded[:, 0] = bias_q28.astype("<i8")
    return padded.tobytes(order="C")


def build_resident_uploads(
    context: BlockContext | None = None,
    *,
    image_path: Path = DEFAULT_IMAGE,
) -> list[DDRUpload]:
    """构造 layer0 完整 Block 的常驻参数上传事务。

    动态生成的 combined-scale 区不在这里上传；它们由四个运行时量化阶段
    在每次 Block 执行时重建。
    """

    validate_memory_layout()
    resolved = load_context(image_path) if context is None else context
    regions = _regions_by_name()
    uploads: list[DDRUpload] = []

    input_gamma = quantize_gamma_q6_10(resolved.attention.gamma)
    post_gamma = quantize_gamma_q6_10(resolved.post_attention_gamma)
    if input_gamma.clipped_count or post_gamma.clipped_count:
        raise TransformerBlockPayloadError("RMSNorm gamma Q6.10 不应出现饱和")

    uploads.extend(
        (
            _upload(
                regions,
                "input_rms_gamma_q10",
                np.asarray(input_gamma.quantized, dtype="<i2").tobytes(order="C"),
                persistent=True,
            ),
            _upload(
                regions,
                "post_rms_gamma_q10",
                np.asarray(post_gamma.quantized, dtype="<i2").tobytes(order="C"),
                persistent=True,
            ),
            _upload(
                regions,
                "rms_lut_uq12_20",
                np.asarray(
                    build_rsqrt_lut(LUT_ONLY_INDEX_BITS), dtype="<u4"
                ).tobytes(order="C"),
                persistent=True,
            ),
            _upload(
                regions,
                "softmax_exp_lut_q31",
                build_exp_lut_payload()[: regions["softmax_exp_lut_q31"].size_bytes],
                persistent=True,
                allow_padding=True,
            ),
        )
    )

    silu_endpoints = np.zeros(80, dtype="<i2")
    endpoints = np.asarray(build_silu_pwl_endpoints(), dtype="<i2")
    if endpoints.shape != (65,):
        raise TransformerBlockPayloadError("SiLU PWL 端点 shape 必须为 (65,)")
    silu_endpoints[:65] = endpoints
    uploads.append(
        _upload(
            regions,
            "silu_pwl_q10",
            silu_endpoints.tobytes(order="C"),
            persistent=True,
        )
    )

    qkv = resolved.attention.qkv_models
    model_by_region = {
        "q": qkv["q"],
        "k": qkv["k"],
        "v": qkv["v"],
    }
    for key in ("q", "k", "v"):
        model = model_by_region[key]
        weight_region = regions[f"{key}_weight_int4"]
        scale_region = regions[f"{key}_weight_scale_fp16"]
        bias_region = regions[f"{key}_bias_q28"]
        uploads.extend(
            (
                _upload(
                    regions,
                    f"{key}_weight_int4",
                    _packed_int4_bytes(
                        model.weights, weight_region.size_bytes, f"{key}_proj"
                    ),
                    persistent=True,
                ),
                _upload(
                    regions,
                    f"{key}_weight_scale_fp16",
                    _raw_fp16_scale_bytes(
                        model.weight_scales,
                        scale_region.size_bytes,
                        f"{key}_proj",
                    ),
                    persistent=True,
                ),
                _upload(
                    regions,
                    f"{key}_bias_q28",
                    _padded_bias_q28(model.bias, model.spec.rows, f"{key}_proj"),
                    persistent=True,
                ),
            )
        )
        if len(uploads[-1].payload) != bias_region.size_bytes:
            raise TransformerBlockPayloadError(f"{key}_proj padded bias 长度错误")

    projection_models = (
        (
            "oproj",
            resolved.attention.oproj_model,
            "oproj_weight_int4",
            "oproj_weight_scale_fp16",
        ),
        (
            "gate",
            resolved.gate_model,
            "gate_weight_int4",
            "gate_weight_scale_fp16",
        ),
        (
            "up",
            resolved.up_model,
            "up_weight_int4",
            "up_weight_scale_fp16",
        ),
        (
            "down",
            resolved.down_model,
            "down_weight_int4",
            "down_weight_scale_fp16",
        ),
    )
    for label, model, weight_name, scale_name in projection_models:
        uploads.extend(
            (
                _upload(
                    regions,
                    weight_name,
                    _packed_int4_bytes(
                        model.weights, regions[weight_name].size_bytes, label
                    ),
                    persistent=True,
                ),
                _upload(
                    regions,
                    scale_name,
                    _raw_fp16_scale_bytes(
                        model.weight_scales,
                        regions[scale_name].size_bytes,
                        label,
                    ),
                    persistent=True,
                ),
            )
        )

    validate_uploads(uploads)
    return uploads


def build_stress_case(
    context: BlockContext,
    *,
    seed: int,
    index: int,
) -> TransformerBlockCase:
    """构造确定性的完整 Block 随机/地址边界用例。

    每 8 个全局 index 固定覆盖四类关键边界，其余四类使用随机 count/query：

    - query=0、count=1；
    - query=1、count=2；
    - query=15、count=16；
    - query=16383、count=16（1 GiB KV Cache 末端）；
    - 随机 1..16 窗口和随机合法位置。
    """

    if index < 0:
        raise TransformerBlockPayloadError("stress index 不能为负")
    rng = np.random.default_rng((int(seed) + int(index) * 0x9E3779B1) & 0xFFFFFFFF)
    mode = index & 7
    boundary = {
        0: (0, 0),
        1: (1, 0),
        2: (15, 0),
        3: (16383, 16368),
    }
    if mode in boundary:
        query, window_start = boundary[mode]
    else:
        count = int(rng.integers(1, 17))
        query = int(rng.integers(count - 1, 16384))
        window_start = query - count + 1
    hidden_seed_base = int((int(seed) + int(index) * 1_000_003) & 0x7FFFFFFF)
    return build_case(
        context,
        query_position=query,
        window_start=window_start,
        hidden_seed_base=hidden_seed_base,
        token_cache={},
    )


def build_dynamic_uploads(case: TransformerBlockCase) -> list[DDRUpload]:
    """构造一个完整 Block 用例的 hidden、trig 与历史 K/V 上传事务。"""

    regions = _regions_by_name()
    uploads = [
        _upload(
            regions,
            "block_hidden_q10",
            np.asarray(case.block_input_q10, dtype="<i2").tobytes(order="C"),
            persistent=False,
        )
    ]

    trig = generate_trig_row(case.query_position)
    trig_payload = (
        np.asarray(trig.cos_q30, dtype="<i4").tobytes(order="C")
        + np.asarray(trig.sin_q30, dtype="<i4").tobytes(order="C")
    )
    uploads.append(
        _upload(
            regions,
            "rope_trig_q30",
            trig_payload,
            persistent=False,
        )
    )

    expected_history = case.count - 1
    if case.history_k_q28.shape != (expected_history, 2, 64):
        raise TransformerBlockPayloadError("history K shape 错误")
    if case.history_v_q28.shape != (expected_history, 2, 64):
        raise TransformerBlockPayloadError("history V shape 错误")
    for index in range(expected_history):
        position = case.window_start + index
        k_byte_address, v_byte_address = kv_slot_byte_addresses(0, position)
        k_payload = np.asarray(case.history_k_q28[index], dtype="<i8").tobytes(
            order="C"
        )
        v_payload = np.asarray(case.history_v_q28[index], dtype="<i8").tobytes(
            order="C"
        )
        if len(k_payload) != KV_VECTOR_BYTES or len(v_payload) != KV_VECTOR_BYTES:
            raise TransformerBlockPayloadError("历史 K/V 字节长度错误")
        uploads.extend(
            (
                DDRUpload(
                    name=f"kv_history_k_position_{position}",
                    controller_address=k_byte_address // CONTROLLER_WORD_BYTES,
                    payload=k_payload,
                    persistent=False,
                ),
                DDRUpload(
                    name=f"kv_history_v_position_{position}",
                    controller_address=v_byte_address // CONTROLLER_WORD_BYTES,
                    payload=v_payload,
                    persistent=False,
                ),
            )
        )

    validate_uploads(uploads)
    return uploads


def validate_uploads(uploads: Sequence[DDRUpload]) -> None:
    if not uploads:
        raise TransformerBlockPayloadError("上传事务列表不能为空")
    ordered = sorted(uploads, key=lambda item: item.byte_address)
    names: set[str] = set()
    for upload in ordered:
        if upload.name in names:
            raise TransformerBlockPayloadError(f"重复上传名称：{upload.name}")
        names.add(upload.name)
        if upload.controller_address < 0:
            raise TransformerBlockPayloadError(f"{upload.name} 地址为负")
        if upload.controller_address & 0x7:
            raise TransformerBlockPayloadError(
                f"{upload.name} controller 地址未按 256-bit beat 对齐"
            )
        _require_beat_aligned(upload.payload, upload.name)
    for previous, current in zip(ordered, ordered[1:]):
        if previous.end_byte_address > current.byte_address:
            raise TransformerBlockPayloadError(
                f"上传事务重叠：{previous.name} 与 {current.name}"
            )


def uploads_manifest(uploads: Sequence[DDRUpload]) -> list[dict[str, object]]:
    validate_uploads(uploads)
    return [
        {
            "name": upload.name,
            "controller_address": f"0x{upload.controller_address:07x}",
            "byte_address": f"0x{upload.byte_address:08x}",
            "length": len(upload.payload),
            "sha256": upload.sha256,
            "persistent": upload.persistent,
        }
        for upload in uploads
    ]


def fixed_payload_manifest(
    *, image_path: Path = DEFAULT_IMAGE
) -> dict[str, object]:
    context = load_context(image_path)
    resident = build_resident_uploads(context)
    cases = build_fixed_real_cases(image_path=image_path)
    return {
        "schema": "pangu50k.transformer_block_g2_payload.v1",
        "image": str(image_path),
        "resident_total_bytes": sum(len(item.payload) for item in resident),
        "resident": uploads_manifest(resident),
        "cases": [
            {
                "label": case.label,
                "query_position": case.query_position,
                "window_start": case.window_start,
                "count": case.count,
                "uploads": uploads_manifest(build_dynamic_uploads(case)),
                "expected_output_sha256": sha256_array(case.output_q10, "<i2"),
                "intermediate_sha256": {
                    "input_norm_q10": sha256_array(case.input_norm_q10, "<i2"),
                    "q_rope_q28": sha256_array(case.current_q_rope_q28, "<i8"),
                    "k_rope_q28": sha256_array(case.current_k_rope_q28, "<i8"),
                    "v_q28": sha256_array(case.current_v_q28, "<i8"),
                    "scores_q28": sha256_array(case.scores_q28, "<i8"),
                    "probabilities_q31": sha256_array(
                        case.probabilities_q31, "<u4"
                    ),
                    "attention_concat_q28": sha256_array(
                        case.attention_concat_q28, "<i8"
                    ),
                    "oproj_q28": sha256_array(case.oproj_q28, "<i8"),
                    "first_residual_q10": sha256_array(
                        case.first_residual_q10, "<i2"
                    ),
                    "post_attention_norm_q10": sha256_array(
                        case.post_attention_norm_q10, "<i2"
                    ),
                    "gate_q28": sha256_array(case.gate_q28, "<i8"),
                    "up_q28": sha256_array(case.up_q28, "<i8"),
                    "silu_gate_q10": sha256_array(case.silu_gate_q10, "<i2"),
                    "silu_up_q28": sha256_array(case.silu_up_q28, "<i8"),
                    "down_proj_q28": sha256_array(case.down_proj_q28, "<i8"),
                },
            }
            for case in cases
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="生成 G2 完整 Block DDR3 上传清单")
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = fixed_payload_manifest(image_path=args.image)
    text = json.dumps(manifest, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
        print(f"已写入 {args.output}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
