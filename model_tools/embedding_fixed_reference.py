#!/usr/bin/env python3
"""Qwen2.5 tied Embedding 的 P50 INT4 -> signed Q6.10 硬件等价参考。

真实张量：``model.embed_tokens.weight``，shape=[151936, 896]，每行 14 个
64 元素分组。P50 保存 packed INT4 权重和 FP16 正 scale。

第一版硬件格式：

- packed INT4：低半字节在前，范围 [-7, 7]；
- 每组 scale：FP16 无损转换为 UQ4.28 uint32；
- 乘积：signed INT4 × UQ4.28；
- 输出：乘积使用 RNE 右移 18 位，从 Q28 转为 signed Q6.10，再显式饱和；
- DDR3 稀疏槽：每个 Token ID 占 512 B，即 16 个 256-bit 拍。

由于真实 embedding FP16 scale 全部可被 UQ4.28 精确表示，固定路径应与
``round_to_nearest_even(INT4 * FP16_scale * 2^10)`` 逐位一致。
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

try:
    from .linear_quant_reference import pack_int4_low_nibble_first, quantize_uq4_28
    from .p50_format import P50FormatError, P50Image
except ImportError:
    from linear_quant_reference import pack_int4_low_nibble_first, quantize_uq4_28
    from p50_format import P50FormatError, P50Image

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

DEFAULT_IMAGE = Path("model_output/yanbo_qwen25_0.5b_int4.p50")
DEFAULT_TENSOR = "model.embed_tokens.weight"
DEFAULT_FIXED_TOKEN_IDS = (0, 1, 2026, 151935)
DEFAULT_RANDOM_SEED = 20260728

VOCAB_SIZE = 151936
EMBEDDING_DIM = 896
GROUP_SIZE = 64
GROUPS_PER_ROW = EMBEDDING_DIM // GROUP_SIZE
PACKED_ROW_BYTES = EMBEDDING_DIM // 2
SCALE_Q28_BYTES = GROUPS_PER_ROW * 4
ROW_SLOT_BYTES = 512
ROW_SLOT_BEATS = ROW_SLOT_BYTES // 32
ROW_SLOT_CTRL_STRIDE = ROW_SLOT_BYTES // 4
RESULT_CTRL_ADDR = 0x02000000

Q10_FRACTION_BITS = 10
Q10_FACTOR = 1 << Q10_FRACTION_BITS
Q10_MIN = -(1 << 15)
Q10_MAX = (1 << 15) - 1
SCALE_Q28_FRACTION_BITS = 28
SCALE_TO_OUTPUT_SHIFT = SCALE_Q28_FRACTION_BITS - Q10_FRACTION_BITS


class EmbeddingReferenceError(ValueError):
    """表示 Embedding 元数据、定点参数或输入不合法。"""


@dataclass(frozen=True)
class EmbeddingReferenceResult:
    """一个真实 Token ID 的完整 Embedding 金标准。"""

    tensor_name: str
    token_id: int
    vocab_size: int
    embedding_dim: int
    group_size: int
    quantized_int4: np.ndarray
    scales_fp16: np.ndarray
    scales_q28: np.ndarray
    p50_float: np.ndarray
    direct_q10: np.ndarray
    fixed_q10: np.ndarray
    fixed_float: np.ndarray
    saturated_count: int

    @property
    def fixed_error_lsb(self) -> np.ndarray:
        return self.fixed_q10.astype(np.int32) - self.direct_q10.astype(np.int32)

    @property
    def q10_quantization_error(self) -> np.ndarray:
        return self.fixed_float.astype(np.float64) - self.p50_float.astype(np.float64)


def _round_shift_signed_array(values: np.ndarray, shift: int) -> np.ndarray:
    """对有符号 int64 数组执行对称 round-to-nearest-even 右移。"""

    array = np.asarray(values, dtype=np.int64)
    if shift < 0:
        return np.left_shift(array, -shift)
    if shift == 0:
        return array.copy()
    magnitude = np.abs(array)
    divisor = np.int64(1 << shift)
    quotient = magnitude // divisor
    remainder = magnitude % divisor
    half = np.int64(1 << (shift - 1))
    increment = (remainder > half) | ((remainder == half) & ((quotient & 1) != 0))
    rounded = quotient + increment.astype(np.int64)
    return np.where(array < 0, -rounded, rounded)


def _validate_token_id(token_id: int, vocab_size: int = VOCAB_SIZE) -> int:
    resolved = int(token_id)
    if not 0 <= resolved < vocab_size:
        raise EmbeddingReferenceError(
            f"Token ID 越界：{resolved}，有效范围 0..{vocab_size - 1}"
        )
    return resolved


def embedding_slot_ctrl_addr(token_id: int, vocab_size: int = VOCAB_SIZE) -> int:
    """返回按 32-bit 字寻址的 DDR3 控制器行槽地址。"""

    resolved = _validate_token_id(token_id, vocab_size)
    return resolved * ROW_SLOT_CTRL_STRIDE


def embedding_slot_byte_offset(token_id: int, vocab_size: int = VOCAB_SIZE) -> int:
    """返回真实 DDR3 字节偏移，便于与 512 B 行槽对应。"""

    resolved = _validate_token_id(token_id, vocab_size)
    return resolved * ROW_SLOT_BYTES


def _validate_quantized_row(values: np.ndarray | Iterable[int]) -> np.ndarray:
    row = np.asarray(values)
    if row.shape != (EMBEDDING_DIM,):
        raise EmbeddingReferenceError(
            f"INT4 行形状错误：{row.shape}，预期 ({EMBEDDING_DIM},)"
        )
    if not np.issubdtype(row.dtype, np.integer):
        raise EmbeddingReferenceError("INT4 行必须是整数")
    wide = row.astype(np.int16)
    if np.any(wide < -7) or np.any(wide > 7):
        raise EmbeddingReferenceError("真实 P50 INT4 行必须位于 [-7, 7]")
    return wide.astype(np.int8)


def _validate_scales(values: np.ndarray | Iterable[float]) -> np.ndarray:
    scales = np.asarray(values, dtype=np.float16).reshape(-1)
    if scales.shape != (GROUPS_PER_ROW,):
        raise EmbeddingReferenceError(
            f"scale 形状错误：{scales.shape}，预期 ({GROUPS_PER_ROW},)"
        )
    float64 = scales.astype(np.float64)
    if not np.all(np.isfinite(float64)) or np.any(float64 <= 0.0):
        raise EmbeddingReferenceError("Embedding scale 必须是有限正数")
    return scales


def compute_embedding_reference(
    *,
    token_id: int,
    quantized_int4: np.ndarray | Iterable[int],
    scales_fp16: np.ndarray | Iterable[float],
    tensor_name: str = DEFAULT_TENSOR,
    vocab_size: int = VOCAB_SIZE,
) -> EmbeddingReferenceResult:
    """根据一行真实 INT4 和 FP16 scale 计算浮点与硬件等价 Q6.10。"""

    resolved_id = _validate_token_id(token_id, vocab_size)
    quantized = _validate_quantized_row(quantized_int4)
    scales = _validate_scales(scales_fp16)
    scales_q28, saturated_scale_count = quantize_uq4_28(scales.astype(np.float64))
    if saturated_scale_count:
        raise EmbeddingReferenceError("Embedding scale 转 UQ4.28 时发生饱和")

    # 真实 FP16 scale 应被 UQ4.28 精确表示。
    recovered_scale = scales_q28.astype(np.float64) / (1 << SCALE_Q28_FRACTION_BITS)
    if not np.array_equal(recovered_scale, scales.astype(np.float64)):
        raise EmbeddingReferenceError("Embedding FP16 scale 不能被 UQ4.28 精确表示")

    expanded_scales = np.repeat(scales.astype(np.float64), GROUP_SIZE)
    p50_float = quantized.astype(np.float64) * expanded_scales
    direct_wide = np.rint(p50_float * Q10_FACTOR)
    direct_clipped = np.clip(direct_wide, Q10_MIN, Q10_MAX)
    direct_q10 = direct_clipped.astype(np.int16)

    expanded_q28 = np.repeat(scales_q28.astype(np.int64), GROUP_SIZE)
    product_q28 = quantized.astype(np.int64) * expanded_q28
    fixed_wide = _round_shift_signed_array(product_q28, SCALE_TO_OUTPUT_SHIFT)
    fixed_clipped = np.clip(fixed_wide, Q10_MIN, Q10_MAX)
    fixed_q10 = fixed_clipped.astype(np.int16)
    saturated_count = int(np.count_nonzero(fixed_wide != fixed_clipped))

    if not np.array_equal(fixed_q10, direct_q10):
        mismatch = np.flatnonzero(fixed_q10 != direct_q10)
        first = int(mismatch[0])
        raise EmbeddingReferenceError(
            f"Q28 固定路径与直接 Q10 不一致：index={first}，"
            f"fixed={int(fixed_q10[first])}，direct={int(direct_q10[first])}"
        )

    return EmbeddingReferenceResult(
        tensor_name=tensor_name,
        token_id=resolved_id,
        vocab_size=vocab_size,
        embedding_dim=EMBEDDING_DIM,
        group_size=GROUP_SIZE,
        quantized_int4=quantized,
        scales_fp16=scales,
        scales_q28=scales_q28,
        p50_float=p50_float.astype(np.float32),
        direct_q10=direct_q10,
        fixed_q10=fixed_q10,
        fixed_float=(fixed_q10.astype(np.float64) / Q10_FACTOR).astype(np.float32),
        saturated_count=saturated_count,
    )


def load_embedding_reference(
    image_path: str | Path = DEFAULT_IMAGE,
    token_id: int = 0,
    tensor_name: str = DEFAULT_TENSOR,
) -> EmbeddingReferenceResult:
    """从真实 P50 镜像读取一个 Token ID 对应的 Embedding 行。"""

    image = P50Image(image_path)
    entry = image.tensor(tensor_name)
    shape = tuple(int(item) for item in entry["shape"])
    if shape != (VOCAB_SIZE, EMBEDDING_DIM):
        raise EmbeddingReferenceError(
            f"Embedding shape 异常：{shape}，预期 {(VOCAB_SIZE, EMBEDDING_DIM)}"
        )
    if entry.get("storage") != "int4_groupwise_symmetric":
        raise EmbeddingReferenceError(f"Embedding storage 异常：{entry.get('storage')!r}")
    if image.header.group_size != GROUP_SIZE:
        raise EmbeddingReferenceError(
            f"group_size 异常：{image.header.group_size} != {GROUP_SIZE}"
        )
    if int(entry["groups_per_row"]) != GROUPS_PER_ROW:
        raise EmbeddingReferenceError("groups_per_row 不是 14")
    if not image.header.tied_embedding:
        raise EmbeddingReferenceError("P50 固定头未标记 tied embedding")

    quantized, scales, _ = image.read_int4_row(tensor_name, int(token_id))
    return compute_embedding_reference(
        token_id=int(token_id),
        quantized_int4=quantized,
        scales_fp16=scales,
        tensor_name=tensor_name,
        vocab_size=shape[0],
    )


def pack_embedding_payload(result: EmbeddingReferenceResult) -> bytes:
    """打包 512 B DDR 行槽：448 B INT4 + 56 B Q28 scale + 8 B 0。"""

    packed = pack_int4_low_nibble_first(
        result.quantized_int4.reshape(1, EMBEDDING_DIM)
    ).reshape(-1)
    if packed.size != PACKED_ROW_BYTES:
        raise AssertionError("packed embedding 行长度错误")
    payload = bytearray(ROW_SLOT_BYTES)
    payload[:PACKED_ROW_BYTES] = packed.tobytes(order="C")
    scale_start = PACKED_ROW_BYTES
    payload[scale_start : scale_start + SCALE_Q28_BYTES] = result.scales_q28.astype(
        "<u4"
    ).tobytes(order="C")
    return bytes(payload)


def unpack_embedding_payload(payload: bytes) -> tuple[np.ndarray, np.ndarray, bytes]:
    """用于主机自检的 512 B 载荷反解。"""

    if len(payload) != ROW_SLOT_BYTES:
        raise EmbeddingReferenceError(
            f"Embedding 载荷长度错误：{len(payload)} != {ROW_SLOT_BYTES}"
        )
    packed = np.frombuffer(payload[:PACKED_ROW_BYTES], dtype=np.uint8)
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    quantized = np.empty(EMBEDDING_DIM, dtype=np.int8)
    quantized[0::2] = np.where(low >= 8, low.astype(np.int16) - 16, low).astype(np.int8)
    quantized[1::2] = np.where(high >= 8, high.astype(np.int16) - 16, high).astype(np.int8)
    scale_start = PACKED_ROW_BYTES
    scales_q28 = np.frombuffer(
        payload[scale_start : scale_start + SCALE_Q28_BYTES], dtype="<u4"
    ).copy()
    padding = payload[scale_start + SCALE_Q28_BYTES :]
    return quantized, scales_q28, padding


def make_random_token_ids(
    count: int,
    seed: int = DEFAULT_RANDOM_SEED,
    vocab_size: int = VOCAB_SIZE,
) -> np.ndarray:
    """生成固定边界前缀和跨平台可复现的随机 Token ID。"""

    if count <= 0:
        raise EmbeddingReferenceError("Token ID 数量必须大于 0")
    prefix = [0, 1, vocab_size - 2, vocab_size - 1]
    output = np.empty(count, dtype=np.uint32)
    prefix_count = min(count, len(prefix))
    output[:prefix_count] = prefix[:prefix_count]
    state = int(seed) & 0xFFFFFFFF
    for index in range(prefix_count, count):
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        output[index] = state % vocab_size
    return output


def _sha256_array(array: np.ndarray, dtype: str | np.dtype) -> str:
    return hashlib.sha256(np.asarray(array, dtype=dtype).tobytes(order="C")).hexdigest()


def result_summary(result: EmbeddingReferenceResult) -> dict[str, Any]:
    """生成单个固定 Token ID 的稳定摘要。"""

    error = np.abs(result.q10_quantization_error)
    payload = pack_embedding_payload(result)
    return {
        "token_id": result.token_id,
        "slot_ctrl_addr": embedding_slot_ctrl_addr(result.token_id, result.vocab_size),
        "slot_byte_offset": embedding_slot_byte_offset(result.token_id, result.vocab_size),
        "scale_fp16_min": float(np.min(result.scales_fp16.astype(np.float64))),
        "scale_fp16_max": float(np.max(result.scales_fp16.astype(np.float64))),
        "output_q10_min": int(np.min(result.fixed_q10)),
        "output_q10_max": int(np.max(result.fixed_q10)),
        "output_saturated_count": result.saturated_count,
        "max_abs_q10_quantization_error": float(np.max(error)),
        "mean_abs_q10_quantization_error": float(np.mean(error)),
        "preview": {
            "quantized_int4_first16": result.quantized_int4[:16].tolist(),
            "scales_q28": result.scales_q28.tolist(),
            "output_q10_first16": result.fixed_q10[:16].tolist(),
        },
        "sha256": {
            "quantized_int4": _sha256_array(result.quantized_int4, "<i1"),
            "scales_fp16": _sha256_array(result.scales_fp16, "<f2"),
            "scales_q28": _sha256_array(result.scales_q28, "<u4"),
            "output_q10": _sha256_array(result.fixed_q10, "<i2"),
            "payload_512b": hashlib.sha256(payload).hexdigest(),
        },
    }


def build_manifest(
    image_path: str | Path = DEFAULT_IMAGE,
    token_ids: Iterable[int] = DEFAULT_FIXED_TOKEN_IDS,
) -> dict[str, Any]:
    """生成固定 Token ID 清单。"""

    image = P50Image(image_path)
    entry = image.tensor(DEFAULT_TENSOR)
    summaries = [
        result_summary(load_embedding_reference(image_path, int(token_id)))
        for token_id in token_ids
    ]
    return {
        "format_version": 1,
        "operator": "qwen2_tied_embedding_k896",
        "tensor_name": DEFAULT_TENSOR,
        "shape": entry["shape"],
        "storage": entry["storage"],
        "group_size": image.header.group_size,
        "groups_per_row": entry["groups_per_row"],
        "tied_embedding": image.header.tied_embedding,
        "fixed_formats": {
            "weight": "packed signed INT4 low_nibble_first",
            "scale": "UQ4.28 uint32 converted exactly from FP16",
            "output": "signed Q6.10 int16",
            "rounding": "round_to_nearest_even shift 18",
            "saturation": "explicit signed int16 saturation",
        },
        "ddr_row_slot": {
            "bytes": ROW_SLOT_BYTES,
            "beats_256bit": ROW_SLOT_BEATS,
            "controller_address_stride": ROW_SLOT_CTRL_STRIDE,
            "packed_weight_bytes": PACKED_ROW_BYTES,
            "scale_q28_bytes": SCALE_Q28_BYTES,
            "padding_bytes": ROW_SLOT_BYTES - PACKED_ROW_BYTES - SCALE_Q28_BYTES,
            "result_controller_address": RESULT_CTRL_ADDR,
            "maximum_slot_controller_address": embedding_slot_ctrl_addr(VOCAB_SIZE - 1),
            "maximum_slot_byte_offset": embedding_slot_byte_offset(VOCAB_SIZE - 1),
        },
        "fixed_token_ids": [int(item) for item in token_ids],
        "rows": summaries,
    }


def save_manifest(manifest: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="真实 P50 Embedding INT4 -> Q6.10 金标准")
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--token-id", type=int, default=0)
    parser.add_argument("--manifest", type=Path, help="保存固定 Token ID JSON 清单")
    parser.add_argument("--print-manifest", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        result = load_embedding_reference(args.image, args.token_id)
        summary = result_summary(result)
        print("=== E3 真实 tied Embedding K=896 软件参考 ===")
        print(
            f"tensor={result.tensor_name}, token_id={result.token_id}, "
            f"slot_ctrl=0x{embedding_slot_ctrl_addr(result.token_id):07x}"
        )
        print(
            f"scale FP16=[{float(result.scales_fp16.min()):.9f}, "
            f"{float(result.scales_fp16.max()):.9f}]，"
            f"Q10 range=[{int(result.fixed_q10.min())}, {int(result.fixed_q10.max())}]"
        )
        print(f"输出前16项：{result.fixed_q10[:16].tolist()}")
        print(
            f"最大 Q6.10 量化误差={summary['max_abs_q10_quantization_error']:.9f}，"
            f"固定路径与直接 Q10 逐位一致"
        )
        if args.manifest is not None or args.print_manifest:
            manifest = build_manifest(args.image)
            if args.manifest is not None:
                save_manifest(manifest, args.manifest)
                print(f"固定清单已保存：{args.manifest}")
            if args.print_manifest:
                print("---BEGIN MANIFEST---")
                print(json.dumps(manifest, ensure_ascii=False, indent=2))
                print("---END MANIFEST---")
        return 0
    except (
        EmbeddingReferenceError,
        P50FormatError,
        FileNotFoundError,
        IndexError,
        KeyError,
        ValueError,
    ) as error:
        print(f"错误：{error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
