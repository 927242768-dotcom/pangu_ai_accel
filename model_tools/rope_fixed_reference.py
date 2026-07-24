#!/usr/bin/env python3
"""Qwen2.5-0.5B layer0 Q/K RoPE 定点软件参考。

Qwen2 的 RoPE 使用 split-half ``rotate_half`` 规则，而不是相邻偶奇维配对：

- ``x_first = x[..., 0:32]``；
- ``x_second = x[..., 32:64]``；
- ``rotate_half(x) = concat(-x_second, x_first)``。

本模块接收 F1 已验证的 signed int64 Q28 Q/K 输出，使用 signed Q1.30
sin/cos，并在 97 位全精度乘加结果上执行一次 round-to-nearest-even（RNE）
右移 30 位，输出仍为 signed int64 Q28。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

try:
    from .linear_quant_reference import DEFAULT_ACTIVATION_SEED
    from .p50_format import P50Image
    from .qkv_linear_reference import (
        DEFAULT_IMAGE,
        HEAD_DIM,
        KV_HEADS,
        Q_HEADS,
        build_qkv_cases,
        load_qkv_models,
        reshape_heads,
    )
except ImportError:
    from linear_quant_reference import DEFAULT_ACTIVATION_SEED
    from p50_format import P50Image
    from qkv_linear_reference import (
        DEFAULT_IMAGE,
        HEAD_DIM,
        KV_HEADS,
        Q_HEADS,
        build_qkv_cases,
        load_qkv_models,
        reshape_heads,
    )

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METADATA = PROJECT_ROOT / "model_output/yanbo_qwen25_0.5b_int4.json"
DEFAULT_MANIFEST = Path(__file__).with_name("rope_layer0_reference.json")

ROTARY_DIM = HEAD_DIM
HALF_DIM = ROTARY_DIM // 2
ROPE_THETA = 1_000_000.0
MAX_POSITION_EMBEDDINGS = 32_768
Q28_FRACTION_BITS = 28
TRIG_FRACTION_BITS = 30
Q28_SCALE = 1 << Q28_FRACTION_BITS
TRIG_SCALE = 1 << TRIG_FRACTION_BITS
Q_VALUES = Q_HEADS * HEAD_DIM
K_VALUES = KV_HEADS * HEAD_DIM
INPUT_BYTES = (Q_VALUES + K_VALUES) * 8
TRIG_ROW_BYTES = HALF_DIM * 4 * 2
DEFAULT_POSITIONS = (0, 1, 2026, MAX_POSITION_EMBEDDINGS - 1)


class RoPEReferenceError(ValueError):
    """表示 RoPE 配置、形状、位置或定点结果不合法。"""


@dataclass(frozen=True)
class RoPETrigRow:
    """一个位置的 32 组 Q1.30 cos/sin。"""

    position: int
    cos_float: np.ndarray
    sin_float: np.ndarray
    cos_q30: np.ndarray
    sin_q30: np.ndarray


@dataclass(frozen=True)
class RoPECase:
    """同一位置下的真实 Q/K 输入、定点输出和误差统计。"""

    position: int
    q_input_q28: np.ndarray
    k_input_q28: np.ndarray
    trig: RoPETrigRow
    q_output_q28: np.ndarray
    k_output_q28: np.ndarray
    q_float_reference: np.ndarray
    k_float_reference: np.ndarray
    q_error_bound: np.ndarray
    k_error_bound: np.ndarray

    @property
    def max_abs_error(self) -> float:
        q_error = np.max(
            np.abs(self.q_output_q28.astype(np.float64) / Q28_SCALE - self.q_float_reference)
        )
        k_error = np.max(
            np.abs(self.k_output_q28.astype(np.float64) / Q28_SCALE - self.k_float_reference)
        )
        return float(max(q_error, k_error))

    @property
    def max_error_bound(self) -> float:
        return float(max(np.max(self.q_error_bound), np.max(self.k_error_bound)))


def _require_shape(array: np.ndarray, shape: tuple[int, ...], label: str) -> None:
    if array.shape != shape:
        raise RoPEReferenceError(f"{label} 形状错误：{array.shape}，预期 {shape}")


def sha256_array(array: np.ndarray, dtype: str | np.dtype) -> str:
    normalized = np.asarray(array, dtype=dtype)
    return hashlib.sha256(normalized.tobytes(order="C")).hexdigest()


def load_rope_config(metadata_path: Path = DEFAULT_METADATA) -> dict[str, int | float]:
    """读取并验证导出镜像中的 RoPE 关键配置。"""

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    model = metadata["model"]
    hidden_size = int(model["hidden_size"])
    attention_heads = int(model["num_attention_heads"])
    kv_heads = int(model["num_key_value_heads"])
    head_dim = hidden_size // attention_heads
    rope_theta = float(model["rope_theta"])
    max_position = int(model["max_position_embeddings"])

    if hidden_size != 896:
        raise RoPEReferenceError(f"hidden_size 错误：{hidden_size}")
    if attention_heads != Q_HEADS or kv_heads != KV_HEADS:
        raise RoPEReferenceError(
            f"GQA 配置错误：Q heads={attention_heads}，KV heads={kv_heads}"
        )
    if head_dim != HEAD_DIM:
        raise RoPEReferenceError(f"head_dim 错误：{head_dim}")
    if rope_theta != ROPE_THETA:
        raise RoPEReferenceError(f"rope_theta 错误：{rope_theta}")
    if max_position != MAX_POSITION_EMBEDDINGS:
        raise RoPEReferenceError(f"max_position_embeddings 错误：{max_position}")

    return {
        "hidden_size": hidden_size,
        "num_attention_heads": attention_heads,
        "num_key_value_heads": kv_heads,
        "head_dim": head_dim,
        "rotary_dim": head_dim,
        "rope_theta": rope_theta,
        "max_position_embeddings": max_position,
    }


def validate_position(position: int) -> int:
    resolved = int(position)
    if not 0 <= resolved < MAX_POSITION_EMBEDDINGS:
        raise RoPEReferenceError(
            f"位置索引越界：{resolved}，有效范围 0..{MAX_POSITION_EMBEDDINGS - 1}"
        )
    return resolved


def inverse_frequencies() -> np.ndarray:
    """生成 Qwen2 默认 RoPE 的 32 个 inverse frequency。"""

    indices = np.arange(0, ROTARY_DIM, 2, dtype=np.float64)
    return 1.0 / np.power(ROPE_THETA, indices / ROTARY_DIM)


def quantize_trig_q30(values: np.ndarray | Sequence[float]) -> np.ndarray:
    """使用 RNE 将 [-1,1] 浮点值量化为 signed Q1.30。"""

    array = np.asarray(values, dtype=np.float64)
    quantized = np.rint(array * TRIG_SCALE)
    quantized = np.clip(quantized, np.iinfo(np.int32).min, np.iinfo(np.int32).max)
    return quantized.astype(np.int32)


def generate_trig_row(position: int) -> RoPETrigRow:
    resolved = validate_position(position)
    angles = inverse_frequencies() * np.float64(resolved)
    cos_float = np.cos(angles).astype(np.float64)
    sin_float = np.sin(angles).astype(np.float64)
    cos_q30 = quantize_trig_q30(cos_float)
    sin_q30 = quantize_trig_q30(sin_float)
    _require_shape(cos_q30, (HALF_DIM,), "cos_q30")
    _require_shape(sin_q30, (HALF_DIM,), "sin_q30")
    return RoPETrigRow(
        position=resolved,
        cos_float=cos_float,
        sin_float=sin_float,
        cos_q30=cos_q30,
        sin_q30=sin_q30,
    )


def round_shift_rne(value: int, shift: int) -> int:
    """对任意精度有符号整数执行 RNE 右移。"""

    if shift <= 0:
        raise RoPEReferenceError("RNE 右移位数必须为正")
    sign = -1 if value < 0 else 1
    magnitude = -value if value < 0 else value
    quotient, remainder = divmod(magnitude, 1 << shift)
    halfway = 1 << (shift - 1)
    if remainder > halfway or (remainder == halfway and (quotient & 1)):
        quotient += 1
    return -quotient if sign < 0 else quotient


def _saturate_int64(value: int) -> int:
    return min(max(value, -(1 << 63)), (1 << 63) - 1)


def _normalize_heads(
    values: np.ndarray | Sequence[int], heads: int, label: str
) -> np.ndarray:
    array = np.asarray(values, dtype=np.int64)
    if array.ndim == 1:
        _require_shape(array, (heads * HEAD_DIM,), label)
        array = array.reshape(heads, HEAD_DIM)
    _require_shape(array, (heads, HEAD_DIM), label)
    return array


def apply_rope_fixed_q28(
    values_q28: np.ndarray | Sequence[int],
    trig: RoPETrigRow,
    *,
    heads: int,
) -> np.ndarray:
    """按硬件定义执行 split-half Q28 × Q1.30 RoPE。"""

    values = _normalize_heads(values_q28, heads, "values_q28")
    output = np.empty_like(values, dtype=np.int64)

    for head in range(heads):
        for pair in range(HALF_DIM):
            first = int(values[head, pair])
            second = int(values[head, pair + HALF_DIM])
            cos_q30 = int(trig.cos_q30[pair])
            sin_q30 = int(trig.sin_q30[pair])

            first_full = first * cos_q30 - second * sin_q30
            second_full = second * cos_q30 + first * sin_q30
            first_rounded = round_shift_rne(first_full, TRIG_FRACTION_BITS)
            second_rounded = round_shift_rne(second_full, TRIG_FRACTION_BITS)
            output[head, pair] = _saturate_int64(first_rounded)
            output[head, pair + HALF_DIM] = _saturate_int64(second_rounded)

    return output


def apply_rope_float(
    values_q28: np.ndarray | Sequence[int],
    trig: RoPETrigRow,
    *,
    heads: int,
) -> np.ndarray:
    """以 Q28 输入的精确实数值计算 float64 RoPE 基线。"""

    values = _normalize_heads(values_q28, heads, "values_q28").astype(np.float64)
    values /= Q28_SCALE
    first = values[:, :HALF_DIM]
    second = values[:, HALF_DIM:]
    output = np.empty_like(values)
    output[:, :HALF_DIM] = (
        first * trig.cos_float[np.newaxis, :]
        - second * trig.sin_float[np.newaxis, :]
    )
    output[:, HALF_DIM:] = (
        second * trig.cos_float[np.newaxis, :]
        + first * trig.sin_float[np.newaxis, :]
    )
    return output


def fixed_error_bound(
    values_q28: np.ndarray | Sequence[int], *, heads: int
) -> np.ndarray:
    """给出每个输出的保守绝对误差界。

    界由两部分组成：两个 trig 系数各至多 0.5 个 Q30 LSB，以及最终一次
    Q28 RNE 至多 0.5 个 Q28 LSB。
    """

    values = _normalize_heads(values_q28, heads, "values_q28").astype(np.float64)
    values = np.abs(values) / Q28_SCALE
    pair_bound = (
        values[:, :HALF_DIM] + values[:, HALF_DIM:]
    ) * (0.5 / TRIG_SCALE) + (0.5 / Q28_SCALE)
    return np.concatenate((pair_bound, pair_bound), axis=1)


def validate_fixed_against_float(
    values_q28: np.ndarray | Sequence[int],
    trig: RoPETrigRow,
    *,
    heads: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fixed = apply_rope_fixed_q28(values_q28, trig, heads=heads)
    floating = apply_rope_float(values_q28, trig, heads=heads)
    bound = fixed_error_bound(values_q28, heads=heads)
    error = np.abs(fixed.astype(np.float64) / Q28_SCALE - floating)
    tolerance = np.finfo(np.float64).eps * 16.0
    if np.any(error > bound + tolerance):
        index = np.unravel_index(int(np.argmax(error - bound)), error.shape)
        raise RoPEReferenceError(
            f"定点误差超过理论界：index={index}，error={error[index]}，bound={bound[index]}"
        )
    return fixed, floating, bound


def build_rope_case(
    q_input_q28: np.ndarray | Sequence[int],
    k_input_q28: np.ndarray | Sequence[int],
    position: int,
) -> RoPECase:
    q_input = _normalize_heads(q_input_q28, Q_HEADS, "q_input_q28")
    k_input = _normalize_heads(k_input_q28, KV_HEADS, "k_input_q28")
    trig = generate_trig_row(position)
    q_fixed, q_float, q_bound = validate_fixed_against_float(
        q_input, trig, heads=Q_HEADS
    )
    k_fixed, k_float, k_bound = validate_fixed_against_float(
        k_input, trig, heads=KV_HEADS
    )
    return RoPECase(
        position=trig.position,
        q_input_q28=q_input,
        k_input_q28=k_input,
        trig=trig,
        q_output_q28=q_fixed,
        k_output_q28=k_fixed,
        q_float_reference=q_float,
        k_float_reference=k_float,
        q_error_bound=q_bound,
        k_error_bound=k_bound,
    )


def load_real_qk_inputs(
    image_path: Path = DEFAULT_IMAGE,
    *,
    activation_seed: int = DEFAULT_ACTIVATION_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """读取 F1 同一固定 hidden state 对应的真实 Q/K Q28 输出。"""

    image = P50Image(image_path)
    image.validate()
    cases = build_qkv_cases(
        load_qkv_models(image), activation_seed=activation_seed
    )
    q_heads = reshape_heads(cases["q"].expected_q28, cases["q"].spec)
    k_heads = reshape_heads(cases["k"].expected_q28, cases["k"].spec)
    return q_heads.astype(np.int64), k_heads.astype(np.int64)


def build_real_rope_cases(
    positions: Iterable[int] = DEFAULT_POSITIONS,
    *,
    image_path: Path = DEFAULT_IMAGE,
    activation_seed: int = DEFAULT_ACTIVATION_SEED,
) -> list[RoPECase]:
    q_input, k_input = load_real_qk_inputs(
        image_path, activation_seed=activation_seed
    )
    return [build_rope_case(q_input, k_input, position) for position in positions]


def build_upload_payload(
    q_input_q28: np.ndarray | Sequence[int],
    k_input_q28: np.ndarray | Sequence[int],
    positions: Sequence[int],
) -> bytes:
    """生成硬件载荷：Q、K、每个位置的 cos[32]、sin[32]。"""

    if not positions:
        raise RoPEReferenceError("至少需要一个位置")
    q_input = _normalize_heads(q_input_q28, Q_HEADS, "q_input_q28")
    k_input = _normalize_heads(k_input_q28, KV_HEADS, "k_input_q28")
    payload = bytearray()
    payload.extend(np.asarray(q_input, dtype="<i8").tobytes(order="C"))
    payload.extend(np.asarray(k_input, dtype="<i8").tobytes(order="C"))
    for position in positions:
        trig = generate_trig_row(position)
        payload.extend(np.asarray(trig.cos_q30, dtype="<i4").tobytes(order="C"))
        payload.extend(np.asarray(trig.sin_q30, dtype="<i4").tobytes(order="C"))
    expected_bytes = INPUT_BYTES + len(positions) * TRIG_ROW_BYTES
    if len(payload) != expected_bytes:
        raise RoPEReferenceError(
            f"RoPE 上传载荷长度错误：{len(payload)} != {expected_bytes}"
        )
    return bytes(payload)


def verify_payload_roundtrip(
    q_input_q28: np.ndarray,
    k_input_q28: np.ndarray,
    positions: Sequence[int],
) -> str:
    payload = build_upload_payload(q_input_q28, k_input_q28, positions)
    q_end = Q_VALUES * 8
    k_end = INPUT_BYTES
    q_decoded = np.frombuffer(payload[:q_end], dtype="<i8").reshape(
        Q_HEADS, HEAD_DIM
    )
    k_decoded = np.frombuffer(payload[q_end:k_end], dtype="<i8").reshape(
        KV_HEADS, HEAD_DIM
    )
    if not np.array_equal(q_decoded, q_input_q28):
        raise RoPEReferenceError("Q Q28 上传往返不一致")
    if not np.array_equal(k_decoded, k_input_q28):
        raise RoPEReferenceError("K Q28 上传往返不一致")
    offset = k_end
    for position in positions:
        trig = generate_trig_row(position)
        row = payload[offset : offset + TRIG_ROW_BYTES]
        cos_decoded = np.frombuffer(row[: HALF_DIM * 4], dtype="<i4")
        sin_decoded = np.frombuffer(row[HALF_DIM * 4 :], dtype="<i4")
        if not np.array_equal(cos_decoded, trig.cos_q30):
            raise RoPEReferenceError(f"position={position} cos 上传往返不一致")
        if not np.array_equal(sin_decoded, trig.sin_q30):
            raise RoPEReferenceError(f"position={position} sin 上传往返不一致")
        offset += TRIG_ROW_BYTES
    return hashlib.sha256(payload).hexdigest()


def case_manifest(case: RoPECase) -> dict[str, object]:
    q_error = np.abs(
        case.q_output_q28.astype(np.float64) / Q28_SCALE - case.q_float_reference
    )
    k_error = np.abs(
        case.k_output_q28.astype(np.float64) / Q28_SCALE - case.k_float_reference
    )
    return {
        "position": case.position,
        "expected": {
            "q_head0_first8_q28": case.q_output_q28[0, :8].tolist(),
            "q_head0_second_half_first8_q28": case.q_output_q28[0, 32:40].tolist(),
            "k_head0_first8_q28": case.k_output_q28[0, :8].tolist(),
            "k_head0_second_half_first8_q28": case.k_output_q28[0, 32:40].tolist(),
        },
        "error": {
            "max_abs": f"{max(float(np.max(q_error)), float(np.max(k_error))):.12e}",
            "max_bound": f"{case.max_error_bound:.12e}",
        },
        "sha256": {
            "cos_q1_30": sha256_array(case.trig.cos_q30, "<i4"),
            "sin_q1_30": sha256_array(case.trig.sin_q30, "<i4"),
            "q_output_q28": sha256_array(case.q_output_q28, "<i8"),
            "k_output_q28": sha256_array(case.k_output_q28, "<i8"),
        },
    }


def rope_manifest(
    cases: Sequence[RoPECase],
    *,
    activation_seed: int = DEFAULT_ACTIVATION_SEED,
) -> dict[str, object]:
    if not cases:
        raise RoPEReferenceError("固定清单至少需要一个位置")
    q_input = cases[0].q_input_q28
    k_input = cases[0].k_input_q28
    for case in cases[1:]:
        if not np.array_equal(case.q_input_q28, q_input) or not np.array_equal(
            case.k_input_q28, k_input
        ):
            raise RoPEReferenceError("固定清单中的所有位置必须复用同一 Q/K 输入")
    positions = [case.position for case in cases]
    config = load_rope_config()
    return {
        "format_version": 1,
        "layer": 0,
        "model": config,
        "pairing_rule": {
            "name": "qwen2_split_half_rotate_half",
            "first_half": "dims[0:32]",
            "second_half": "dims[32:64]",
            "pairs": "dim i <-> dim i+32",
            "formula": "[x0*cos-x1*sin, x1*cos+x0*sin]",
        },
        "fixed_point": {
            "input_output": "signed int64 Q28",
            "trig": "signed int32 Q1.30",
            "multiply_accumulate": "signed 97-bit full precision",
            "rounding": "single RNE shift by 30 after add/sub",
            "saturation": "signed int64",
        },
        "activation_seed": activation_seed,
        "positions": positions,
        "input_sha256": {
            "q_q28": sha256_array(q_input, "<i8"),
            "k_q28": sha256_array(k_input, "<i8"),
            "upload_payload": verify_payload_roundtrip(q_input, k_input, positions),
        },
        "cases": [case_manifest(case) for case in cases],
    }


def validate_manifest(
    cases: Sequence[RoPECase],
    manifest_path: Path = DEFAULT_MANIFEST,
    *,
    activation_seed: int = DEFAULT_ACTIVATION_SEED,
) -> dict[str, object]:
    generated = rope_manifest(cases, activation_seed=activation_seed)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"RoPE 固定清单不存在：{manifest_path}")
    committed = json.loads(manifest_path.read_text(encoding="utf-8"))
    if generated != committed:
        raise RoPEReferenceError("RoPE 固定向量与已提交 JSON 清单不一致")
    return generated


def software_stress(
    *,
    rounds: int = 1000,
    seed: int = 20260730,
) -> None:
    """随机 Q/K 和位置的定点误差界压力测试。"""

    if rounds <= 0:
        raise RoPEReferenceError("rounds 必须为正数")
    rng = np.random.default_rng(seed)
    for round_index in range(rounds):
        position = int(rng.integers(0, MAX_POSITION_EMBEDDINGS))
        q_real = rng.uniform(-8.0, 8.0, size=(Q_HEADS, HEAD_DIM))
        k_real = rng.uniform(-8.0, 8.0, size=(KV_HEADS, HEAD_DIM))
        q_q28 = np.rint(q_real * Q28_SCALE).astype(np.int64)
        k_q28 = np.rint(k_real * Q28_SCALE).astype(np.int64)
        case = build_rope_case(q_q28, k_q28, position)
        if case.max_abs_error > case.max_error_bound + np.finfo(np.float64).eps * 16:
            raise RoPEReferenceError(
                f"随机第 {round_index} 轮误差越界："
                f"{case.max_abs_error} > {case.max_error_bound}"
            )


def parse_positions(text: str) -> tuple[int, ...]:
    positions = tuple(validate_position(int(item.strip())) for item in text.split(",") if item.strip())
    if not positions:
        raise argparse.ArgumentTypeError("positions 不能为空")
    return positions


def _print_summary(cases: Sequence[RoPECase]) -> None:
    print("=== layer0 Q/K RoPE 定点参考 ===")
    print(
        "Qwen2 split-half：dim i 与 dim i+32 配对；"
        "Q/K signed Q28，sin/cos signed Q1.30"
    )
    for case in cases:
        print(
            f"position={case.position:5d}，max_error={case.max_abs_error:.12e}，"
            f"bound={case.max_error_bound:.12e}，"
            f"Q_SHA256={sha256_array(case.q_output_q28, '<i8')}，"
            f"K_SHA256={sha256_array(case.k_output_q28, '<i8')}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Qwen2.5 layer0 Q/K RoPE 定点参考")
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--activation-seed", type=int, default=DEFAULT_ACTIVATION_SEED)
    parser.add_argument(
        "--positions",
        type=parse_positions,
        default=DEFAULT_POSITIONS,
        help="逗号分隔的位置索引",
    )
    parser.add_argument("--json", action="store_true", help="输出固定清单 JSON")
    parser.add_argument("--stress-rounds", type=int, default=0)
    parser.add_argument("--stress-seed", type=int, default=20260730)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        load_rope_config(args.metadata)
        cases = build_real_rope_cases(
            args.positions,
            image_path=args.image,
            activation_seed=args.activation_seed,
        )
        if args.stress_rounds:
            software_stress(rounds=args.stress_rounds, seed=args.stress_seed)
            print(
                f"RoPE 软件随机压力：{args.stress_rounds}/{args.stress_rounds} PASS，"
                f"seed={args.stress_seed}"
            )
        if args.json:
            print(
                json.dumps(
                    rope_manifest(cases, activation_seed=args.activation_seed),
                    ensure_ascii=False,
                    indent=2,
                )
            )
        else:
            _print_summary(cases)
        return 0
    except (FileNotFoundError, KeyError, IndexError, ValueError, OverflowError) as error:
        print(f"错误：{error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
