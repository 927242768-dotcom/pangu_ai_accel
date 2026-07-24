#!/usr/bin/env python3
"""Qwen2.5-0.5B layer0 MLP ``SiLU(gate) × up`` 硬件等价软件参考。

两路输入直接来自已经分别真实上板逐位通过的阶段：

- ``SiLU(gate)``：``[4864]`` signed int16 Q6.10；
- ``up_proj``：``[4864]`` signed int64 Q28。

本阶段冻结的硬件定义：

1. 每项执行完整 signed 16×64 乘法，保留 signed 80 bit Q38；
2. 对乘积绝对值执行 round-to-nearest-even（RNE）右移 10 位；
3. 恢复符号后显式饱和到 signed int64；
4. 输出为 ``[4864]`` signed int64 Q28。

该输出格式可在下一阶段直接作为 ``down_proj`` 的 Q28 输入；本模块本身不执行
``down_proj``。
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
    from .mlp_gate_up_reference import (
        DEFAULT_IMAGE,
        GateUpCase,
        build_fixed_real_cases as build_gate_up_fixed_cases,
    )
    from .mlp_silu_reference import case_from_gate_q28
except ImportError:
    from mlp_gate_up_reference import (
        DEFAULT_IMAGE,
        GateUpCase,
        build_fixed_real_cases as build_gate_up_fixed_cases,
    )
    from mlp_silu_reference import case_from_gate_q28

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

DEFAULT_MANIFEST = Path(__file__).with_name("mlp_silu_up_mul_g1_reference.json")
DEFAULT_STRESS_SEED = 20260815

M = 4864
SILU_FRACTION_BITS = 10
UP_FRACTION_BITS = 28
PRODUCT_FRACTION_BITS = SILU_FRACTION_BITS + UP_FRACTION_BITS
OUTPUT_FRACTION_BITS = 28
OUTPUT_SHIFT = PRODUCT_FRACTION_BITS - OUTPUT_FRACTION_BITS
FULL_PRODUCT_BITS = 80
SILU_BYTES = M * 2
UP_BYTES = M * 8
UPLOAD_BYTES = SILU_BYTES + UP_BYTES
RESULT_BYTES = M * 8
INT64_MIN = -(1 << 63)
INT64_MAX = (1 << 63) - 1


class MLPSiLUUpMulReferenceError(ValueError):
    """表示 MLP SiLU×up 输入、定点规则、载荷或结果不合法。"""


@dataclass(frozen=True)
class SiLUUpMulCase:
    label: str
    query_position: int | None
    count: int | None
    silu_q10: np.ndarray
    up_q28: np.ndarray
    output_q28: np.ndarray
    saturated_count: int


def _require_shape(array: np.ndarray, shape: tuple[int, ...], label: str) -> None:
    if array.shape != shape:
        raise MLPSiLUUpMulReferenceError(f"{label} 形状错误：{array.shape}，预期 {shape}")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_array(array: np.ndarray, dtype: str | np.dtype) -> str:
    return sha256_bytes(np.asarray(array, dtype=dtype).tobytes(order="C"))


def multiply_scalar_q10_q28_to_q28(silu_raw: int, up_raw: int) -> tuple[int, bool]:
    """完整 80 位乘法后执行对称 RNE >>10，并饱和到 signed int64。"""

    if not -32768 <= silu_raw <= 32767:
        raise MLPSiLUUpMulReferenceError("SiLU(gate) 原始值超出 signed int16")
    if not INT64_MIN <= up_raw <= INT64_MAX:
        raise MLPSiLUUpMulReferenceError("up_proj 原始值超出 signed int64")

    product = int(silu_raw) * int(up_raw)
    # signed 16×64 的数学乘积必须落在 signed 80 bit 表示范围内。
    if not -(1 << (FULL_PRODUCT_BITS - 1)) <= product <= (1 << (FULL_PRODUCT_BITS - 1)) - 1:
        raise MLPSiLUUpMulReferenceError("完整乘积超出 signed 80 bit")

    negative = product < 0
    magnitude = -product if negative else product
    quotient, remainder = divmod(magnitude, 1 << OUTPUT_SHIFT)
    half = 1 << (OUTPUT_SHIFT - 1)
    if remainder > half or (remainder == half and (quotient & 1)):
        quotient += 1
    rounded = -quotient if negative else quotient

    if rounded > INT64_MAX:
        return INT64_MAX, True
    if rounded < INT64_MIN:
        return INT64_MIN, True
    return rounded, False


def multiply_q10_q28_to_q28(
    silu_q10: np.ndarray | Sequence[int],
    up_q28: np.ndarray | Sequence[int],
) -> tuple[np.ndarray, int]:
    """逐元素执行完整 80 位 Q38 乘法、RNE >>10 和 int64 饱和。"""

    silu_source = np.asarray(silu_q10)
    up_source = np.asarray(up_q28)
    if silu_source.ndim != 1:
        silu_source = silu_source.reshape(-1)
    if up_source.ndim != 1:
        up_source = up_source.reshape(-1)
    if not np.issubdtype(silu_source.dtype, np.integer):
        raise MLPSiLUUpMulReferenceError("SiLU(gate) 输入必须为整数")
    if not np.issubdtype(up_source.dtype, np.integer):
        raise MLPSiLUUpMulReferenceError("up_proj 输入必须为整数")

    silu_wide = silu_source.astype(np.int64, copy=False)
    up_wide = up_source.astype(np.int64, copy=False)
    _require_shape(silu_wide, (M,), "silu_q10")
    _require_shape(up_wide, (M,), "up_q28")
    if np.any(silu_wide < -32768) or np.any(silu_wide > 32767):
        raise MLPSiLUUpMulReferenceError("SiLU(gate) 输入超出 signed int16")

    output = np.empty(M, dtype=np.int64)
    saturated = 0
    for index, (silu_value, up_value) in enumerate(zip(silu_wide, up_wide, strict=True)):
        result, did_saturate = multiply_scalar_q10_q28_to_q28(
            int(silu_value), int(up_value)
        )
        output[index] = result
        saturated += int(did_saturate)
    return output, saturated


def case_from_inputs(
    silu_q10: np.ndarray | Sequence[int],
    up_q28: np.ndarray | Sequence[int],
    *,
    label: str,
    query_position: int | None = None,
    count: int | None = None,
) -> SiLUUpMulCase:
    silu = np.asarray(silu_q10, dtype=np.int16).reshape(-1)
    up = np.asarray(up_q28, dtype=np.int64).reshape(-1)
    _require_shape(silu, (M,), "silu_q10")
    _require_shape(up, (M,), "up_q28")
    output, saturated = multiply_q10_q28_to_q28(silu, up)
    return SiLUUpMulCase(
        label=label,
        query_position=query_position,
        count=count,
        silu_q10=silu.copy(),
        up_q28=up.copy(),
        output_q28=output,
        saturated_count=saturated,
    )


def _from_gate_up_case(case: GateUpCase) -> SiLUUpMulCase:
    silu_case = case_from_gate_q28(
        case.gate.expected_q28,
        label=f"SiLU source query={case.query_position}, count={case.count}",
        query_position=case.query_position,
        count=case.count,
    )
    return case_from_inputs(
        silu_case.output_pwl_q10,
        case.up.expected_q28,
        label=f"layer0 MLP SiLU(gate) × up query={case.query_position}, count={case.count}",
        query_position=case.query_position,
        count=case.count,
    )


def build_fixed_real_cases(*, image_path: Path = DEFAULT_IMAGE) -> list[SiLUUpMulCase]:
    return [_from_gate_up_case(case) for case in build_gate_up_fixed_cases(image_path=image_path)]


def build_upload_payload(case: SiLUUpMulCase) -> bytes:
    _require_shape(case.silu_q10, (M,), "silu_q10")
    _require_shape(case.up_q28, (M,), "up_q28")
    payload = (
        np.asarray(case.silu_q10, dtype="<i2").tobytes(order="C")
        + np.asarray(case.up_q28, dtype="<i8").tobytes(order="C")
    )
    if len(payload) != UPLOAD_BYTES:
        raise MLPSiLUUpMulReferenceError(
            f"SiLU×up 上传载荷长度错误：{len(payload)} != {UPLOAD_BYTES}"
        )
    return payload


def verify_upload_payload(case: SiLUUpMulCase) -> str:
    payload = build_upload_payload(case)
    decoded_silu = np.frombuffer(payload[:SILU_BYTES], dtype="<i2").copy()
    decoded_up = np.frombuffer(payload[SILU_BYTES:], dtype="<i8").copy()
    if not np.array_equal(decoded_silu, case.silu_q10):
        raise MLPSiLUUpMulReferenceError("SiLU(gate) 上传载荷往返不一致")
    if not np.array_equal(decoded_up, case.up_q28):
        raise MLPSiLUUpMulReferenceError("up_proj 上传载荷往返不一致")
    return sha256_bytes(payload)


def fixed_manifest(cases: Sequence[SiLUUpMulCase]) -> dict[str, object]:
    return {
        "format_version": 1,
        "operator": "qwen2_layer0_mlp_silu_gate_times_up",
        "definition": {
            "silu_source": "verified layer0 SiLU(gate) output",
            "silu_format": "signed int16 Q6.10 [4864]",
            "up_source": "verified layer0 up_proj output",
            "up_format": "signed int64 Q28 [4864]",
            "full_product": "signed 80 bit Q38",
            "rounding": "symmetric magnitude RNE right shift 10",
            "output_format": "signed int64 Q28 [4864] with explicit saturation",
            "silu_bytes": SILU_BYTES,
            "up_bytes": UP_BYTES,
            "upload_bytes": UPLOAD_BYTES,
            "result_bytes": RESULT_BYTES,
            "forbidden_next_operation": "down_proj is not part of this stage",
        },
        "cases": [
            {
                "label": case.label,
                "query_position": case.query_position,
                "count": case.count,
                "saturated_count": case.saturated_count,
                "range": {
                    "silu_q10_min": int(np.min(case.silu_q10)),
                    "silu_q10_max": int(np.max(case.silu_q10)),
                    "up_q28_min": int(np.min(case.up_q28)),
                    "up_q28_max": int(np.max(case.up_q28)),
                    "output_q28_min": int(np.min(case.output_q28)),
                    "output_q28_max": int(np.max(case.output_q28)),
                },
                "sha256": {
                    "silu_q10": sha256_array(case.silu_q10, "<i2"),
                    "up_q28": sha256_array(case.up_q28, "<i8"),
                    "output_q28": sha256_array(case.output_q28, "<i8"),
                    "upload_payload": verify_upload_payload(case),
                },
                "preview": {
                    "silu_q10_first16": case.silu_q10[:16].tolist(),
                    "up_q28_first8": case.up_q28[:8].tolist(),
                    "output_q28_first8": case.output_q28[:8].tolist(),
                    "output_q28_last8": case.output_q28[-8:].tolist(),
                },
            }
            for case in cases
        ],
    }


def validate_manifest(
    cases: Sequence[SiLUUpMulCase], manifest_path: Path = DEFAULT_MANIFEST
) -> dict[str, object]:
    actual = fixed_manifest(cases)
    expected = json.loads(manifest_path.read_text(encoding="utf-8"))
    if actual != expected:
        raise MLPSiLUUpMulReferenceError(f"MLP SiLU×up 固定清单不一致：{manifest_path}")
    return expected


def make_random_inputs(
    rng: np.random.Generator, index: int
) -> tuple[np.ndarray, np.ndarray]:
    """生成真实范围、RNE tie、全位宽和饱和边界输入。"""

    mode = index % 8
    if mode == 0:
        return np.zeros(M, dtype=np.int16), np.zeros(M, dtype=np.int64)
    if mode == 1:
        silu = np.empty(M, dtype=np.int16)
        up = np.empty(M, dtype=np.int64)
        silu[0::2] = np.iinfo(np.int16).min
        silu[1::2] = np.iinfo(np.int16).max
        up[0::4] = np.iinfo(np.int64).min
        up[1::4] = np.iinfo(np.int64).max
        up[2::4] = np.iinfo(np.int64).max
        up[3::4] = np.iinfo(np.int64).min
        return silu, up
    if mode == 2:
        # silu_raw=±1 时，up 的低 10 位直接构造正负 half-way tie。
        silu = rng.choice(np.asarray([-1, 1], dtype=np.int16), size=M)
        quotient = rng.integers(-(1 << 40), (1 << 40), size=M, dtype=np.int64)
        up = quotient * np.int64(1 << OUTPUT_SHIFT)
        signs = rng.choice(np.asarray([-1, 1], dtype=np.int64), size=M)
        up = up + signs * np.int64(1 << (OUTPUT_SHIFT - 1))
        return silu.astype(np.int16), up.astype(np.int64)
    if mode == 3:
        silu = rng.integers(-4096, 4097, size=M, dtype=np.int16)
        up = rng.integers(
            -(8 << UP_FRACTION_BITS),
            (8 << UP_FRACTION_BITS) + 1,
            size=M,
            dtype=np.int64,
        )
        return silu, up
    if mode == 4:
        silu = np.zeros(M, dtype=np.int16)
        up = np.zeros(M, dtype=np.int64)
        positions = rng.choice(M, size=64, replace=False)
        silu[positions] = rng.integers(-4096, 4097, size=positions.size, dtype=np.int16)
        up[positions] = rng.integers(
            -(8 << UP_FRACTION_BITS),
            (8 << UP_FRACTION_BITS) + 1,
            size=positions.size,
            dtype=np.int64,
        )
        return silu, up
    if mode == 5:
        silu = rng.integers(
            np.iinfo(np.int16).min,
            np.iinfo(np.int16).max + 1,
            size=M,
            dtype=np.int16,
        )
        up = rng.integers(-(1 << 52), (1 << 52), size=M, dtype=np.int64)
        return silu, up
    if mode == 6:
        silu = rng.integers(
            np.iinfo(np.int16).min,
            np.iinfo(np.int16).max + 1,
            size=M,
            dtype=np.int16,
        )
        raw_up = rng.integers(0, np.iinfo(np.uint64).max, size=M, dtype=np.uint64)
        return silu, raw_up.view(np.int64)

    silu = rng.integers(
        np.iinfo(np.int16).min,
        np.iinfo(np.int16).max + 1,
        size=M,
        dtype=np.int16,
    )
    up = np.empty(M, dtype=np.int64)
    up[0::4] = np.iinfo(np.int64).min
    up[1::4] = np.iinfo(np.int64).max
    up[2::4] = np.int64(1 << 47)
    up[3::4] = np.int64(-(1 << 47))
    return silu, up


def software_stress(*, rounds: int = 1000, seed: int = DEFAULT_STRESS_SEED) -> None:
    if rounds <= 0:
        raise MLPSiLUUpMulReferenceError("rounds 必须大于 0")
    rng = np.random.default_rng(seed)
    for index in range(rounds):
        silu, up = make_random_inputs(rng, index)
        case = case_from_inputs(
            silu,
            up,
            label=f"MLP SiLU×up stress {index + 1}/{rounds} mode={index % 8}",
        )
        if case.output_q28.dtype != np.int64:
            raise MLPSiLUUpMulReferenceError("压力测试输出格式错误")
        if index < 8:
            verify_upload_payload(case)
        if not np.any(silu) or not np.any(up):
            if np.any(case.output_q28):
                raise MLPSiLUUpMulReferenceError("零乘数没有严格输出全零")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="layer0 MLP SiLU(gate) × up 软件参考")
    sub = parser.add_subparsers(dest="command", required=True)
    manifest = sub.add_parser("manifest", help="输出四组真实固定清单")
    manifest.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    check = sub.add_parser("check", help="校验已提交固定清单")
    check.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    check.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    stress = sub.add_parser("stress", help="运行随机/边界软件压力")
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
            print("layer0 MLP SiLU(gate) × up 固定清单：PASS")
            for item in manifest["cases"]:
                print(
                    f"query={item['query_position']} count={item['count']} "
                    f"output={item['sha256']['output_q28']}"
                )
        elif args.command == "stress":
            software_stress(rounds=args.rounds, seed=args.seed)
            print(f"MLP SiLU×up 软件压力 PASS：{args.rounds}/{args.rounds}，seed={args.seed}")
        else:  # pragma: no cover
            raise AssertionError(args.command)
        return 0
    except (FileNotFoundError, KeyError, ValueError, OverflowError, RuntimeError) as error:
        print(f"错误：{error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
