#!/usr/bin/env python3
"""Qwen2.5-0.5B layer0 MLP ``SiLU(gate)`` 硬件等价软件参考。

输入直接来自已经真实上板逐位通过的 layer0 ``gate_proj``：
``[4864]`` signed int64 Q28。

本阶段冻结的硬件定义：

1. gate Q28 对称 signed RNE 右移 18 位，转换到 Q6.10；
2. 转换结果显式饱和到 signed int16；
3. 复用 E2 已验证的 64 段端点 PWL SiLU；
4. PWL 覆盖 ``[-8,8)``，尾部规则为 ``x<-8 -> 0``、``x>=8 -> x``；
5. 输出为 ``[4864]`` signed Q6.10 int16。

``SiLU(gate)`` 单独通过前，本模块不执行与 ``up_proj`` 的逐元素乘法。
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
    from .elementwise_fixed_reference import (
        Q_MAX,
        Q_MIN,
        build_silu_pwl_endpoints,
        silu_exact_q10,
        silu_pwl_q10,
    )
    from .mlp_gate_up_reference import (
        DEFAULT_IMAGE,
        GateUpCase,
        build_fixed_real_cases as build_gate_up_fixed_cases,
    )
except ImportError:
    from elementwise_fixed_reference import (
        Q_MAX,
        Q_MIN,
        build_silu_pwl_endpoints,
        silu_exact_q10,
        silu_pwl_q10,
    )
    from mlp_gate_up_reference import (
        DEFAULT_IMAGE,
        GateUpCase,
        build_fixed_real_cases as build_gate_up_fixed_cases,
    )

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

DEFAULT_MANIFEST = Path(__file__).with_name("mlp_silu_g1_reference.json")
DEFAULT_STRESS_SEED = 20260809

M = 4864
Q28_FRACTION_BITS = 28
Q10_FRACTION_BITS = 10
Q28_TO_Q10_SHIFT = Q28_FRACTION_BITS - Q10_FRACTION_BITS
PWL_ENTRIES = 65
PWL_PADDED_ENTRIES = 80
INPUT_BYTES = M * 8
PWL_BYTES = PWL_PADDED_ENTRIES * 2
UPLOAD_BYTES = INPUT_BYTES + PWL_BYTES
RESULT_BYTES = M * 2


class MLPSiLUReferenceError(ValueError):
    """表示 MLP SiLU 输入、定点规则、载荷或结果不合法。"""


@dataclass(frozen=True)
class SiLUCase:
    label: str
    query_position: int | None
    count: int | None
    gate_q28: np.ndarray
    gate_q10: np.ndarray
    output_pwl_q10: np.ndarray
    output_exact_q10: np.ndarray
    rescale_saturated_count: int
    pwl_max_abs_error_lsb: int


def _require_shape(array: np.ndarray, shape: tuple[int, ...], label: str) -> None:
    if array.shape != shape:
        raise MLPSiLUReferenceError(f"{label} 形状错误：{array.shape}，预期 {shape}")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_array(array: np.ndarray, dtype: str | np.dtype) -> str:
    return sha256_bytes(np.asarray(array, dtype=dtype).tobytes(order="C"))


def rne_q28_to_q10(values: np.ndarray | Sequence[int]) -> tuple[np.ndarray, int]:
    """对 signed int64 Q28 执行对称 RNE >>18，并饱和到 signed Q6.10。"""

    source = np.asarray(values)
    if source.ndim != 1:
        source = source.reshape(-1)
    if not np.issubdtype(source.dtype, np.integer):
        raise MLPSiLUReferenceError("gate Q28 输入必须为整数")
    signed = source.astype(np.int64, copy=False)

    # 使用 uint64 二补码绝对值，正确覆盖 INT64_MIN。
    bits = signed.view(np.uint64)
    negative = signed < 0
    magnitude = np.where(negative, (~bits) + np.uint64(1), bits)
    quotient = magnitude >> np.uint64(Q28_TO_Q10_SHIFT)
    remainder_mask = np.uint64((1 << Q28_TO_Q10_SHIFT) - 1)
    remainder = magnitude & remainder_mask
    half = np.uint64(1 << (Q28_TO_Q10_SHIFT - 1))
    increment = (remainder > half) | ((remainder == half) & ((quotient & 1) != 0))
    rounded_magnitude = quotient + increment.astype(np.uint64)
    rounded_positive = rounded_magnitude.astype(np.int64)
    rounded = np.where(negative, -rounded_positive, rounded_positive)

    clipped = np.clip(rounded, Q_MIN, Q_MAX)
    return clipped.astype(np.int16), int(np.count_nonzero(rounded != clipped))


def case_from_gate_q28(
    gate_q28: np.ndarray | Sequence[int],
    *,
    label: str,
    query_position: int | None = None,
    count: int | None = None,
) -> SiLUCase:
    gate = np.asarray(gate_q28, dtype=np.int64).reshape(-1)
    _require_shape(gate, (M,), "gate_q28")
    gate_q10, saturated = rne_q28_to_q10(gate)
    output_pwl = silu_pwl_q10(gate_q10).astype(np.int16)
    output_exact = silu_exact_q10(gate_q10).astype(np.int16)
    error = np.abs(output_pwl.astype(np.int32) - output_exact.astype(np.int32))
    return SiLUCase(
        label=label,
        query_position=query_position,
        count=count,
        gate_q28=gate.copy(),
        gate_q10=gate_q10,
        output_pwl_q10=output_pwl,
        output_exact_q10=output_exact,
        rescale_saturated_count=saturated,
        pwl_max_abs_error_lsb=int(np.max(error)),
    )


def _from_gate_up_case(case: GateUpCase) -> SiLUCase:
    return case_from_gate_q28(
        case.gate.expected_q28,
        label=f"layer0 MLP SiLU(gate) query={case.query_position}, count={case.count}",
        query_position=case.query_position,
        count=case.count,
    )


def build_fixed_real_cases(*, image_path: Path = DEFAULT_IMAGE) -> list[SiLUCase]:
    return [_from_gate_up_case(case) for case in build_gate_up_fixed_cases(image_path=image_path)]


def build_upload_payload(case: SiLUCase) -> bytes:
    _require_shape(case.gate_q28, (M,), "gate_q28")
    endpoints = build_silu_pwl_endpoints().astype(np.int16)
    _require_shape(endpoints, (PWL_ENTRIES,), "SiLU PWL endpoints")
    padded = np.zeros(PWL_PADDED_ENTRIES, dtype="<i2")
    padded[:PWL_ENTRIES] = endpoints.astype("<i2")
    payload = (
        np.asarray(case.gate_q28, dtype="<i8").tobytes(order="C")
        + padded.tobytes(order="C")
    )
    if len(payload) != UPLOAD_BYTES:
        raise MLPSiLUReferenceError(
            f"SiLU 上传载荷长度错误：{len(payload)} != {UPLOAD_BYTES}"
        )
    return payload


def verify_upload_payload(case: SiLUCase) -> str:
    payload = build_upload_payload(case)
    decoded_gate = np.frombuffer(payload[:INPUT_BYTES], dtype="<i8").copy()
    decoded_pwl = np.frombuffer(payload[INPUT_BYTES:], dtype="<i2").copy()
    expected_pwl = build_silu_pwl_endpoints().astype(np.int16)
    if not np.array_equal(decoded_gate, case.gate_q28):
        raise MLPSiLUReferenceError("gate Q28 上传载荷往返不一致")
    if not np.array_equal(decoded_pwl[:PWL_ENTRIES], expected_pwl):
        raise MLPSiLUReferenceError("SiLU PWL 端点上传载荷往返不一致")
    if np.any(decoded_pwl[PWL_ENTRIES:]):
        raise MLPSiLUReferenceError("SiLU PWL 补齐区域非零")
    return sha256_bytes(payload)


def fixed_manifest(cases: Sequence[SiLUCase]) -> dict[str, object]:
    return {
        "format_version": 1,
        "operator": "qwen2_layer0_mlp_silu_gate",
        "definition": {
            "input_source": "verified layer0 gate_proj output",
            "input_format": "signed int64 Q28 [4864]",
            "rescale": "symmetric signed RNE right shift 18",
            "silu_input_format": "signed Q6.10 int16 [4864] with explicit saturation",
            "silu_scheme": "E2 verified PWL64 endpoints",
            "silu_range": "[-8,8), x<-8 -> 0, x>=8 -> x",
            "output_format": "signed Q6.10 int16 [4864]",
            "pwl_entries": PWL_ENTRIES,
            "pwl_padded_entries": PWL_PADDED_ENTRIES,
            "upload_bytes": UPLOAD_BYTES,
            "result_bytes": RESULT_BYTES,
            "forbidden_next_operation": "SiLU(gate) * up is not part of this stage",
        },
        "sha256": {
            "pwl65_q6_10": sha256_array(build_silu_pwl_endpoints(), "<i2"),
        },
        "cases": [
            {
                "label": case.label,
                "query_position": case.query_position,
                "count": case.count,
                "rescale_saturated_count": case.rescale_saturated_count,
                "pwl_max_abs_error_lsb": case.pwl_max_abs_error_lsb,
                "range": {
                    "gate_q28_min": int(np.min(case.gate_q28)),
                    "gate_q28_max": int(np.max(case.gate_q28)),
                    "gate_q10_min": int(np.min(case.gate_q10)),
                    "gate_q10_max": int(np.max(case.gate_q10)),
                },
                "sha256": {
                    "gate_q28": sha256_array(case.gate_q28, "<i8"),
                    "gate_q10": sha256_array(case.gate_q10, "<i2"),
                    "silu_exact_q10": sha256_array(case.output_exact_q10, "<i2"),
                    "silu_pwl_q10": sha256_array(case.output_pwl_q10, "<i2"),
                    "upload_payload": verify_upload_payload(case),
                },
                "preview": {
                    "gate_q28_first8": case.gate_q28[:8].tolist(),
                    "gate_q10_first16": case.gate_q10[:16].tolist(),
                    "silu_pwl_first16": case.output_pwl_q10[:16].tolist(),
                    "silu_pwl_last16": case.output_pwl_q10[-16:].tolist(),
                },
            }
            for case in cases
        ],
    }


def validate_manifest(
    cases: Sequence[SiLUCase], manifest_path: Path = DEFAULT_MANIFEST
) -> dict[str, object]:
    actual = fixed_manifest(cases)
    expected = json.loads(manifest_path.read_text(encoding="utf-8"))
    if actual != expected:
        raise MLPSiLUReferenceError(f"MLP SiLU(gate) 固定清单不一致：{manifest_path}")
    return expected


def make_random_gate_q28(rng: np.random.Generator, index: int) -> np.ndarray:
    """生成覆盖真实范围、RNE tie、尾部和 int64 极值的随机/边界输入。"""

    mode = index % 8
    if mode == 0:
        return np.zeros(M, dtype=np.int64)
    if mode == 1:
        result = np.empty(M, dtype=np.int64)
        result[0::2] = np.iinfo(np.int64).min
        result[1::2] = np.iinfo(np.int64).max
        return result
    if mode == 2:
        base = rng.integers(-40000, 40001, size=M, dtype=np.int64)
        tie = np.int64(1 << (Q28_TO_Q10_SHIFT - 1))
        result = base * np.int64(1 << Q28_TO_Q10_SHIFT)
        signs = rng.choice(np.asarray([-1, 1], dtype=np.int64), size=M)
        return result + signs * tie
    if mode == 3:
        centers = np.asarray([-8193, -8192, -8191, 8191, 8192, 8193], dtype=np.int64)
        q10 = rng.choice(centers, size=M)
        jitter = rng.integers(-(1 << 17), (1 << 17) + 1, size=M, dtype=np.int64)
        return q10 * np.int64(1 << Q28_TO_Q10_SHIFT) + jitter
    if mode == 4:
        centers = np.asarray([Q_MIN - 1, Q_MIN, Q_MIN + 1, Q_MAX - 1, Q_MAX, Q_MAX + 1], dtype=np.int64)
        q10 = rng.choice(centers, size=M)
        jitter = rng.integers(-(1 << 17), (1 << 17) + 1, size=M, dtype=np.int64)
        return q10 * np.int64(1 << Q28_TO_Q10_SHIFT) + jitter
    if mode == 5:
        result = np.zeros(M, dtype=np.int64)
        positions = rng.choice(M, size=64, replace=False)
        result[positions] = rng.integers(
            -(6 << Q28_FRACTION_BITS),
            (6 << Q28_FRACTION_BITS) + 1,
            size=positions.size,
            dtype=np.int64,
        )
        return result
    if mode == 6:
        return rng.integers(
            -(6 << Q28_FRACTION_BITS),
            (6 << Q28_FRACTION_BITS) + 1,
            size=M,
            dtype=np.int64,
        )
    raw = rng.integers(0, np.iinfo(np.uint64).max, size=M, dtype=np.uint64)
    return raw.view(np.int64)


def software_stress(*, rounds: int = 1000, seed: int = DEFAULT_STRESS_SEED) -> None:
    if rounds <= 0:
        raise MLPSiLUReferenceError("rounds 必须大于 0")
    rng = np.random.default_rng(seed)
    for index in range(rounds):
        gate = make_random_gate_q28(rng, index)
        case = case_from_gate_q28(gate, label=f"MLP SiLU stress {index + 1}/{rounds}")
        if case.gate_q10.dtype != np.int16 or case.output_pwl_q10.dtype != np.int16:
            raise MLPSiLUReferenceError("压力测试输出格式错误")
        if case.pwl_max_abs_error_lsb > 4:
            raise MLPSiLUReferenceError(
                f"PWL64 误差超界：{case.pwl_max_abs_error_lsb} LSB"
            )
        if index < 8:
            verify_upload_payload(case)
        if not np.any(gate) and np.any(case.output_pwl_q10):
            raise MLPSiLUReferenceError("全零 gate 没有严格输出全零")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="layer0 MLP SiLU(gate) 软件参考")
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
            print("layer0 MLP SiLU(gate) 固定清单：PASS")
            for item in manifest["cases"]:
                print(
                    f"query={item['query_position']} count={item['count']} "
                    f"output={item['sha256']['silu_pwl_q10']}"
                )
        elif args.command == "stress":
            software_stress(rounds=args.rounds, seed=args.seed)
            print(f"MLP SiLU(gate) 软件压力 PASS：{args.rounds}/{args.rounds}，seed={args.seed}")
        else:  # pragma: no cover
            raise AssertionError(args.command)
        return 0
    except (FileNotFoundError, KeyError, ValueError, OverflowError, RuntimeError) as error:
        print(f"错误：{error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
