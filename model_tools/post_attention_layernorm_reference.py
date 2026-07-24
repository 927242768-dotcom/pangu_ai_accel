#!/usr/bin/env python3
"""Qwen2.5-0.5B layer0 post_attention_layernorm 定点软件参考。

本模块把已经真实上板逐位通过的完整 layer0 Attention 子层输出作为 G1 MLP
入口，执行真实 ``model.layers.0.post_attention_layernorm.weight`` RMSNorm：

    attention_output_q10[896]
    -> mean(x^2) + epsilon
    -> LUT256 rsqrt
    -> gamma * x * rsqrt
    -> signed Q6.10 int16[896]

定点格式、RNE、饱和和 LUT 规则严格复用 E1 ``rmsnorm_k896`` 已验证定义，
但固定清单、真实输入来源、上位机和硬件工程保持独立。
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
    from .attention_residual_reference import (
        DEFAULT_HIDDEN_SEED_BASE,
        build_fixed_real_cases as build_attention_fixed_cases,
    )
    from .p50_format import P50Image
    from .rmsnorm_fixed_reference import (
        ACTIVATION_FACTOR,
        DEFAULT_EPSILON,
        LUT_ONLY_INDEX_BITS,
        RMSNormReferenceResult,
        build_rsqrt_lut,
        compute_rmsnorm_reference,
    )
except ImportError:
    from attention_residual_reference import (
        DEFAULT_HIDDEN_SEED_BASE,
        build_fixed_real_cases as build_attention_fixed_cases,
    )
    from p50_format import P50Image
    from rmsnorm_fixed_reference import (
        ACTIVATION_FACTOR,
        DEFAULT_EPSILON,
        LUT_ONLY_INDEX_BITS,
        RMSNormReferenceResult,
        build_rsqrt_lut,
        compute_rmsnorm_reference,
    )

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE = PROJECT_ROOT / "model_output/yanbo_qwen25_0.5b_int4.p50"
DEFAULT_MANIFEST = Path(__file__).with_name("post_attention_layernorm_g1_reference.json")
DEFAULT_GAMMA = "model.layers.0.post_attention_layernorm.weight"
DEFAULT_STRESS_SEED = 20260807
DEFAULT_FIXED_QUERIES = (0, 1, 5, 15)

K = 896
DATA_BYTES = K * 2
LUT_ENTRIES = 1 << LUT_ONLY_INDEX_BITS
LUT_BYTES = LUT_ENTRIES * 4
UPLOAD_BYTES = DATA_BYTES + DATA_BYTES + LUT_BYTES

Q_MIN = -32768
Q_MAX = 32767


class PostAttentionLayerNormError(ValueError):
    """表示 G1 post_attention_layernorm 输入、格式或清单不合法。"""


@dataclass(frozen=True)
class PostAttentionLayerNormCase:
    """一组完整 G1 post_attention_layernorm 硬件等价用例。"""

    label: str
    query_position: int | None
    count: int | None
    input_q10: np.ndarray
    gamma_q10: np.ndarray
    lut_q20: np.ndarray
    output_lut_q10: np.ndarray
    output_exact_q10: np.ndarray
    sum_squares: int
    variance_q20: int
    lut_rsqrt_q20: int
    input_clipped_count: int
    gamma_clipped_count: int
    output_saturated_count: int


def _require_q10_vector(values: np.ndarray | Sequence[int], label: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        array = array.reshape(-1)
    if array.shape != (K,):
        raise PostAttentionLayerNormError(
            f"{label} 形状错误：{array.shape}，预期 ({K},)"
        )
    if not np.issubdtype(array.dtype, np.integer):
        raise PostAttentionLayerNormError(f"{label} 必须是整数 Q6.10")
    wide = array.astype(np.int64)
    if np.any(wide < Q_MIN) or np.any(wide > Q_MAX):
        raise PostAttentionLayerNormError(f"{label} 超出 signed int16 范围")
    return wide.astype(np.int16)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_array(array: np.ndarray, dtype: str | np.dtype) -> str:
    return sha256_bytes(np.asarray(array, dtype=dtype).tobytes(order="C"))


def load_gamma(
    image: P50Image,
    gamma_name: str = DEFAULT_GAMMA,
) -> np.ndarray:
    gamma = image.read_float16_tensor(gamma_name).astype(np.float32).reshape(-1)
    if gamma.shape != (K,):
        raise PostAttentionLayerNormError(
            f"post_attention_layernorm gamma 形状错误：{gamma.shape}"
        )
    return gamma


def case_from_input_q10(
    *,
    input_q10: np.ndarray | Sequence[int],
    gamma_values: np.ndarray | Sequence[float],
    label: str,
    query_position: int | None = None,
    count: int | None = None,
    epsilon: float = DEFAULT_EPSILON,
) -> PostAttentionLayerNormCase:
    """从精确 signed Q6.10 输入构造硬件等价 RMSNorm 用例。"""

    resolved_input = _require_q10_vector(input_q10, "post-attention 输入")
    gamma = np.asarray(gamma_values, dtype=np.float32).reshape(-1)
    if gamma.shape != (K,):
        raise PostAttentionLayerNormError(
            f"gamma 形状错误：{gamma.shape}，预期 ({K},)"
        )

    # Q6.10 的所有 int16 值均可被 float32 精确表示，重新量化必须逐位还原。
    input_float = resolved_input.astype(np.float32) / np.float32(ACTIVATION_FACTOR)
    result: RMSNormReferenceResult = compute_rmsnorm_reference(
        activation_values=input_float,
        gamma_values=gamma,
        epsilon=epsilon,
        gamma_name=DEFAULT_GAMMA,
    )
    if not np.array_equal(result.activation.quantized, resolved_input):
        raise PostAttentionLayerNormError("Q6.10 输入经软件参考后未逐位还原")

    return PostAttentionLayerNormCase(
        label=label,
        query_position=query_position,
        count=count,
        input_q10=resolved_input.copy(),
        gamma_q10=result.gamma.quantized.astype(np.int16),
        lut_q20=build_rsqrt_lut(LUT_ONLY_INDEX_BITS).astype(np.uint32),
        output_lut_q10=result.output_lut_q10.astype(np.int16),
        output_exact_q10=result.output_exact_q10.astype(np.int16),
        sum_squares=result.sum_squares,
        variance_q20=result.variance_q20,
        lut_rsqrt_q20=result.lut_rsqrt_q20,
        input_clipped_count=result.activation.clipped_count,
        gamma_clipped_count=result.gamma.clipped_count,
        output_saturated_count=result.lut_output_saturated_count,
    )


def build_fixed_real_cases(
    *,
    image_path: Path = DEFAULT_IMAGE,
    hidden_seed_base: int = DEFAULT_HIDDEN_SEED_BASE,
) -> list[PostAttentionLayerNormCase]:
    """构造 query=0/1/5/15 四组连贯真实 Attention 输出输入。"""

    image = P50Image(image_path)
    image.validate()
    gamma = load_gamma(image)
    attention_cases = build_attention_fixed_cases(
        image_path=image_path,
        hidden_seed_base=hidden_seed_base,
        queries=DEFAULT_FIXED_QUERIES,
    )
    return [
        case_from_input_q10(
            input_q10=case.output_q10,
            gamma_values=gamma,
            label=(
                f"layer0 post_attention_layernorm query={case.query_position}, "
                f"count={case.count}"
            ),
            query_position=case.query_position,
            count=case.count,
        )
        for case in attention_cases
    ]


def build_upload_payload(case: PostAttentionLayerNormCase) -> bytes:
    payload = (
        np.asarray(case.input_q10, dtype="<i2").tobytes(order="C")
        + np.asarray(case.gamma_q10, dtype="<i2").tobytes(order="C")
        + np.asarray(case.lut_q20, dtype="<u4").tobytes(order="C")
    )
    if len(payload) != UPLOAD_BYTES:
        raise PostAttentionLayerNormError(
            f"上传载荷长度错误：{len(payload)} != {UPLOAD_BYTES}"
        )
    return payload


def verify_payload_roundtrip(case: PostAttentionLayerNormCase) -> str:
    payload = build_upload_payload(case)
    input_end = DATA_BYTES
    gamma_end = input_end + DATA_BYTES
    decoded_input = np.frombuffer(payload[:input_end], dtype="<i2").copy()
    decoded_gamma = np.frombuffer(payload[input_end:gamma_end], dtype="<i2").copy()
    decoded_lut = np.frombuffer(payload[gamma_end:], dtype="<u4").copy()
    if not np.array_equal(decoded_input, case.input_q10):
        raise PostAttentionLayerNormError("输入载荷往返不一致")
    if not np.array_equal(decoded_gamma, case.gamma_q10):
        raise PostAttentionLayerNormError("gamma 载荷往返不一致")
    if not np.array_equal(decoded_lut, case.lut_q20):
        raise PostAttentionLayerNormError("LUT 载荷往返不一致")
    return sha256_bytes(payload)


def fixed_manifest(cases: Sequence[PostAttentionLayerNormCase]) -> dict[str, object]:
    return {
        "format_version": 1,
        "operator": "qwen2_layer0_post_attention_layernorm",
        "definition": {
            "input_source": "verified coherent layer0 Attention residual output",
            "gamma_tensor": DEFAULT_GAMMA,
            "length": K,
            "epsilon": DEFAULT_EPSILON,
            "input": "signed Q6.10 int16 [896]",
            "gamma": "signed Q6.10 int16 [896]",
            "sum_squares": "unsigned 40-bit with 20 fractional bits",
            "mean_and_epsilon": "Q12.20, epsilon_q20=1",
            "rsqrt": "LUT256 midpoint unsigned UQ12.20",
            "rounding": "round_to_nearest_even",
            "output": "signed Q6.10 int16 [896] with explicit saturation",
            "upload_bytes": UPLOAD_BYTES,
            "result_bytes": DATA_BYTES,
            "fixed_queries": [case.query_position for case in cases],
        },
        "cases": [
            {
                "label": case.label,
                "query_position": case.query_position,
                "count": case.count,
                "sum_squares": case.sum_squares,
                "variance_q20": case.variance_q20,
                "lut_rsqrt_q20": case.lut_rsqrt_q20,
                "input_clipped_count": case.input_clipped_count,
                "gamma_clipped_count": case.gamma_clipped_count,
                "output_saturated_count": case.output_saturated_count,
                "sha256": {
                    "attention_output_input_q10": sha256_array(case.input_q10, "<i2"),
                    "gamma_q10": sha256_array(case.gamma_q10, "<i2"),
                    "lut256_uq12_20": sha256_array(case.lut_q20, "<u4"),
                    "output_exact_q10": sha256_array(case.output_exact_q10, "<i2"),
                    "output_lut_q10": sha256_array(case.output_lut_q10, "<i2"),
                    "upload_payload": verify_payload_roundtrip(case),
                },
                "preview": {
                    "input_first8_q10": case.input_q10[:8].tolist(),
                    "output_first8_q10": case.output_lut_q10[:8].tolist(),
                    "output_last8_q10": case.output_lut_q10[-8:].tolist(),
                },
            }
            for case in cases
        ],
    }


def validate_manifest(
    cases: Sequence[PostAttentionLayerNormCase],
    manifest_path: Path = DEFAULT_MANIFEST,
) -> dict[str, object]:
    actual = fixed_manifest(cases)
    expected = json.loads(manifest_path.read_text(encoding="utf-8"))
    if actual != expected:
        raise PostAttentionLayerNormError(
            f"post_attention_layernorm 固定清单不一致：{manifest_path}"
        )
    return expected


def make_random_input_q10(
    rng: np.random.Generator,
    round_index: int,
) -> np.ndarray:
    """生成覆盖全零、极值、常量、稀疏和一般 int16 的压力输入。"""

    mode = round_index % 8
    if mode == 0:
        return np.zeros(K, dtype=np.int16)
    if mode == 1:
        output = np.empty(K, dtype=np.int16)
        output[0::2] = np.int16(Q_MAX)
        output[1::2] = np.int16(Q_MIN)
        return output
    if mode == 2:
        value = int(rng.integers(Q_MIN, Q_MAX + 1))
        return np.full(K, value, dtype=np.int16)
    if mode == 3:
        output = np.zeros(K, dtype=np.int16)
        indices = rng.choice(K, size=32, replace=False)
        output[indices] = rng.integers(Q_MIN, Q_MAX + 1, size=32, dtype=np.int32).astype(
            np.int16
        )
        return output
    if mode == 4:
        return rng.integers(-8, 9, size=K, dtype=np.int16)
    if mode == 5:
        return rng.integers(-1024, 1025, size=K, dtype=np.int16)
    return rng.integers(Q_MIN, Q_MAX + 1, size=K, dtype=np.int32).astype(np.int16)


def software_stress(
    *,
    image_path: Path = DEFAULT_IMAGE,
    rounds: int = 1000,
    seed: int = DEFAULT_STRESS_SEED,
) -> int:
    if rounds <= 0:
        raise PostAttentionLayerNormError("rounds 必须大于 0")
    image = P50Image(image_path)
    image.validate()
    gamma = load_gamma(image)
    rng = np.random.default_rng(seed)
    max_lut_exact_lsb = 0
    for index in range(rounds):
        input_q10 = make_random_input_q10(rng, index)
        case = case_from_input_q10(
            input_q10=input_q10,
            gamma_values=gamma,
            label=f"post_attention_layernorm stress {index + 1}/{rounds}",
        )
        if case.input_clipped_count or case.gamma_clipped_count:
            raise PostAttentionLayerNormError("软件压力中输入或 gamma 发生意外量化截断")
        verify_payload_roundtrip(case)
        delta = int(
            np.max(
                np.abs(
                    case.output_lut_q10.astype(np.int32)
                    - case.output_exact_q10.astype(np.int32)
                )
            )
        )
        max_lut_exact_lsb = max(max_lut_exact_lsb, delta)
    return max_lut_exact_lsb


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="layer0 post_attention_layernorm 连贯软件参考"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    manifest = sub.add_parser("manifest", help="输出四组连贯固定清单")
    manifest.add_argument("--image", type=Path, default=DEFAULT_IMAGE)

    check = sub.add_parser("check", help="校验已提交固定清单")
    check.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    check.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)

    stress = sub.add_parser("stress", help="运行随机和边界软件压力")
    stress.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
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
            committed = validate_manifest(cases, args.manifest)
            print("layer0 post_attention_layernorm 固定清单：PASS")
            for item in committed["cases"]:
                print(
                    f"query={item['query_position']} count={item['count']} "
                    f"output_sha256={item['sha256']['output_lut_q10']}"
                )
        elif args.command == "stress":
            max_delta = software_stress(
                image_path=args.image,
                rounds=args.rounds,
                seed=args.seed,
            )
            print(
                f"post_attention_layernorm 软件压力 PASS：{args.rounds}/{args.rounds}，"
                f"seed={args.seed}，LUT相对精确路径最大差值={max_delta} Q10 LSB"
            )
        else:  # pragma: no cover
            raise AssertionError(args.command)
        return 0
    except (
        FileNotFoundError,
        KeyError,
        ValueError,
        OverflowError,
        RuntimeError,
    ) as error:
        print(f"错误：{error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
