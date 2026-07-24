#!/usr/bin/env python3
"""Qwen2.5-0.5B F5 Softmax 定点软件金标准。

第一版固定消费 F4 的 ``[14,16]`` signed int64 Q28 Attention Score：

- ``INT64_MIN`` 作为 causal mask 哨兵，mask 槽概率严格为 0；
- 每个 head 独立执行 mask 感知的 max reduction；
- 有效 score 减去最大值，差值始终不大于 0；
- exp 使用 ``[-16,0]``、步长 ``1/32`` 的 513 点 UQ1.31 端点 LUT，
  区间内线性插值，低于 -16 的尾部直接置 0；
- 最多 16 个 UQ1.31 exp 求和，和使用无符号 36 位语义；
- 每个 head 计算一次 ``reciprocal_q31 = RNE(2^62 / sum_exp_q31)``；
- 概率为 ``RNE(exp_q31 * reciprocal_q31 / 2^31)``，输出 unsigned UQ1.31；
- 全 mask head 输出全 0；单有效 token 精确输出 ``0x80000000``（1.0）。

该定义优先保证硬件可复现、数值稳定和未来 V 加权时的精度。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

try:
    from .attention_score_reference import (
        DEFAULT_IMAGE,
        MASK_VALUE,
        MAX_TOKENS,
        Q_HEADS,
        AttentionScoreReferenceError,
        build_fixed_real_cases,
        decode_scores,
        encode_scores,
    )
except ImportError:
    from attention_score_reference import (
        DEFAULT_IMAGE,
        MASK_VALUE,
        MAX_TOKENS,
        Q_HEADS,
        AttentionScoreReferenceError,
        build_fixed_real_cases,
        decode_scores,
        encode_scores,
    )

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

DEFAULT_MANIFEST = Path(__file__).with_name("softmax_f5_reference.json")

SCORE_FRACTION_BITS = 28
PROB_FRACTION_BITS = 31
PROB_ONE = 1 << PROB_FRACTION_BITS
PROB_MAX = PROB_ONE
EXP_MIN_INTEGER = -16
EXP_INTERVALS = 512
EXP_STEP_DENOMINATOR = 32
EXP_STEP_SHIFT = SCORE_FRACTION_BITS - 5  # Q28 中 1/32 对应 2^23
EXP_REMAINDER_MASK = (1 << EXP_STEP_SHIFT) - 1
EXP_MIN_Q28 = EXP_MIN_INTEGER << SCORE_FRACTION_BITS
EXP_LUT_ENTRIES = EXP_INTERVALS + 1
EXP_LUT_RAW_BYTES = EXP_LUT_ENTRIES * 4
EXP_LUT_BEATS = (EXP_LUT_RAW_BYTES + 31) // 32
EXP_LUT_PADDED_BYTES = EXP_LUT_BEATS * 32
PROB_VALUES = Q_HEADS * MAX_TOKENS
PROB_BYTES = PROB_VALUES * 4
DEFAULT_STRESS_SEED = 20260803
DEFAULT_FLOAT_TOLERANCE = 2.0e-4


class SoftmaxReferenceError(ValueError):
    """表示 F5 Softmax 配置、定点值、布局或载荷不合法。"""


@dataclass(frozen=True)
class SoftmaxHeadDebug:
    """单个 head 的定点中间量。"""

    all_masked: bool
    max_score_q28: int
    exp_q31: np.ndarray
    sum_exp_q31: int
    reciprocal_q31: int


@dataclass(frozen=True)
class SoftmaxCase:
    """一个来自 F4 固定窗口的 Softmax 用例。"""

    label: str
    layer: int
    query_position: int
    window_start: int
    count: int
    scores_q28: np.ndarray
    expected_probs_q31: np.ndarray
    debug: tuple[SoftmaxHeadDebug, ...]

    @property
    def score_payload(self) -> bytes:
        return encode_scores(self.scores_q28)

    @property
    def probability_payload(self) -> bytes:
        return encode_probabilities(self.expected_probs_q31)


def _require_shape(array: np.ndarray, shape: tuple[int, ...], label: str) -> None:
    if array.shape != shape:
        raise SoftmaxReferenceError(
            f"{label} 形状错误：{array.shape}，预期 {shape}"
        )


def round_shift_rne_unsigned(value: int, shift: int) -> int:
    """对非负整数执行 round-to-nearest-even 右移。"""

    resolved = int(value)
    if resolved < 0:
        raise SoftmaxReferenceError("unsigned RNE 输入不能为负")
    if shift <= 0:
        raise SoftmaxReferenceError("RNE 右移位数必须为正")
    quotient, remainder = divmod(resolved, 1 << shift)
    halfway = 1 << (shift - 1)
    if remainder > halfway or (remainder == halfway and (quotient & 1)):
        quotient += 1
    return quotient


def round_div_rne_unsigned(numerator: int, denominator: int) -> int:
    """对非负整数除法执行 round-to-nearest-even。"""

    resolved_numerator = int(numerator)
    resolved_denominator = int(denominator)
    if resolved_numerator < 0:
        raise SoftmaxReferenceError("unsigned RNE 除法分子不能为负")
    if resolved_denominator <= 0:
        raise SoftmaxReferenceError("unsigned RNE 除法分母必须大于 0")
    quotient, remainder = divmod(resolved_numerator, resolved_denominator)
    doubled = remainder << 1
    if doubled > resolved_denominator or (
        doubled == resolved_denominator and (quotient & 1)
    ):
        quotient += 1
    return quotient


def build_exp_lut_q31() -> np.ndarray:
    """生成 ``exp(-index/32)`` 的 513 点 UQ1.31 端点表。"""

    values = [
        min(PROB_ONE, max(0, round(math.exp(-index / 32.0) * PROB_ONE)))
        for index in range(EXP_LUT_ENTRIES)
    ]
    lut = np.asarray(values, dtype=np.uint32)
    if int(lut[0]) != PROB_ONE:
        raise SoftmaxReferenceError("exp LUT 首项必须精确等于 1.0")
    if np.any(lut[1:] > lut[:-1]):
        raise SoftmaxReferenceError("exp LUT 必须单调不增")
    return lut


EXP_LUT_Q31 = build_exp_lut_q31()


def exp_pwl_q31(diff_q28: int, lut_q31: np.ndarray = EXP_LUT_Q31) -> int:
    """对非正 Q28 差值执行端点 LUT + 线性插值 exp 近似。"""

    resolved = int(diff_q28)
    lut = np.asarray(lut_q31, dtype=np.uint32)
    _require_shape(lut, (EXP_LUT_ENTRIES,), "exp_lut_q31")
    if resolved >= 0:
        return PROB_ONE
    if resolved < EXP_MIN_Q28:
        return 0

    magnitude = -resolved
    interval = magnitude >> EXP_STEP_SHIFT
    remainder = magnitude & EXP_REMAINDER_MASK
    if interval >= EXP_INTERVALS:
        return int(lut[EXP_INTERVALS])

    left = int(lut[interval])
    right = int(lut[interval + 1])
    delta = left - right
    correction = round_shift_rne_unsigned(delta * remainder, EXP_STEP_SHIFT)
    return left - correction


def saturate_probability_q31(value: int) -> int:
    return min(max(int(value), 0), PROB_MAX)


def softmax_head_q31(
    scores_q28: np.ndarray | Sequence[int],
    *,
    lut_q31: np.ndarray = EXP_LUT_Q31,
) -> tuple[np.ndarray, SoftmaxHeadDebug]:
    """对 16 个 score 执行 mask 感知定点 Softmax。"""

    scores = np.asarray(scores_q28, dtype=np.int64)
    _require_shape(scores, (MAX_TOKENS,), "scores_head_q28")
    valid = scores != MASK_VALUE
    probabilities = np.zeros(MAX_TOKENS, dtype=np.uint32)
    exp_values = np.zeros(MAX_TOKENS, dtype=np.uint32)

    if not np.any(valid):
        return probabilities, SoftmaxHeadDebug(
            all_masked=True,
            max_score_q28=MASK_VALUE,
            exp_q31=exp_values,
            sum_exp_q31=0,
            reciprocal_q31=0,
        )

    max_score = int(np.max(scores[valid]))
    for index in np.flatnonzero(valid):
        difference = int(scores[index]) - max_score
        if difference > 0:
            raise SoftmaxReferenceError("减最大值后差值不能为正")
        exp_values[index] = exp_pwl_q31(difference, lut_q31)

    sum_exp = sum(int(value) for value in exp_values)
    if sum_exp <= 0:
        raise SoftmaxReferenceError("存在有效 score 时 exp 求和必须大于 0")
    reciprocal = round_div_rne_unsigned(1 << (2 * PROB_FRACTION_BITS), sum_exp)

    for index in np.flatnonzero(valid):
        product = int(exp_values[index]) * reciprocal
        probabilities[index] = saturate_probability_q31(
            round_shift_rne_unsigned(product, PROB_FRACTION_BITS)
        )

    return probabilities, SoftmaxHeadDebug(
        all_masked=False,
        max_score_q28=max_score,
        exp_q31=exp_values,
        sum_exp_q31=sum_exp,
        reciprocal_q31=reciprocal,
    )


def softmax_scores_q31(
    scores_q28: np.ndarray | Sequence[int],
    *,
    lut_q31: np.ndarray = EXP_LUT_Q31,
) -> tuple[np.ndarray, tuple[SoftmaxHeadDebug, ...]]:
    """对固定 ``[14,16]`` score 矩阵逐 head 执行 Softmax。"""

    scores = np.asarray(scores_q28, dtype=np.int64)
    _require_shape(scores, (Q_HEADS, MAX_TOKENS), "scores_q28")
    probabilities = np.zeros((Q_HEADS, MAX_TOKENS), dtype=np.uint32)
    debug: list[SoftmaxHeadDebug] = []
    for head in range(Q_HEADS):
        probabilities[head], head_debug = softmax_head_q31(
            scores[head], lut_q31=lut_q31
        )
        debug.append(head_debug)
    return probabilities, tuple(debug)


def float_softmax_reference(scores_q28: np.ndarray | Sequence[int]) -> np.ndarray:
    """生成仅用于误差评估的 float64 mask 感知 Softmax。"""

    scores = np.asarray(scores_q28, dtype=np.int64)
    _require_shape(scores, (Q_HEADS, MAX_TOKENS), "scores_q28")
    result = np.zeros((Q_HEADS, MAX_TOKENS), dtype=np.float64)
    for head in range(Q_HEADS):
        valid = scores[head] != MASK_VALUE
        if not np.any(valid):
            continue
        values = scores[head, valid].astype(np.float64) / float(1 << SCORE_FRACTION_BITS)
        shifted = values - np.max(values)
        exp_values = np.exp(shifted)
        result[head, valid] = exp_values / np.sum(exp_values)
    return result


def max_probability_error(
    scores_q28: np.ndarray | Sequence[int],
    probabilities_q31: np.ndarray | Sequence[int],
) -> float:
    scores = np.asarray(scores_q28, dtype=np.int64)
    probabilities = np.asarray(probabilities_q31, dtype=np.uint32)
    _require_shape(scores, (Q_HEADS, MAX_TOKENS), "scores_q28")
    _require_shape(probabilities, (Q_HEADS, MAX_TOKENS), "probabilities_q31")
    fixed = probabilities.astype(np.float64) / float(PROB_ONE)
    return float(np.max(np.abs(fixed - float_softmax_reference(scores))))


def encode_probabilities(probabilities_q31: np.ndarray | Sequence[int]) -> bytes:
    probabilities = np.asarray(probabilities_q31, dtype=np.uint32)
    _require_shape(probabilities, (Q_HEADS, MAX_TOKENS), "probabilities_q31")
    payload = np.asarray(probabilities, dtype="<u4").tobytes(order="C")
    if len(payload) != PROB_BYTES:
        raise SoftmaxReferenceError(
            f"概率载荷长度错误：{len(payload)} != {PROB_BYTES}"
        )
    return payload


def decode_probabilities(payload: bytes) -> np.ndarray:
    if len(payload) != PROB_BYTES:
        raise SoftmaxReferenceError(
            f"概率载荷长度错误：{len(payload)} != {PROB_BYTES}"
        )
    return np.frombuffer(payload, dtype="<u4").copy().reshape(Q_HEADS, MAX_TOKENS)


def build_exp_lut_payload(lut_q31: np.ndarray = EXP_LUT_Q31) -> bytes:
    lut = np.asarray(lut_q31, dtype=np.uint32)
    _require_shape(lut, (EXP_LUT_ENTRIES,), "exp_lut_q31")
    raw = np.asarray(lut, dtype="<u4").tobytes(order="C")
    if len(raw) != EXP_LUT_RAW_BYTES:
        raise SoftmaxReferenceError("exp LUT 原始载荷长度错误")
    return raw + bytes(EXP_LUT_PADDED_BYTES - len(raw))


def decode_exp_lut_payload(payload: bytes) -> np.ndarray:
    if len(payload) != EXP_LUT_PADDED_BYTES:
        raise SoftmaxReferenceError(
            f"exp LUT 载荷长度错误：{len(payload)} != {EXP_LUT_PADDED_BYTES}"
        )
    if any(payload[EXP_LUT_RAW_BYTES:]):
        raise SoftmaxReferenceError("exp LUT padding 必须全为 0")
    return np.frombuffer(payload[:EXP_LUT_RAW_BYTES], dtype="<u4").copy()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_array(array: np.ndarray, dtype: str | np.dtype) -> str:
    return sha256_bytes(np.asarray(array, dtype=dtype).tobytes(order="C"))


def build_fixed_real_cases(
    *,
    image_path: Path = DEFAULT_IMAGE,
) -> list[SoftmaxCase]:
    """直接消费 F4 的四组真实固定 score，建立 F5 固定用例。"""

    cases: list[SoftmaxCase] = []
    for source in build_fixed_real_cases_f4(image_path=image_path):
        probabilities, debug = softmax_scores_q31(source.expected_scores_q28)
        cases.append(
            SoftmaxCase(
                label=source.label,
                layer=source.layer,
                query_position=source.query_position,
                window_start=source.window_start,
                count=source.count,
                scores_q28=source.expected_scores_q28,
                expected_probs_q31=probabilities,
                debug=debug,
            )
        )
    return cases


def build_fixed_real_cases_f4(*, image_path: Path = DEFAULT_IMAGE):
    """隔离 F4 导入名称，避免与 F5 构造函数重名。"""

    try:
        from .attention_score_reference import build_fixed_real_cases as builder
    except ImportError:
        from attention_score_reference import build_fixed_real_cases as builder
    return builder(image_path=image_path)


def fixed_manifest(cases: Sequence[SoftmaxCase]) -> dict[str, object]:
    return {
        "format_version": 1,
        "definition": {
            "input_format": "signed int64 Q28",
            "input_layout": "head-major [14,16]",
            "mask_value": MASK_VALUE,
            "max_reduction": "ignore mask sentinel; all-mask head outputs zero",
            "exp": "513 endpoint LUT on [-16,0], step 1/32, linear interpolation",
            "exp_format": "unsigned UQ1.31 uint32",
            "exp_tail": "difference < -16 -> 0; difference == -16 uses LUT endpoint",
            "sum_format": "unsigned integer carrying Q31, maximum 16*2^31",
            "reciprocal_rule": "RNE(2^62 / sum_exp_q31)",
            "normalization_rule": "RNE(exp_q31 * reciprocal_q31 / 2^31)",
            "output_format": "unsigned UQ1.31 uint32",
            "one": PROB_ONE,
            "output_layout": "head-major [14,16]",
            "probability_bytes": PROB_BYTES,
            "exp_lut_entries": EXP_LUT_ENTRIES,
            "exp_lut_raw_bytes": EXP_LUT_RAW_BYTES,
            "exp_lut_padded_bytes": EXP_LUT_PADDED_BYTES,
            "float_error_tolerance": DEFAULT_FLOAT_TOLERANCE,
            "boundary_behavior": {
                "all_mask": "all probabilities are zero",
                "single_valid": "valid probability is exactly 0x80000000",
                "masked_slot": "probability is exactly zero",
            },
        },
        "sha256": {
            "exp_lut_q31": sha256_array(EXP_LUT_Q31, "<u4"),
            "exp_lut_payload": sha256_bytes(build_exp_lut_payload()),
        },
        "cases": [
            {
                "label": case.label,
                "layer": case.layer,
                "query_position": case.query_position,
                "window_start": case.window_start,
                "count": case.count,
                "max_probability_error": max_probability_error(
                    case.scores_q28, case.expected_probs_q31
                ),
                "head_debug": {
                    f"head{head}": {
                        "all_masked": case.debug[head].all_masked,
                        "max_score_q28": case.debug[head].max_score_q28,
                        "sum_exp_q31": case.debug[head].sum_exp_q31,
                        "reciprocal_q31": case.debug[head].reciprocal_q31,
                    }
                    for head in (0, 7, 13)
                },
                "sha256": {
                    "scores_q28": sha256_array(case.scores_q28, "<i8"),
                    "probabilities_q31": sha256_array(
                        case.expected_probs_q31, "<u4"
                    ),
                    "score_payload": sha256_bytes(case.score_payload),
                    "probability_payload": sha256_bytes(case.probability_payload),
                },
                "preview": {
                    "head0_first4": case.expected_probs_q31[0, :4].tolist(),
                    "head7_first4": case.expected_probs_q31[7, :4].tolist(),
                    "head13_first4": case.expected_probs_q31[13, :4].tolist(),
                    "sum_head0": int(np.sum(case.expected_probs_q31[0], dtype=np.uint64)),
                    "sum_head7": int(np.sum(case.expected_probs_q31[7], dtype=np.uint64)),
                    "sum_head13": int(np.sum(case.expected_probs_q31[13], dtype=np.uint64)),
                },
            }
            for case in cases
        ],
    }


def validate_manifest(
    cases: Sequence[SoftmaxCase],
    manifest_path: Path = DEFAULT_MANIFEST,
) -> dict[str, object]:
    actual = fixed_manifest(cases)
    expected = json.loads(manifest_path.read_text(encoding="utf-8"))
    if actual != expected:
        raise SoftmaxReferenceError(f"Softmax 固定清单不一致：{manifest_path}")
    return expected


def _random_scores(rng: np.random.Generator) -> np.ndarray:
    scores = np.full((Q_HEADS, MAX_TOKENS), MASK_VALUE, dtype=np.int64)
    for head in range(Q_HEADS):
        valid_count = int(rng.integers(0, MAX_TOKENS + 1))
        if valid_count == 0:
            continue
        indices = rng.choice(MAX_TOKENS, size=valid_count, replace=False)
        center = int(rng.integers(-(8 << SCORE_FRACTION_BITS), 8 << SCORE_FRACTION_BITS))
        spread = rng.integers(
            -(24 << SCORE_FRACTION_BITS),
            1,
            size=valid_count,
            dtype=np.int64,
        )
        values = np.asarray(center + spread, dtype=np.int64)
        scores[head, indices] = values
    return scores


def software_stress(
    rounds: int = 1000,
    seed: int = DEFAULT_STRESS_SEED,
    *,
    float_tolerance: float = DEFAULT_FLOAT_TOLERANCE,
) -> float:
    """随机验证 mask、LUT、倒数、载荷和 float64 误差。"""

    if rounds <= 0:
        raise SoftmaxReferenceError("rounds 必须大于 0")
    if float_tolerance <= 0:
        raise SoftmaxReferenceError("float_tolerance 必须大于 0")
    rng = np.random.default_rng(seed)
    worst_error = 0.0

    for _ in range(rounds):
        scores = _random_scores(rng)
        probabilities, debug = softmax_scores_q31(scores)
        if not np.array_equal(decode_scores(encode_scores(scores)), scores):
            raise SoftmaxReferenceError("score 载荷往返不一致")
        if not np.array_equal(
            decode_probabilities(encode_probabilities(probabilities)), probabilities
        ):
            raise SoftmaxReferenceError("概率载荷往返不一致")
        if not np.array_equal(
            decode_exp_lut_payload(build_exp_lut_payload()), EXP_LUT_Q31
        ):
            raise SoftmaxReferenceError("exp LUT 载荷往返不一致")

        for head in range(Q_HEADS):
            valid = scores[head] != MASK_VALUE
            if np.any(probabilities[head, ~valid] != 0):
                raise SoftmaxReferenceError("mask 槽概率不是 0")
            if not np.any(valid):
                if np.any(probabilities[head] != 0) or not debug[head].all_masked:
                    raise SoftmaxReferenceError("全 mask 行为错误")
            else:
                if debug[head].all_masked or int(np.max(probabilities[head])) > PROB_ONE:
                    raise SoftmaxReferenceError("有效 head 概率或状态错误")
                if np.count_nonzero(valid) == 1:
                    if int(probabilities[head, valid][0]) != PROB_ONE:
                        raise SoftmaxReferenceError("单有效 token 没有精确归一化到 1.0")

        error = max_probability_error(scores, probabilities)
        worst_error = max(worst_error, error)
        if error > float_tolerance:
            raise SoftmaxReferenceError(
                f"Softmax float64 最大误差超限：{error} > {float_tolerance}"
            )

    return worst_error


def _print_summary(manifest: dict[str, object]) -> None:
    definition = manifest["definition"]
    print("F5 Softmax 定点软件金标准：PASS")
    print(
        f"输入：{definition['input_layout']} {definition['input_format']}，"
        f"输出：{definition['output_layout']} {definition['output_format']}"
    )
    print(
        f"exp：{definition['exp']}，LUT={definition['exp_lut_entries']} 项，"
        f"padding={definition['exp_lut_padded_bytes']} B"
    )
    for case in manifest["cases"]:
        print(
            f"{case['label']}，prob_sha256={case['sha256']['probabilities_q31']}，"
            f"max_error={case['max_probability_error']:.12g}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="F5 Softmax 定点软件金标准")
    sub = parser.add_subparsers(dest="command", required=True)

    manifest = sub.add_parser("manifest", help="输出真实固定清单 JSON")
    manifest.add_argument("--image", type=Path, default=DEFAULT_IMAGE)

    verify = sub.add_parser("verify", help="校验固定清单并运行随机压力")
    verify.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    verify.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    verify.add_argument("--rounds", type=int, default=1000)
    verify.add_argument("--seed", type=int, default=DEFAULT_STRESS_SEED)
    verify.add_argument(
        "--float-tolerance", type=float, default=DEFAULT_FLOAT_TOLERANCE
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        cases = build_fixed_real_cases(image_path=args.image)
        manifest = fixed_manifest(cases)
        if args.command == "manifest":
            print(json.dumps(manifest, ensure_ascii=False, indent=2))
            return 0
        validate_manifest(cases, args.manifest)
        worst_error = software_stress(
            rounds=args.rounds,
            seed=args.seed,
            float_tolerance=args.float_tolerance,
        )
        _print_summary(manifest)
        print(
            f"Softmax 软件随机压力 PASS：{args.rounds}/{args.rounds}，"
            f"seed={args.seed}，worst_error={worst_error:.12g}"
        )
        return 0
    except (
        FileNotFoundError,
        OSError,
        AttentionScoreReferenceError,
        SoftmaxReferenceError,
        OverflowError,
    ) as error:
        print(f"错误：{error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
