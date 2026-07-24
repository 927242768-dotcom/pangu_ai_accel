#!/usr/bin/env python3
"""Qwen2.5-0.5B F6 Attention 输出加权和软件金标准。

第一版只完成 F6 的独立 ``probability × V`` 闭环，不包含 O_proj 和残差：

- 概率直接消费 F5 的 ``[14,16]`` unsigned UQ1.31；
- V 直接消费 F3 KV Cache 的历史 ``[count,2,64]`` signed int64 Q28；
- GQA 映射为 Q head ``0..6 -> KV0``、``7..13 -> KV1``；
- 每项乘积为 signed Q59，最多 16 项使用 Python 任意精度整数精确累加；
- 每个输出只在累加结束后执行一次 signed round-to-nearest-even 右移 31 位；
- 最终显式饱和为 signed int64 Q28；
- 输出为 ``[14,64]`` head-major，并可无损拼接为 ``[896]``。

全 mask（概率全 0）严格输出全 0；单 token 概率 1.0 时严格复制对应 V；
未使用的固定 16 槽必须为 0。该定义用于后续独立 RTL、PDS 和真实上板逐位验证。
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
    from .attention_score_reference import (
        GQA_GROUP_SIZE,
        HEAD_DIM,
        KV_HEADS,
        MAX_TOKENS,
        Q_HEADS,
        gqa_kv_head,
    )
    from .linear_quant_reference import DEFAULT_ACTIVATION_SEED
    from .p50_format import P50Image
    from .qkv_linear_reference import (
        DEFAULT_IMAGE,
        case_from_model,
        load_projection_model,
        reshape_heads,
    )
    from .softmax_fixed_reference import (
        PROB_BYTES,
        PROB_FRACTION_BITS,
        PROB_ONE,
        SoftmaxCase,
        build_fixed_real_cases as build_softmax_fixed_real_cases,
        decode_probabilities,
        encode_probabilities,
        softmax_scores_q31,
    )
except ImportError:
    from attention_score_reference import (
        GQA_GROUP_SIZE,
        HEAD_DIM,
        KV_HEADS,
        MAX_TOKENS,
        Q_HEADS,
        gqa_kv_head,
    )
    from linear_quant_reference import DEFAULT_ACTIVATION_SEED
    from p50_format import P50Image
    from qkv_linear_reference import (
        DEFAULT_IMAGE,
        case_from_model,
        load_projection_model,
        reshape_heads,
    )
    from softmax_fixed_reference import (
        PROB_BYTES,
        PROB_FRACTION_BITS,
        PROB_ONE,
        SoftmaxCase,
        build_fixed_real_cases as build_softmax_fixed_real_cases,
        decode_probabilities,
        encode_probabilities,
        softmax_scores_q31,
    )

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

DEFAULT_MANIFEST = Path(__file__).with_name("attention_output_f6_reference.json")

V_FRACTION_BITS = 28
PRODUCT_FRACTION_BITS = PROB_FRACTION_BITS + V_FRACTION_BITS
OUTPUT_SHIFT = PROB_FRACTION_BITS
OUTPUT_VALUES = Q_HEADS * HEAD_DIM
OUTPUT_BYTES = OUTPUT_VALUES * 8
V_VALUES = KV_HEADS * HEAD_DIM
V_BYTES = V_VALUES * 8
INT64_MIN = -(1 << 63)
INT64_MAX = (1 << 63) - 1
DEFAULT_STRESS_SEED = 20260804
DEFAULT_V_ACTIVATION_SEED_BASE = 20260804


class AttentionOutputReferenceError(ValueError):
    """表示 F6 Attention 输出配置、定点值、布局或载荷不合法。"""


@dataclass(frozen=True)
class AttentionOutputDebug:
    """一次完整加权和的精确累加与饱和统计。"""

    max_abs_accumulator_q59: int
    saturated_values: int


@dataclass(frozen=True)
class AttentionOutputCase:
    """一个直接消费 F5 概率和 F3 V Cache 的固定窗口用例。"""

    label: str
    layer: int
    query_position: int
    window_start: int
    count: int
    probabilities_q31: np.ndarray
    v_history_q28: np.ndarray
    v_activation_seeds: tuple[int, ...]
    expected_heads_q28: np.ndarray
    debug: AttentionOutputDebug

    @property
    def positions(self) -> tuple[int, ...]:
        return tuple(range(self.window_start, self.window_start + self.count))

    @property
    def probability_payload(self) -> bytes:
        return encode_probabilities(self.probabilities_q31)

    @property
    def output_payload(self) -> bytes:
        return encode_attention_output(self.expected_heads_q28)

    def v_payload(self, index: int) -> bytes:
        if not 0 <= int(index) < self.count:
            raise AttentionOutputReferenceError(f"V token 索引越界：{index}")
        return encode_v_vector(self.v_history_q28[int(index)])


def _require_shape(array: np.ndarray, shape: tuple[int, ...], label: str) -> None:
    if array.shape != shape:
        raise AttentionOutputReferenceError(
            f"{label} 形状错误：{array.shape}，预期 {shape}"
        )


def round_shift_rne_signed(value: int, shift: int) -> int:
    """对任意精度有符号整数执行 round-to-nearest-even 右移。"""

    if shift <= 0:
        raise AttentionOutputReferenceError("RNE 右移位数必须为正")
    resolved = int(value)
    negative = resolved < 0
    magnitude = -resolved if negative else resolved
    quotient, remainder = divmod(magnitude, 1 << shift)
    halfway = 1 << (shift - 1)
    if remainder > halfway or (remainder == halfway and (quotient & 1)):
        quotient += 1
    return -quotient if negative else quotient


def saturate_int64(value: int) -> int:
    return min(max(int(value), INT64_MIN), INT64_MAX)


def _normalize_probabilities(
    probabilities_q31: np.ndarray | Sequence[int],
) -> np.ndarray:
    raw = np.asarray(probabilities_q31)
    _require_shape(raw, (Q_HEADS, MAX_TOKENS), "probabilities_q31")
    if np.issubdtype(raw.dtype, np.signedinteger) and np.any(raw < 0):
        raise AttentionOutputReferenceError("UQ1.31 概率不能为负")
    values = raw.astype(np.uint64)
    if np.any(values > PROB_ONE):
        raise AttentionOutputReferenceError("UQ1.31 概率不能大于 1.0")
    return values.astype(np.uint32)


def _normalize_v_history(
    v_history_q28: np.ndarray | Sequence[int], count: int | None,
) -> tuple[np.ndarray, int]:
    values = np.asarray(v_history_q28, dtype=np.int64)
    if values.ndim != 3:
        raise AttentionOutputReferenceError(
            f"v_history_q28 形状错误：{values.shape}，预期 [count,2,64]"
        )
    resolved_count = int(values.shape[0]) if count is None else int(count)
    if not 1 <= resolved_count <= MAX_TOKENS:
        raise AttentionOutputReferenceError(
            f"count 越界：{resolved_count}，有效范围 1..{MAX_TOKENS}"
        )
    _require_shape(
        values, (resolved_count, KV_HEADS, HEAD_DIM), "v_history_q28"
    )
    return values, resolved_count


def weighted_sum_head_q28(
    probabilities_q31: np.ndarray | Sequence[int],
    v_history_head_q28: np.ndarray | Sequence[int],
) -> tuple[np.ndarray, int, int]:
    """计算一个 Q head 对一个 KV head 的 16-token 加权和。

    返回 ``(output_q28[64], max_abs_accumulator_q59, saturated_values)``。
    """

    probabilities_raw = np.asarray(probabilities_q31)
    _require_shape(probabilities_raw, (MAX_TOKENS,), "probabilities_head_q31")
    if np.issubdtype(probabilities_raw.dtype, np.signedinteger) and np.any(
        probabilities_raw < 0
    ):
        raise AttentionOutputReferenceError("UQ1.31 概率不能为负")
    probabilities = probabilities_raw.astype(np.uint64)
    if np.any(probabilities > PROB_ONE):
        raise AttentionOutputReferenceError("UQ1.31 概率不能大于 1.0")

    history = np.asarray(v_history_head_q28, dtype=np.int64)
    if history.ndim != 2 or history.shape[1:] != (HEAD_DIM,):
        raise AttentionOutputReferenceError(
            f"v_history_head_q28 形状错误：{history.shape}，预期 [count,64]"
        )
    count = int(history.shape[0])
    if not 1 <= count <= MAX_TOKENS:
        raise AttentionOutputReferenceError(
            f"V history count 越界：{count}，有效范围 1..{MAX_TOKENS}"
        )
    if np.any(probabilities[count:] != 0):
        raise AttentionOutputReferenceError("count 之外的固定概率槽必须为 0")

    outputs: list[int] = []
    max_abs_accumulator = 0
    saturated_values = 0
    for dimension in range(HEAD_DIM):
        accumulator_q59 = sum(
            int(probabilities[token]) * int(history[token, dimension])
            for token in range(count)
        )
        max_abs_accumulator = max(max_abs_accumulator, abs(accumulator_q59))
        rounded_q28 = round_shift_rne_signed(accumulator_q59, OUTPUT_SHIFT)
        saturated_q28 = saturate_int64(rounded_q28)
        saturated_values += int(saturated_q28 != rounded_q28)
        outputs.append(saturated_q28)
    return (
        np.asarray(outputs, dtype=np.int64),
        max_abs_accumulator,
        saturated_values,
    )


def attention_output_q28(
    probabilities_q31: np.ndarray | Sequence[int],
    v_history_q28: np.ndarray | Sequence[int],
    *,
    count: int | None = None,
) -> tuple[np.ndarray, AttentionOutputDebug]:
    """生成 ``[14,64]`` head-major Attention 加权和输出。"""

    probabilities = _normalize_probabilities(probabilities_q31)
    history, resolved_count = _normalize_v_history(v_history_q28, count)
    if np.any(probabilities[:, resolved_count:] != 0):
        raise AttentionOutputReferenceError("count 之外的固定概率槽必须为 0")

    outputs = np.zeros((Q_HEADS, HEAD_DIM), dtype=np.int64)
    max_abs_accumulator = 0
    saturated_values = 0
    for q_head in range(Q_HEADS):
        kv_head = gqa_kv_head(q_head)
        head_output, head_max, head_saturated = weighted_sum_head_q28(
            probabilities[q_head], history[:, kv_head, :]
        )
        outputs[q_head] = head_output
        max_abs_accumulator = max(max_abs_accumulator, head_max)
        saturated_values += head_saturated
    return outputs, AttentionOutputDebug(
        max_abs_accumulator_q59=max_abs_accumulator,
        saturated_values=saturated_values,
    )


def flatten_attention_heads(
    heads_q28: np.ndarray | Sequence[int],
) -> np.ndarray:
    heads = np.asarray(heads_q28, dtype=np.int64)
    _require_shape(heads, (Q_HEADS, HEAD_DIM), "heads_q28")
    flat = heads.reshape(OUTPUT_VALUES).copy()
    if not np.array_equal(flat.reshape(Q_HEADS, HEAD_DIM), heads):
        raise AttentionOutputReferenceError("[14,64] 到 [896] 拼接无法无损还原")
    return flat


def reshape_attention_heads(
    flat_q28: np.ndarray | Sequence[int],
) -> np.ndarray:
    flat = np.asarray(flat_q28, dtype=np.int64)
    _require_shape(flat, (OUTPUT_VALUES,), "flat_q28")
    return flat.reshape(Q_HEADS, HEAD_DIM).copy()


def encode_v_vector(v_q28: np.ndarray | Sequence[int]) -> bytes:
    values = np.asarray(v_q28, dtype=np.int64)
    _require_shape(values, (KV_HEADS, HEAD_DIM), "v_q28")
    payload = np.asarray(values, dtype="<i8").tobytes(order="C")
    if len(payload) != V_BYTES:
        raise AttentionOutputReferenceError(
            f"V 载荷长度错误：{len(payload)} != {V_BYTES}"
        )
    return payload


def decode_v_vector(payload: bytes) -> np.ndarray:
    if len(payload) != V_BYTES:
        raise AttentionOutputReferenceError(
            f"V 载荷长度错误：{len(payload)} != {V_BYTES}"
        )
    return np.frombuffer(payload, dtype="<i8").copy().reshape(KV_HEADS, HEAD_DIM)


def encode_attention_output(heads_q28: np.ndarray | Sequence[int]) -> bytes:
    flat = flatten_attention_heads(heads_q28)
    payload = np.asarray(flat, dtype="<i8").tobytes(order="C")
    if len(payload) != OUTPUT_BYTES:
        raise AttentionOutputReferenceError(
            f"Attention 输出载荷长度错误：{len(payload)} != {OUTPUT_BYTES}"
        )
    return payload


def decode_attention_output(payload: bytes) -> np.ndarray:
    if len(payload) != OUTPUT_BYTES:
        raise AttentionOutputReferenceError(
            f"Attention 输出载荷长度错误：{len(payload)} != {OUTPUT_BYTES}"
        )
    flat = np.frombuffer(payload, dtype="<i8").copy()
    return reshape_attention_heads(flat)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_array(array: np.ndarray, dtype: str | np.dtype) -> str:
    return sha256_bytes(np.asarray(array, dtype=dtype).tobytes(order="C"))


def _v_seed_for_position(position: int) -> int:
    return DEFAULT_V_ACTIVATION_SEED_BASE + int(position)


def build_fixed_real_cases(
    *, image_path: Path = DEFAULT_IMAGE,
) -> list[AttentionOutputCase]:
    """复用 F5 四组固定概率，并为每个位置生成不同的真实 layer0 V。"""

    softmax_cases: list[SoftmaxCase] = build_softmax_fixed_real_cases(
        image_path=image_path
    )
    image = P50Image(image_path)
    image.validate()
    v_model = load_projection_model(image, "v")

    v_by_position: dict[int, np.ndarray] = {}
    seed_by_position: dict[int, int] = {}
    cases: list[AttentionOutputCase] = []
    for source in softmax_cases:
        positions = range(source.window_start, source.window_start + source.count)
        history: list[np.ndarray] = []
        seeds: list[int] = []
        for position in positions:
            if position not in v_by_position:
                seed = _v_seed_for_position(position)
                v_case = case_from_model(
                    v_model,
                    activation_seed=seed,
                    label=f"layer0 v_proj position={position} seed={seed}",
                )
                v_by_position[position] = reshape_heads(
                    v_case.expected_q28, v_case.spec
                ).astype(np.int64)
                seed_by_position[position] = seed
            history.append(v_by_position[position])
            seeds.append(seed_by_position[position])
        v_history = np.stack(history, axis=0)
        output, debug = attention_output_q28(
            source.expected_probs_q31, v_history, count=source.count
        )
        cases.append(
            AttentionOutputCase(
                label=source.label,
                layer=source.layer,
                query_position=source.query_position,
                window_start=source.window_start,
                count=source.count,
                probabilities_q31=source.expected_probs_q31,
                v_history_q28=v_history,
                v_activation_seeds=tuple(seeds),
                expected_heads_q28=output,
                debug=debug,
            )
        )
    return cases


def fixed_manifest(cases: Sequence[AttentionOutputCase]) -> dict[str, object]:
    return {
        "format_version": 1,
        "definition": {
            "probability_format": "unsigned UQ1.31 uint32",
            "probability_layout": "head-major [14,16]",
            "v_format": "signed int64 Q28",
            "v_history_layout": "token-major [count,2,64]",
            "gqa_mapping": "Q head 0..6 -> KV0; Q head 7..13 -> KV1",
            "product_format": "signed Q59 exact integer product",
            "accumulator": "exact signed integer sum across at most 16 tokens",
            "rounding": "single signed RNE right shift by 31 after full accumulation",
            "saturation": "explicit signed int64 saturation after RNE",
            "output_format": "signed int64 Q28",
            "output_heads_layout": "head-major [14,64]",
            "output_flat_layout": "head-major contiguous [896]",
            "probability_bytes": PROB_BYTES,
            "v_token_bytes": V_BYTES,
            "output_bytes": OUTPUT_BYTES,
            "q_heads": Q_HEADS,
            "kv_heads": KV_HEADS,
            "gqa_group_size": GQA_GROUP_SIZE,
            "head_dim": HEAD_DIM,
            "max_tokens": MAX_TOKENS,
            "v_fixed_generator": (
                "real layer0 v_proj; activation_seed="
                f"{DEFAULT_V_ACTIVATION_SEED_BASE}+position"
            ),
            "boundary_behavior": {
                "all_mask": "all-zero probabilities produce all-zero output",
                "single_token": "0x80000000 probability copies mapped V exactly",
                "unused_slot": "all probability slots at index >= count must be zero",
                "full_window": "all 16 tokens are accumulated before one RNE",
            },
        },
        "cases": [
            {
                "label": case.label,
                "layer": case.layer,
                "query_position": case.query_position,
                "window_start": case.window_start,
                "count": case.count,
                "positions": list(case.positions),
                "v_activation_seeds": list(case.v_activation_seeds),
                "debug": {
                    "max_abs_accumulator_q59": case.debug.max_abs_accumulator_q59,
                    "saturated_values": case.debug.saturated_values,
                },
                "sha256": {
                    "probabilities_q31": sha256_array(
                        case.probabilities_q31, "<u4"
                    ),
                    "probability_payload": sha256_bytes(case.probability_payload),
                    "v_history_q28": sha256_array(case.v_history_q28, "<i8"),
                    "v_payloads": sha256_bytes(
                        b"".join(case.v_payload(index) for index in range(case.count))
                    ),
                    "output_heads_q28": sha256_array(
                        case.expected_heads_q28, "<i8"
                    ),
                    "output_payload": sha256_bytes(case.output_payload),
                },
                "preview": {
                    "head0_first8": case.expected_heads_q28[0, :8].tolist(),
                    "head7_first8": case.expected_heads_q28[7, :8].tolist(),
                    "head13_last8": case.expected_heads_q28[13, -8:].tolist(),
                    "flat_first8": flatten_attention_heads(
                        case.expected_heads_q28
                    )[:8].tolist(),
                    "flat_last8": flatten_attention_heads(
                        case.expected_heads_q28
                    )[-8:].tolist(),
                },
            }
            for case in cases
        ],
    }


def validate_manifest(
    cases: Sequence[AttentionOutputCase],
    manifest_path: Path = DEFAULT_MANIFEST,
) -> dict[str, object]:
    actual = fixed_manifest(cases)
    expected = json.loads(manifest_path.read_text(encoding="utf-8"))
    if actual != expected:
        raise AttentionOutputReferenceError(
            f"Attention 输出固定清单不一致：{manifest_path}"
        )
    return expected


def _random_scores(rng: np.random.Generator) -> np.ndarray:
    mask_value = -(1 << 63)
    scores = np.full((Q_HEADS, MAX_TOKENS), mask_value, dtype=np.int64)
    for head in range(Q_HEADS):
        valid_count = int(rng.integers(0, MAX_TOKENS + 1))
        if valid_count == 0:
            continue
        indices = rng.choice(MAX_TOKENS, size=valid_count, replace=False)
        values = rng.integers(
            -(12 << 28), 12 << 28, size=valid_count, dtype=np.int64
        )
        scores[head, indices] = values
    return scores


def _random_v_history(
    rng: np.random.Generator, count: int, round_index: int
) -> np.ndarray:
    if round_index % 8 == 0:
        raw = rng.integers(
            0, 1 << 64, size=count * V_VALUES, dtype=np.uint64
        )
        return raw.view(np.int64).reshape(count, KV_HEADS, HEAD_DIM).copy()
    limit = 16 << V_FRACTION_BITS
    return rng.integers(
        -limit,
        limit + 1,
        size=(count, KV_HEADS, HEAD_DIM),
        dtype=np.int64,
    )


def software_stress(
    rounds: int = 1000, seed: int = DEFAULT_STRESS_SEED
) -> int:
    """随机验证 F5 概率、GQA、Q59 累加、RNE、饱和和载荷。"""

    if rounds <= 0:
        raise AttentionOutputReferenceError("rounds 必须大于 0")
    rng = np.random.default_rng(seed)
    total_saturated = 0

    for round_index in range(rounds):
        count = int(rng.integers(1, MAX_TOKENS + 1))
        scores = _random_scores(rng)
        scores[:, count:] = -(1 << 63)
        probabilities, _ = softmax_scores_q31(scores)
        history = _random_v_history(rng, count, round_index)
        output, debug = attention_output_q28(
            probabilities, history, count=count
        )
        total_saturated += debug.saturated_values

        if not np.array_equal(
            decode_probabilities(encode_probabilities(probabilities)), probabilities
        ):
            raise AttentionOutputReferenceError("概率载荷往返不一致")
        for token in range(count):
            if not np.array_equal(
                decode_v_vector(encode_v_vector(history[token])), history[token]
            ):
                raise AttentionOutputReferenceError("V 载荷往返不一致")
        if not np.array_equal(
            decode_attention_output(encode_attention_output(output)), output
        ):
            raise AttentionOutputReferenceError("Attention 输出载荷往返不一致")
        flat = flatten_attention_heads(output)
        if not np.array_equal(reshape_attention_heads(flat), output):
            raise AttentionOutputReferenceError("[14,64]/[896] 拼接往返不一致")

        if not np.any(probabilities):
            if np.any(output):
                raise AttentionOutputReferenceError("全 mask 概率没有输出全 0")

        for q_head in range(Q_HEADS):
            independent: list[int] = []
            kv_head = gqa_kv_head(q_head)
            for dimension in range(HEAD_DIM):
                exact = sum(
                    int(probabilities[q_head, token])
                    * int(history[token, kv_head, dimension])
                    for token in range(count)
                )
                independent.append(
                    saturate_int64(round_shift_rne_signed(exact, OUTPUT_SHIFT))
                )
            if not np.array_equal(output[q_head], np.asarray(independent, dtype=np.int64)):
                raise AttentionOutputReferenceError("独立逐元素 Q59 重算不一致")

    return total_saturated


def _print_summary(manifest: dict[str, object]) -> None:
    definition = manifest["definition"]
    print("F6 Attention 输出加权和软件金标准：PASS")
    print(
        f"输入：{definition['probability_layout']} {definition['probability_format']} + "
        f"{definition['v_history_layout']} {definition['v_format']}"
    )
    print(
        f"输出：{definition['output_heads_layout']} -> "
        f"{definition['output_flat_layout']} {definition['output_format']}"
    )
    for case in manifest["cases"]:
        print(
            f"{case['label']}，count={case['count']}，"
            f"output_sha256={case['sha256']['output_heads_q28']}，"
            f"saturated={case['debug']['saturated_values']}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="F6 Attention 输出加权和软件金标准")
    sub = parser.add_subparsers(dest="command", required=True)

    manifest = sub.add_parser("manifest", help="输出真实固定清单 JSON")
    manifest.add_argument("--image", type=Path, default=DEFAULT_IMAGE)

    verify = sub.add_parser("verify", help="校验固定清单并运行随机压力")
    verify.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    verify.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    verify.add_argument("--rounds", type=int, default=1000)
    verify.add_argument("--seed", type=int, default=DEFAULT_STRESS_SEED)
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
        saturated = software_stress(rounds=args.rounds, seed=args.seed)
        _print_summary(manifest)
        print(
            f"Attention 输出软件随机压力 PASS：{args.rounds}/{args.rounds}，"
            f"seed={args.seed}，累计饱和值={saturated}"
        )
        return 0
    except (
        FileNotFoundError,
        OSError,
        OverflowError,
        AttentionOutputReferenceError,
    ) as error:
        print(f"错误：{error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
