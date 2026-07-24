#!/usr/bin/env python3
"""Qwen2.5-0.5B F4 Attention Score 软件金标准。

第一版固定模型结构：

- Q heads = 14，KV heads = 2，head_dim = 64；
- GQA 映射：每 7 个连续 Q head 共用一个 KV head；
- Q/K 输入均为 signed int64 Q28；
- 64 维点积得到精确 signed Q56；
- 乘以 ``1/sqrt(64)=1/8``，并以 RNE 右移 31 位转换回 signed Q28；
- 未来位置和窗口未使用槽统一输出 ``INT64_MIN``，作为 causal mask 哨兵；
- 第一版窗口上限为 16 token，输出固定布局为 ``[14,16]`` head-major。

K 的 DDR3 地址布局完全复用 F3 KV Cache：每个 token 的 K 位于 token 槽前 1024 B。
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
    from .kv_cache_reference import (
        DEFAULT_IMAGE,
        MAX_CONTEXT,
        NUM_LAYERS,
        VECTOR_BYTES,
        kv_slot_address,
    )
    from .linear_quant_reference import DEFAULT_ACTIVATION_SEED
    from .p50_format import P50Image
    from .qkv_linear_reference import (
        HEAD_DIM,
        KV_HEADS,
        Q_HEADS,
        build_qkv_cases,
        load_qkv_models,
        reshape_heads,
    )
    from .rope_fixed_reference import apply_rope_fixed_q28, generate_trig_row
except ImportError:
    from kv_cache_reference import (
        DEFAULT_IMAGE,
        MAX_CONTEXT,
        NUM_LAYERS,
        VECTOR_BYTES,
        kv_slot_address,
    )
    from linear_quant_reference import DEFAULT_ACTIVATION_SEED
    from p50_format import P50Image
    from qkv_linear_reference import (
        HEAD_DIM,
        KV_HEADS,
        Q_HEADS,
        build_qkv_cases,
        load_qkv_models,
        reshape_heads,
    )
    from rope_fixed_reference import apply_rope_fixed_q28, generate_trig_row

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = Path(__file__).with_name("attention_score_f4_reference.json")

Q_FRACTION_BITS = 28
DOT_FRACTION_BITS = Q_FRACTION_BITS * 2
HEAD_SCALE_SHIFT = 3
OUTPUT_SHIFT = Q_FRACTION_BITS + HEAD_SCALE_SHIFT
MAX_TOKENS = 16
GQA_GROUP_SIZE = Q_HEADS // KV_HEADS
MASK_VALUE = -(1 << 63)
SCORE_VALUES = Q_HEADS * MAX_TOKENS
SCORE_BYTES = SCORE_VALUES * 8
Q_BYTES = Q_HEADS * HEAD_DIM * 8
K_BYTES = KV_HEADS * HEAD_DIM * 8
DEFAULT_STRESS_SEED = 20260802
DEFAULT_FIXED_WINDOWS = (
    (0, 0, 0, 1),
    (0, 1, 0, 2),
    (13, 2026, 2023, 6),
    (27, MAX_CONTEXT - 1, MAX_CONTEXT - MAX_TOKENS, MAX_TOKENS),
)


class AttentionScoreReferenceError(ValueError):
    """表示 F4 配置、定点值、布局或载荷不合法。"""


@dataclass(frozen=True)
class AttentionScoreCase:
    """一个固定窗口 Attention Score 用例。"""

    layer: int
    query_position: int
    window_start: int
    count: int
    q_q28: np.ndarray
    k_history_q28: np.ndarray
    expected_scores_q28: np.ndarray
    label: str

    @property
    def positions(self) -> tuple[int, ...]:
        return tuple(range(self.window_start, self.window_start + self.count))

    @property
    def q_payload(self) -> bytes:
        return build_q_payload(self.q_q28)

    def k_payload(self, index: int) -> bytes:
        if not 0 <= int(index) < self.count:
            raise AttentionScoreReferenceError(f"K token 索引越界：{index}")
        return build_k_payload(self.k_history_q28[int(index)])


def _require_shape(array: np.ndarray, shape: tuple[int, ...], label: str) -> None:
    if array.shape != shape:
        raise AttentionScoreReferenceError(
            f"{label} 形状错误：{array.shape}，预期 {shape}"
        )


def validate_layer(layer: int) -> int:
    resolved = int(layer)
    if not 0 <= resolved < NUM_LAYERS:
        raise AttentionScoreReferenceError(
            f"layer 越界：{resolved}，有效范围 0..{NUM_LAYERS - 1}"
        )
    return resolved


def validate_window(
    query_position: int, window_start: int, count: int
) -> tuple[int, int, int]:
    query = int(query_position)
    start = int(window_start)
    resolved_count = int(count)
    if not 0 <= query < MAX_CONTEXT:
        raise AttentionScoreReferenceError(
            f"query_position 越界：{query}，有效范围 0..{MAX_CONTEXT - 1}"
        )
    if not 0 <= start < MAX_CONTEXT:
        raise AttentionScoreReferenceError(
            f"window_start 越界：{start}，有效范围 0..{MAX_CONTEXT - 1}"
        )
    if not 1 <= resolved_count <= MAX_TOKENS:
        raise AttentionScoreReferenceError(
            f"count 越界：{resolved_count}，有效范围 1..{MAX_TOKENS}"
        )
    if start + resolved_count > MAX_CONTEXT:
        raise AttentionScoreReferenceError(
            f"窗口越界：start={start}, count={resolved_count}, max={MAX_CONTEXT}"
        )
    return query, start, resolved_count


def gqa_kv_head(q_head: int) -> int:
    resolved = int(q_head)
    if not 0 <= resolved < Q_HEADS:
        raise AttentionScoreReferenceError(
            f"Q head 越界：{resolved}，有效范围 0..{Q_HEADS - 1}"
        )
    kv_head = resolved // GQA_GROUP_SIZE
    if not 0 <= kv_head < KV_HEADS:
        raise AttentionScoreReferenceError("GQA head 映射越界")
    return kv_head


def round_shift_rne(value: int, shift: int) -> int:
    """对任意精度有符号整数执行 round-to-nearest-even 右移。"""

    if shift <= 0:
        raise AttentionScoreReferenceError("RNE 右移位数必须为正")
    sign = -1 if value < 0 else 1
    magnitude = -value if value < 0 else value
    quotient, remainder = divmod(magnitude, 1 << shift)
    halfway = 1 << (shift - 1)
    if remainder > halfway or (remainder == halfway and (quotient & 1)):
        quotient += 1
    return -quotient if sign < 0 else quotient


def saturate_int64(value: int) -> int:
    return min(max(int(value), -(1 << 63)), (1 << 63) - 1)


def scaled_dot_q28(
    q_q28: np.ndarray | Sequence[int],
    k_q28: np.ndarray | Sequence[int],
) -> int:
    """计算一个 head 的精确 Q·K/8，并返回 signed int64 Q28。"""

    q = np.asarray(q_q28, dtype=np.int64)
    k = np.asarray(k_q28, dtype=np.int64)
    _require_shape(q, (HEAD_DIM,), "q_head_q28")
    _require_shape(k, (HEAD_DIM,), "k_head_q28")
    dot_q56 = sum(int(left) * int(right) for left, right in zip(q, k))
    return saturate_int64(round_shift_rne(dot_q56, OUTPUT_SHIFT))


def attention_scores_q28(
    q_q28: np.ndarray | Sequence[int],
    k_history_q28: np.ndarray | Sequence[int],
    *,
    query_position: int,
    window_start: int,
    count: int | None = None,
) -> np.ndarray:
    """生成固定 ``[14,16]`` head-major score 矩阵。

    ``k_history_q28`` 的形状为 ``[count,2,64]``。窗口中位置大于
    ``query_position`` 的项和 ``count`` 之外的固定槽均写入 ``MASK_VALUE``。
    """

    q = np.asarray(q_q28, dtype=np.int64)
    history = np.asarray(k_history_q28, dtype=np.int64)
    if count is None:
        if history.ndim != 3:
            raise AttentionScoreReferenceError("无法从 K history 推导 count")
        count = int(history.shape[0])
    query, start, resolved_count = validate_window(
        query_position, window_start, count
    )
    _require_shape(q, (Q_HEADS, HEAD_DIM), "q_q28")
    _require_shape(
        history,
        (resolved_count, KV_HEADS, HEAD_DIM),
        "k_history_q28",
    )

    scores = np.full((Q_HEADS, MAX_TOKENS), MASK_VALUE, dtype=np.int64)
    for token_index in range(resolved_count):
        token_position = start + token_index
        if token_position > query:
            continue
        for q_head in range(Q_HEADS):
            scores[q_head, token_index] = scaled_dot_q28(
                q[q_head], history[token_index, gqa_kv_head(q_head)]
            )
    return scores


def build_q_payload(q_q28: np.ndarray | Sequence[int]) -> bytes:
    q = np.asarray(q_q28, dtype=np.int64)
    _require_shape(q, (Q_HEADS, HEAD_DIM), "q_q28")
    payload = np.asarray(q, dtype="<i8").tobytes(order="C")
    if len(payload) != Q_BYTES:
        raise AttentionScoreReferenceError(
            f"Q 载荷长度错误：{len(payload)} != {Q_BYTES}"
        )
    return payload


def decode_q_payload(payload: bytes) -> np.ndarray:
    if len(payload) != Q_BYTES:
        raise AttentionScoreReferenceError(
            f"Q 载荷长度错误：{len(payload)} != {Q_BYTES}"
        )
    return np.frombuffer(payload, dtype="<i8").copy().reshape(Q_HEADS, HEAD_DIM)


def build_k_payload(k_q28: np.ndarray | Sequence[int]) -> bytes:
    k = np.asarray(k_q28, dtype=np.int64)
    _require_shape(k, (KV_HEADS, HEAD_DIM), "k_q28")
    payload = np.asarray(k, dtype="<i8").tobytes(order="C")
    if len(payload) != K_BYTES or len(payload) != VECTOR_BYTES:
        raise AttentionScoreReferenceError(
            f"K 载荷长度错误：{len(payload)} != {K_BYTES}"
        )
    return payload


def decode_k_payload(payload: bytes) -> np.ndarray:
    if len(payload) != K_BYTES:
        raise AttentionScoreReferenceError(
            f"K 载荷长度错误：{len(payload)} != {K_BYTES}"
        )
    return np.frombuffer(payload, dtype="<i8").copy().reshape(KV_HEADS, HEAD_DIM)


def encode_scores(scores_q28: np.ndarray | Sequence[int]) -> bytes:
    scores = np.asarray(scores_q28, dtype=np.int64)
    _require_shape(scores, (Q_HEADS, MAX_TOKENS), "scores_q28")
    payload = np.asarray(scores, dtype="<i8").tobytes(order="C")
    if len(payload) != SCORE_BYTES:
        raise AttentionScoreReferenceError(
            f"score 载荷长度错误：{len(payload)} != {SCORE_BYTES}"
        )
    return payload


def decode_scores(payload: bytes) -> np.ndarray:
    if len(payload) != SCORE_BYTES:
        raise AttentionScoreReferenceError(
            f"score 载荷长度错误：{len(payload)} != {SCORE_BYTES}"
        )
    return np.frombuffer(payload, dtype="<i8").copy().reshape(Q_HEADS, MAX_TOKENS)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_array(array: np.ndarray, dtype: str | np.dtype = "<i8") -> str:
    return sha256_bytes(np.asarray(array, dtype=dtype).tobytes(order="C"))


def build_case(
    *,
    layer: int,
    query_position: int,
    window_start: int,
    q_q28: np.ndarray,
    k_history_q28: np.ndarray,
    label: str,
) -> AttentionScoreCase:
    resolved_layer = validate_layer(layer)
    history = np.asarray(k_history_q28, dtype=np.int64)
    if history.ndim != 3:
        raise AttentionScoreReferenceError(
            f"k_history_q28 形状错误：{history.shape}"
        )
    count = int(history.shape[0])
    validate_window(query_position, window_start, count)
    q = np.asarray(q_q28, dtype=np.int64)
    _require_shape(q, (Q_HEADS, HEAD_DIM), "q_q28")
    _require_shape(history, (count, KV_HEADS, HEAD_DIM), "k_history_q28")
    scores = attention_scores_q28(
        q,
        history,
        query_position=query_position,
        window_start=window_start,
        count=count,
    )
    return AttentionScoreCase(
        layer=resolved_layer,
        query_position=int(query_position),
        window_start=int(window_start),
        count=count,
        q_q28=q,
        k_history_q28=history,
        expected_scores_q28=scores,
        label=label,
    )


def build_fixed_real_cases(
    windows: Iterable[tuple[int, int, int, int]] = DEFAULT_FIXED_WINDOWS,
    *,
    image_path: Path = DEFAULT_IMAGE,
    activation_seed: int = DEFAULT_ACTIVATION_SEED,
) -> list[AttentionScoreCase]:
    """基于真实 F1 Q/K 和 F2 RoPE 建立 F4 固定用例。"""

    resolved_windows = [
        (
            validate_layer(layer),
            *validate_window(query_position, window_start, count),
        )
        for layer, query_position, window_start, count in windows
    ]

    image = P50Image(image_path)
    image.validate()
    qkv_cases = build_qkv_cases(
        load_qkv_models(image), activation_seed=activation_seed
    )
    q_before = reshape_heads(
        qkv_cases["q"].expected_q28, qkv_cases["q"].spec
    ).astype(np.int64)
    k_before = reshape_heads(
        qkv_cases["k"].expected_q28, qkv_cases["k"].spec
    ).astype(np.int64)

    q_by_position: dict[int, np.ndarray] = {}
    k_by_position: dict[int, np.ndarray] = {}
    cases: list[AttentionScoreCase] = []
    for layer, query_position, window_start, count in resolved_windows:
        if query_position not in q_by_position:
            q_by_position[query_position] = apply_rope_fixed_q28(
                q_before,
                generate_trig_row(query_position),
                heads=Q_HEADS,
            )
        history: list[np.ndarray] = []
        for position in range(window_start, window_start + count):
            if position not in k_by_position:
                k_by_position[position] = apply_rope_fixed_q28(
                    k_before,
                    generate_trig_row(position),
                    heads=KV_HEADS,
                )
            history.append(k_by_position[position])
        cases.append(
            build_case(
                layer=layer,
                query_position=query_position,
                window_start=window_start,
                q_q28=q_by_position[query_position],
                k_history_q28=np.stack(history, axis=0),
                label=(
                    f"layer={layer}, query={query_position}, "
                    f"window={window_start}..{window_start + count - 1}"
                ),
            )
        )
    return cases


def fixed_manifest(cases: Sequence[AttentionScoreCase]) -> dict[str, object]:
    return {
        "format_version": 1,
        "definition": {
            "q_heads": Q_HEADS,
            "kv_heads": KV_HEADS,
            "head_dim": HEAD_DIM,
            "gqa_group_size": GQA_GROUP_SIZE,
            "input_format": "signed int64 Q28",
            "dot_format": "signed exact Q56",
            "scale": "1/sqrt(64)=1/8",
            "output_rule": "RNE signed dot right shift 31, saturate int64",
            "output_format": "signed int64 Q28",
            "mask_value": MASK_VALUE,
            "max_tokens": MAX_TOKENS,
            "output_layout": "head-major [14,16]",
            "score_bytes": SCORE_BYTES,
            "k_address_formula_ctrl": (
                "0x02000000 + layer*0x00800000 + position*0x00000200"
            ),
        },
        "cases": [
            {
                "label": case.label,
                "layer": case.layer,
                "query_position": case.query_position,
                "window_start": case.window_start,
                "count": case.count,
                "positions": list(case.positions),
                "valid_positions": [
                    position
                    for position in case.positions
                    if position <= case.query_position
                ],
                "k_base_ctrl": [
                    kv_slot_address(case.layer, position).k_base_ctrl
                    for position in case.positions
                ],
                "sha256": {
                    "q_q28": sha256_array(case.q_q28),
                    "k_history_q28": sha256_array(case.k_history_q28),
                    "scores_q28": sha256_array(case.expected_scores_q28),
                    "q_payload": sha256_bytes(case.q_payload),
                    "score_payload": sha256_bytes(
                        encode_scores(case.expected_scores_q28)
                    ),
                },
                "preview": {
                    "head0": case.expected_scores_q28[0, : case.count].tolist(),
                    "head6": case.expected_scores_q28[6, : case.count].tolist(),
                    "head7": case.expected_scores_q28[7, : case.count].tolist(),
                    "head13": case.expected_scores_q28[13, : case.count].tolist(),
                },
            }
            for case in cases
        ],
    }


def validate_manifest(
    cases: Sequence[AttentionScoreCase],
    manifest_path: Path = DEFAULT_MANIFEST,
) -> dict[str, object]:
    actual = fixed_manifest(cases)
    expected = json.loads(manifest_path.read_text(encoding="utf-8"))
    if actual != expected:
        raise AttentionScoreReferenceError(
            f"Attention Score 固定清单不一致：{manifest_path}"
        )
    return expected


def software_stress(
    rounds: int = 1000,
    seed: int = DEFAULT_STRESS_SEED,
) -> None:
    """随机验证 GQA 映射、RNE、causal mask、载荷和固定输出布局。"""

    if rounds <= 0:
        raise AttentionScoreReferenceError("rounds 必须大于 0")
    rng = np.random.default_rng(seed)
    limit = 8 << Q_FRACTION_BITS

    for _ in range(rounds):
        count = int(rng.integers(1, MAX_TOKENS + 1))
        start = int(rng.integers(0, MAX_CONTEXT - count + 1))
        query = int(
            rng.integers(max(0, start - 2), min(MAX_CONTEXT, start + count + 2))
        )
        q = rng.integers(
            -limit, limit + 1, size=(Q_HEADS, HEAD_DIM), dtype=np.int64
        )
        history = rng.integers(
            -limit,
            limit + 1,
            size=(count, KV_HEADS, HEAD_DIM),
            dtype=np.int64,
        )
        scores = attention_scores_q28(
            q,
            history,
            query_position=query,
            window_start=start,
            count=count,
        )
        decoded_q = decode_q_payload(build_q_payload(q))
        decoded_scores = decode_scores(encode_scores(scores))
        if not np.array_equal(decoded_q, q):
            raise AttentionScoreReferenceError("Q 载荷往返不一致")
        if not np.array_equal(decoded_scores, scores):
            raise AttentionScoreReferenceError("score 载荷往返不一致")
        for token_index in range(count):
            decoded_k = decode_k_payload(build_k_payload(history[token_index]))
            if not np.array_equal(decoded_k, history[token_index]):
                raise AttentionScoreReferenceError("K 载荷往返不一致")
            position = start + token_index
            for q_head in (0, 6, 7, 13):
                actual = int(scores[q_head, token_index])
                if position > query:
                    if actual != MASK_VALUE:
                        raise AttentionScoreReferenceError("causal mask 结果错误")
                else:
                    expected = scaled_dot_q28(
                        q[q_head], history[token_index, gqa_kv_head(q_head)]
                    )
                    if actual != expected:
                        raise AttentionScoreReferenceError("随机 score 结果错误")
        if np.any(scores[:, count:] != MASK_VALUE):
            raise AttentionScoreReferenceError("固定输出未使用槽没有保持 mask")


def _print_summary(manifest: dict[str, object]) -> None:
    definition = manifest["definition"]
    print("F4 Attention Score 软件金标准：PASS")
    print(
        f"Q={definition['q_heads']} heads，KV={definition['kv_heads']} heads，"
        f"head_dim={definition['head_dim']}，每 {definition['gqa_group_size']} 个 Q head 共用一个 KV head"
    )
    print(
        f"输出：{definition['output_layout']}，{definition['output_format']}，"
        f"mask={definition['mask_value']}"
    )
    for case in manifest["cases"]:
        print(
            f"{case['label']}，scores_sha256={case['sha256']['scores_q28']}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="F4 Attention Score 软件金标准")
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
        software_stress(rounds=args.rounds, seed=args.seed)
        _print_summary(manifest)
        print(
            f"Attention Score 软件随机压力 PASS：{args.rounds}/{args.rounds}，seed={args.seed}"
        )
        return 0
    except (
        FileNotFoundError,
        OSError,
        AttentionScoreReferenceError,
        OverflowError,
    ) as error:
        print(f"错误：{error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
