#!/usr/bin/env python3
"""Qwen2.5-0.5B layer0 Attention 残差与完整子层软件参考。

本模块把此前已经分别验证的 layer0 算子串成一条连贯的软件路径：

hidden state(Q6.10)
-> input RMSNorm(Q6.10)
-> Q/K/V 真实 INT4 Linear(Q28)
-> RoPE(Q28)
-> Attention Score(Q28)
-> Softmax(UQ1.31)
-> probability x V(Q28)
-> O_proj(Q28)
-> Q28 到 Q6.10 的 signed RNE 重标定
-> 与原 hidden state 残差相加并 signed int16 饱和。

硬件残差入口只需要接收两路已经明确格式的数据：

- residual hidden state: signed int16 Q6.10 [896]
- O_proj output: signed int64 Q28 [896]

O_proj 先执行 signed RNE 右移 18 位并饱和到 signed Q6.10，再与 residual
扩展相加，最终再次饱和到 signed int16。该规则与已验证 elementwise_k896
的残差输出格式保持一致。
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
    from .attention_oproj_reference import (
        AttentionOProjModel,
        case_from_attention_q28,
        load_oproj_model,
    )
    from .attention_output_reference import (
        attention_output_q28,
        flatten_attention_heads,
    )
    from .attention_score_reference import attention_scores_q28
    from .elementwise_fixed_reference import Q_MAX, Q_MIN, residual_add_q10
    from .p50_format import P50Image
    from .qkv_linear_reference import (
        ProjectionModel,
        case_from_model,
        load_qkv_models,
        reshape_heads,
    )
    from .rmsnorm_fixed_reference import (
        DEFAULT_EPSILON,
        DEFAULT_GAMMA,
        compute_rmsnorm_reference,
        make_deterministic_input,
    )
    from .rope_fixed_reference import apply_rope_fixed_q28, generate_trig_row
    from .softmax_fixed_reference import softmax_scores_q31
except ImportError:
    from attention_oproj_reference import (
        AttentionOProjModel,
        case_from_attention_q28,
        load_oproj_model,
    )
    from attention_output_reference import attention_output_q28, flatten_attention_heads
    from attention_score_reference import attention_scores_q28
    from elementwise_fixed_reference import Q_MAX, Q_MIN, residual_add_q10
    from p50_format import P50Image
    from qkv_linear_reference import (
        ProjectionModel,
        case_from_model,
        load_qkv_models,
        reshape_heads,
    )
    from rmsnorm_fixed_reference import (
        DEFAULT_EPSILON,
        DEFAULT_GAMMA,
        compute_rmsnorm_reference,
        make_deterministic_input,
    )
    from rope_fixed_reference import apply_rope_fixed_q28, generate_trig_row
    from softmax_fixed_reference import softmax_scores_q31

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE = PROJECT_ROOT / "model_output/yanbo_qwen25_0.5b_int4.p50"
DEFAULT_MANIFEST = Path(__file__).with_name("attention_residual_f6_reference.json")
DEFAULT_HIDDEN_SEED_BASE = 20260806
DEFAULT_STRESS_SEED = 20260806
DEFAULT_FIXED_QUERIES = (0, 1, 5, 15)

K = 896
Q28_FRACTION_BITS = 28
Q10_FRACTION_BITS = 10
RESCALE_SHIFT = Q28_FRACTION_BITS - Q10_FRACTION_BITS


class AttentionResidualReferenceError(ValueError):
    """表示完整 Attention 子层或残差定点输入不合法。"""


@dataclass(frozen=True)
class TokenAttentionState:
    """一个 token 在 layer0 Attention 前半段的连贯中间结果。"""

    position: int
    hidden_seed: int
    hidden_q10: np.ndarray
    norm_q10: np.ndarray
    q_rope_q28: np.ndarray
    k_rope_q28: np.ndarray
    v_q28: np.ndarray


@dataclass(frozen=True)
class AttentionResidualCase:
    """一个完整 layer0 Attention 子层固定用例。"""

    label: str
    query_position: int
    window_start: int
    count: int
    hidden_seed_base: int
    residual_hidden_q10: np.ndarray
    scores_q28: np.ndarray
    probabilities_q31: np.ndarray
    attention_concat_q28: np.ndarray
    oproj_q28: np.ndarray
    oproj_q10: np.ndarray
    output_q10: np.ndarray
    oproj_rescale_saturated_count: int
    residual_saturated_count: int


@dataclass(frozen=True)
class AttentionSublayerContext:
    image: P50Image
    gamma: np.ndarray
    qkv_models: dict[str, ProjectionModel]
    oproj_model: AttentionOProjModel


def _require_shape(array: np.ndarray, shape: tuple[int, ...], label: str) -> None:
    if array.shape != shape:
        raise AttentionResidualReferenceError(
            f"{label} 形状错误：{array.shape}，预期 {shape}"
        )


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_array(array: np.ndarray, dtype: str | np.dtype) -> str:
    return sha256_bytes(np.asarray(array, dtype=dtype).tobytes(order="C"))


def round_shift_rne_signed(value: int, shift: int) -> int:
    """对任意 Python signed int 执行对称 round-to-nearest-even 右移。"""

    resolved = int(value)
    if shift <= 0:
        return resolved << (-shift)
    magnitude = abs(resolved)
    quotient, remainder = divmod(magnitude, 1 << shift)
    half = 1 << (shift - 1)
    if remainder > half or (remainder == half and (quotient & 1)):
        quotient += 1
    return -quotient if resolved < 0 else quotient


def rescale_oproj_q28_to_q10(
    values_q28: np.ndarray | Sequence[int],
) -> tuple[np.ndarray, int]:
    """把 signed int64 Q28 使用 signed RNE 转换并饱和到 signed Q6.10。"""

    source = np.asarray(values_q28, dtype=np.int64)
    _require_shape(source, (K,), "oproj_q28")
    rounded = np.fromiter(
        (round_shift_rne_signed(int(value), RESCALE_SHIFT) for value in source),
        dtype=np.int64,
        count=K,
    )
    clipped = np.clip(rounded, Q_MIN, Q_MAX)
    saturated_count = int(np.count_nonzero(rounded != clipped))
    return clipped.astype(np.int16), saturated_count


def attention_residual_q10(
    residual_hidden_q10: np.ndarray | Sequence[int],
    oproj_q28: np.ndarray | Sequence[int],
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """执行 O_proj 重标定和最终残差相加。

    返回 ``(output_q10, oproj_q10, rescale_saturated_count,
    residual_saturated_count)``。
    """

    residual = np.asarray(residual_hidden_q10)
    if residual.ndim != 1:
        residual = residual.reshape(-1)
    _require_shape(residual, (K,), "residual_hidden_q10")
    if not np.issubdtype(residual.dtype, np.integer):
        raise AttentionResidualReferenceError("residual hidden 必须是整数 Q6.10")
    residual_wide = residual.astype(np.int64)
    if np.any(residual_wide < Q_MIN) or np.any(residual_wide > Q_MAX):
        raise AttentionResidualReferenceError("residual hidden 超出 signed int16")

    oproj_q10, rescale_saturated = rescale_oproj_q28_to_q10(oproj_q28)
    output, residual_saturated = residual_add_q10(
        residual_wide.astype(np.int16), oproj_q10
    )
    return output, oproj_q10, rescale_saturated, residual_saturated


def load_context(image_path: Path = DEFAULT_IMAGE) -> AttentionSublayerContext:
    image = P50Image(image_path)
    image.validate()
    gamma = image.read_float16_tensor(DEFAULT_GAMMA).astype(np.float32).reshape(-1)
    _require_shape(gamma, (K,), "layer0 input_layernorm gamma")
    return AttentionSublayerContext(
        image=image,
        gamma=gamma,
        qkv_models=load_qkv_models(image),
        oproj_model=load_oproj_model(image),
    )


def build_token_state(
    context: AttentionSublayerContext,
    *,
    position: int,
    hidden_seed: int,
) -> TokenAttentionState:
    """从一个真实 hidden state 构造同一 token 的 RMSNorm/Q/K/V/RoPE。"""

    if position < 0:
        raise AttentionResidualReferenceError("position 不能为负")
    hidden_float = make_deterministic_input(K, hidden_seed)
    rms = compute_rmsnorm_reference(
        activation_values=hidden_float,
        gamma_values=context.gamma,
        epsilon=DEFAULT_EPSILON,
        gamma_name=DEFAULT_GAMMA,
    )
    norm_float = (rms.output_lut_q10.astype(np.float64) / (1 << Q10_FRACTION_BITS)).astype(
        np.float32
    )
    q_case = case_from_model(
        context.qkv_models["q"],
        activation_values=norm_float,
        label=f"coherent q_proj position={position}",
    )
    k_case = case_from_model(
        context.qkv_models["k"],
        activation_values=norm_float,
        label=f"coherent k_proj position={position}",
    )
    v_case = case_from_model(
        context.qkv_models["v"],
        activation_values=norm_float,
        label=f"coherent v_proj position={position}",
    )
    q_before = reshape_heads(q_case.expected_q28, q_case.spec).astype(np.int64)
    k_before = reshape_heads(k_case.expected_q28, k_case.spec).astype(np.int64)
    v_heads = reshape_heads(v_case.expected_q28, v_case.spec).astype(np.int64)
    trig = generate_trig_row(position)
    return TokenAttentionState(
        position=position,
        hidden_seed=hidden_seed,
        hidden_q10=rms.activation.quantized.astype(np.int16),
        norm_q10=rms.output_lut_q10.astype(np.int16),
        q_rope_q28=apply_rope_fixed_q28(q_before, trig, heads=14).astype(np.int64),
        k_rope_q28=apply_rope_fixed_q28(k_before, trig, heads=2).astype(np.int64),
        v_q28=v_heads,
    )


def build_coherent_case(
    context: AttentionSublayerContext,
    *,
    query_position: int,
    window_start: int,
    hidden_seed_base: int = DEFAULT_HIDDEN_SEED_BASE,
    token_cache: dict[int, TokenAttentionState] | None = None,
) -> AttentionResidualCase:
    """构造一个从 hidden state 到最终残差输出的完整 layer0 Attention 用例。"""

    query = int(query_position)
    start = int(window_start)
    if query < start:
        raise AttentionResidualReferenceError("query_position 不能小于 window_start")
    count = query - start + 1
    if not 1 <= count <= 16:
        raise AttentionResidualReferenceError("当前完整子层窗口只支持 1..16 token")

    cache = {} if token_cache is None else token_cache
    states: list[TokenAttentionState] = []
    for position in range(start, query + 1):
        if position not in cache:
            cache[position] = build_token_state(
                context,
                position=position,
                hidden_seed=hidden_seed_base + position,
            )
        states.append(cache[position])

    current = states[-1]
    k_history = np.stack([state.k_rope_q28 for state in states], axis=0)
    v_history = np.stack([state.v_q28 for state in states], axis=0)
    scores = attention_scores_q28(
        current.q_rope_q28,
        k_history,
        query_position=query,
        window_start=start,
        count=count,
    )
    probabilities, _ = softmax_scores_q31(scores)
    attention_heads, _ = attention_output_q28(
        probabilities,
        v_history,
        count=count,
    )
    attention_concat = flatten_attention_heads(attention_heads)
    oproj_case = case_from_attention_q28(
        context.oproj_model,
        attention_concat,
        label=f"coherent layer0 o_proj query={query}",
    )
    output, oproj_q10, rescale_sat, residual_sat = attention_residual_q10(
        current.hidden_q10,
        oproj_case.expected_q28,
    )
    return AttentionResidualCase(
        label=f"layer0 coherent attention query={query}, window={start}..{query}",
        query_position=query,
        window_start=start,
        count=count,
        hidden_seed_base=hidden_seed_base,
        residual_hidden_q10=current.hidden_q10.copy(),
        scores_q28=scores,
        probabilities_q31=probabilities,
        attention_concat_q28=attention_concat,
        oproj_q28=oproj_case.expected_q28.astype(np.int64),
        oproj_q10=oproj_q10,
        output_q10=output,
        oproj_rescale_saturated_count=rescale_sat,
        residual_saturated_count=residual_sat,
    )


def build_fixed_real_cases(
    *,
    image_path: Path = DEFAULT_IMAGE,
    hidden_seed_base: int = DEFAULT_HIDDEN_SEED_BASE,
    queries: Iterable[int] = DEFAULT_FIXED_QUERIES,
) -> list[AttentionResidualCase]:
    context = load_context(image_path)
    cache: dict[int, TokenAttentionState] = {}
    cases: list[AttentionResidualCase] = []
    for query in queries:
        resolved = int(query)
        start = max(0, resolved - 15)
        cases.append(
            build_coherent_case(
                context,
                query_position=resolved,
                window_start=start,
                hidden_seed_base=hidden_seed_base,
                token_cache=cache,
            )
        )
    return cases


def build_upload_payload(case: AttentionResidualCase) -> bytes:
    """硬件载荷：1792 B residual Q10 + 7168 B O_proj Q28。"""

    _require_shape(case.residual_hidden_q10, (K,), "residual_hidden_q10")
    _require_shape(case.oproj_q28, (K,), "oproj_q28")
    payload = (
        np.asarray(case.residual_hidden_q10, dtype="<i2").tobytes(order="C")
        + np.asarray(case.oproj_q28, dtype="<i8").tobytes(order="C")
    )
    if len(payload) != K * 2 + K * 8:
        raise AttentionResidualReferenceError("Attention residual 上传载荷长度错误")
    return payload


def fixed_manifest(cases: Sequence[AttentionResidualCase]) -> dict[str, object]:
    return {
        "format_version": 1,
        "definition": {
            "model": "Qwen2.5-0.5B layer0",
            "complete_path": (
                "hidden Q6.10 -> RMSNorm -> QKV -> RoPE -> score -> softmax -> "
                "probability*V -> O_proj Q28 -> residual"
            ),
            "residual_input": "signed int16 Q6.10 [896]",
            "oproj_input": "signed int64 Q28 [896]",
            "oproj_rescale": "signed RNE right shift 18, then signed int16 saturation",
            "residual_add": "sign-extended Q6.10 add, then signed int16 saturation",
            "output": "signed int16 Q6.10 [896]",
            "upload_bytes": K * 10,
            "result_bytes": K * 2,
            "fixed_queries": [case.query_position for case in cases],
        },
        "cases": [
            {
                "label": case.label,
                "query_position": case.query_position,
                "window_start": case.window_start,
                "count": case.count,
                "hidden_seed_base": case.hidden_seed_base,
                "oproj_rescale_saturated_count": case.oproj_rescale_saturated_count,
                "residual_saturated_count": case.residual_saturated_count,
                "sha256": {
                    "residual_hidden_q10": sha256_array(case.residual_hidden_q10, "<i2"),
                    "scores_q28": sha256_array(case.scores_q28, "<i8"),
                    "probabilities_q31": sha256_array(case.probabilities_q31, "<u4"),
                    "attention_concat_q28": sha256_array(case.attention_concat_q28, "<i8"),
                    "oproj_q28": sha256_array(case.oproj_q28, "<i8"),
                    "oproj_q10": sha256_array(case.oproj_q10, "<i2"),
                    "output_q10": sha256_array(case.output_q10, "<i2"),
                    "upload_payload": sha256_bytes(build_upload_payload(case)),
                },
                "preview": {
                    "residual_first8_q10": case.residual_hidden_q10[:8].tolist(),
                    "oproj_first8_q28": case.oproj_q28[:8].tolist(),
                    "oproj_first8_q10": case.oproj_q10[:8].tolist(),
                    "output_first8_q10": case.output_q10[:8].tolist(),
                    "output_last8_q10": case.output_q10[-8:].tolist(),
                },
            }
            for case in cases
        ],
    }


def validate_manifest(
    cases: Sequence[AttentionResidualCase],
    manifest_path: Path = DEFAULT_MANIFEST,
) -> dict[str, object]:
    actual = fixed_manifest(cases)
    expected = json.loads(manifest_path.read_text(encoding="utf-8"))
    if actual != expected:
        raise AttentionResidualReferenceError(
            f"Attention residual 固定清单不一致：{manifest_path}"
        )
    return expected


def make_random_residual_inputs(
    rng: np.random.Generator, round_index: int
) -> tuple[np.ndarray, np.ndarray]:
    """生成覆盖 RNE tie、极值和随机范围的 residual/O_proj 输入。"""

    hidden = rng.integers(Q_MIN, Q_MAX + 1, size=K, dtype=np.int32).astype(np.int16)
    mode = round_index % 6
    if mode == 0:
        oproj = np.zeros(K, dtype=np.int64)
    elif mode == 1:
        # 正负 half-way ties，覆盖偶数保持与奇数进位。
        base = rng.integers(-40000, 40001, size=K, dtype=np.int64)
        signs = np.where((np.arange(K) & 1) == 0, 1, -1).astype(np.int64)
        oproj = base * (1 << RESCALE_SHIFT) + signs * (1 << (RESCALE_SHIFT - 1))
    elif mode == 2:
        oproj = np.empty(K, dtype=np.int64)
        oproj[0::2] = np.iinfo(np.int64).max
        oproj[1::2] = np.iinfo(np.int64).min
    else:
        limit = 64 << Q28_FRACTION_BITS
        oproj = rng.integers(-limit, limit, size=K, dtype=np.int64)
    return hidden, oproj


def software_stress(
    *, rounds: int = 1000, seed: int = DEFAULT_STRESS_SEED
) -> None:
    if rounds <= 0:
        raise AttentionResidualReferenceError("rounds 必须大于 0")
    rng = np.random.default_rng(seed)
    for index in range(rounds):
        hidden, oproj = make_random_residual_inputs(rng, index)
        output, oproj_q10, _, _ = attention_residual_q10(hidden, oproj)
        # 独立逐元素 Python int 重算，避免 NumPy 溢出掩盖错误。
        expected_oproj: list[int] = []
        expected_output: list[int] = []
        for hidden_value, oproj_value in zip(hidden, oproj):
            scaled = round_shift_rne_signed(int(oproj_value), RESCALE_SHIFT)
            scaled = min(max(scaled, Q_MIN), Q_MAX)
            total = min(max(int(hidden_value) + scaled, Q_MIN), Q_MAX)
            expected_oproj.append(scaled)
            expected_output.append(total)
        if not np.array_equal(oproj_q10, np.asarray(expected_oproj, dtype=np.int16)):
            raise AttentionResidualReferenceError("O_proj Q28->Q10 独立重算不一致")
        if not np.array_equal(output, np.asarray(expected_output, dtype=np.int16)):
            raise AttentionResidualReferenceError("Attention residual 独立重算不一致")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="layer0 Attention 残差与完整子层软件参考")
    sub = parser.add_subparsers(dest="command", required=True)

    manifest = sub.add_parser("manifest", help="生成四组连贯真实固定清单")
    manifest.add_argument("--image", type=Path, default=DEFAULT_IMAGE)

    check = sub.add_parser("check", help="校验已提交固定清单")
    check.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    check.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)

    stress = sub.add_parser("stress", help="运行残差定点随机压力")
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
            print("layer0 Attention 完整软件参考与残差固定清单：PASS")
            for item in committed["cases"]:
                print(
                    f"query={item['query_position']} count={item['count']} "
                    f"output_sha256={item['sha256']['output_q10']}"
                )
        elif args.command == "stress":
            software_stress(rounds=args.rounds, seed=args.seed)
            print(
                f"Attention residual 软件随机压力 PASS：{args.rounds}/{args.rounds}，"
                f"seed={args.seed}"
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
