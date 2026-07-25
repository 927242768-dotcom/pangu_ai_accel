#!/usr/bin/env python3
"""Qwen2.5-0.5B layer0 MLP 第二处残差硬件等价软件参考。

本阶段只消费两路已经分别真实上板逐位通过的数据：

- residual hidden：进入 ``post_attention_layernorm`` 之前，也就是完整 Attention
  第一处残差后的 ``[896]`` signed int16 Q6.10；
- down_proj output：真实 layer0 ``down_proj`` 的 ``[896]`` signed int64 Q28。

冻结的硬件规则：

1. down_proj Q28 执行正负对称 round-to-nearest-even 右移 18 位；
2. 显式饱和到 signed int16 Q6.10；
3. 与 residual hidden 符号扩展后相加；
4. 再次显式饱和到 signed int16 Q6.10。

严禁把 ``post_attention_layernorm`` 的归一化输出错误接到 residual 分支。
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
        DEFAULT_IMAGE,
        Q_MAX,
        Q_MIN,
        RESCALE_SHIFT,
        build_fixed_real_cases as build_attention_fixed_cases,
        round_shift_rne_signed,
    )
    from .elementwise_fixed_reference import residual_add_q10
    from .mlp_down_proj_reference import (
        build_fixed_real_cases as build_down_fixed_cases,
    )
except ImportError:
    from attention_residual_reference import (
        DEFAULT_IMAGE,
        Q_MAX,
        Q_MIN,
        RESCALE_SHIFT,
        build_fixed_real_cases as build_attention_fixed_cases,
        round_shift_rne_signed,
    )
    from elementwise_fixed_reference import residual_add_q10
    from mlp_down_proj_reference import build_fixed_real_cases as build_down_fixed_cases

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

DEFAULT_MANIFEST = Path(__file__).with_name("mlp_residual_g1_reference.json")
DEFAULT_STRESS_SEED = 20260817
K = 896
Q28_FRACTION_BITS = 28
Q10_FRACTION_BITS = 10
UPLOAD_BYTES = K * 10
RESULT_BYTES = K * 2


class MLPResidualReferenceError(ValueError):
    """表示 MLP 第二处残差输入、定点结果或固定清单不合法。"""


@dataclass(frozen=True)
class MLPResidualCase:
    label: str
    query_position: int
    count: int
    residual_hidden_q10: np.ndarray
    down_proj_q28: np.ndarray
    down_proj_q10: np.ndarray
    output_q10: np.ndarray
    down_rescale_saturated_count: int
    residual_saturated_count: int


def _require_shape(array: np.ndarray, shape: tuple[int, ...], label: str) -> None:
    if array.shape != shape:
        raise MLPResidualReferenceError(f"{label} 形状错误：{array.shape}，预期 {shape}")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_array(array: np.ndarray, dtype: str | np.dtype) -> str:
    return sha256_bytes(np.asarray(array, dtype=dtype).tobytes(order="C"))


def rescale_down_q28_to_q10(
    values_q28: np.ndarray | Sequence[int],
) -> tuple[np.ndarray, int]:
    """把 signed int64 Q28 使用 signed RNE 转换并饱和到 signed Q6.10。"""

    source = np.asarray(values_q28, dtype=np.int64)
    if source.ndim != 1:
        source = source.reshape(-1)
    _require_shape(source, (K,), "down_proj_q28")
    rounded = np.fromiter(
        (round_shift_rne_signed(int(value), RESCALE_SHIFT) for value in source),
        dtype=np.int64,
        count=K,
    )
    clipped = np.clip(rounded, Q_MIN, Q_MAX)
    saturated_count = int(np.count_nonzero(rounded != clipped))
    return clipped.astype(np.int16), saturated_count


def mlp_residual_q10(
    residual_hidden_q10: np.ndarray | Sequence[int],
    down_proj_q28: np.ndarray | Sequence[int],
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """执行 down_proj 重标定和 MLP 第二处残差。

    返回 ``(output_q10, down_q10, rescale_saturated_count,
    residual_saturated_count)``。
    """

    residual = np.asarray(residual_hidden_q10)
    if residual.ndim != 1:
        residual = residual.reshape(-1)
    _require_shape(residual, (K,), "residual_hidden_q10")
    if not np.issubdtype(residual.dtype, np.integer):
        raise MLPResidualReferenceError("residual hidden 必须是整数 Q6.10")
    residual_wide = residual.astype(np.int64)
    if np.any(residual_wide < Q_MIN) or np.any(residual_wide > Q_MAX):
        raise MLPResidualReferenceError("residual hidden 超出 signed int16")

    down_q10, rescale_saturated = rescale_down_q28_to_q10(down_proj_q28)
    output, residual_saturated = residual_add_q10(
        residual_wide.astype(np.int16), down_q10
    )
    return output, down_q10, rescale_saturated, residual_saturated


def build_fixed_real_cases(*, image_path: Path = DEFAULT_IMAGE) -> list[MLPResidualCase]:
    """配对同一 query/count 的真实 Attention residual hidden 与 down_proj 输出。"""

    attention_cases = build_attention_fixed_cases(image_path=image_path)
    down_cases = build_down_fixed_cases(image_path=image_path)
    if len(attention_cases) != len(down_cases):
        raise MLPResidualReferenceError("Attention 与 down_proj 固定用例数量不一致")

    cases: list[MLPResidualCase] = []
    for attention, down in zip(attention_cases, down_cases):
        if attention.query_position != down.query_position or attention.count != down.count:
            raise MLPResidualReferenceError(
                "Attention residual 与 down_proj 固定用例 query/count 不一致"
            )
        output, down_q10, rescale_sat, residual_sat = mlp_residual_q10(
            attention.output_q10,
            down.expected_q28,
        )
        cases.append(
            MLPResidualCase(
                label=(
                    f"layer0 MLP second residual query={attention.query_position}, "
                    f"count={attention.count}"
                ),
                query_position=int(attention.query_position),
                count=int(attention.count),
                residual_hidden_q10=attention.output_q10.astype(np.int16).copy(),
                down_proj_q28=down.expected_q28.astype(np.int64).copy(),
                down_proj_q10=down_q10,
                output_q10=output,
                down_rescale_saturated_count=rescale_sat,
                residual_saturated_count=residual_sat,
            )
        )
    return cases


def build_upload_payload(case: MLPResidualCase) -> bytes:
    """硬件载荷：1792 B residual Q10 + 7168 B down_proj Q28。"""

    _require_shape(case.residual_hidden_q10, (K,), "residual_hidden_q10")
    _require_shape(case.down_proj_q28, (K,), "down_proj_q28")
    payload = (
        np.asarray(case.residual_hidden_q10, dtype="<i2").tobytes(order="C")
        + np.asarray(case.down_proj_q28, dtype="<i8").tobytes(order="C")
    )
    if len(payload) != UPLOAD_BYTES:
        raise MLPResidualReferenceError(
            f"MLP residual 上传载荷长度错误：{len(payload)} != {UPLOAD_BYTES}"
        )
    return payload


def verify_upload_payload(case: MLPResidualCase) -> str:
    payload = build_upload_payload(case)
    hidden_bytes = K * 2
    hidden = np.frombuffer(payload[:hidden_bytes], dtype="<i2").copy()
    down = np.frombuffer(payload[hidden_bytes:], dtype="<i8").copy()
    if not np.array_equal(hidden, case.residual_hidden_q10):
        raise MLPResidualReferenceError("residual hidden 载荷往返不一致")
    if not np.array_equal(down, case.down_proj_q28):
        raise MLPResidualReferenceError("down_proj Q28 载荷往返不一致")
    return sha256_bytes(payload)


def fixed_manifest(cases: Sequence[MLPResidualCase]) -> dict[str, object]:
    return {
        "format_version": 1,
        "operator": "qwen2_layer0_mlp_second_residual",
        "definition": {
            "residual_source": (
                "verified complete Attention first-residual output before "
                "post_attention_layernorm"
            ),
            "residual_input": "signed int16 Q6.10 [896]",
            "down_source": "verified layer0 MLP down_proj output",
            "down_input": "signed int64 Q28 [896]",
            "down_rescale": "signed RNE right shift 18, then signed int16 saturation",
            "residual_add": "sign-extended Q6.10 add, then signed int16 saturation",
            "output": "signed int16 Q6.10 [896]",
            "upload_bytes": UPLOAD_BYTES,
            "result_bytes": RESULT_BYTES,
            "fixed_queries": [case.query_position for case in cases],
            "forbidden_residual_source": "post_attention_layernorm output",
            "forbidden_next_operation": (
                "complete MLP / Transformer Block before this residual passes"
            ),
        },
        "cases": [
            {
                "label": case.label,
                "query_position": case.query_position,
                "count": case.count,
                "down_rescale_saturated_count": case.down_rescale_saturated_count,
                "residual_saturated_count": case.residual_saturated_count,
                "range": {
                    "residual_hidden_q10_min": int(np.min(case.residual_hidden_q10)),
                    "residual_hidden_q10_max": int(np.max(case.residual_hidden_q10)),
                    "down_proj_q28_min": int(np.min(case.down_proj_q28)),
                    "down_proj_q28_max": int(np.max(case.down_proj_q28)),
                    "output_q10_min": int(np.min(case.output_q10)),
                    "output_q10_max": int(np.max(case.output_q10)),
                },
                "sha256": {
                    "residual_hidden_q10": sha256_array(case.residual_hidden_q10, "<i2"),
                    "down_proj_q28": sha256_array(case.down_proj_q28, "<i8"),
                    "down_proj_q10": sha256_array(case.down_proj_q10, "<i2"),
                    "output_q10": sha256_array(case.output_q10, "<i2"),
                    "upload_payload": verify_upload_payload(case),
                },
                "preview": {
                    "residual_first8_q10": case.residual_hidden_q10[:8].tolist(),
                    "down_first8_q28": case.down_proj_q28[:8].tolist(),
                    "down_first8_q10": case.down_proj_q10[:8].tolist(),
                    "output_first8_q10": case.output_q10[:8].tolist(),
                    "output_last8_q10": case.output_q10[-8:].tolist(),
                },
            }
            for case in cases
        ],
    }


def validate_manifest(
    cases: Sequence[MLPResidualCase],
    manifest_path: Path = DEFAULT_MANIFEST,
) -> dict[str, object]:
    actual = fixed_manifest(cases)
    expected = json.loads(manifest_path.read_text(encoding="utf-8"))
    if actual != expected:
        raise MLPResidualReferenceError(f"MLP residual 固定清单不一致：{manifest_path}")
    return expected


def make_random_residual_inputs(
    rng: np.random.Generator, round_index: int
) -> tuple[np.ndarray, np.ndarray]:
    """生成覆盖 RNE tie、INT64 极值和两级饱和的输入。"""

    hidden = rng.integers(Q_MIN, Q_MAX + 1, size=K, dtype=np.int32).astype(np.int16)
    mode = round_index % 6
    if mode == 0:
        down = np.zeros(K, dtype=np.int64)
    elif mode == 1:
        base = rng.integers(-40000, 40001, size=K, dtype=np.int64)
        signs = np.where((np.arange(K) & 1) == 0, 1, -1).astype(np.int64)
        down = base * (1 << RESCALE_SHIFT) + signs * (1 << (RESCALE_SHIFT - 1))
    elif mode == 2:
        down = np.empty(K, dtype=np.int64)
        down[0::2] = np.iinfo(np.int64).max
        down[1::2] = np.iinfo(np.int64).min
    elif mode == 3:
        # 精确覆盖 Q10 正负饱和边缘。
        q10 = np.resize(
            np.asarray([Q_MAX - 1, Q_MAX, Q_MAX + 1, Q_MIN + 1, Q_MIN, Q_MIN - 1]),
            K,
        ).astype(np.int64)
        down = q10 * (1 << RESCALE_SHIFT)
    else:
        limit = 64 << Q28_FRACTION_BITS
        down = rng.integers(-limit, limit, size=K, dtype=np.int64)
    return hidden, down


def software_stress(
    *, rounds: int = 1000, seed: int = DEFAULT_STRESS_SEED
) -> None:
    if rounds <= 0:
        raise MLPResidualReferenceError("rounds 必须大于 0")
    rng = np.random.default_rng(seed)
    for index in range(rounds):
        hidden, down = make_random_residual_inputs(rng, index)
        output, down_q10, _, _ = mlp_residual_q10(hidden, down)
        expected_down: list[int] = []
        expected_output: list[int] = []
        for hidden_value, down_value in zip(hidden, down):
            scaled = round_shift_rne_signed(int(down_value), RESCALE_SHIFT)
            scaled = min(max(scaled, Q_MIN), Q_MAX)
            total = min(max(int(hidden_value) + scaled, Q_MIN), Q_MAX)
            expected_down.append(scaled)
            expected_output.append(total)
        if not np.array_equal(down_q10, np.asarray(expected_down, dtype=np.int16)):
            raise MLPResidualReferenceError("down_proj Q28->Q10 独立重算不一致")
        if not np.array_equal(output, np.asarray(expected_output, dtype=np.int16)):
            raise MLPResidualReferenceError("MLP residual 独立重算不一致")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="layer0 MLP 第二处残差软件参考")
    sub = parser.add_subparsers(dest="command", required=True)
    manifest = sub.add_parser("manifest", help="输出四组连贯真实固定清单")
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
            print("layer0 MLP 第二处残差固定清单：PASS")
            for item in manifest["cases"]:
                print(
                    f"query={item['query_position']} count={item['count']} "
                    f"output={item['sha256']['output_q10']}"
                )
        elif args.command == "stress":
            software_stress(rounds=args.rounds, seed=args.seed)
            print(
                f"MLP residual 软件压力 PASS：{args.rounds}/{args.rounds}，seed={args.seed}"
            )
        else:  # pragma: no cover
            raise AssertionError(args.command)
        return 0
    except (FileNotFoundError, KeyError, ValueError, OverflowError, RuntimeError) as error:
        print(f"错误：{error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
