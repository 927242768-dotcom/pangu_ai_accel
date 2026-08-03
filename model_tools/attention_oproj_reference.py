#!/usr/bin/env python3
"""Qwen2.5-0.5B layer0 Attention O_proj 软件金标准。

本模块承接 F6 已验证的 ``[14,64] -> [896]`` signed int64 Q28 Attention
多头拼接结果，并完成真实 ``self_attn.o_proj=[896,896]`` 的分组 INT4
量化线性计算：

1. 将输入 Q28 精确解释为实数，并按逐向量对称规则量化为 INT8；
2. 从真实 ``.p50`` 读取 O_proj packed INT4 权重与 FP16 group scale；
3. 生成 ``activation_scale * weight_scale`` 的 UQ4.28 combined scale；
4. 每 64 元素先形成 INT32 点积，再在 signed int64 Q28 中跨组累加；
5. 本模型的 O_proj 不含 bias，因此 bias_q28 固定为 0。

硬件计算定义与 ``linear_quant_reference.py`` 保持一致，便于直接复用已经
真实上板验证的完整 Linear 数据通路，同时通过独立清单、上位机和 PDS 工程
隔离本阶段成果。
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
    from .attention_output_reference import (
        OUTPUT_VALUES,
        build_fixed_real_cases as build_attention_output_fixed_cases,
        flatten_attention_heads,
    )
    from .linear_quant_reference import (
        LinearReferenceResult,
        compute_groupwise_linear_reference,
        pack_int4_low_nibble_first,
    )
    from .p50_format import P50Image
except ImportError:
    from attention_output_reference import (
        OUTPUT_VALUES,
        build_fixed_real_cases as build_attention_output_fixed_cases,
        flatten_attention_heads,
    )
    from linear_quant_reference import (
        LinearReferenceResult,
        compute_groupwise_linear_reference,
        pack_int4_low_nibble_first,
    )
    from p50_format import P50Image

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

DEFAULT_IMAGE = Path("model_output/yanbo_qwen25_0.5b_int4.p50")
DEFAULT_MANIFEST = Path(__file__).with_name("attention_oproj_f6_reference.json")
DEFAULT_WEIGHT = "model.layers.0.self_attn.o_proj.weight"
DEFAULT_STRESS_SEED = 20260805

M = 896
K = OUTPUT_VALUES
GROUP_SIZE = 64
GROUPS = K // GROUP_SIZE
INPUT_FRACTION_BITS = 28
INPUT_FACTOR = 1 << INPUT_FRACTION_BITS


class AttentionOProjReferenceError(ValueError):
    """表示 O_proj 输入、张量、定点值或固定清单不合法。"""


@dataclass(frozen=True)
class AttentionOProjModel:
    """一个真实 O_proj 的 INT4 权重、FP16 group scale 与张量名。"""

    weights: np.ndarray
    weight_scales: np.ndarray
    weight_name: str = DEFAULT_WEIGHT


@dataclass(frozen=True)
class AttentionOProjCase:
    """一个由 Attention Q28 输入生成的完整 O_proj 硬件用例。"""

    label: str
    source_attention_q28: np.ndarray
    activation_int8: np.ndarray
    activation_scale: float
    weights: np.ndarray
    scales_q28: np.ndarray
    bias_q28: np.ndarray
    expected_q28: np.ndarray
    linear_result: LinearReferenceResult


def _require_shape(array: np.ndarray, shape: tuple[int, ...], label: str) -> None:
    if array.shape != shape:
        raise AttentionOProjReferenceError(
            f"{label} 形状错误：{array.shape}，预期 {shape}"
        )


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_array(array: np.ndarray, dtype: str | np.dtype) -> str:
    return sha256_bytes(np.asarray(array, dtype=dtype).tobytes(order="C"))


def attention_q28_to_float32(values: np.ndarray | Sequence[int]) -> np.ndarray:
    """把 ``[896]`` signed int64 Q28 输入转换为 float32 实数向量。"""

    q28 = np.asarray(values, dtype=np.int64)
    _require_shape(q28, (K,), "attention_q28")
    result = (q28.astype(np.float64) / INPUT_FACTOR).astype(np.float32)
    if not np.all(np.isfinite(result)):
        raise AttentionOProjReferenceError("Attention Q28 转 float32 后出现非有限值")
    return result


def load_oproj_model(
    image: P50Image,
    weight_name: str = DEFAULT_WEIGHT,
) -> AttentionOProjModel:
    """从真实 P50 镜像读取一个完整 O_proj；默认保持 layer0。"""

    block = image.extract_block(weight_name, 0, M, 0, K)
    if block.quantized is None or block.scales is None:
        raise AttentionOProjReferenceError("O_proj 权重不是分组 INT4 张量")
    weights = block.quantized.astype(np.int8)
    scales = block.scales.astype(np.float32)
    _require_shape(weights, (M, K), "O_proj weights")
    _require_shape(scales, (M, GROUPS), "O_proj scales")
    return AttentionOProjModel(
        weights=weights,
        weight_scales=scales,
        weight_name=weight_name,
    )


def compute_q28_reference(
    activation_int8: np.ndarray | Sequence[int],
    weights: np.ndarray,
    scales_q28: np.ndarray,
    bias_q28: np.ndarray | Sequence[int] | None = None,
) -> np.ndarray:
    """按 FPGA 精确定义独立重算完整 O_proj 的 896 个 Q28 输出。"""

    activation = np.asarray(activation_int8, dtype=np.int8)
    weight_values = np.asarray(weights, dtype=np.int8)
    scales = np.asarray(scales_q28, dtype=np.uint32)
    bias = (
        np.zeros(M, dtype=np.int64)
        if bias_q28 is None
        else np.asarray(bias_q28, dtype=np.int64)
    )
    _require_shape(activation, (K,), "activation_int8")
    _require_shape(weight_values, (M, K), "weights")
    _require_shape(scales, (M, GROUPS), "scales_q28")
    _require_shape(bias, (M,), "bias_q28")

    grouped_weights = weight_values.astype(np.int32).reshape(
        M, GROUPS, GROUP_SIZE
    )
    grouped_activation = activation.astype(np.int32).reshape(
        GROUPS, GROUP_SIZE
    )
    accumulators = np.sum(
        grouped_weights * grouped_activation[np.newaxis, :, :],
        axis=2,
        dtype=np.int64,
    )
    if np.any(accumulators < np.iinfo(np.int32).min) or np.any(
        accumulators > np.iinfo(np.int32).max
    ):
        raise AttentionOProjReferenceError("O_proj 分组点积超出 signed int32")

    outputs: list[int] = []
    for row in range(M):
        total = int(bias[row])
        for group in range(GROUPS):
            total += int(accumulators[row, group]) * int(scales[row, group])
        if not -(1 << 63) <= total <= (1 << 63) - 1:
            raise AttentionOProjReferenceError(
                f"O_proj 第 {row} 行 Q28 累加超出 signed int64"
            )
        outputs.append(total)
    return np.asarray(outputs, dtype=np.int64)


def case_from_attention_q28(
    model: AttentionOProjModel,
    attention_q28: np.ndarray | Sequence[int],
    *,
    label: str,
) -> AttentionOProjCase:
    """把一个 F6 Attention Q28 向量转换为完整真实 O_proj 用例。"""

    source = np.asarray(attention_q28, dtype=np.int64)
    _require_shape(source, (K,), "source_attention_q28")
    activation_float = attention_q28_to_float32(source)
    result = compute_groupwise_linear_reference(
        weight_quantized=model.weights,
        weight_scales=model.weight_scales,
        activation_values=activation_float,
        bias=None,
        group_size=GROUP_SIZE,
        weight_name=model.weight_name,
        bias_name=None,
    )
    if result.combined_scale_saturated_count:
        raise AttentionOProjReferenceError("O_proj combined scale 出现 UQ4.28 饱和")
    if np.any(result.bias_q28):
        raise AttentionOProjReferenceError("无 bias 的 O_proj 产生了非零 bias_q28")

    independent = compute_q28_reference(
        result.activation.quantized,
        model.weights,
        result.combined_scale_q28,
        result.bias_q28,
    )
    if not np.array_equal(independent, result.output_fixed_q28):
        raise AttentionOProjReferenceError(
            "O_proj 独立 Q28 重算与 linear_quant_reference 不一致"
        )

    return AttentionOProjCase(
        label=label,
        source_attention_q28=source.copy(),
        activation_int8=result.activation.quantized.astype(np.int8),
        activation_scale=float(result.activation.scale),
        weights=model.weights,
        scales_q28=result.combined_scale_q28.astype(np.uint32),
        bias_q28=result.bias_q28.astype(np.int64),
        expected_q28=result.output_fixed_q28.astype(np.int64),
        linear_result=result,
    )


def build_fixed_real_cases(
    *, image_path: Path = DEFAULT_IMAGE,
) -> list[AttentionOProjCase]:
    """使用 F6 四组真实固定 Attention 输出建立 O_proj 固定用例。"""

    attention_cases = build_attention_output_fixed_cases(image_path=image_path)
    image = P50Image(image_path)
    image.validate()
    model = load_oproj_model(image)
    cases: list[AttentionOProjCase] = []
    for source in attention_cases:
        flat = flatten_attention_heads(source.expected_heads_q28)
        cases.append(
            case_from_attention_q28(
                model,
                flat,
                label=(
                    f"{source.label} -> layer0 o_proj，"
                    f"window={source.window_start}+{source.count}"
                ),
            )
        )
    return cases


def fixed_manifest(cases: Sequence[AttentionOProjCase]) -> dict[str, object]:
    """生成可提交到 Git 的 O_proj 固定清单。"""

    return {
        "format_version": 1,
        "definition": {
            "source": "F6 real attention output fixed cases",
            "input_format": "signed int64 Q28 [896] head-major contiguous",
            "activation_quantization": (
                "dequantize Q28 to float32, symmetric per-vector INT8 [-127,127], "
                "RNE, zero_point=0"
            ),
            "weight_tensor": DEFAULT_WEIGHT,
            "weight_shape": [M, K],
            "weight_storage": "signed INT4 groupwise symmetric",
            "group_size": GROUP_SIZE,
            "groups_per_row": GROUPS,
            "combined_scale": "unsigned UQ4.28",
            "bias": "absent in P50; bias_q28 is all zero",
            "output_format": "signed int64 Q28 [896]",
            "formula": (
                "output_q28[row] = sum(group_acc_int32 * "
                "combined_scale_uq4_28)"
            ),
        },
        "cases": [
            {
                "label": case.label,
                "activation_scale": case.activation_scale,
                "activation_clipped_count": case.linear_result.activation.clipped_count,
                "combined_scale_saturated_count": (
                    case.linear_result.combined_scale_saturated_count
                ),
                "max_output_error_from_activation_quantization": float(
                    np.max(np.abs(case.linear_result.activation_error))
                ),
                "max_fixed_scale_error": float(
                    np.max(np.abs(case.linear_result.fixed_error))
                ),
                "max_fixed_error_bound": float(
                    np.max(case.linear_result.fixed_error_bound)
                ),
                "sha256": {
                    "source_attention_q28": sha256_array(
                        case.source_attention_q28, "<i8"
                    ),
                    "activation_int8": sha256_array(case.activation_int8, np.int8),
                    "packed_weight_int4": sha256_array(
                        pack_int4_low_nibble_first(case.weights), np.uint8
                    ),
                    "combined_scale_uq4_28": sha256_array(
                        case.scales_q28, "<u4"
                    ),
                    "bias_q28": sha256_array(case.bias_q28, "<i8"),
                    "output_fixed_q28": sha256_array(case.expected_q28, "<i8"),
                },
                "preview": {
                    "input_first8_q28": case.source_attention_q28[:8].tolist(),
                    "input_last8_q28": case.source_attention_q28[-8:].tolist(),
                    "activation_first16_int8": case.activation_int8[:16].tolist(),
                    "output_first8_q28": case.expected_q28[:8].tolist(),
                    "output_last8_q28": case.expected_q28[-8:].tolist(),
                },
            }
            for case in cases
        ],
    }


def validate_manifest(
    cases: Sequence[AttentionOProjCase],
    manifest_path: Path = DEFAULT_MANIFEST,
) -> dict[str, object]:
    actual = fixed_manifest(cases)
    expected = json.loads(manifest_path.read_text(encoding="utf-8"))
    if actual != expected:
        raise AttentionOProjReferenceError(
            f"Attention O_proj 固定清单不一致：{manifest_path}"
        )
    return expected


def make_random_attention_q28(
    rng: np.random.Generator, round_index: int
) -> np.ndarray:
    """生成覆盖零向量、常量、边界和一般分布的随机 Q28 输入。"""

    mode = round_index % 8
    if mode == 0:
        return np.zeros(K, dtype=np.int64)
    if mode == 1:
        value = int(rng.integers(-(8 << INPUT_FRACTION_BITS), 8 << INPUT_FRACTION_BITS))
        return np.full(K, value, dtype=np.int64)
    if mode == 2:
        values = np.zeros(K, dtype=np.int64)
        values[0] = 16 << INPUT_FRACTION_BITS
        values[1] = -(16 << INPUT_FRACTION_BITS)
        return values
    limit = 16 << INPUT_FRACTION_BITS
    return rng.integers(-limit, limit + 1, size=K, dtype=np.int64)


def software_stress(
    *,
    image_path: Path = DEFAULT_IMAGE,
    rounds: int = 1000,
    seed: int = DEFAULT_STRESS_SEED,
) -> None:
    """运行 O_proj 输入量化、真实参数和 Q28 定点路径随机压力测试。"""

    if rounds <= 0:
        raise AttentionOProjReferenceError("rounds 必须大于 0")
    image = P50Image(image_path)
    image.validate()
    model = load_oproj_model(image)
    rng = np.random.default_rng(seed)
    for round_index in range(rounds):
        source = make_random_attention_q28(rng, round_index)
        case = case_from_attention_q28(
            model,
            source,
            label=f"O_proj software stress {round_index + 1}/{rounds}",
        )
        if source.max(initial=0) == 0 and source.min(initial=0) == 0:
            if np.any(case.activation_int8) or np.any(case.expected_q28):
                raise AttentionOProjReferenceError("O_proj 全零输入没有严格输出全零")


def _print_summary(manifest: dict[str, object]) -> None:
    print("F6 Attention O_proj 软件金标准：PASS")
    definition = manifest["definition"]
    print(f"权重：{definition['weight_tensor']}，shape={definition['weight_shape']}")
    print(
        f"输入：{definition['input_format']} -> {definition['activation_quantization']}"
    )
    for index, case in enumerate(manifest["cases"]):
        print(
            f"case{index}: {case['label']}，activation_scale="
            f"{case['activation_scale']:.12g}，output_sha256="
            f"{case['sha256']['output_fixed_q28']}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="F6 layer0 Attention O_proj 软件金标准")
    sub = parser.add_subparsers(dest="command", required=True)

    manifest = sub.add_parser("manifest", help="输出四组真实固定清单 JSON")
    manifest.add_argument("--image", type=Path, default=DEFAULT_IMAGE)

    check = sub.add_parser("check", help="校验已提交固定清单")
    check.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    check.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)

    stress = sub.add_parser("stress", help="运行随机软件压力测试")
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
            _print_summary(committed)
        elif args.command == "stress":
            software_stress(
                image_path=args.image,
                rounds=args.rounds,
                seed=args.seed,
            )
            print(
                f"Attention O_proj 软件随机压力 PASS：{args.rounds}/{args.rounds}，"
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
