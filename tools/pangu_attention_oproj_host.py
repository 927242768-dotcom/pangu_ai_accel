#!/usr/bin/env python3
"""盘古 PGL50H layer0 Attention O_proj 独立闭环验证工具。"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model_tools.attention_oproj_reference import (  # noqa: E402
    DEFAULT_IMAGE,
    DEFAULT_MANIFEST,
    DEFAULT_STRESS_SEED,
    AttentionOProjCase,
    build_fixed_real_cases,
    case_from_attention_q28,
    load_oproj_model,
    make_random_attention_q28,
    validate_manifest,
)
from model_tools.p50_format import P50Image  # noqa: E402
from tools.pangu_gemv_qproj_full_host import (  # noqa: E402
    FullLayerCase,
    command_info,
    command_status,
    open_port,
    run_and_compare,
    sha256_array,
    show_ports,
    verify_payload_roundtrip,
    wait_until_ready,
)

DEFAULT_PORT = "COM20"


def to_full_layer_case(case: AttentionOProjCase) -> FullLayerCase:
    """转换为已验证完整 Linear UART/DDR3 数据通路可直接消费的载荷。"""

    return FullLayerCase(
        activation=case.activation_int8.astype(np.int8),
        weights=case.weights.astype(np.int8),
        scales_q28=case.scales_q28.astype(np.uint32),
        bias_q28=case.bias_q28.astype(np.int64),
        expected_q28=case.expected_q28.astype(np.int64),
        activation_scale=case.activation_scale,
        label=case.label,
    )


def validate_fixed_cases(
    cases: list[AttentionOProjCase], manifest_path: Path
) -> dict[str, object]:
    committed = validate_manifest(cases, manifest_path)
    for index, case in enumerate(cases):
        expected_hash = committed["cases"][index]["sha256"]["output_fixed_q28"]
        actual_hash = sha256_array(case.expected_q28, "<i8")
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"O_proj case{index} 输出哈希变化：{actual_hash} != {expected_hash}"
            )
        verify_payload_roundtrip(to_full_layer_case(case))
    return committed


def select_cases(
    cases: list[AttentionOProjCase], case_index: str
) -> list[tuple[int, AttentionOProjCase]]:
    if case_index.lower() == "all":
        return list(enumerate(cases))
    index = int(case_index)
    if not 0 <= index < len(cases):
        raise ValueError(f"case 必须为 0..{len(cases)-1} 或 all")
    return [(index, cases[index])]


def command_selftest(
    image_path: Path,
    manifest_path: Path,
    rounds: int,
    seed: int,
) -> None:
    if rounds <= 0:
        raise ValueError("rounds 必须大于 0")
    started = time.monotonic()
    cases = build_fixed_real_cases(image_path=image_path)
    committed = validate_fixed_cases(cases, manifest_path)
    print("O_proj 四组真实 F6 固定输入、清单和载荷往返：PASS")
    for index, case in enumerate(cases):
        payload_hash = verify_payload_roundtrip(to_full_layer_case(case))
        print(
            f"case{index}: input_sha256="
            f"{committed['cases'][index]['sha256']['source_attention_q28']}，"
            f"output_sha256={committed['cases'][index]['sha256']['output_fixed_q28']}，"
            f"upload_sha256={payload_hash}"
        )

    image = P50Image(image_path)
    image.validate()
    model = load_oproj_model(image)
    rng = np.random.default_rng(seed)
    for index in range(rounds):
        source = make_random_attention_q28(rng, index)
        case = case_from_attention_q28(
            model,
            source,
            label=f"O_proj 软件随机 {index + 1}/{rounds}",
        )
        verify_payload_roundtrip(to_full_layer_case(case))
        if index == 0 or index + 1 == rounds or (index + 1) % 100 == 0:
            print(f"O_proj 软件随机已通过 {index + 1}/{rounds}")
    elapsed = time.monotonic() - started
    print(
        f"O_proj 软件参考与完整上传载荷压力 PASS：{rounds}/{rounds}，"
        f"seed={seed}，耗时 {elapsed:.2f} 秒"
    )


def command_fixed(
    port_name: str,
    image_path: Path,
    manifest_path: Path,
    case_index: str,
) -> None:
    cases = build_fixed_real_cases(image_path=image_path)
    committed = validate_fixed_cases(cases, manifest_path)
    selected = select_cases(cases, case_index)
    with open_port(port_name) as port:
        wait_until_ready(port)
        for index, case in selected:
            started = time.monotonic()
            fpga = run_and_compare(port, to_full_layer_case(case))
            elapsed = time.monotonic() - started
            print(f"Attention O_proj 固定 case{index} 逐位一致：PASS")
            print(
                "输入 SHA256："
                f"{committed['cases'][index]['sha256']['source_attention_q28']}"
            )
            print(
                "输出 SHA256："
                f"{committed['cases'][index]['sha256']['output_fixed_q28']}"
            )
            print(f"前 8 行：{fpga[:8].tolist()}")
            print(f"后 8 行：{fpga[-8:].tolist()}")
            print(f"上传、计算与回读总耗时：{elapsed:.2f} 秒")


def command_stress(
    port_name: str,
    image_path: Path,
    rounds: int,
    seed: int,
) -> None:
    if rounds <= 0:
        raise ValueError("rounds 必须大于 0")
    image = P50Image(image_path)
    image.validate()
    model = load_oproj_model(image)
    rng = np.random.default_rng(seed)
    started = time.monotonic()
    with open_port(port_name) as port:
        wait_until_ready(port)
        for index in range(rounds):
            source = make_random_attention_q28(rng, index)
            case = case_from_attention_q28(
                model,
                source,
                label=f"O_proj 真实 FPGA 随机 {index + 1}/{rounds}",
            )
            run_and_compare(port, to_full_layer_case(case))
            print(f"O_proj 真实 FPGA 随机已通过 {index + 1}/{rounds}")
    elapsed = time.monotonic() - started
    print(
        f"Attention O_proj 真实 FPGA 随机回归 PASS：{rounds}/{rounds}，"
        f"seed={seed}，耗时 {elapsed:.2f} 秒"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PGL50H layer0 Attention O_proj M896K896 分组 Q28 上位机"
    )
    parser.add_argument("--port", default=DEFAULT_PORT)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("ports", help="列出串口")
    sub.add_parser("info", help="读取固件信息（复用完整 Linear V1 协议）")
    sub.add_parser("status", help="读取状态")

    fixed = sub.add_parser("fixed", help="运行真实 F6 输入到 O_proj 的固定用例")
    fixed.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    fixed.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    fixed.add_argument("--case", default="all", help="0..3 或 all")

    stress = sub.add_parser("stress", help="运行随机 Attention Q28 输入上板回归")
    stress.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    stress.add_argument("--rounds", type=int, default=4)
    stress.add_argument("--seed", type=int, default=DEFAULT_STRESS_SEED)

    selftest = sub.add_parser("selftest", help="运行软件金标准和上传载荷自检")
    selftest.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    selftest.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    selftest.add_argument("--rounds", type=int, default=1000)
    selftest.add_argument("--seed", type=int, default=DEFAULT_STRESS_SEED)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.command == "ports":
            show_ports()
        elif args.command == "selftest":
            command_selftest(args.image, args.manifest, args.rounds, args.seed)
        elif args.command == "fixed":
            command_fixed(args.port, args.image, args.manifest, args.case)
        elif args.command == "stress":
            command_stress(args.port, args.image, args.rounds, args.seed)
        elif args.command in {"info", "status"}:
            with open_port(args.port) as port:
                if args.command == "info":
                    command_info(port)
                else:
                    command_status(port)
        else:  # pragma: no cover
            raise AssertionError(args.command)
        return 0
    except (
        FileNotFoundError,
        KeyError,
        ValueError,
        OverflowError,
        RuntimeError,
        TimeoutError,
    ) as error:
        print(f"错误：{error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
