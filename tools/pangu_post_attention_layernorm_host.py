#!/usr/bin/env python3
"""盘古 PGL50H G1 layer0 post_attention_layernorm 验证工具。"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

try:
    import serial  # type: ignore
    from serial.tools import list_ports  # type: ignore
except ImportError:  # pragma: no cover
    serial = None
    list_ports = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model_tools.p50_format import P50Image  # noqa: E402
from model_tools.post_attention_layernorm_reference import (  # noqa: E402
    DATA_BYTES,
    DEFAULT_GAMMA,
    DEFAULT_IMAGE,
    DEFAULT_MANIFEST,
    DEFAULT_STRESS_SEED,
    PostAttentionLayerNormCase,
    build_fixed_real_cases,
    build_upload_payload,
    case_from_input_q10,
    load_gamma,
    make_random_input_q10,
    software_stress,
    validate_manifest,
    verify_payload_roundtrip,
)

BAUD_RATE = 115200
RESULT_BYTES = DATA_BYTES
DEFAULT_IMAGE_PATH = Path(DEFAULT_IMAGE)

ERROR_MESSAGES = {
    0x01: "未知命令",
    0x02: "DDR3 尚未初始化完成",
    0x04: "尚未加载 RMSNorm 数据",
    0xFF: "FPGA 状态机异常",
}


@dataclass(frozen=True)
class FpgaStatus:
    ddr_ready: bool
    data_loaded: bool
    result_valid: bool
    core_busy: bool


def require_pyserial() -> None:
    if serial is None:
        raise SystemExit("缺少 pyserial，请运行：python -m pip install pyserial")


def read_exact(port: "serial.Serial", size: int, timeout: float = 30.0) -> bytes:
    data = bytearray()
    deadline = time.monotonic() + timeout
    while len(data) < size:
        chunk = port.read(size - len(data))
        if chunk:
            data.extend(chunk)
        elif time.monotonic() >= deadline:
            raise TimeoutError(f"串口超时：期望 {size} 字节，只收到 {len(data)} 字节")
    return bytes(data)


def raise_if_error_frame(frame: bytes) -> None:
    if len(frame) >= 2 and frame[:1] == b"E":
        code = frame[1]
        raise RuntimeError(
            f"FPGA 返回错误 0x{code:02X}：{ERROR_MESSAGES.get(code, '未知错误')}"
        )


def read_ack(port: "serial.Serial") -> None:
    reply = read_exact(port, 3, timeout=15.0)
    raise_if_error_frame(reply)
    if reply != b"K\r\n":
        raise RuntimeError(f"FPGA 确认帧错误：{reply!r}")


def open_port(name: str) -> "serial.Serial":
    require_pyserial()
    port = serial.Serial(
        port=name,
        baudrate=BAUD_RATE,
        bytesize=serial.EIGHTBITS,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        timeout=0.2,
        write_timeout=30.0,
    )
    time.sleep(0.08)
    port.reset_input_buffer()
    return port


def show_ports() -> None:
    require_pyserial()
    ports = list(list_ports.comports())
    if not ports:
        print("未发现串口设备。")
        return
    for item in ports:
        print(f"{item.device:8s}  {item.description}")


def command_info(port: "serial.Serial") -> str:
    port.write(b"I")
    reply = port.read_until(b"\n", 128)
    if not reply.endswith(b"\n"):
        raise TimeoutError(f"信息回复不完整：{reply!r}")
    raise_if_error_frame(reply)
    text = reply.decode("ascii", errors="replace").strip()
    print(text)
    return text


def command_status(port: "serial.Serial") -> FpgaStatus:
    port.write(b"S")
    reply = read_exact(port, 4)
    raise_if_error_frame(reply)
    if reply[:1] != b"S" or reply[2:] != b"\r\n":
        raise RuntimeError(f"状态帧格式错误：{reply!r}")
    flags = reply[1]
    status = FpgaStatus(
        ddr_ready=bool(flags & 0x01),
        data_loaded=bool(flags & 0x02),
        result_valid=bool(flags & 0x04),
        core_busy=bool(flags & 0x08),
    )
    print(
        "DDR3初始化={}，数据已加载={}，结果有效={}，计算核心忙={}".format(
            "是" if status.ddr_ready else "否",
            "是" if status.data_loaded else "否",
            "是" if status.result_valid else "否",
            "是" if status.core_busy else "否",
        )
    )
    return status


def wait_until_ready(port: "serial.Serial", timeout: float = 15.0) -> None:
    deadline = time.monotonic() + timeout
    while True:
        status = command_status(port)
        if status.ddr_ready:
            return
        if time.monotonic() >= deadline:
            raise TimeoutError("等待 DDR3 初始化完成超时")
        time.sleep(0.25)


def load_case(port: "serial.Serial", case: PostAttentionLayerNormCase) -> None:
    port.write(b"L")
    port.write(build_upload_payload(case))
    port.flush()
    read_ack(port)


def run_loaded_case(port: "serial.Serial") -> np.ndarray:
    port.write(b"G")
    reply = read_exact(port, 1 + RESULT_BYTES, timeout=30.0)
    raise_if_error_frame(reply)
    if reply[:1] != b"R":
        raise RuntimeError(f"RMSNorm 结果帧头错误：{reply[:16]!r}")
    return np.frombuffer(reply[1:], dtype="<i2").copy()


def run_and_compare(
    port: "serial.Serial",
    case: PostAttentionLayerNormCase,
) -> np.ndarray:
    load_case(port, case)
    fpga = run_loaded_case(port)
    if not np.array_equal(fpga, case.output_lut_q10):
        mismatch = np.flatnonzero(fpga != case.output_lut_q10)
        first = int(mismatch[0])
        raise RuntimeError(
            f"{case.label} 不一致：首错={first}，FPGA={int(fpga[first])}，"
            f"Python={int(case.output_lut_q10[first])}，总错误数={mismatch.size}"
        )
    return fpga


def command_fixed(
    port: "serial.Serial",
    image_path: Path,
    manifest_path: Path,
) -> None:
    wait_until_ready(port)
    cases = build_fixed_real_cases(image_path=image_path)
    manifest = validate_manifest(cases, manifest_path)
    started = time.monotonic()
    for index, case in enumerate(cases, start=1):
        fpga = run_and_compare(port, case)
        expected_hash = manifest["cases"][index - 1]["sha256"]["output_lut_q10"]
        print(
            f"固定 {index}/4 PASS：query={case.query_position} count={case.count}，"
            f"output_sha256={expected_hash}，前8项={fpga[:8].tolist()}"
        )
    print(
        "四组连贯真实 Attention 输出的 post_attention_layernorm 全部逐位一致：PASS，"
        f"耗时 {time.monotonic() - started:.2f} 秒"
    )


def command_stress(
    port: "serial.Serial",
    image_path: Path,
    rounds: int,
    seed: int,
) -> None:
    if rounds <= 0:
        raise ValueError("rounds 必须大于 0")
    wait_until_ready(port)
    image = P50Image(image_path)
    image.validate()
    gamma = load_gamma(image, DEFAULT_GAMMA)
    rng = np.random.default_rng(seed)
    started = time.monotonic()
    for index in range(rounds):
        input_q10 = make_random_input_q10(rng, index)
        case = case_from_input_q10(
            input_q10=input_q10,
            gamma_values=gamma,
            label=f"post_attention_layernorm FPGA stress {index + 1}/{rounds}",
        )
        run_and_compare(port, case)
        if index == 0 or index + 1 == rounds or (index + 1) % 25 == 0:
            print(f"真实 FPGA 随机/边界已通过 {index + 1}/{rounds}")
    print(
        f"post_attention_layernorm 真实 FPGA 压力 PASS：{rounds}/{rounds}，"
        f"seed={seed}，耗时 {time.monotonic() - started:.2f} 秒"
    )


def command_selftest(
    image_path: Path,
    manifest_path: Path,
    rounds: int,
    seed: int,
) -> None:
    cases = build_fixed_real_cases(image_path=image_path)
    validate_manifest(cases, manifest_path)
    for case in cases:
        verify_payload_roundtrip(case)
    print("四组真实 Attention 输出、post-attention gamma、固定清单和载荷往返：PASS")
    for case in cases:
        delta = int(
            np.max(
                np.abs(
                    case.output_lut_q10.astype(np.int32)
                    - case.output_exact_q10.astype(np.int32)
                )
            )
        )
        print(
            f"query={case.query_position} count={case.count} "
            f"sum_sq={case.sum_squares} variance_q20={case.variance_q20} "
            f"lut_rsqrt_q20={case.lut_rsqrt_q20} LUT偏差={delta} LSB"
        )
    max_delta = software_stress(
        image_path=image_path,
        rounds=rounds,
        seed=seed,
    )
    print(
        f"软件随机/边界压力 PASS：{rounds}/{rounds}，seed={seed}，"
        f"LUT相对精确路径最大差值={max_delta} Q10 LSB"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PGL50H layer0 post_attention_layernorm G1 上位机"
    )
    parser.add_argument("--port", default="COM20")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("ports", help="列出串口")
    sub.add_parser("info", help="读取固件信息")
    sub.add_parser("status", help="读取状态")

    fixed = sub.add_parser("fixed", help="运行四组连贯真实固定输入")
    fixed.add_argument("--image", type=Path, default=DEFAULT_IMAGE_PATH)
    fixed.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)

    stress = sub.add_parser("stress", help="运行随机/边界真实 FPGA 压力")
    stress.add_argument("--image", type=Path, default=DEFAULT_IMAGE_PATH)
    stress.add_argument("--rounds", type=int, default=100)
    stress.add_argument("--seed", type=int, default=DEFAULT_STRESS_SEED)

    selftest = sub.add_parser("selftest", help="只运行软件、载荷和清单自检")
    selftest.add_argument("--image", type=Path, default=DEFAULT_IMAGE_PATH)
    selftest.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    selftest.add_argument("--rounds", type=int, default=1000)
    selftest.add_argument("--seed", type=int, default=DEFAULT_STRESS_SEED)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.command == "ports":
            show_ports()
            return 0
        if args.command == "selftest":
            command_selftest(args.image, args.manifest, args.rounds, args.seed)
            return 0

        with open_port(args.port) as port:
            if args.command == "info":
                command_info(port)
            elif args.command == "status":
                command_status(port)
            elif args.command == "fixed":
                command_fixed(port, args.image, args.manifest)
            elif args.command == "stress":
                command_stress(port, args.image, args.rounds, args.seed)
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
