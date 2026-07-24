#!/usr/bin/env python3
"""盘古 PGL50H layer0 MLP ``SiLU(gate)`` 验证工具。"""

from __future__ import annotations

import argparse
import hashlib
import json
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

from model_tools.mlp_silu_reference import (  # noqa: E402
    DEFAULT_MANIFEST,
    DEFAULT_STRESS_SEED,
    M,
    RESULT_BYTES,
    UPLOAD_BYTES,
    SiLUCase,
    build_fixed_real_cases,
    build_upload_payload,
    case_from_gate_q28,
    make_random_gate_q28,
    sha256_array,
    validate_manifest,
    verify_upload_payload,
)

BAUD_RATE = 115200
EXPECTED_FIRMWARE = "PANGU50K MLP SILU V1"

ERROR_MESSAGES = {
    0x01: "未知命令",
    0x02: "DDR3 尚未初始化完成",
    0x04: "尚未加载 SiLU 数据",
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


def read_exact(port: "serial.Serial", size: int, timeout: float = 120.0) -> bytes:
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
    if len(frame) >= 2 and frame[0:1] == b"E":
        code = frame[1]
        raise RuntimeError(
            f"FPGA 返回错误 0x{code:02X}：{ERROR_MESSAGES.get(code, '未知错误')}"
        )


def read_ack(port: "serial.Serial") -> None:
    reply = read_exact(port, 3, timeout=30.0)
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
        write_timeout=90.0,
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
    reply = read_exact(port, 4, timeout=15.0)
    raise_if_error_frame(reply)
    if reply[0:1] != b"S" or reply[2:] != b"\r\n":
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


def wait_until_ready(port: "serial.Serial", timeout: float = 20.0) -> None:
    deadline = time.monotonic() + timeout
    while True:
        status = command_status(port)
        if status.ddr_ready:
            return
        if time.monotonic() >= deadline:
            raise TimeoutError("等待 DDR3 初始化完成超时")
        time.sleep(0.25)


def load_case(port: "serial.Serial", case: SiLUCase) -> None:
    payload = build_upload_payload(case)
    if len(payload) != UPLOAD_BYTES:
        raise RuntimeError(f"上传载荷长度错误：{len(payload)} != {UPLOAD_BYTES}")
    port.write(b"L")
    written = port.write(payload)
    if written != len(payload):
        raise RuntimeError(f"串口只写入 {written}/{len(payload)} 字节")
    port.flush()
    read_ack(port)


def run_loaded(port: "serial.Serial") -> np.ndarray:
    port.write(b"G")
    reply = read_exact(port, 1 + RESULT_BYTES, timeout=120.0)
    raise_if_error_frame(reply)
    if reply[0:1] != b"R":
        raise RuntimeError(f"SiLU 结果帧头错误：{reply[:16]!r}")
    result = np.frombuffer(reply[1:], dtype="<i2").copy()
    if result.shape != (M,):
        raise RuntimeError(f"SiLU 结果形状错误：{result.shape}")
    return result


def run_and_compare(port: "serial.Serial", case: SiLUCase) -> np.ndarray:
    load_case(port, case)
    fpga = run_loaded(port)
    expected = case.output_pwl_q10
    if not np.array_equal(fpga, expected):
        mismatch = np.flatnonzero(fpga != expected)
        first = int(mismatch[0])
        raise RuntimeError(
            f"{case.label} 不一致：首个错误元素={first}，"
            f"FPGA={int(fpga[first])}，Python={int(expected[first])}，"
            f"总错误数={mismatch.size}"
        )
    return fpga


def command_fixed(port: "serial.Serial", manifest_path: Path) -> None:
    wait_until_ready(port)
    firmware = command_info(port)
    if firmware != EXPECTED_FIRMWARE:
        raise RuntimeError(f"固件标识不匹配：{firmware!r}")
    cases = build_fixed_real_cases()
    manifest = validate_manifest(cases, manifest_path)
    started = time.monotonic()
    for index, case in enumerate(cases):
        fpga = run_and_compare(port, case)
        expected_hash = manifest["cases"][index]["sha256"]["silu_pwl_q10"]
        actual_hash = sha256_array(fpga, "<i2")
        if actual_hash != expected_hash:
            raise RuntimeError(f"{case.label} 输出 SHA256 不一致")
        print(
            f"{case.label}：4864/4864 逐位一致，SHA256={actual_hash}，"
            f"前16项={fpga[:16].tolist()}"
        )
    print(
        f"四组连贯真实 SiLU(gate) 固定输入全部通过，耗时 {time.monotonic() - started:.2f} 秒"
    )


def command_stress(port: "serial.Serial", rounds: int, seed: int) -> None:
    if rounds <= 0:
        raise ValueError("rounds 必须大于 0")
    wait_until_ready(port)
    firmware = command_info(port)
    if firmware != EXPECTED_FIRMWARE:
        raise RuntimeError(f"固件标识不匹配：{firmware!r}")
    rng = np.random.default_rng(seed)
    started = time.monotonic()
    for index in range(rounds):
        gate_q28 = make_random_gate_q28(rng, index)
        case = case_from_gate_q28(
            gate_q28,
            label=f"MLP SiLU FPGA stress {index + 1}/{rounds} mode={index % 8}",
        )
        run_and_compare(port, case)
        if index == 0 or index + 1 == rounds or (index + 1) % 10 == 0:
            print(
                f"MLP SiLU 真实 FPGA 已通过 {index + 1}/{rounds}，"
                f"mode={index % 8}，seed={seed}"
            )
    print(
        f"MLP SiLU 真实 FPGA 随机/边界回归 PASS：{rounds}/{rounds}，"
        f"seed={seed}，耗时 {time.monotonic() - started:.2f} 秒"
    )


def command_selftest(manifest_path: Path, rounds: int, seed: int) -> None:
    if rounds <= 0:
        raise ValueError("rounds 必须大于 0")
    started = time.monotonic()
    cases = build_fixed_real_cases()
    manifest = validate_manifest(cases, manifest_path)
    print("MLP SiLU 四组真实固定清单：PASS")
    print(f"上传载荷={UPLOAD_BYTES} B，结果={RESULT_BYTES} B")
    for index, case in enumerate(cases):
        payload_hash = verify_upload_payload(case)
        expected_hash = manifest["cases"][index]["sha256"]["upload_payload"]
        if payload_hash != expected_hash:
            raise RuntimeError(f"{case.label} 上传载荷 SHA256 不一致")
        print(
            f"query={case.query_position} count={case.count} "
            f"input={sha256_array(case.gate_q28, '<i8')} "
            f"output={sha256_array(case.output_pwl_q10, '<i2')}"
        )

    rng = np.random.default_rng(seed)
    for index in range(rounds):
        case = case_from_gate_q28(
            make_random_gate_q28(rng, index),
            label=f"MLP SiLU selftest {index + 1}/{rounds}",
        )
        if index < 8:
            verify_upload_payload(case)
        if case.pwl_max_abs_error_lsb > 4:
            raise RuntimeError("PWL64 软件误差超过 4 LSB")
        if index == 0 or index + 1 == rounds or (index + 1) % 100 == 0:
            print(f"MLP SiLU 软件随机/边界已通过 {index + 1}/{rounds}")
    print(
        f"MLP SiLU 软件参考与载荷压力 PASS：{rounds}/{rounds}，"
        f"seed={seed}，耗时 {time.monotonic() - started:.2f} 秒"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PGL50H layer0 MLP SiLU(gate) 上位机")
    parser.add_argument("--port", default="COM20")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("ports", help="列出串口")
    sub.add_parser("info", help="读取固件信息")
    sub.add_parser("status", help="读取状态")

    fixed = sub.add_parser("fixed", help="运行四组连贯真实固定输入")
    fixed.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)

    stress = sub.add_parser("stress", help="运行随机/边界真实 FPGA 回归")
    stress.add_argument("--rounds", type=int, default=100)
    stress.add_argument("--seed", type=int, default=DEFAULT_STRESS_SEED)

    selftest = sub.add_parser("selftest", help="只运行软件参考和载荷自检")
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
            command_selftest(args.manifest, args.rounds, args.seed)
            return 0

        with open_port(args.port) as port:
            if args.command == "info":
                command_info(port)
            elif args.command == "status":
                command_status(port)
            elif args.command == "fixed":
                command_fixed(port, args.manifest)
            elif args.command == "stress":
                command_stress(port, args.rounds, args.seed)
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
