#!/usr/bin/env python3
"""盘古50K INT8 MAC16 V1 串口测试工具。"""

from __future__ import annotations

import argparse
import struct
import sys
import time
from typing import Sequence

# 避免部分 Windows 终端默认使用 cp1252 时无法输出中文。
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

try:
    import serial  # type: ignore
    from serial.tools import list_ports  # type: ignore
except ImportError:  # pragma: no cover - depends on host environment
    serial = None
    list_ports = None

BAUD_RATE = 115200
VECTOR_LENGTH = 16


def parse_int8_vector(text: str) -> list[int]:
    values = [int(item.strip(), 0) for item in text.split(",") if item.strip()]
    if len(values) != VECTOR_LENGTH:
        raise argparse.ArgumentTypeError(
            f"必须提供 {VECTOR_LENGTH} 个数，当前为 {len(values)} 个"
        )
    for value in values:
        if not -128 <= value <= 127:
            raise argparse.ArgumentTypeError(f"{value} 超出 INT8 范围 -128..127")
    return values


def dot_reference(a: Sequence[int], b: Sequence[int]) -> int:
    return sum(x * y for x, y in zip(a, b))


def read_exact(port: "serial.Serial", size: int) -> bytes:
    data = bytearray()
    deadline = time.monotonic() + max(2.0, port.timeout or 0.0)
    while len(data) < size:
        chunk = port.read(size - len(data))
        if chunk:
            data.extend(chunk)
            continue
        if time.monotonic() >= deadline:
            raise TimeoutError(f"串口超时：期望 {size} 字节，只收到 {len(data)} 字节")
    return bytes(data)


def require_pyserial() -> None:
    if serial is None:
        raise SystemExit(
            "缺少 pyserial。请先运行：python -m pip install pyserial"
        )


def show_ports() -> None:
    require_pyserial()
    ports = list(list_ports.comports())
    if not ports:
        print("未发现串口设备。")
        return
    for item in ports:
        print(f"{item.device:8s}  {item.description}")


def open_port(name: str) -> "serial.Serial":
    require_pyserial()
    port = serial.Serial(
        port=name,
        baudrate=BAUD_RATE,
        bytesize=serial.EIGHTBITS,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        timeout=1.0,
        write_timeout=1.0,
    )
    time.sleep(0.05)
    port.reset_input_buffer()
    return port


def command_info(port: "serial.Serial") -> None:
    port.write(b"I")
    reply = port.read_until(b"\n", 128)
    if not reply.endswith(b"\n"):
        raise TimeoutError(f"信息回复不完整：{reply!r}")
    print(reply.decode("ascii", errors="replace").strip())


def command_test(port: "serial.Serial") -> None:
    port.write(b"T")
    reply = port.read_until(b"\n", 32)
    text = reply.decode("ascii", errors="replace").strip()
    print(f"FPGA 自检结果：{text}")
    if text != "PASS":
        raise RuntimeError("FPGA 自检未通过")


def command_dot(port: "serial.Serial", a: Sequence[int], b: Sequence[int]) -> None:
    payload = struct.pack("16b", *a) + struct.pack("16b", *b)
    port.write(b"D" + payload)
    reply = read_exact(port, 5)
    if reply[0:1] != b"R":
        raise RuntimeError(f"回复帧头错误：{reply!r}")

    fpga_result = struct.unpack("<i", reply[1:5])[0]
    reference = dot_reference(a, b)
    print(f"FPGA 结果：{fpga_result}")
    print(f"PC   结果：{reference}")
    if fpga_result != reference:
        raise RuntimeError("FPGA 与 PC 计算结果不一致")
    print("点积验证：PASS")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MES50HP/盘古50K INT8 MAC16 串口测试工具"
    )
    parser.add_argument("--port", help="串口号，例如 COM5")

    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("ports", help="列出可用串口")
    subparsers.add_parser("info", help="读取 FPGA 版本信息")
    subparsers.add_parser("test", help="运行 FPGA 固定向量自检")

    dot_parser = subparsers.add_parser("dot", help="执行 16 路 INT8 点积")
    dot_parser.add_argument(
        "--a",
        type=parse_int8_vector,
        default=list(range(1, 17)),
        help="16 个 INT8，逗号分隔；默认 1..16",
    )
    dot_parser.add_argument(
        "--b",
        type=parse_int8_vector,
        default=list(range(-8, 8)),
        help="16 个 INT8，逗号分隔；默认 -8..7",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "ports":
        show_ports()
        return 0

    if not args.port:
        parser.error("info/test/dot 命令必须指定 --port，例如 --port COM5")

    with open_port(args.port) as port:
        if args.command == "info":
            command_info(port)
        elif args.command == "test":
            command_test(port)
        elif args.command == "dot":
            command_dot(port, args.a, args.b)
        else:  # pragma: no cover
            parser.error(f"未知命令：{args.command}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (TimeoutError, RuntimeError, serial.SerialException if serial else OSError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        raise SystemExit(1)
