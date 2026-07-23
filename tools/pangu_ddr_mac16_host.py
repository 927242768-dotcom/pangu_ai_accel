#!/usr/bin/env python3
"""盘古 PGL50H DDR3 + MAC16（INT8/INT4 权重）集成验证工具。"""

from __future__ import annotations

import argparse
import random
import struct
import sys
import time
from dataclasses import dataclass
from typing import Sequence

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

try:
    import serial  # type: ignore
    from serial.tools import list_ports  # type: ignore
except ImportError:  # pragma: no cover
    serial = None
    list_ports = None

BAUD_RATE = 115200
VECTOR_LENGTH = 16

ERROR_MESSAGES = {
    0x01: "未知命令",
    0x02: "DDR3 尚未初始化完成",
    0x03: "尚未向 DDR3 加载输入和权重",
    0xFF: "FPGA 状态机异常",
}


@dataclass(frozen=True)
class FpgaStatus:
    ddr_ready: bool
    vectors_loaded: bool
    result_valid: bool
    int4_weight_mode: bool


def require_pyserial() -> None:
    if serial is None:
        raise SystemExit("缺少 pyserial，请运行：python -m pip install pyserial")


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


def parse_int4_vector(text: str) -> list[int]:
    values = [int(item.strip(), 0) for item in text.split(",") if item.strip()]
    if len(values) != VECTOR_LENGTH:
        raise argparse.ArgumentTypeError(
            f"必须提供 {VECTOR_LENGTH} 个数，当前为 {len(values)} 个"
        )
    for value in values:
        if not -8 <= value <= 7:
            raise argparse.ArgumentTypeError(f"{value} 超出有符号 INT4 范围 -8..7")
    return values


def pack_int4_vector(values: Sequence[int]) -> bytes:
    if len(values) != VECTOR_LENGTH:
        raise ValueError(f"INT4 权重必须为 {VECTOR_LENGTH} 个")
    packed = bytearray()
    for index in range(0, VECTOR_LENGTH, 2):
        low = values[index] & 0x0F
        high = values[index + 1] & 0x0F
        packed.append(low | (high << 4))
    return bytes(packed)


def dot_reference(a: Sequence[int], b: Sequence[int]) -> int:
    return sum(x * y for x, y in zip(a, b))


def read_exact(port: "serial.Serial", size: int, timeout: float = 3.0) -> bytes:
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
        message = ERROR_MESSAGES.get(code, "未知 FPGA 错误")
        raise RuntimeError(f"FPGA 返回错误 0x{code:02X}：{message}")


def open_port(name: str) -> "serial.Serial":
    require_pyserial()
    port = serial.Serial(
        port=name,
        baudrate=BAUD_RATE,
        bytesize=serial.EIGHTBITS,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        timeout=0.2,
        write_timeout=1.0,
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
    if reply[0:1] != b"S" or reply[2:] != b"\r\n":
        raise RuntimeError(f"状态帧格式错误：{reply!r}")
    flags = reply[1]
    status = FpgaStatus(
        ddr_ready=bool(flags & 0x01),
        vectors_loaded=bool(flags & 0x02),
        result_valid=bool(flags & 0x04),
        int4_weight_mode=bool(flags & 0x08),
    )
    print(
        "DDR3初始化={}，向量已加载={}，结果有效={}，权重模式={}".format(
            "是" if status.ddr_ready else "否",
            "是" if status.vectors_loaded else "否",
            "是" if status.result_valid else "否",
            "INT4" if status.int4_weight_mode else "INT8",
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


def read_load_ack(port: "serial.Serial") -> None:
    reply = port.read_until(b"\n", 16)
    raise_if_error_frame(reply)
    if reply != b"K\r\n":
        raise RuntimeError(f"DDR3 写入确认帧错误：{reply!r}")


def load_int8_vectors(
    port: "serial.Serial", a: Sequence[int], b: Sequence[int]
) -> None:
    payload = struct.pack("16b", *a) + struct.pack("16b", *b)
    port.write(b"L" + payload)
    read_load_ack(port)


def load_int4_vectors(
    port: "serial.Serial", a: Sequence[int], weights: Sequence[int]
) -> None:
    payload = struct.pack("16b", *a) + pack_int4_vector(weights)
    port.write(b"Q" + payload)
    read_load_ack(port)


def run_loaded_dot(port: "serial.Serial") -> int:
    port.write(b"G")
    reply = read_exact(port, 5)
    raise_if_error_frame(reply)
    if reply[0:1] != b"R":
        raise RuntimeError(f"计算结果帧头错误：{reply!r}")
    return struct.unpack("<i", reply[1:5])[0]


def command_dot(port: "serial.Serial", a: Sequence[int], b: Sequence[int]) -> int:
    wait_until_ready(port)
    load_int8_vectors(port, a, b)
    fpga_result = run_loaded_dot(port)
    reference = dot_reference(a, b)
    print(f"FPGA 结果：{fpga_result}")
    print(f"Python结果：{reference}")
    if fpga_result != reference:
        raise RuntimeError("DDR3+INT8 MAC16 闭环结果与 Python 不一致")
    print("INT8写入→256位burst读回→MAC16→结果回写：PASS")
    return fpga_result


def command_stress(port: "serial.Serial", rounds: int, seed: int) -> None:
    wait_until_ready(port)
    rng = random.Random(seed)
    started = time.monotonic()
    for index in range(1, rounds + 1):
        a = [rng.randint(-128, 127) for _ in range(VECTOR_LENGTH)]
        b = [rng.randint(-128, 127) for _ in range(VECTOR_LENGTH)]
        load_int8_vectors(port, a, b)
        fpga_result = run_loaded_dot(port)
        reference = dot_reference(a, b)
        if fpga_result != reference:
            raise RuntimeError(
                f"第 {index} 轮失败：FPGA={fpga_result}，Python={reference}\n"
                f"a={a}\nb={b}"
            )
        if index == 1 or index == rounds or index % 100 == 0:
            print(f"INT8 已通过 {index}/{rounds} 轮")
    elapsed = time.monotonic() - started
    print(f"INT8 压力测试 PASS：{rounds} 轮，耗时 {elapsed:.2f} 秒")


def command_dot_int4(
    port: "serial.Serial", a: Sequence[int], weights: Sequence[int]
) -> int:
    wait_until_ready(port)
    load_int4_vectors(port, a, weights)
    fpga_result = run_loaded_dot(port)
    reference = dot_reference(a, weights)
    print(f"FPGA 结果：{fpga_result}")
    print(f"Python结果：{reference}")
    if fpga_result != reference:
        raise RuntimeError("INT4 解包后的 DDR3+MAC16 结果与 Python 不一致")
    print("packed INT4写入→burst读回→符号扩展→INT8 MAC16：PASS")
    return fpga_result


def command_stress_int4(port: "serial.Serial", rounds: int, seed: int) -> None:
    wait_until_ready(port)
    rng = random.Random(seed)
    started = time.monotonic()
    for index in range(1, rounds + 1):
        a = [rng.randint(-128, 127) for _ in range(VECTOR_LENGTH)]
        weights = [rng.randint(-8, 7) for _ in range(VECTOR_LENGTH)]
        load_int4_vectors(port, a, weights)
        fpga_result = run_loaded_dot(port)
        reference = dot_reference(a, weights)
        if fpga_result != reference:
            raise RuntimeError(
                f"INT4 第 {index} 轮失败：FPGA={fpga_result}，Python={reference}\n"
                f"a={a}\nweights={weights}"
            )
        if index == 1 or index == rounds or index % 100 == 0:
            print(f"INT4 已通过 {index}/{rounds} 轮")
    elapsed = time.monotonic() - started
    print(f"INT4 压力测试 PASS：{rounds} 轮，耗时 {elapsed:.2f} 秒")


def add_vector_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--a",
        type=parse_int8_vector,
        default=list(range(1, 17)),
        help="16 个 INT8，逗号分隔；默认 1..16",
    )
    parser.add_argument(
        "--b",
        type=parse_int8_vector,
        default=list(range(-8, 8)),
        help="16 个 INT8，逗号分隔；默认 -8..7",
    )


def add_int4_vector_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--a",
        type=parse_int8_vector,
        default=list(range(1, 17)),
        help="16 个 INT8 激活，逗号分隔；默认 1..16",
    )
    parser.add_argument(
        "--w",
        type=parse_int4_vector,
        default=list(range(-8, 8)),
        help="16 个有符号 INT4 权重，逗号分隔；默认 -8..7",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MES50HP/PGL50H DDR3 + INT8/INT4 MAC16 集成验证工具"
    )
    parser.add_argument("--port", help="串口号，例如 COM20")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("ports", help="列出串口")
    subparsers.add_parser("info", help="读取固件信息")
    subparsers.add_parser("status", help="读取 DDR3/计算状态")
    subparsers.add_parser("wait-ready", help="等待 DDR3 初始化完成")

    dot_parser = subparsers.add_parser("dot", help="执行一次 INT8 DDR3+MAC16 闭环")
    add_vector_args(dot_parser)

    stress_parser = subparsers.add_parser("stress", help="执行 INT8 随机闭环压力测试")
    stress_parser.add_argument("--rounds", type=int, default=1000)
    stress_parser.add_argument("--seed", type=int, default=20260723)

    dot_int4_parser = subparsers.add_parser(
        "dot-int4", help="执行 packed INT4 权重 × INT8 激活点积"
    )
    add_int4_vector_args(dot_int4_parser)

    stress_int4_parser = subparsers.add_parser(
        "stress-int4", help="执行 INT4×INT8 随机压力测试"
    )
    stress_int4_parser.add_argument("--rounds", type=int, default=1000)
    stress_int4_parser.add_argument("--seed", type=int, default=20260724)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "ports":
        show_ports()
        return 0
    if not args.port:
        parser.error("该命令必须指定 --port，例如 --port COM20")

    with open_port(args.port) as port:
        if args.command == "info":
            command_info(port)
        elif args.command == "status":
            command_status(port)
        elif args.command == "wait-ready":
            wait_until_ready(port)
            print("DDR3 初始化完成")
        elif args.command == "dot":
            command_dot(port, args.a, args.b)
        elif args.command == "stress":
            if args.rounds <= 0:
                parser.error("--rounds 必须大于 0")
            command_stress(port, args.rounds, args.seed)
        elif args.command == "dot-int4":
            command_dot_int4(port, args.a, args.w)
        elif args.command == "stress-int4":
            if args.rounds <= 0:
                parser.error("--rounds 必须大于 0")
            command_stress_int4(port, args.rounds, args.seed)
        else:  # pragma: no cover
            parser.error(f"未知命令：{args.command}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (TimeoutError, RuntimeError, serial.SerialException if serial else OSError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        raise SystemExit(1)
