#!/usr/bin/env python3
"""盘古 PGL50H packed INT4 GEMV（M=4、K=64）验证工具。"""

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
M = 4
K = 64

ERROR_MESSAGES = {
    0x01: "未知命令",
    0x02: "DDR3 尚未初始化完成",
    0x03: "尚未加载 GEMV 输入和权重",
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


def parse_int_list(text: str, expected: int, minimum: int, maximum: int) -> list[int]:
    values = [int(item.strip(), 0) for item in text.split(",") if item.strip()]
    if len(values) != expected:
        raise argparse.ArgumentTypeError(f"必须提供 {expected} 个数，当前为 {len(values)} 个")
    for value in values:
        if not minimum <= value <= maximum:
            raise argparse.ArgumentTypeError(
                f"{value} 超出允许范围 {minimum}..{maximum}"
            )
    return values


def parse_x_vector(text: str) -> list[int]:
    return parse_int_list(text, K, -128, 127)


def parse_weight_matrix(text: str) -> list[list[int]]:
    row_texts = [row.strip() for row in text.split(";") if row.strip()]
    if len(row_texts) != M:
        raise argparse.ArgumentTypeError(f"权重必须包含 {M} 行，以分号分隔")
    return [parse_int_list(row, K, -8, 7) for row in row_texts]


def pack_int4_row(values: Sequence[int]) -> bytes:
    if len(values) != K:
        raise ValueError(f"每行 INT4 权重必须为 {K} 个")
    packed = bytearray()
    for index in range(0, K, 2):
        low = values[index] & 0x0F
        high = values[index + 1] & 0x0F
        packed.append(low | (high << 4))
    return bytes(packed)


def unpack_int4_row(payload: bytes) -> list[int]:
    if len(payload) != K // 2:
        raise ValueError(f"packed INT4 行必须为 {K // 2} 字节")
    values: list[int] = []
    for byte in payload:
        for nibble in (byte & 0x0F, (byte >> 4) & 0x0F):
            values.append(nibble - 16 if nibble & 0x08 else nibble)
    return values


def pack_weight_matrix(weights: Sequence[Sequence[int]]) -> bytes:
    if len(weights) != M:
        raise ValueError(f"权重矩阵必须为 {M} 行")
    return b"".join(pack_int4_row(row) for row in weights)


def gemv_reference(x: Sequence[int], weights: Sequence[Sequence[int]]) -> list[int]:
    if len(x) != K or len(weights) != M:
        raise ValueError(f"输入形状必须为 x=[{K}]、W=[{M},{K}]")
    return [sum(int(a) * int(w) for a, w in zip(x, row)) for row in weights]


def gemv_blocked_reference(
    x: Sequence[int], weights: Sequence[Sequence[int]]
) -> list[int]:
    outputs: list[int] = []
    for row in weights:
        accumulator = 0
        for block in range(K // 16):
            start = block * 16
            accumulator += sum(
                int(x[index]) * int(row[index])
                for index in range(start, start + 16)
            )
        outputs.append(accumulator)
    return outputs


def read_exact(port: "serial.Serial", size: int, timeout: float = 5.0) -> bytes:
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
        write_timeout=3.0,
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


def load_gemv_data(
    port: "serial.Serial", x: Sequence[int], weights: Sequence[Sequence[int]]
) -> None:
    if len(x) != K:
        raise ValueError(f"激活向量必须为 {K} 个 INT8")
    payload = struct.pack(f"{K}b", *x) + pack_weight_matrix(weights)
    if len(payload) != 192:
        raise AssertionError(f"GEMV 载荷长度错误：{len(payload)}")
    port.write(b"M" + payload)
    reply = port.read_until(b"\n", 16)
    raise_if_error_frame(reply)
    if reply != b"K\r\n":
        raise RuntimeError(f"DDR3 写入确认帧错误：{reply!r}")


def run_loaded_gemv(port: "serial.Serial") -> list[int]:
    port.write(b"G")
    reply = read_exact(port, 17)
    raise_if_error_frame(reply)
    if reply[0:1] != b"R":
        raise RuntimeError(f"GEMV 结果帧头错误：{reply!r}")
    return list(struct.unpack("<4i", reply[1:17]))


def command_gemv(
    port: "serial.Serial", x: Sequence[int], weights: Sequence[Sequence[int]]
) -> list[int]:
    wait_until_ready(port)
    load_gemv_data(port, x, weights)
    fpga_result = run_loaded_gemv(port)
    reference = gemv_reference(x, weights)
    print(f"FPGA 结果：{fpga_result}")
    print(f"Python结果：{reference}")
    if fpga_result != reference:
        raise RuntimeError("M=4、K=64 packed INT4 GEMV 与 Python 不一致")
    print("激活单次读取→4拍权重burst→16次MAC16→4个INT32输出：PASS")
    return fpga_result


def command_stress(port: "serial.Serial", rounds: int, seed: int) -> None:
    wait_until_ready(port)
    rng = random.Random(seed)
    started = time.monotonic()
    for index in range(1, rounds + 1):
        x = [rng.randint(-128, 127) for _ in range(K)]
        weights = [[rng.randint(-8, 7) for _ in range(K)] for _ in range(M)]
        load_gemv_data(port, x, weights)
        fpga_result = run_loaded_gemv(port)
        reference = gemv_reference(x, weights)
        if fpga_result != reference:
            raise RuntimeError(
                f"第 {index} 轮失败：FPGA={fpga_result}，Python={reference}\n"
                f"x={x}\nW={weights}"
            )
        if index == 1 or index == rounds or index % 100 == 0:
            print(f"GEMV 已通过 {index}/{rounds} 轮")
    elapsed = time.monotonic() - started
    print(f"GEMV 压力测试 PASS：{rounds} 轮，耗时 {elapsed:.2f} 秒")


def command_selftest(rounds: int, seed: int) -> None:
    rng = random.Random(seed)
    for index in range(1, rounds + 1):
        x = [rng.randint(-128, 127) for _ in range(K)]
        weights = [[rng.randint(-8, 7) for _ in range(K)] for _ in range(M)]
        packed = pack_weight_matrix(weights)
        if len(packed) != M * K // 2:
            raise RuntimeError(f"第 {index} 轮 packed 长度错误：{len(packed)}")
        unpacked = [
            unpack_int4_row(packed[row * (K // 2) : (row + 1) * (K // 2)])
            for row in range(M)
        ]
        if unpacked != weights:
            raise RuntimeError(f"第 {index} 轮 INT4 打包/解包不一致")
        direct = gemv_reference(x, weights)
        blocked = gemv_blocked_reference(x, weights)
        if direct != blocked:
            raise RuntimeError(
                f"第 {index} 轮直接参考与 MAC16 分块参考不一致：{direct} != {blocked}"
            )
    print(f"Python 金标准自检 PASS：{rounds} 轮，seed={seed}")


def default_x() -> list[int]:
    return [index - 32 for index in range(K)]


def default_weights() -> list[list[int]]:
    return [
        [(index % 16) - 8 for index in range(K)],
        [7 - (index % 16) for index in range(K)],
        [((index * 3) % 16) - 8 for index in range(K)],
        [-8 if index % 2 == 0 else 7 for index in range(K)],
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MES50HP/PGL50H packed INT4 GEMV（M=4、K=64）验证工具"
    )
    parser.add_argument("--port", help="串口号，例如 COM20")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("ports", help="列出串口")
    subparsers.add_parser("info", help="读取固件信息")
    subparsers.add_parser("status", help="读取 DDR3/GEMV 状态")
    subparsers.add_parser("wait-ready", help="等待 DDR3 初始化完成")

    gemv_parser = subparsers.add_parser("gemv", help="执行一次固定 M4K64 GEMV")
    gemv_parser.add_argument(
        "--x",
        type=parse_x_vector,
        default=default_x(),
        help="64 个 INT8 激活，逗号分隔",
    )
    gemv_parser.add_argument(
        "--w",
        type=parse_weight_matrix,
        default=default_weights(),
        help="4 行×64 个 INT4；行内逗号分隔，行间分号分隔",
    )

    stress_parser = subparsers.add_parser("stress", help="执行真实 FPGA 随机压力测试")
    stress_parser.add_argument("--rounds", type=int, default=1000)
    stress_parser.add_argument("--seed", type=int, default=20260725)

    selftest_parser = subparsers.add_parser(
        "selftest", help="不连接 FPGA，验证 Python 打包和分块金标准"
    )
    selftest_parser.add_argument("--rounds", type=int, default=1000)
    selftest_parser.add_argument("--seed", type=int, default=20260725)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "ports":
        show_ports()
        return 0
    if args.command == "selftest":
        if args.rounds <= 0:
            parser.error("--rounds 必须大于 0")
        command_selftest(args.rounds, args.seed)
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
        elif args.command == "gemv":
            command_gemv(port, args.x, args.w)
        elif args.command == "stress":
            if args.rounds <= 0:
                parser.error("--rounds 必须大于 0")
            command_stress(port, args.rounds, args.seed)
        else:  # pragma: no cover
            parser.error(f"未知命令：{args.command}")
    return 0


if __name__ == "__main__":
    serial_exception = serial.SerialException if serial else OSError
    try:
        raise SystemExit(main())
    except (TimeoutError, RuntimeError, ValueError, serial_exception) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        raise SystemExit(1)
