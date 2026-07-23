#!/usr/bin/env python3
"""盘古 PGL50H 参数化 packed INT4 GEMV 验证与性能分析工具。"""

from __future__ import annotations

import argparse
import random
import struct
import sys
import time
from dataclasses import dataclass
from typing import Iterable, Sequence

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
MAX_M = 64
MAX_K = 896
CORE_CLOCK_HZ = 100_000_000
MAC_LANES = 16
REGRESSION_M = (1, 4, 16, 64)
REGRESSION_K = (16, 64, 256, 896)
TAIL_REGRESSION_SHAPES = (
    (1, 1),
    (4, 15),
    (4, 17),
    (16, 63),
    (16, 65),
    (64, 255),
    (4, 257),
    (1, 895),
)

ERROR_MESSAGES = {
    0x01: "未知命令",
    0x02: "DDR3 尚未初始化完成",
    0x03: "尚未配置有效的 M/K",
    0x04: "尚未加载 GEMV 输入和权重",
    0x05: "M/K 配置超出支持范围",
    0x06: "尚无有效的 GEMV 性能计数",
    0xFF: "FPGA 状态机异常",
}


@dataclass(frozen=True)
class FpgaStatus:
    ddr_ready: bool
    config_valid: bool
    data_loaded: bool
    result_valid: bool
    core_busy: bool
    perf_valid: bool


@dataclass(frozen=True)
class PerformanceCounters:
    activation_read_cycles: int
    weight_read_cycles: int
    mac_cycles: int
    total_cycles: int


@dataclass(frozen=True)
class PerformanceMetrics:
    activation_bytes: int
    weight_bytes: int
    ddr_read_cycles: int
    control_cycles: int
    activation_bandwidth_mb_s: float
    weight_bandwidth_mb_s: float
    combined_bandwidth_mb_s: float
    compute_gmac_s: float
    end_to_end_gmac_s: float
    compute_utilization_percent: float
    end_to_end_utilization_percent: float
    bottleneck: str


def require_pyserial() -> None:
    if serial is None:
        raise SystemExit("缺少 pyserial，请运行：python -m pip install pyserial")


def validate_shape(m: int, k: int) -> None:
    if not 1 <= m <= MAX_M:
        raise ValueError(f"M 必须在 1..{MAX_M}，当前为 {m}")
    if not 1 <= k <= MAX_K:
        raise ValueError(f"K 必须在 1..{MAX_K}，当前为 {k}")


def ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def pad_to(payload: bytes, alignment: int) -> bytes:
    padded_size = ceil_div(len(payload), alignment) * alignment
    return payload + bytes(padded_size - len(payload))


def pack_int4_row(values: Sequence[int], k: int) -> bytes:
    if len(values) != k:
        raise ValueError(f"每行 INT4 权重必须为 {k} 个")
    packed = bytearray()
    for index in range(0, k, 2):
        low_value = int(values[index])
        high_value = int(values[index + 1]) if index + 1 < k else 0
        if not -8 <= low_value <= 7 or not -8 <= high_value <= 7:
            raise ValueError("INT4 权重必须位于 -8..7")
        packed.append((low_value & 0x0F) | ((high_value & 0x0F) << 4))
    return bytes(packed)


def unpack_int4_row(payload: bytes, k: int) -> list[int]:
    packed_size = ceil_div(k, 2)
    if len(payload) != packed_size:
        raise ValueError(f"packed INT4 行必须为 {packed_size} 字节")
    values: list[int] = []
    for byte in payload:
        for nibble in (byte & 0x0F, (byte >> 4) & 0x0F):
            values.append(nibble - 16 if nibble & 0x08 else nibble)
    return values[:k]


def build_upload_payload(
    x: Sequence[int], weights: Sequence[Sequence[int]], m: int, k: int
) -> bytes:
    validate_shape(m, k)
    if len(x) != k:
        raise ValueError(f"激活向量必须为 {k} 个 INT8")
    if len(weights) != m:
        raise ValueError(f"权重矩阵必须为 {m} 行")
    for value in x:
        if not -128 <= int(value) <= 127:
            raise ValueError("INT8 激活必须位于 -128..127")

    activation = pad_to(struct.pack(f"{k}b", *x), 32)
    packed_rows = b"".join(pad_to(pack_int4_row(row, k), 32) for row in weights)
    payload = activation + packed_rows

    expected = ceil_div(k, 32) * 32 + m * ceil_div(k, 64) * 32
    if len(payload) != expected:
        raise AssertionError(f"上传载荷长度错误：{len(payload)} != {expected}")
    return payload


def gemv_reference(
    x: Sequence[int], weights: Sequence[Sequence[int]], m: int, k: int
) -> list[int]:
    validate_shape(m, k)
    if len(x) != k or len(weights) != m:
        raise ValueError(f"输入形状必须为 x=[{k}]、W=[{m},{k}]")
    return [sum(int(a) * int(w) for a, w in zip(x, row)) for row in weights]


def gemv_blocked_reference(
    x: Sequence[int], weights: Sequence[Sequence[int]], m: int, k: int
) -> list[int]:
    validate_shape(m, k)
    outputs: list[int] = []
    for row in weights:
        accumulator = 0
        for block_start in range(0, k, 16):
            block_end = min(block_start + 16, k)
            accumulator += sum(
                int(x[index]) * int(row[index])
                for index in range(block_start, block_end)
            )
        outputs.append(accumulator)
    if len(outputs) != m:
        raise ValueError(f"权重矩阵必须为 {m} 行")
    return outputs


def iter_regression_shapes() -> Iterable[tuple[int, int]]:
    for m in REGRESSION_M:
        for k in REGRESSION_K:
            yield m, k
    yield from TAIL_REGRESSION_SHAPES


def read_exact(port: "serial.Serial", size: int, timeout: float = 20.0) -> bytes:
    data = bytearray()
    deadline = time.monotonic() + timeout
    while len(data) < size:
        chunk = port.read(size - len(data))
        if chunk:
            data.extend(chunk)
        elif time.monotonic() >= deadline:
            raise TimeoutError(f"串口超时：期望 {size} 字节，只收到 {len(data)} 字节")
    return bytes(data)


def read_ack(port: "serial.Serial") -> None:
    reply = port.read_until(b"\n", 16)
    raise_if_error_frame(reply)
    if reply != b"K\r\n":
        raise RuntimeError(f"FPGA 确认帧错误：{reply!r}")


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
        write_timeout=15.0,
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
        config_valid=bool(flags & 0x02),
        data_loaded=bool(flags & 0x04),
        result_valid=bool(flags & 0x08),
        core_busy=bool(flags & 0x10),
        perf_valid=bool(flags & 0x20),
    )
    print(
        "DDR3初始化={}，配置有效={}，数据已加载={}，结果有效={}，计算核心忙={}，性能计数有效={}".format(
            "是" if status.ddr_ready else "否",
            "是" if status.config_valid else "否",
            "是" if status.data_loaded else "否",
            "是" if status.result_valid else "否",
            "是" if status.core_busy else "否",
            "是" if status.perf_valid else "否",
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


def configure_gemv(port: "serial.Serial", m: int, k: int) -> None:
    validate_shape(m, k)
    port.write(b"C" + struct.pack("<HH", m, k))
    read_ack(port)


def load_gemv_data(
    port: "serial.Serial",
    x: Sequence[int],
    weights: Sequence[Sequence[int]],
    m: int,
    k: int,
) -> None:
    payload = build_upload_payload(x, weights, m, k)
    port.write(b"L" + payload)
    read_ack(port)


def run_loaded_gemv(port: "serial.Serial", m: int) -> list[int]:
    port.write(b"G")
    reply = read_exact(port, 1 + 4 * m, timeout=30.0)
    raise_if_error_frame(reply)
    if reply[0:1] != b"R":
        raise RuntimeError(f"GEMV 结果帧头错误：{reply[:16]!r}")
    return list(struct.unpack(f"<{m}i", reply[1:]))


def read_performance_counters(port: "serial.Serial") -> PerformanceCounters:
    port.write(b"P")
    reply = read_exact(port, 17)
    raise_if_error_frame(reply)
    if reply[0:1] != b"P":
        raise RuntimeError(f"性能计数帧头错误：{reply[:16]!r}")
    values = struct.unpack("<4I", reply[1:])
    counters = PerformanceCounters(*values)
    if counters.total_cycles == 0:
        raise RuntimeError("FPGA 返回的 GEMV 总周期为 0")
    return counters


def calculate_performance_metrics(
    counters: PerformanceCounters, m: int, k: int
) -> PerformanceMetrics:
    validate_shape(m, k)
    for name, value in (
        ("激活读取周期", counters.activation_read_cycles),
        ("权重读取周期", counters.weight_read_cycles),
        ("MAC 计算周期", counters.mac_cycles),
        ("GEMV 总周期", counters.total_cycles),
    ):
        if value <= 0:
            raise ValueError(f"{name}必须大于 0，当前为 {value}")

    activation_bytes = ceil_div(k, 32) * 32
    weight_bytes = m * ceil_div(k, 64) * 32
    ddr_read_cycles = counters.activation_read_cycles + counters.weight_read_cycles
    measured_cycles = ddr_read_cycles + counters.mac_cycles
    if counters.total_cycles < measured_cycles:
        raise ValueError(
            "总周期小于 DDR3 读取周期与 MAC 周期之和，计数口径不一致"
        )
    control_cycles = counters.total_cycles - measured_cycles

    def bandwidth_mb_s(byte_count: int, cycles: int) -> float:
        return byte_count * CORE_CLOCK_HZ / cycles / 1_000_000.0

    mac_count = m * k
    compute_gmac_s = mac_count * CORE_CLOCK_HZ / counters.mac_cycles / 1_000_000_000.0
    end_to_end_gmac_s = (
        mac_count * CORE_CLOCK_HZ / counters.total_cycles / 1_000_000_000.0
    )
    peak_gmac_s = MAC_LANES * CORE_CLOCK_HZ / 1_000_000_000.0

    cycle_groups = {
        "DDR3 读取": ddr_read_cycles,
        "MAC 数量/计算": counters.mac_cycles,
        "控制与结果写回开销": control_cycles,
    }
    bottleneck = max(cycle_groups, key=cycle_groups.get)

    return PerformanceMetrics(
        activation_bytes=activation_bytes,
        weight_bytes=weight_bytes,
        ddr_read_cycles=ddr_read_cycles,
        control_cycles=control_cycles,
        activation_bandwidth_mb_s=bandwidth_mb_s(
            activation_bytes, counters.activation_read_cycles
        ),
        weight_bandwidth_mb_s=bandwidth_mb_s(weight_bytes, counters.weight_read_cycles),
        combined_bandwidth_mb_s=bandwidth_mb_s(
            activation_bytes + weight_bytes, ddr_read_cycles
        ),
        compute_gmac_s=compute_gmac_s,
        end_to_end_gmac_s=end_to_end_gmac_s,
        compute_utilization_percent=compute_gmac_s / peak_gmac_s * 100.0,
        end_to_end_utilization_percent=end_to_end_gmac_s / peak_gmac_s * 100.0,
        bottleneck=bottleneck,
    )


def print_performance_report(
    counters: PerformanceCounters, metrics: PerformanceMetrics, m: int, k: int
) -> None:
    total_us = counters.total_cycles / CORE_CLOCK_HZ * 1_000_000.0
    print(f"性能计数：M={m}、K={k}，core_clk={CORE_CLOCK_HZ / 1e6:.0f} MHz")
    print(
        "  周期：激活读取={}，权重读取={}，MAC={}，总周期={}（{:.3f} us）".format(
            counters.activation_read_cycles,
            counters.weight_read_cycles,
            counters.mac_cycles,
            counters.total_cycles,
            total_us,
        )
    )
    print(
        "  DDR3：激活 {:.2f} MB/s，权重 {:.2f} MB/s，合并 {:.2f} MB/s".format(
            metrics.activation_bandwidth_mb_s,
            metrics.weight_bandwidth_mb_s,
            metrics.combined_bandwidth_mb_s,
        )
    )
    print(
        "  计算：核心阶段 {:.4f} GMAC/s，端到端 {:.4f} GMAC/s".format(
            metrics.compute_gmac_s,
            metrics.end_to_end_gmac_s,
        )
    )
    print(
        "  MAC16 利用率：核心阶段 {:.2f}% ，端到端 {:.2f}%".format(
            metrics.compute_utilization_percent,
            metrics.end_to_end_utilization_percent,
        )
    )
    print(
        "  周期构成：DDR3读取={}，MAC={}，控制/结果写回={}；当前主瓶颈={}".format(
            metrics.ddr_read_cycles,
            counters.mac_cycles,
            metrics.control_cycles,
            metrics.bottleneck,
        )
    )


def run_case(
    port: "serial.Serial",
    x: Sequence[int],
    weights: Sequence[Sequence[int]],
    m: int,
    k: int,
) -> list[int]:
    configure_gemv(port, m, k)
    load_gemv_data(port, x, weights, m, k)
    fpga_result = run_loaded_gemv(port, m)
    reference = gemv_reference(x, weights, m, k)
    if fpga_result != reference:
        raise RuntimeError(
            f"M={m}、K={k} GEMV 不一致：FPGA={fpga_result}，Python={reference}"
        )
    return fpga_result


def deterministic_case(m: int, k: int) -> tuple[list[int], list[list[int]]]:
    x = [((index * 13 + 17) % 256) - 128 for index in range(k)]
    weights = [
        [((row * 5 + index * 3 + 1) % 16) - 8 for index in range(k)]
        for row in range(m)
    ]
    return x, weights


def random_case(rng: random.Random, m: int, k: int) -> tuple[list[int], list[list[int]]]:
    x = [rng.randint(-128, 127) for _ in range(k)]
    weights = [[rng.randint(-8, 7) for _ in range(k)] for _ in range(m)]
    return x, weights


def command_gemv(port: "serial.Serial", m: int, k: int) -> None:
    wait_until_ready(port)
    x, weights = deterministic_case(m, k)
    result = run_case(port, x, weights, m, k)
    print(f"M={m}、K={k} FPGA/Python 逐元素一致：PASS")
    print(f"输出前 16 项：{result[:16]}")


def command_performance(port: "serial.Serial", m: int, k: int) -> None:
    wait_until_ready(port)
    x, weights = deterministic_case(m, k)
    result = run_case(port, x, weights, m, k)
    counters = read_performance_counters(port)
    metrics = calculate_performance_metrics(counters, m, k)
    print(f"M={m}、K={k} FPGA/Python 逐元素一致：PASS")
    print(f"输出前 16 项：{result[:16]}")
    print_performance_report(counters, metrics, m, k)


def command_boundary(port: "serial.Serial") -> None:
    """验证当前最大 K 下的 INT32 正负累加边界。"""
    wait_until_ready(port)
    m = 4
    k = MAX_K
    x = [-128] * k
    weights = [
        [-8] * k,
        [7] * k,
        [-8 if index % 2 == 0 else 7 for index in range(k)],
        [7 if index % 2 == 0 else -8 for index in range(k)],
    ]
    result = run_case(port, x, weights, m, k)
    limit = max(abs(value) for value in result)
    theoretical_limit = MAX_K * 128 * 8
    if limit > theoretical_limit:
        raise RuntimeError("累加结果超过 INT8×INT4 理论边界")
    print(f"INT32 累加边界 FPGA/Python 一致：PASS，结果={result}")
    print(
        f"当前 K<={MAX_K} 的绝对理论上界为 {theoretical_limit}，"
        f"远小于 INT32_MAX={2**31 - 1}"
    )


def command_stress(port: "serial.Serial", m: int, k: int, rounds: int, seed: int) -> None:
    wait_until_ready(port)
    rng = random.Random(seed)
    started = time.monotonic()
    configure_gemv(port, m, k)
    for index in range(1, rounds + 1):
        x, weights = random_case(rng, m, k)
        load_gemv_data(port, x, weights, m, k)
        fpga_result = run_loaded_gemv(port, m)
        reference = gemv_reference(x, weights, m, k)
        if fpga_result != reference:
            raise RuntimeError(
                f"第 {index} 轮失败，M={m}、K={k}：\n"
                f"FPGA={fpga_result}\nPython={reference}\nx={x}\nW={weights}"
            )
        if index == 1 or index == rounds or index % 100 == 0:
            print(f"M={m}、K={k} 已通过 {index}/{rounds} 轮")
    elapsed = time.monotonic() - started
    print(
        f"参数化 GEMV 压力测试 PASS：M={m}、K={k}，"
        f"{rounds} 轮，seed={seed}，耗时 {elapsed:.2f} 秒"
    )


def command_regression(port: "serial.Serial", rounds_per_shape: int, seed: int) -> None:
    wait_until_ready(port)
    rng = random.Random(seed)
    total = 0
    started = time.monotonic()
    for m, k in iter_regression_shapes():
        configure_gemv(port, m, k)
        x, weights = deterministic_case(m, k)
        load_gemv_data(port, x, weights, m, k)
        if run_loaded_gemv(port, m) != gemv_reference(x, weights, m, k):
            raise RuntimeError(f"固定回归失败：M={m}、K={k}")
        total += 1
        for _ in range(rounds_per_shape):
            x, weights = random_case(rng, m, k)
            load_gemv_data(port, x, weights, m, k)
            if run_loaded_gemv(port, m) != gemv_reference(x, weights, m, k):
                raise RuntimeError(f"随机回归失败：M={m}、K={k}")
            total += 1
        print(f"形状 M={m:2d}、K={k:3d}：PASS")
    elapsed = time.monotonic() - started
    shape_count = len(REGRESSION_M) * len(REGRESSION_K) + len(TAIL_REGRESSION_SHAPES)
    print(
        f"多尺寸真实 FPGA 回归 PASS：{shape_count} 种形状，共 {total} 例，"
        f"seed={seed}，耗时 {elapsed:.2f} 秒"
    )


def verify_python_case(x: list[int], weights: list[list[int]], m: int, k: int) -> None:
    payload = build_upload_payload(x, weights, m, k)
    expected_payload_size = ceil_div(k, 32) * 32 + m * ceil_div(k, 64) * 32
    if len(payload) != expected_payload_size:
        raise RuntimeError("上传载荷尺寸计算不一致")

    activation_bytes = ceil_div(k, 32) * 32
    weight_row_bytes = ceil_div(k, 64) * 32
    packed_area = payload[activation_bytes:]
    unpacked = []
    for row in range(m):
        row_start = row * weight_row_bytes
        row_payload = packed_area[row_start : row_start + ceil_div(k, 2)]
        unpacked.append(unpack_int4_row(row_payload, k))
    if unpacked != weights:
        raise RuntimeError(f"M={m}、K={k} INT4 打包/解包不一致")

    direct = gemv_reference(x, weights, m, k)
    blocked = gemv_blocked_reference(x, weights, m, k)
    if direct != blocked:
        raise RuntimeError(f"M={m}、K={k} 直接参考与 MAC16 分块参考不一致")


def command_selftest(rounds: int, seed: int) -> None:
    rng = random.Random(seed)
    checked = 0

    for m, k in iter_regression_shapes():
        x, weights = deterministic_case(m, k)
        verify_python_case(x, weights, m, k)
        checked += 1

    for _ in range(rounds):
        m = rng.randint(1, MAX_M)
        k = rng.randint(1, MAX_K)
        x, weights = random_case(rng, m, k)
        verify_python_case(x, weights, m, k)
        checked += 1

    # 显式保留已验证固定 M4K64 基线形状。
    x, weights = deterministic_case(4, 64)
    verify_python_case(x, weights, 4, 64)
    checked += 1

    # D1.3 性能公式自检：确保带宽、GMAC/s、利用率和瓶颈分类可计算。
    sample_counters = PerformanceCounters(
        activation_read_cycles=12,
        weight_read_cycles=80,
        mac_cycles=64,
        total_cycles=180,
    )
    sample_metrics = calculate_performance_metrics(sample_counters, 4, 64)
    if sample_metrics.control_cycles != 24:
        raise RuntimeError("性能公式控制周期计算错误")
    if sample_metrics.bottleneck != "DDR3 读取":
        raise RuntimeError("性能公式瓶颈分类错误")

    print(
        f"Python 参数化金标准自检 PASS：{checked} 例，"
        f"含整数倍与尾块边界形状、固定 M4K64 回归，seed={seed}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MES50HP/PGL50H 运行时参数化 packed INT4 GEMV 验证工具"
    )
    parser.add_argument("--port", help="串口号，例如 COM20")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("ports", help="列出串口")
    subparsers.add_parser("info", help="读取固件信息")
    subparsers.add_parser("status", help="读取 DDR3/GEMV 状态")
    subparsers.add_parser("wait-ready", help="等待 DDR3 初始化完成")
    subparsers.add_parser("boundary", help="验证最大 K 下的 INT32 累加边界")

    gemv_parser = subparsers.add_parser("gemv", help="执行一次确定性参数化 GEMV")
    gemv_parser.add_argument("--m", type=int, default=4)
    gemv_parser.add_argument("--k", type=int, default=64)

    perf_parser = subparsers.add_parser(
        "perf", help="执行一次 GEMV 并读取周期、带宽、GMAC/s 和利用率"
    )
    perf_parser.add_argument("--m", type=int, default=4)
    perf_parser.add_argument("--k", type=int, default=64)

    stress_parser = subparsers.add_parser("stress", help="指定 M/K 的真实 FPGA 随机压力测试")
    stress_parser.add_argument("--m", type=int, default=4)
    stress_parser.add_argument("--k", type=int, default=64)
    stress_parser.add_argument("--rounds", type=int, default=1000)
    stress_parser.add_argument("--seed", type=int, default=20260726)

    regression_parser = subparsers.add_parser(
        "regression", help="覆盖标准尺寸和 K 尾块边界尺寸"
    )
    regression_parser.add_argument("--rounds-per-shape", type=int, default=1)
    regression_parser.add_argument("--seed", type=int, default=20260726)

    selftest_parser = subparsers.add_parser(
        "selftest", help="不连接 FPGA，验证参数、布局、INT4 打包和 Python 金标准"
    )
    selftest_parser.add_argument("--rounds", type=int, default=1000)
    selftest_parser.add_argument("--seed", type=int, default=20260726)
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
        elif args.command == "boundary":
            command_boundary(port)
        elif args.command == "gemv":
            validate_shape(args.m, args.k)
            command_gemv(port, args.m, args.k)
        elif args.command == "perf":
            validate_shape(args.m, args.k)
            command_performance(port, args.m, args.k)
        elif args.command == "stress":
            validate_shape(args.m, args.k)
            if args.rounds <= 0:
                parser.error("--rounds 必须大于 0")
            command_stress(port, args.m, args.k, args.rounds, args.seed)
        elif args.command == "regression":
            if args.rounds_per_shape < 0:
                parser.error("--rounds-per-shape 不能小于 0")
            command_regression(port, args.rounds_per_shape, args.seed)
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
