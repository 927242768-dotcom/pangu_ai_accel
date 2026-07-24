#!/usr/bin/env python3
"""盘古 PGL50H layer0 Q/K RoPE 定点闭环验证工具。"""

from __future__ import annotations

import argparse
import hashlib
import struct
import sys
import time
from pathlib import Path
from typing import Sequence

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

from model_tools.linear_quant_reference import DEFAULT_ACTIVATION_SEED  # noqa: E402
from model_tools.rope_fixed_reference import (  # noqa: E402
    DEFAULT_IMAGE,
    DEFAULT_MANIFEST,
    DEFAULT_POSITIONS,
    HEAD_DIM,
    INPUT_BYTES,
    K_VALUES,
    MAX_POSITION_EMBEDDINGS,
    Q_HEADS,
    Q_VALUES,
    TRIG_ROW_BYTES,
    RoPECase,
    build_real_rope_cases,
    build_rope_case,
    build_upload_payload,
    load_real_qk_inputs,
    sha256_array,
    software_stress,
    validate_manifest,
)

BAUD_RATE = 115200
MAX_TABLE_ROWS = 16
RESULT_BYTES = INPUT_BYTES
DEFAULT_STRESS_SEED = 20260731

ERROR_MESSAGES = {
    0x01: "未知命令",
    0x02: "DDR3 尚未初始化完成",
    0x03: "尚未配置位置表",
    0x04: "尚未加载 Q/K 与三角函数表",
    0x05: "位置表配置非法",
    0x06: "当前位置表已经执行完毕",
    0xFF: "FPGA 状态机异常",
}


class FpgaStatus:
    def __init__(
        self,
        flags: int,
        current_position: int,
        table_index: int,
        table_count: int,
    ):
        self.ddr_ready = bool(flags & 0x01)
        self.configured = bool(flags & 0x02)
        self.data_loaded = bool(flags & 0x04)
        self.result_valid = bool(flags & 0x08)
        self.core_busy = bool(flags & 0x10)
        self.sequence_exhausted = bool(flags & 0x20)
        self.current_position = current_position
        self.table_index = table_index
        self.table_count = table_count


def require_pyserial() -> None:
    if serial is None:
        raise SystemExit("缺少 pyserial，请运行：python -m pip install pyserial")


def read_exact(port: "serial.Serial", size: int, timeout: float = 180.0) -> bytes:
    data = bytearray()
    deadline = time.monotonic() + timeout
    while len(data) < size:
        chunk = port.read(size - len(data))
        if chunk:
            data.extend(chunk)
        elif time.monotonic() >= deadline:
            raise TimeoutError(f"串口超时：期望 {size} 字节，只收到 {len(data)} 字节")
    return bytes(data)


def raise_error_code(code: int) -> None:
    raise RuntimeError(
        f"FPGA 返回错误 0x{code:02X}：{ERROR_MESSAGES.get(code, '未知错误')}"
    )


def read_ack(port: "serial.Serial") -> None:
    first = read_exact(port, 1, timeout=30.0)
    if first == b"E":
        rest = read_exact(port, 3, timeout=5.0)
        raise_error_code(rest[0])
    rest = read_exact(port, 2, timeout=5.0)
    reply = first + rest
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
        write_timeout=180.0,
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
    if reply.startswith(b"E") and len(reply) >= 2:
        raise_error_code(reply[1])
    text = reply.decode("ascii", errors="replace").strip()
    print(text)
    return text


def command_status(port: "serial.Serial", *, quiet: bool = False) -> FpgaStatus:
    port.write(b"S")
    first = read_exact(port, 1)
    if first == b"E":
        rest = read_exact(port, 3)
        raise_error_code(rest[0])
    reply = first + read_exact(port, 7)
    if reply[0:1] != b"S" or reply[-2:] != b"\r\n":
        raise RuntimeError(f"状态帧格式错误：{reply!r}")
    status = FpgaStatus(
        reply[1],
        int.from_bytes(reply[2:4], "little"),
        reply[4],
        reply[5],
    )
    if not quiet:
        print(
            "DDR3={}，已配置={}，已加载={}，结果有效={}，核心忙={}，"
            "序列结束={}，当前位置={}，表索引={}/{}".format(
                "是" if status.ddr_ready else "否",
                "是" if status.configured else "否",
                "是" if status.data_loaded else "否",
                "是" if status.result_valid else "否",
                "是" if status.core_busy else "否",
                "是" if status.sequence_exhausted else "否",
                status.current_position,
                status.table_index,
                status.table_count,
            )
        )
    return status


def wait_until_ready(port: "serial.Serial", timeout: float = 20.0) -> None:
    deadline = time.monotonic() + timeout
    while True:
        status = command_status(port, quiet=True)
        if status.ddr_ready:
            return
        if time.monotonic() >= deadline:
            raise TimeoutError("等待 DDR3 初始化完成超时")
        time.sleep(0.25)


def configure_sequence(port: "serial.Serial", start_position: int, count: int) -> None:
    if not 0 <= start_position < MAX_POSITION_EMBEDDINGS:
        raise ValueError("start_position 越界")
    if not 1 <= count <= MAX_TABLE_ROWS:
        raise ValueError(f"count 必须位于 1..{MAX_TABLE_ROWS}")
    if start_position + count > MAX_POSITION_EMBEDDINGS:
        raise ValueError("连续位置表超出 max_position_embeddings")
    port.write(b"C" + struct.pack("<HH", start_position, count))
    read_ack(port)


def reset_sequence(port: "serial.Serial") -> None:
    port.write(b"Z")
    read_ack(port)


def write_all(port: "serial.Serial", payload: bytes, chunk_size: int = 4096) -> None:
    sent = 0
    view = memoryview(payload)
    while sent < len(payload):
        end = min(sent + chunk_size, len(payload))
        count = port.write(view[sent:end])
        if not count:
            raise TimeoutError("串口写入没有前进")
        sent += count
    port.flush()


def load_sequence(
    port: "serial.Serial",
    q_input_q28: np.ndarray,
    k_input_q28: np.ndarray,
    positions: Sequence[int],
) -> str:
    if not positions:
        raise ValueError("positions 不能为空")
    expected = list(range(positions[0], positions[0] + len(positions)))
    if list(positions) != expected:
        raise ValueError("FPGA 位置表必须为连续位置")
    configure_sequence(port, positions[0], len(positions))
    payload = build_upload_payload(q_input_q28, k_input_q28, positions)
    port.write(b"L")
    write_all(port, payload)
    read_ack(port)
    return hashlib.sha256(payload).hexdigest()


def run_loaded_position(port: "serial.Serial") -> tuple[int, np.ndarray, np.ndarray]:
    port.write(b"G")
    first = read_exact(port, 1, timeout=30.0)
    if first == b"E":
        rest = read_exact(port, 3, timeout=5.0)
        raise_error_code(rest[0])
    if first != b"R":
        raise RuntimeError(f"RoPE 结果帧头错误：{first!r}")
    position = int.from_bytes(read_exact(port, 2, timeout=5.0), "little")
    payload = read_exact(port, RESULT_BYTES, timeout=180.0)
    values = np.frombuffer(payload, dtype="<i8").copy()
    q = values[:Q_VALUES].reshape(Q_HEADS, HEAD_DIM)
    k = values[Q_VALUES : Q_VALUES + K_VALUES].reshape(2, HEAD_DIM)
    return position, q, k


def compare_case(
    position: int,
    q_fpga: np.ndarray,
    k_fpga: np.ndarray,
    case: RoPECase,
) -> None:
    if position != case.position:
        raise RuntimeError(
            f"返回位置错误：FPGA={position}，Python={case.position}"
        )
    for label, fpga, expected in (
        ("Q", q_fpga, case.q_output_q28),
        ("K", k_fpga, case.k_output_q28),
    ):
        if not np.array_equal(fpga, expected):
            mismatch = np.argwhere(fpga != expected)
            head, dim = map(int, mismatch[0])
            raise RuntimeError(
                f"position={position} {label} 不一致：首错 head={head}, dim={dim}，"
                f"FPGA={int(fpga[head, dim])}，Python={int(expected[head, dim])}，"
                f"总错误数={mismatch.shape[0]}"
            )


def run_sequence_and_compare(
    port: "serial.Serial",
    q_input_q28: np.ndarray,
    k_input_q28: np.ndarray,
    positions: Sequence[int],
) -> list[tuple[int, str, str]]:
    load_sequence(port, q_input_q28, k_input_q28, positions)
    summaries: list[tuple[int, str, str]] = []
    for position in positions:
        case = build_rope_case(q_input_q28, k_input_q28, position)
        returned_position, q_fpga, k_fpga = run_loaded_position(port)
        compare_case(returned_position, q_fpga, k_fpga, case)
        summaries.append(
            (
                position,
                sha256_array(q_fpga, "<i8"),
                sha256_array(k_fpga, "<i8"),
            )
        )
    return summaries


def command_fixed(
    port: "serial.Serial",
    image_path: Path,
    manifest_path: Path,
) -> None:
    wait_until_ready(port)
    cases = build_real_rope_cases(
        DEFAULT_POSITIONS,
        image_path=image_path,
        activation_seed=DEFAULT_ACTIVATION_SEED,
    )
    validate_manifest(
        cases,
        manifest_path,
        activation_seed=DEFAULT_ACTIVATION_SEED,
    )
    for case in cases:
        started = time.monotonic()
        summaries = run_sequence_and_compare(
            port,
            case.q_input_q28,
            case.k_input_q28,
            (case.position,),
        )
        elapsed = time.monotonic() - started
        _, q_hash, k_hash = summaries[0]
        print(
            f"真实 layer0 Q/K RoPE position={case.position} 全输出逐位一致：PASS"
        )
        print(f"Q SHA256：{q_hash}")
        print(f"K SHA256：{k_hash}")
        print(f"上传、计算与回读耗时：{elapsed:.2f} 秒")


def command_sequence(
    port: "serial.Serial",
    image_path: Path,
    start_position: int,
    count: int,
    verify_reset: bool,
) -> None:
    wait_until_ready(port)
    q_input, k_input = load_real_qk_inputs(image_path)
    positions = tuple(range(start_position, start_position + count))
    started = time.monotonic()
    summaries = run_sequence_and_compare(port, q_input, k_input, positions)
    status = command_status(port, quiet=True)
    if status.current_position != start_position + count:
        raise RuntimeError(
            f"位置递增错误：{status.current_position} != {start_position + count}"
        )
    if status.table_index != count or not status.sequence_exhausted:
        raise RuntimeError(
            f"位置表结束状态错误：index={status.table_index}, count={status.table_count}, "
            f"exhausted={status.sequence_exhausted}"
        )
    if verify_reset:
        reset_sequence(port)
        reset_status = command_status(port, quiet=True)
        if reset_status.current_position != start_position or reset_status.table_index != 0:
            raise RuntimeError("Z 命令未正确复位位置与表索引")
        case0 = build_rope_case(q_input, k_input, start_position)
        returned_position, q_fpga, k_fpga = run_loaded_position(port)
        compare_case(returned_position, q_fpga, k_fpga, case0)
        print("位置表 Z 复位后首位置重放：PASS")
    elapsed = time.monotonic() - started
    print(
        f"连续位置自动递增 PASS：{count}/{count}，start={start_position}，"
        f"last={positions[-1]}，耗时 {elapsed:.2f} 秒"
    )
    print(f"首位置 Q/K SHA256：{summaries[0][1]} / {summaries[0][2]}")
    print(f"末位置 Q/K SHA256：{summaries[-1][1]} / {summaries[-1][2]}")


def command_stress(
    port: "serial.Serial",
    image_path: Path,
    positions: int,
    seed: int,
) -> None:
    if positions <= 0:
        raise ValueError("positions 必须大于 0")
    wait_until_ready(port)
    q_input, k_input = load_real_qk_inputs(image_path)
    rng = np.random.default_rng(seed)
    passed = 0
    started = time.monotonic()
    while passed < positions:
        count = min(MAX_TABLE_ROWS, positions - passed)
        start = int(rng.integers(0, MAX_POSITION_EMBEDDINGS - count + 1))
        sequence = tuple(range(start, start + count))
        run_sequence_and_compare(port, q_input, k_input, sequence)
        passed += count
        print(
            f"真实 Q/K 随机位置已通过 {passed}/{positions}，"
            f"本批 start={start}, count={count}"
        )
    elapsed = time.monotonic() - started
    print(
        f"RoPE 真实 FPGA 随机位置回归 PASS：{positions}/{positions}，"
        f"seed={seed}，耗时 {elapsed:.2f} 秒"
    )


def command_selftest(
    image_path: Path,
    manifest_path: Path,
    rounds: int,
    seed: int,
) -> None:
    cases = build_real_rope_cases(
        DEFAULT_POSITIONS,
        image_path=image_path,
        activation_seed=DEFAULT_ACTIVATION_SEED,
    )
    manifest = validate_manifest(
        cases,
        manifest_path,
        activation_seed=DEFAULT_ACTIVATION_SEED,
    )
    software_stress(rounds=rounds, seed=seed)
    print("RoPE 配置、split-half 配对、固定清单、载荷往返与误差界：PASS")
    print(f"真实固定位置：{manifest['positions']}")
    for case in manifest["cases"]:
        print(
            f"position={case['position']}，max_error={case['error']['max_abs']}，"
            f"Q={case['sha256']['q_output_q28']}，K={case['sha256']['k_output_q28']}"
        )
    print(f"RoPE 软件随机压力 PASS：{rounds}/{rounds}，seed={seed}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PGL50H layer0 Q/K RoPE 上位机")
    parser.add_argument("--port", default="COM20")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("ports", help="列出串口")
    sub.add_parser("info", help="读取固件信息")
    sub.add_parser("status", help="读取状态")

    fixed = sub.add_parser("fixed", help="运行真实 Q/K 固定位置清单")
    fixed.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    fixed.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)

    sequence = sub.add_parser("sequence", help="验证连续位置与自动递增")
    sequence.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    sequence.add_argument("--start", type=int, default=2026)
    sequence.add_argument("--count", type=int, default=8)
    sequence.add_argument("--verify-reset", action="store_true")

    stress = sub.add_parser("stress", help="运行真实 Q/K 随机位置上板回归")
    stress.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    stress.add_argument("--positions", type=int, default=100)
    stress.add_argument("--seed", type=int, default=DEFAULT_STRESS_SEED)

    selftest = sub.add_parser("selftest", help="只运行软件参考和载荷自检")
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
            elif args.command == "sequence":
                command_sequence(
                    port,
                    args.image,
                    args.start,
                    args.count,
                    args.verify_reset,
                )
            elif args.command == "stress":
                command_stress(
                    port,
                    args.image,
                    args.positions,
                    args.seed,
                )
            else:  # pragma: no cover
                raise AssertionError(args.command)
        return 0
    except (FileNotFoundError, OSError, RuntimeError, TimeoutError, ValueError) as error:
        print(f"错误：{error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
