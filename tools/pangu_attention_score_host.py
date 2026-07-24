#!/usr/bin/env python3
"""盘古 PGL50H F4 Attention Score 定点闭环验证工具。"""

from __future__ import annotations

import argparse
import hashlib
import struct
import sys
import time
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

from model_tools.attention_score_reference import (  # noqa: E402
    DEFAULT_FIXED_WINDOWS,
    DEFAULT_IMAGE,
    DEFAULT_MANIFEST,
    DEFAULT_STRESS_SEED,
    HEAD_DIM,
    KV_HEADS,
    MASK_VALUE,
    MAX_CONTEXT,
    MAX_TOKENS,
    NUM_LAYERS,
    Q_HEADS,
    SCORE_BYTES,
    AttentionScoreCase,
    attention_scores_q28,
    build_fixed_real_cases,
    build_k_payload,
    build_q_payload,
    decode_scores,
    encode_scores,
    sha256_bytes,
    software_stress,
    validate_manifest,
)

BAUD_RATE = 115200
Q_BYTES = Q_HEADS * HEAD_DIM * 8
K_BYTES = KV_HEADS * HEAD_DIM * 8
DEFAULT_BOARD_WINDOWS = 100

ERROR_MESSAGES = {
    0x01: "未知命令",
    0x02: "DDR3 尚未初始化完成",
    0x03: "尚未配置 layer/query/window",
    0x04: "layer/query/window 配置非法",
    0x05: "尚未上传 Q",
    0x06: "K position 非法",
    0x08: "尚无可读取的 score 结果",
    0xFF: "FPGA 状态机异常",
}


class FpgaProtocolError(RuntimeError):
    def __init__(self, code: int):
        self.code = int(code)
        super().__init__(
            f"FPGA 返回错误 0x{self.code:02X}："
            f"{ERROR_MESSAGES.get(self.code, '未知错误')}"
        )


class FpgaStatus:
    def __init__(
        self,
        flags: int,
        layer: int,
        query_position: int,
        window_start: int,
        count: int,
        k_loaded: int,
    ):
        self.ddr_ready = bool(flags & 0x01)
        self.configured = bool(flags & 0x02)
        self.q_loaded = bool(flags & 0x04)
        self.k_loaded = bool(flags & 0x08)
        self.result_valid = bool(flags & 0x10)
        self.core_busy = bool(flags & 0x20)
        self.protocol_error = bool(flags & 0x40)
        self.layer = layer
        self.query_position = query_position
        self.window_start = window_start
        self.count = count
        self.k_loaded_count = k_loaded


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


def read_frame_prefix(port: "serial.Serial", expected: bytes) -> None:
    first = read_exact(port, 1, timeout=30.0)
    if first == b"E":
        rest = read_exact(port, 3, timeout=5.0)
        if rest[-2:] != b"\r\n":
            raise RuntimeError(f"错误帧格式错误：{first + rest!r}")
        raise FpgaProtocolError(rest[0])
    if first != expected:
        raise RuntimeError(f"FPGA 帧头错误：收到 {first!r}，预期 {expected!r}")


def read_ack(port: "serial.Serial", timeout: float = 30.0) -> None:
    read_frame_prefix(port, b"K")
    tail = read_exact(port, 2, timeout=timeout)
    if tail != b"\r\n":
        raise RuntimeError(f"FPGA 确认帧错误：{b'K' + tail!r}")


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
        raise FpgaProtocolError(reply[1])
    text = reply.decode("ascii", errors="replace").strip()
    print(text)
    return text


def command_status(port: "serial.Serial", *, quiet: bool = False) -> FpgaStatus:
    port.write(b"S")
    read_frame_prefix(port, b"S")
    body = read_exact(port, 10, timeout=10.0)
    if body[-2:] != b"\r\n":
        raise RuntimeError(f"状态帧格式错误：{b'S' + body!r}")
    status = FpgaStatus(
        flags=body[0],
        layer=body[1],
        query_position=int.from_bytes(body[2:4], "little"),
        window_start=int.from_bytes(body[4:6], "little"),
        count=body[6],
        k_loaded=body[7],
    )
    if not quiet:
        print(
            "DDR3={}，已配置={}，Q={}，K={}，结果={}，核心忙={}，协议错误={}，"
            "layer={}，query={}，start={}，count={}，K上传={} 次".format(
                "是" if status.ddr_ready else "否",
                "是" if status.configured else "否",
                "已加载" if status.q_loaded else "未加载",
                "已加载" if status.k_loaded else "未加载",
                "有效" if status.result_valid else "无效",
                "是" if status.core_busy else "否",
                "是" if status.protocol_error else "否",
                status.layer,
                status.query_position,
                status.window_start,
                status.count,
                status.k_loaded_count,
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


def configure(
    port: "serial.Serial",
    layer: int,
    query_position: int,
    window_start: int,
    count: int,
) -> None:
    if not 0 <= layer < NUM_LAYERS:
        raise ValueError(f"layer 必须位于 0..{NUM_LAYERS - 1}")
    if not 0 <= query_position < MAX_CONTEXT:
        raise ValueError("query_position 越界")
    if not 0 <= window_start < MAX_CONTEXT:
        raise ValueError("window_start 越界")
    if not 1 <= count <= MAX_TOKENS:
        raise ValueError(f"count 必须位于 1..{MAX_TOKENS}")
    if window_start + count > MAX_CONTEXT:
        raise ValueError("窗口超出硬件上下文")
    port.write(b"C" + struct.pack("<BHHB", layer, query_position, window_start, count))
    read_ack(port)


def upload_q(port: "serial.Serial", q_q28: np.ndarray) -> None:
    payload = build_q_payload(q_q28)
    if len(payload) != Q_BYTES:
        raise ValueError(f"Q 载荷必须为 {Q_BYTES} B")
    write_all(port, b"Q" + payload)
    read_ack(port, timeout=60.0)


def upload_k(port: "serial.Serial", position: int, k_q28: np.ndarray) -> int:
    if not 0 <= position < MAX_CONTEXT:
        raise ValueError("K position 越界")
    payload = build_k_payload(k_q28)
    write_all(port, b"K" + struct.pack("<H", position) + payload)
    read_frame_prefix(port, b"K")
    body = read_exact(port, 4, timeout=30.0)
    if body[-2:] != b"\r\n":
        raise RuntimeError(f"K 上传确认帧错误：{b'K' + body!r}")
    returned_position = int.from_bytes(body[:2], "little")
    if returned_position != position:
        raise RuntimeError(
            f"K 上传确认 position 错误：{returned_position} != {position}"
        )
    return returned_position


def run_compute(port: "serial.Serial") -> None:
    port.write(b"G")
    read_ack(port, timeout=120.0)


def read_scores(
    port: "serial.Serial",
    *,
    expected_layer: int,
    expected_query: int,
    expected_start: int,
    expected_count: int,
) -> np.ndarray:
    port.write(b"R")
    read_frame_prefix(port, b"D")
    header = read_exact(port, 6, timeout=10.0)
    layer = header[0]
    query = int.from_bytes(header[1:3], "little")
    start = int.from_bytes(header[3:5], "little")
    count = header[5]
    if (layer, query, start, count) != (
        expected_layer,
        expected_query,
        expected_start,
        expected_count,
    ):
        raise RuntimeError(
            "score 结果头错误："
            f"收到 {(layer, query, start, count)}，"
            f"预期 {(expected_layer, expected_query, expected_start, expected_count)}"
        )
    return decode_scores(read_exact(port, SCORE_BYTES, timeout=60.0))


def compare_scores(actual: np.ndarray, expected: np.ndarray, label: str) -> None:
    actual_values = np.asarray(actual, dtype=np.int64)
    expected_values = np.asarray(expected, dtype=np.int64)
    if np.array_equal(actual_values, expected_values):
        return
    mismatch = np.argwhere(actual_values != expected_values)
    head, token = (int(value) for value in mismatch[0])
    raise RuntimeError(
        f"{label} 不一致：head={head}, token_slot={token}，"
        f"FPGA={int(actual_values[head, token])}，"
        f"Python={int(expected_values[head, token])}，"
        f"总错误数={len(mismatch)}"
    )


def execute_case(port: "serial.Serial", case: AttentionScoreCase) -> np.ndarray:
    configure(
        port,
        case.layer,
        case.query_position,
        case.window_start,
        case.count,
    )
    upload_q(port, case.q_q28)
    for index, position in enumerate(case.positions):
        upload_k(port, position, case.k_history_q28[index])
    run_compute(port)
    return read_scores(
        port,
        expected_layer=case.layer,
        expected_query=case.query_position,
        expected_start=case.window_start,
        expected_count=case.count,
    )


def command_fixed(port: "serial.Serial", image_path: Path, manifest_path: Path) -> None:
    wait_until_ready(port)
    cases = build_fixed_real_cases(DEFAULT_FIXED_WINDOWS, image_path=image_path)
    validate_manifest(cases, manifest_path)
    started = time.monotonic()

    for case in cases:
        actual = execute_case(port, case)
        compare_scores(actual, case.expected_scores_q28, case.label)
        score_hash = sha256_bytes(encode_scores(actual))
        print(f"{case.label} 逐位一致：PASS，SHA256={score_hash}")

    causal_case = cases[2]
    if not np.all(causal_case.expected_scores_q28[:, 4:] == MASK_VALUE):
        raise RuntimeError("固定 causal mask 用例本身不完整")
    print("未来位置 causal mask 与固定未使用槽 INT64_MIN：PASS")
    print(f"F4 真实固定用例全部通过，耗时 {time.monotonic() - started:.2f} 秒")


def command_stress(port: "serial.Serial", windows: int, seed: int) -> None:
    if windows <= 0:
        raise ValueError("windows 必须大于 0")
    wait_until_ready(port)
    rng = np.random.default_rng(seed)
    limit = 8 << 28
    started = time.monotonic()

    for index in range(windows):
        count = int(rng.integers(1, MAX_TOKENS + 1))
        start = int(rng.integers(0, MAX_CONTEXT - count + 1))
        query = int(
            rng.integers(max(0, start - 2), min(MAX_CONTEXT, start + count + 2))
        )
        layer = int(rng.integers(0, NUM_LAYERS))
        q = rng.integers(
            -limit, limit + 1, size=(Q_HEADS, HEAD_DIM), dtype=np.int64
        )
        history = rng.integers(
            -limit,
            limit + 1,
            size=(count, KV_HEADS, HEAD_DIM),
            dtype=np.int64,
        )
        expected = attention_scores_q28(
            q,
            history,
            query_position=query,
            window_start=start,
            count=count,
        )

        configure(port, layer, query, start, count)
        upload_q(port, q)
        for token_index in range(count):
            upload_k(port, start + token_index, history[token_index])
        run_compute(port)
        actual = read_scores(
            port,
            expected_layer=layer,
            expected_query=query,
            expected_start=start,
            expected_count=count,
        )
        compare_scores(
            actual,
            expected,
            f"stress {index + 1}/{windows}, layer={layer}, query={query}, start={start}, count={count}",
        )
        print(
            f"随机 Attention Score 已通过 {index + 1}/{windows} 窗口，"
            f"layer={layer}, query={query}, start={start}, count={count}"
        )

    print(
        f"F4 真实 FPGA 随机窗口回归 PASS：{windows}/{windows}，"
        f"seed={seed}，耗时 {time.monotonic() - started:.2f} 秒"
    )


def command_selftest(
    image_path: Path,
    manifest_path: Path,
    rounds: int,
    seed: int,
) -> None:
    cases = build_fixed_real_cases(DEFAULT_FIXED_WINDOWS, image_path=image_path)
    manifest = validate_manifest(cases, manifest_path)
    software_stress(rounds=rounds, seed=seed)
    print("F4 真实 Q/K、GQA、1/8 缩放、RNE、causal mask 和载荷：PASS")
    print(manifest["definition"]["output_rule"])
    print(f"Attention Score 软件随机压力 PASS：{rounds}/{rounds}，seed={seed}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PGL50H F4 Attention Score 上位机")
    parser.add_argument("--port", default="COM20")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("ports", help="列出串口")
    sub.add_parser("info", help="读取固件信息")
    sub.add_parser("status", help="读取状态")

    fixed = sub.add_parser("fixed", help="运行真实固定窗口和 causal mask 测试")
    fixed.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    fixed.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)

    stress = sub.add_parser("stress", help="随机层/窗口/Q/K 逐位回归")
    stress.add_argument("--windows", type=int, default=DEFAULT_BOARD_WINDOWS)
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
            elif args.command == "stress":
                command_stress(port, args.windows, args.seed)
            else:  # pragma: no cover
                raise AssertionError(args.command)
        return 0
    except (
        FileNotFoundError,
        OSError,
        RuntimeError,
        TimeoutError,
        ValueError,
    ) as error:
        print(f"错误：{error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
