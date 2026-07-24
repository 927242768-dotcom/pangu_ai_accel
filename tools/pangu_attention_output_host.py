#!/usr/bin/env python3
"""盘古 PGL50H F6 Attention 输出定点闭环验证工具。"""

from __future__ import annotations

import argparse
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

from model_tools.attention_output_reference import (  # noqa: E402
    DEFAULT_IMAGE,
    DEFAULT_MANIFEST,
    DEFAULT_STRESS_SEED,
    HEAD_DIM,
    INT64_MAX,
    INT64_MIN,
    KV_HEADS,
    MAX_TOKENS,
    OUTPUT_BYTES,
    PROB_ONE,
    Q_HEADS,
    AttentionOutputCase,
    attention_output_q28,
    build_fixed_real_cases,
    decode_attention_output,
    encode_attention_output,
    encode_v_vector,
    sha256_bytes,
    software_stress,
    validate_manifest,
)
from model_tools.kv_cache_reference import MAX_CONTEXT, NUM_LAYERS  # noqa: E402
from model_tools.softmax_fixed_reference import (  # noqa: E402
    PROB_BYTES,
    encode_probabilities,
    softmax_scores_q31,
)

BAUD_RATE = 115200
DEFAULT_BOARD_WINDOWS = 100

ERROR_MESSAGES = {
    0x01: "未知命令",
    0x02: "DDR3 尚未初始化完成",
    0x03: "尚未配置 layer/window",
    0x04: "layer/window 配置非法",
    0x05: "尚未上传概率",
    0x06: "V position 非法",
    0x07: "V 上传数量不足",
    0x08: "尚无可读取的 Attention 输出",
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
        window_start: int,
        count: int,
        v_loaded: int,
    ):
        self.ddr_ready = bool(flags & 0x01)
        self.configured = bool(flags & 0x02)
        self.probabilities_loaded = bool(flags & 0x04)
        self.v_loaded = bool(flags & 0x08)
        self.result_valid = bool(flags & 0x10)
        self.core_busy = bool(flags & 0x20)
        self.protocol_error = bool(flags & 0x40)
        self.layer = layer
        self.window_start = window_start
        self.count = count
        self.v_loaded_count = v_loaded


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
    body = read_exact(port, 8, timeout=10.0)
    if body[-2:] != b"\r\n":
        raise RuntimeError(f"状态帧格式错误：{b'S' + body!r}")
    status = FpgaStatus(
        flags=body[0],
        layer=body[1],
        window_start=int.from_bytes(body[2:4], "little"),
        count=body[4],
        v_loaded=body[5],
    )
    if not quiet:
        print(
            "DDR3={}，已配置={}，概率={}，V={}，结果={}，核心忙={}，协议错误={}，"
            "layer={}，start={}，count={}，V上传={} 次".format(
                "是" if status.ddr_ready else "否",
                "是" if status.configured else "否",
                "已加载" if status.probabilities_loaded else "未加载",
                "已加载" if status.v_loaded else "未加载",
                "有效" if status.result_valid else "无效",
                "是" if status.core_busy else "否",
                "是" if status.protocol_error else "否",
                status.layer,
                status.window_start,
                status.count,
                status.v_loaded_count,
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


def configure(port: "serial.Serial", layer: int, window_start: int, count: int) -> None:
    if not 0 <= layer < NUM_LAYERS:
        raise ValueError(f"layer 必须位于 0..{NUM_LAYERS - 1}")
    if not 0 <= window_start < MAX_CONTEXT:
        raise ValueError("window_start 越界")
    if not 1 <= count <= MAX_TOKENS:
        raise ValueError(f"count 必须位于 1..{MAX_TOKENS}")
    if window_start + count > MAX_CONTEXT:
        raise ValueError("窗口超出硬件上下文")
    port.write(b"C" + struct.pack("<BHB", layer, window_start, count))
    read_ack(port)


def upload_probabilities(port: "serial.Serial", probabilities_q31: np.ndarray) -> None:
    payload = encode_probabilities(probabilities_q31)
    if len(payload) != PROB_BYTES:
        raise ValueError(f"概率载荷必须为 {PROB_BYTES} B")
    write_all(port, b"P" + payload)
    read_ack(port, timeout=60.0)


def upload_v(port: "serial.Serial", position: int, v_q28: np.ndarray) -> int:
    if not 0 <= position < MAX_CONTEXT:
        raise ValueError("V position 越界")
    payload = encode_v_vector(v_q28)
    write_all(port, b"V" + struct.pack("<H", position) + payload)
    read_frame_prefix(port, b"K")
    body = read_exact(port, 4, timeout=30.0)
    if body[-2:] != b"\r\n":
        raise RuntimeError(f"V 上传确认帧错误：{b'K' + body!r}")
    returned_position = int.from_bytes(body[:2], "little")
    if returned_position != position:
        raise RuntimeError(
            f"V 上传确认 position 错误：{returned_position} != {position}"
        )
    return returned_position


def run_compute(port: "serial.Serial") -> None:
    port.write(b"G")
    read_ack(port, timeout=180.0)


def read_output(
    port: "serial.Serial",
    *,
    expected_layer: int,
    expected_start: int,
    expected_count: int,
) -> np.ndarray:
    port.write(b"R")
    read_frame_prefix(port, b"D")
    header = read_exact(port, 4, timeout=10.0)
    layer = header[0]
    start = int.from_bytes(header[1:3], "little")
    count = header[3]
    if (layer, start, count) != (
        expected_layer,
        expected_start,
        expected_count,
    ):
        raise RuntimeError(
            "Attention 输出结果头错误："
            f"收到 {(layer, start, count)}，"
            f"预期 {(expected_layer, expected_start, expected_count)}"
        )
    return decode_attention_output(read_exact(port, OUTPUT_BYTES, timeout=120.0))


def compare_output(actual: np.ndarray, expected: np.ndarray, label: str) -> None:
    actual_values = np.asarray(actual, dtype=np.int64)
    expected_values = np.asarray(expected, dtype=np.int64)
    if np.array_equal(actual_values, expected_values):
        return
    mismatch = np.argwhere(actual_values != expected_values)
    head, dimension = (int(value) for value in mismatch[0])
    raise RuntimeError(
        f"{label} 不一致：head={head}, dimension={dimension}，"
        f"FPGA={int(actual_values[head, dimension])}，"
        f"Python={int(expected_values[head, dimension])}，"
        f"总错误数={len(mismatch)}"
    )


def execute_arrays(
    port: "serial.Serial",
    *,
    layer: int,
    window_start: int,
    probabilities_q31: np.ndarray,
    v_history_q28: np.ndarray,
) -> np.ndarray:
    history = np.asarray(v_history_q28, dtype=np.int64)
    count = int(history.shape[0])
    configure(port, layer, window_start, count)
    upload_probabilities(port, probabilities_q31)
    for index in range(count):
        upload_v(port, window_start + index, history[index])
    run_compute(port)
    return read_output(
        port,
        expected_layer=layer,
        expected_start=window_start,
        expected_count=count,
    )


def execute_case(port: "serial.Serial", case: AttentionOutputCase) -> np.ndarray:
    return execute_arrays(
        port,
        layer=case.layer,
        window_start=case.window_start,
        probabilities_q31=case.probabilities_q31,
        v_history_q28=case.v_history_q28,
    )


def command_fixed(port: "serial.Serial", image_path: Path, manifest_path: Path) -> None:
    wait_until_ready(port)
    cases = build_fixed_real_cases(image_path=image_path)
    validate_manifest(cases, manifest_path)
    started = time.monotonic()

    for case in cases:
        actual = execute_case(port, case)
        compare_output(actual, case.expected_heads_q28, case.label)
        output_hash = sha256_bytes(encode_attention_output(actual))
        print(f"{case.label} 逐位一致：PASS，SHA256={output_hash}")

    rng = np.random.default_rng(20260804)
    all_mask_probabilities = np.zeros((Q_HEADS, MAX_TOKENS), dtype=np.uint32)
    all_mask_v = rng.integers(
        -(16 << 28),
        (16 << 28) + 1,
        size=(4, KV_HEADS, HEAD_DIM),
        dtype=np.int64,
    )
    actual = execute_arrays(
        port,
        layer=0,
        window_start=100,
        probabilities_q31=all_mask_probabilities,
        v_history_q28=all_mask_v,
    )
    np.testing.assert_array_equal(actual, 0)
    print("全 mask / 全零概率输出严格全 0：PASS")

    single_probabilities = np.zeros((Q_HEADS, MAX_TOKENS), dtype=np.uint32)
    single_probabilities[:, 0] = PROB_ONE
    extreme_v = np.zeros((1, KV_HEADS, HEAD_DIM), dtype=np.int64)
    pattern = np.asarray(
        [INT64_MIN, INT64_MAX, -1, 0, 1, -(1 << 62), 1 << 62, 20260804],
        dtype=np.int64,
    )
    extreme_v[0, 0] = np.resize(pattern, HEAD_DIM)
    extreme_v[0, 1] = np.resize(pattern[::-1], HEAD_DIM)
    expected, _ = attention_output_q28(single_probabilities, extreme_v, count=1)
    actual = execute_arrays(
        port,
        layer=0,
        window_start=200,
        probabilities_q31=single_probabilities,
        v_history_q28=extreme_v,
    )
    compare_output(actual, expected, "单 token 极端 V")
    print("单 token 1.0 概率、GQA 映射与 INT64 极端 V 精确复制：PASS")

    saturation_probabilities = np.full(
        (Q_HEADS, MAX_TOKENS), PROB_ONE, dtype=np.uint32
    )
    saturation_v = np.empty(
        (MAX_TOKENS, KV_HEADS, HEAD_DIM), dtype=np.int64
    )
    saturation_v[:, 0, :] = INT64_MAX
    saturation_v[:, 1, :] = INT64_MIN
    expected, debug = attention_output_q28(
        saturation_probabilities, saturation_v, count=MAX_TOKENS
    )
    if debug.saturated_values != Q_HEADS * HEAD_DIM:
        raise RuntimeError(
            f"饱和边界软件用例错误：{debug.saturated_values} != "
            f"{Q_HEADS * HEAD_DIM}"
        )
    actual = execute_arrays(
        port,
        layer=0,
        window_start=300,
        probabilities_q31=saturation_probabilities,
        v_history_q28=saturation_v,
    )
    compare_output(actual, expected, "16-token 双向饱和")
    np.testing.assert_array_equal(actual[:7], INT64_MAX)
    np.testing.assert_array_equal(actual[7:], INT64_MIN)
    print("16-token Q59 宽累加与 INT64 正/负双向显式饱和：PASS")

    print(f"F6 真实固定与边界用例全部通过，耗时 {time.monotonic() - started:.2f} 秒")


def _random_scores(rng: np.random.Generator, count: int) -> np.ndarray:
    mask = -(1 << 63)
    scores = np.full((Q_HEADS, MAX_TOKENS), mask, dtype=np.int64)
    for head in range(Q_HEADS):
        valid_count = int(rng.integers(0, count + 1))
        if valid_count:
            indices = rng.choice(count, size=valid_count, replace=False)
            scores[head, indices] = rng.integers(
                -(12 << 28), 12 << 28, size=valid_count, dtype=np.int64
            )
    return scores


def command_stress(port: "serial.Serial", windows: int, seed: int) -> None:
    if windows <= 0:
        raise ValueError("windows 必须大于 0")
    wait_until_ready(port)
    rng = np.random.default_rng(seed)
    started = time.monotonic()

    for index in range(windows):
        count = int(rng.integers(1, MAX_TOKENS + 1))
        start = int(rng.integers(0, MAX_CONTEXT - count + 1))
        layer = int(rng.integers(0, NUM_LAYERS))
        scores = _random_scores(rng, count)
        probabilities, _ = softmax_scores_q31(scores)
        if index % 10 == 0:
            raw = rng.integers(
                0,
                1 << 64,
                size=count * KV_HEADS * HEAD_DIM,
                dtype=np.uint64,
            )
            history = raw.view(np.int64).reshape(count, KV_HEADS, HEAD_DIM).copy()
        else:
            history = rng.integers(
                -(16 << 28),
                (16 << 28) + 1,
                size=(count, KV_HEADS, HEAD_DIM),
                dtype=np.int64,
            )
        expected, _ = attention_output_q28(probabilities, history, count=count)
        actual = execute_arrays(
            port,
            layer=layer,
            window_start=start,
            probabilities_q31=probabilities,
            v_history_q28=history,
        )
        compare_output(
            actual,
            expected,
            f"stress {index + 1}/{windows}, layer={layer}, start={start}, count={count}",
        )
        print(
            f"随机 Attention 输出已通过 {index + 1}/{windows} 窗口，"
            f"layer={layer}, start={start}, count={count}"
        )

    print(
        f"F6 真实 FPGA 随机窗口回归 PASS：{windows}/{windows}，"
        f"seed={seed}，耗时 {time.monotonic() - started:.2f} 秒"
    )


def command_selftest(
    image_path: Path,
    manifest_path: Path,
    rounds: int,
    seed: int,
) -> None:
    cases = build_fixed_real_cases(image_path=image_path)
    manifest = validate_manifest(cases, manifest_path)
    saturated = software_stress(rounds=rounds, seed=seed)
    print("F6 F5概率、F3 V、GQA、Q59、RNE、饱和和载荷：PASS")
    print(manifest["definition"]["rounding"])
    print(
        f"Attention 输出软件随机压力 PASS：{rounds}/{rounds}，"
        f"seed={seed}，累计饱和值={saturated}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PGL50H F6 Attention 输出上位机")
    parser.add_argument("--port", default="COM20")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("ports", help="列出串口")
    sub.add_parser("info", help="读取固件信息")
    sub.add_parser("status", help="读取状态")

    fixed = sub.add_parser("fixed", help="运行真实固定窗口和边界测试")
    fixed.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    fixed.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)

    stress = sub.add_parser("stress", help="随机层/窗口/概率/V 逐位回归")
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
        AssertionError,
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
