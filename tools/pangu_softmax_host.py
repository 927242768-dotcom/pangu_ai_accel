#!/usr/bin/env python3
"""盘古 PGL50H F5 Softmax 定点闭环验证工具。"""

from __future__ import annotations

import argparse
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

from model_tools.softmax_fixed_reference import (  # noqa: E402
    DEFAULT_FLOAT_TOLERANCE,
    DEFAULT_IMAGE,
    DEFAULT_MANIFEST,
    DEFAULT_STRESS_SEED,
    EXP_LUT_PADDED_BYTES,
    MASK_VALUE,
    MAX_TOKENS,
    PROB_BYTES,
    PROB_ONE,
    Q_HEADS,
    SCORE_FRACTION_BITS,
    SoftmaxCase,
    build_exp_lut_payload,
    build_fixed_real_cases,
    decode_probabilities,
    encode_probabilities,
    max_probability_error,
    sha256_bytes,
    softmax_scores_q31,
    software_stress,
    validate_manifest,
)
from model_tools.attention_score_reference import (  # noqa: E402
    SCORE_BYTES,
    encode_scores,
)

BAUD_RATE = 115200
DEFAULT_BOARD_ROUNDS = 100

ERROR_MESSAGES = {
    0x01: "未知命令",
    0x02: "DDR3 尚未初始化完成",
    0x03: "尚未上传 Attention Score",
    0x04: "尚未上传 exp LUT",
    0x05: "尚无可读取的 Softmax 结果",
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
    def __init__(self, flags: int):
        self.ddr_ready = bool(flags & 0x01)
        self.scores_loaded = bool(flags & 0x02)
        self.lut_loaded = bool(flags & 0x04)
        self.result_valid = bool(flags & 0x08)
        self.core_busy = bool(flags & 0x10)
        self.protocol_error = bool(flags & 0x20)


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


def read_ack(port: "serial.Serial", timeout: float = 60.0) -> None:
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
    body = read_exact(port, 3, timeout=10.0)
    if body[-2:] != b"\r\n":
        raise RuntimeError(f"状态帧格式错误：{b'S' + body!r}")
    status = FpgaStatus(body[0])
    if not quiet:
        print(
            "DDR3={}，score={}，LUT={}，结果={}，核心忙={}，协议错误={}".format(
                "是" if status.ddr_ready else "否",
                "已加载" if status.scores_loaded else "未加载",
                "已加载" if status.lut_loaded else "未加载",
                "有效" if status.result_valid else "无效",
                "是" if status.core_busy else "否",
                "是" if status.protocol_error else "否",
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


def upload_scores(port: "serial.Serial", scores_q28: np.ndarray) -> None:
    payload = encode_scores(np.asarray(scores_q28, dtype=np.int64))
    if len(payload) != SCORE_BYTES:
        raise ValueError(f"score 载荷必须为 {SCORE_BYTES} B")
    write_all(port, b"L" + payload)
    read_ack(port)


def upload_exp_lut(port: "serial.Serial") -> None:
    payload = build_exp_lut_payload()
    if len(payload) != EXP_LUT_PADDED_BYTES:
        raise ValueError(f"exp LUT 载荷必须为 {EXP_LUT_PADDED_BYTES} B")
    write_all(port, b"T" + payload)
    read_ack(port)


def run_compute(port: "serial.Serial") -> None:
    port.write(b"G")
    read_ack(port, timeout=120.0)


def read_probabilities(port: "serial.Serial") -> np.ndarray:
    port.write(b"R")
    read_frame_prefix(port, b"D")
    return decode_probabilities(read_exact(port, PROB_BYTES, timeout=60.0))


def compare_probabilities(actual: np.ndarray, expected: np.ndarray, label: str) -> None:
    actual_values = np.asarray(actual, dtype=np.uint32)
    expected_values = np.asarray(expected, dtype=np.uint32)
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


def execute_scores(port: "serial.Serial", scores_q28: np.ndarray) -> np.ndarray:
    upload_scores(port, scores_q28)
    run_compute(port)
    return read_probabilities(port)


def command_fixed(port: "serial.Serial", image_path: Path, manifest_path: Path) -> None:
    wait_until_ready(port)
    cases = build_fixed_real_cases(image_path=image_path)
    validate_manifest(cases, manifest_path)
    upload_exp_lut(port)
    started = time.monotonic()

    for case in cases:
        actual = execute_scores(port, case.scores_q28)
        compare_probabilities(actual, case.expected_probs_q31, case.label)
        probability_hash = sha256_bytes(encode_probabilities(actual))
        error = max_probability_error(case.scores_q28, actual)
        print(
            f"{case.label} 逐位一致：PASS，SHA256={probability_hash}，"
            f"float64最大误差={error:.12g}"
        )

    single = cases[0]
    if not np.all(single.expected_probs_q31[:, 0] == PROB_ONE):
        raise RuntimeError("固定单有效 token 用例本身不完整")
    causal = cases[2]
    if not np.all(causal.expected_probs_q31[:, 4:] == 0):
        raise RuntimeError("固定 causal mask 概率用例本身不完整")
    print("单有效 token 精确 1.0、未来位置和未使用槽概率 0：PASS")
    print(f"F5 真实固定用例全部通过，耗时 {time.monotonic() - started:.2f} 秒")


def _random_scores(rng: np.random.Generator, index: int) -> np.ndarray:
    scores = np.full((Q_HEADS, MAX_TOKENS), MASK_VALUE, dtype=np.int64)
    mode = index % 6
    if mode == 0:
        return scores
    if mode == 1:
        for head in range(Q_HEADS):
            scores[head, int(rng.integers(0, MAX_TOKENS))] = int(
                rng.integers(-(8 << SCORE_FRACTION_BITS), 8 << SCORE_FRACTION_BITS)
            )
        return scores
    if mode == 2:
        count = int(rng.integers(1, MAX_TOKENS + 1))
        scores[:, :count] = rng.integers(
            -(8 << SCORE_FRACTION_BITS),
            8 << SCORE_FRACTION_BITS,
            size=(Q_HEADS, count),
            dtype=np.int64,
        )
        return scores
    if mode == 3:
        scores[:, :] = int(rng.integers(-(4 << SCORE_FRACTION_BITS), 4 << SCORE_FRACTION_BITS))
        return scores
    if mode == 4:
        for head in range(Q_HEADS):
            scores[head, 0] = 4 << SCORE_FRACTION_BITS
            scores[head, 1] = -(12 << SCORE_FRACTION_BITS)
            scores[head, 2] = -(13 << SCORE_FRACTION_BITS) - 1
        return scores

    for head in range(Q_HEADS):
        valid_count = int(rng.integers(0, MAX_TOKENS + 1))
        if valid_count == 0:
            continue
        indices = rng.choice(MAX_TOKENS, size=valid_count, replace=False)
        center = int(rng.integers(-(8 << SCORE_FRACTION_BITS), 8 << SCORE_FRACTION_BITS))
        differences = rng.integers(
            -(24 << SCORE_FRACTION_BITS),
            1,
            size=valid_count,
            dtype=np.int64,
        )
        scores[head, indices] = center + differences
    return scores


def command_stress(port: "serial.Serial", rounds: int, seed: int) -> None:
    if rounds <= 0:
        raise ValueError("rounds 必须大于 0")
    wait_until_ready(port)
    upload_exp_lut(port)
    rng = np.random.default_rng(seed)
    started = time.monotonic()
    worst_error = 0.0

    for index in range(rounds):
        scores = _random_scores(rng, index)
        expected, _ = softmax_scores_q31(scores)
        actual = execute_scores(port, scores)
        compare_probabilities(actual, expected, f"stress {index + 1}/{rounds}")
        error = max_probability_error(scores, actual)
        worst_error = max(worst_error, error)
        if error > DEFAULT_FLOAT_TOLERANCE:
            raise RuntimeError(
                f"stress {index + 1} float64 误差超限：{error}"
            )
        print(
            f"随机 Softmax 已通过 {index + 1}/{rounds}，"
            f"当前误差={error:.8g}，最坏误差={worst_error:.8g}"
        )

    print(
        f"F5 真实 FPGA 随机回归 PASS：{rounds}/{rounds}，seed={seed}，"
        f"worst_error={worst_error:.12g}，耗时 {time.monotonic() - started:.2f} 秒"
    )


def command_selftest(
    image_path: Path,
    manifest_path: Path,
    rounds: int,
    seed: int,
) -> None:
    cases = build_fixed_real_cases(image_path=image_path)
    manifest = validate_manifest(cases, manifest_path)
    worst_error = software_stress(rounds=rounds, seed=seed)
    print("F5 mask、max、减最大值、PWL exp、sum、reciprocal 和归一化：PASS")
    print(manifest["definition"]["normalization_rule"])
    print(
        f"Softmax 软件随机压力 PASS：{rounds}/{rounds}，seed={seed}，"
        f"worst_error={worst_error:.12g}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PGL50H F5 Softmax 上位机")
    parser.add_argument("--port", default="COM20")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("ports", help="列出串口")
    sub.add_parser("info", help="读取固件信息")
    sub.add_parser("status", help="读取状态")

    fixed = sub.add_parser("fixed", help="运行真实 F4 固定 score Softmax 测试")
    fixed.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    fixed.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)

    stress = sub.add_parser("stress", help="随机 mask/窗口/极端 score 逐位回归")
    stress.add_argument("--rounds", type=int, default=DEFAULT_BOARD_ROUNDS)
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
                command_stress(port, args.rounds, args.seed)
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
