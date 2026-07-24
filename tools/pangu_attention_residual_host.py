#!/usr/bin/env python3
"""盘古 PGL50H layer0 Attention 残差与完整子层验证工具。"""

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

from model_tools.attention_residual_reference import (  # noqa: E402
    DEFAULT_MANIFEST,
    DEFAULT_STRESS_SEED,
    K,
    AttentionResidualCase,
    attention_residual_q10,
    build_fixed_real_cases,
    build_upload_payload,
    fixed_manifest,
    make_random_residual_inputs,
    software_stress,
    validate_manifest,
)

BAUD_RATE = 115200
HIDDEN_BYTES = K * 2
OPROJ_BYTES = K * 8
UPLOAD_BYTES = HIDDEN_BYTES + OPROJ_BYTES
RESULT_BYTES = HIDDEN_BYTES

ERROR_MESSAGES = {
    0x01: "未知命令",
    0x02: "DDR3 尚未初始化完成",
    0x04: "尚未加载 Attention residual 数据",
    0xFF: "FPGA 状态机异常",
}


@dataclass(frozen=True)
class FpgaStatus:
    ddr_ready: bool
    data_loaded: bool
    result_valid: bool
    core_busy: bool


@dataclass(frozen=True)
class ResidualHardwareCase:
    hidden_q10: np.ndarray
    oproj_q28: np.ndarray
    expected_q10: np.ndarray
    label: str


def require_pyserial() -> None:
    if serial is None:
        raise SystemExit("缺少 pyserial，请运行：python -m pip install pyserial")


def sha256_array(array: np.ndarray, dtype: str | np.dtype) -> str:
    return hashlib.sha256(np.asarray(array, dtype=dtype).tobytes(order="C")).hexdigest()


def _require_shape(array: np.ndarray, shape: tuple[int, ...], label: str) -> None:
    if array.shape != shape:
        raise ValueError(f"{label} 形状错误：{array.shape}，预期 {shape}")


def validate_hardware_case(case: ResidualHardwareCase) -> None:
    _require_shape(np.asarray(case.hidden_q10), (K,), "hidden_q10")
    _require_shape(np.asarray(case.oproj_q28), (K,), "oproj_q28")
    _require_shape(np.asarray(case.expected_q10), (K,), "expected_q10")
    if np.asarray(case.hidden_q10).dtype != np.int16:
        raise ValueError("hidden_q10 必须为 int16")
    if np.asarray(case.oproj_q28).dtype != np.int64:
        raise ValueError("oproj_q28 必须为 int64")
    if np.asarray(case.expected_q10).dtype != np.int16:
        raise ValueError("expected_q10 必须为 int16")


def from_reference_case(case: AttentionResidualCase) -> ResidualHardwareCase:
    result = ResidualHardwareCase(
        hidden_q10=case.residual_hidden_q10.astype(np.int16),
        oproj_q28=case.oproj_q28.astype(np.int64),
        expected_q10=case.output_q10.astype(np.int16),
        label=case.label,
    )
    validate_hardware_case(result)
    return result


def make_random_case(seed: int, round_index: int) -> ResidualHardwareCase:
    rng = np.random.default_rng(seed)
    hidden, oproj = make_random_residual_inputs(rng, round_index)
    output, _, _, _ = attention_residual_q10(hidden, oproj)
    case = ResidualHardwareCase(
        hidden_q10=hidden.astype(np.int16),
        oproj_q28=oproj.astype(np.int64),
        expected_q10=output.astype(np.int16),
        label=f"Attention residual 随机/边界 seed={seed} mode={round_index % 6}",
    )
    validate_hardware_case(case)
    return case


def build_hardware_payload(case: ResidualHardwareCase) -> bytes:
    validate_hardware_case(case)
    payload = (
        np.asarray(case.hidden_q10, dtype="<i2").tobytes(order="C")
        + np.asarray(case.oproj_q28, dtype="<i8").tobytes(order="C")
    )
    if len(payload) != UPLOAD_BYTES:
        raise AssertionError(f"上传载荷长度错误：{len(payload)} != {UPLOAD_BYTES}")
    return payload


def verify_payload_roundtrip(case: ResidualHardwareCase) -> str:
    payload = build_hardware_payload(case)
    hidden = np.frombuffer(payload[:HIDDEN_BYTES], dtype="<i2").copy()
    oproj = np.frombuffer(payload[HIDDEN_BYTES:], dtype="<i8").copy()
    if not np.array_equal(hidden, case.hidden_q10):
        raise RuntimeError("residual hidden 上传往返不一致")
    if not np.array_equal(oproj, case.oproj_q28):
        raise RuntimeError("O_proj Q28 上传往返不一致")
    return hashlib.sha256(payload).hexdigest()


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
    if len(frame) >= 2 and frame[0:1] == b"E":
        code = frame[1]
        raise RuntimeError(
            f"FPGA 返回错误 0x{code:02X}：{ERROR_MESSAGES.get(code, '未知错误')}"
        )


def read_ack(port: "serial.Serial") -> None:
    reply = read_exact(port, 3, timeout=20.0)
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


def load_case(port: "serial.Serial", case: ResidualHardwareCase) -> None:
    payload = build_hardware_payload(case)
    port.write(b"L")
    port.write(payload)
    port.flush()
    read_ack(port)


def run_loaded_case(port: "serial.Serial") -> np.ndarray:
    port.write(b"G")
    reply = read_exact(port, 1 + RESULT_BYTES, timeout=30.0)
    raise_if_error_frame(reply)
    if reply[0:1] != b"R":
        raise RuntimeError(f"Attention residual 结果帧头错误：{reply[:16]!r}")
    return np.frombuffer(reply[1:], dtype="<i2").copy()


def run_and_compare(port: "serial.Serial", case: ResidualHardwareCase) -> np.ndarray:
    load_case(port, case)
    fpga = run_loaded_case(port)
    if not np.array_equal(fpga, case.expected_q10):
        mismatch = np.flatnonzero(fpga != case.expected_q10)
        first = int(mismatch[0])
        raise RuntimeError(
            f"{case.label} 不一致：首个错误元素={first}，FPGA={int(fpga[first])}，"
            f"Python={int(case.expected_q10[first])}，总错误数={mismatch.size}"
        )
    return fpga


def command_fixed(port: "serial.Serial", manifest_path: Path) -> None:
    wait_until_ready(port)
    reference_cases = build_fixed_real_cases()
    committed = validate_manifest(reference_cases, manifest_path)
    started = time.monotonic()
    for index, reference in enumerate(reference_cases):
        case = from_reference_case(reference)
        payload_sha = verify_payload_roundtrip(case)
        expected_payload_sha = committed["cases"][index]["sha256"]["upload_payload"]
        if payload_sha != expected_payload_sha:
            raise RuntimeError(f"固定用例 {index} 上传载荷 SHA256 不一致")
        fpga = run_and_compare(port, case)
        print(
            f"query={reference.query_position} count={reference.count} 完整 Attention 残差逐位一致：PASS，"
            f"SHA256={sha256_array(fpga, '<i2')}，前16项={fpga[:16].tolist()}"
        )
    elapsed = time.monotonic() - started
    print(f"四组连贯真实 layer0 Attention 子层固定用例全部通过，耗时 {elapsed:.2f} 秒。")


def command_stress(port: "serial.Serial", rounds: int, seed: int) -> None:
    if rounds <= 0:
        raise ValueError("rounds 必须大于 0")
    wait_until_ready(port)
    started = time.monotonic()
    for index in range(rounds):
        case_seed = seed + index
        case = make_random_case(case_seed, index)
        verify_payload_roundtrip(case)
        run_and_compare(port, case)
        if index == 0 or index + 1 == rounds or (index + 1) % 10 == 0:
            print(
                f"Attention residual 真实 FPGA 随机/边界已通过 {index + 1}/{rounds}，"
                f"seed={case_seed}，mode={index % 6}"
            )
    elapsed = time.monotonic() - started
    print(
        f"Attention residual 真实 FPGA 随机/边界回归 PASS：{rounds}/{rounds}，"
        f"seed_start={seed}，耗时 {elapsed:.2f} 秒"
    )


def command_selftest(manifest_path: Path, rounds: int, seed: int) -> None:
    if rounds <= 0:
        raise ValueError("rounds 必须大于 0")
    started = time.monotonic()
    reference_cases = build_fixed_real_cases()
    committed = validate_manifest(reference_cases, manifest_path)
    generated = fixed_manifest(reference_cases)
    if generated != committed:
        raise RuntimeError("固定清单二次比较不一致")
    for index, reference in enumerate(reference_cases):
        hardware = from_reference_case(reference)
        payload_sha = verify_payload_roundtrip(hardware)
        if payload_sha != committed["cases"][index]["sha256"]["upload_payload"]:
            raise RuntimeError(f"固定用例 {index} 上传载荷 SHA256 不一致")
    software_stress(rounds=rounds, seed=seed)
    elapsed = time.monotonic() - started
    print("完整 layer0 Attention 软件链、固定清单和 8960 B 载荷往返：PASS")
    print(
        f"Attention residual 软件随机/边界压力 PASS：{rounds}/{rounds}，"
        f"seed={seed}，耗时 {elapsed:.2f} 秒"
    )
    print(
        json.dumps(
            {
                "fixed_output_sha256": [
                    item["sha256"]["output_q10"] for item in committed["cases"]
                ],
                "upload_bytes": UPLOAD_BYTES,
                "result_bytes": RESULT_BYTES,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PGL50H layer0 Attention residual 上位机"
    )
    parser.add_argument("--port", default="COM20")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("ports", help="列出串口")
    sub.add_parser("info", help="读取固件信息")
    sub.add_parser("status", help="读取状态")

    fixed = sub.add_parser("fixed", help="运行四组连贯真实 Attention 固定用例")
    fixed.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)

    stress = sub.add_parser("stress", help="运行残差随机/饱和边界上板回归")
    stress.add_argument("--rounds", type=int, default=100)
    stress.add_argument("--seed", type=int, default=DEFAULT_STRESS_SEED)

    selftest = sub.add_parser("selftest", help="只运行完整软件链与载荷自检")
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
