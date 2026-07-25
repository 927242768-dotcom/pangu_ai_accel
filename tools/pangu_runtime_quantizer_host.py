#!/usr/bin/env python3
"""PGL50H G2 Q6.10/Q28 运行时量化器真实上板逐位验证工具。"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path

import numpy as np

try:
    import serial  # type: ignore
    from serial.tools import list_ports  # type: ignore
except ImportError:  # pragma: no cover
    serial = None
    list_ports = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_TOOLS = PROJECT_ROOT / "model_tools"
if str(MODEL_TOOLS) not in sys.path:
    sys.path.insert(0, str(MODEL_TOOLS))

from runtime_quantizer_validation import (  # noqa: E402
    DEFAULT_IMAGE,
    DEFAULT_STRESS_SEED,
    MATRIX_NAMES,
    RESULT_HEADER_STRUCT,
    RuntimeQuantizerValidationCase,
    build_config_payload,
    build_fixed_validation_cases,
    build_upload_payload,
    expected_result_payload,
    random_source_case,
    validate_manifest,
)

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

EXPECTED_INFO = "PANGU50K G2 QUANT V1"
DEFAULT_PORT = "COM20"
DEFAULT_BAUD = 115200
HEADER_FIELDS = (
    "magic",
    "version",
    "matrix_id",
    "source_q28",
    "vector_length",
    "rows",
    "groups",
    "padded_groups",
    "all_zero",
    "max_abs_q10",
    "max_mantissa_binary32",
    "max_exponent_binary32",
    "max_abs_binary32_bits",
    "saturated_count",
    "source_read_commands",
    "source_read_beats",
    "raw_scale_read_commands",
    "raw_scale_read_beats",
    "activation_write_commands",
    "activation_write_beats",
    "combined_write_commands",
    "combined_write_beats",
    "trace_error_code",
)


def require_pyserial() -> None:
    if serial is None:
        raise SystemExit("缺少 pyserial，请运行：python -m pip install pyserial")


def read_exact(port: "serial.Serial", size: int, timeout: float) -> bytes:
    deadline = time.monotonic() + timeout
    output = bytearray()
    while len(output) < size:
        chunk = port.read(size - len(output))
        if chunk:
            output.extend(chunk)
        elif time.monotonic() >= deadline:
            raise TimeoutError(f"串口读取超时：收到 {len(output)}/{size} B")
    return bytes(output)


def open_port(name: str, baud: int) -> "serial.Serial":
    require_pyserial()
    port = serial.Serial(
        name,
        baud,
        timeout=0.25,
        write_timeout=300.0,
    )
    port.reset_input_buffer()
    port.reset_output_buffer()
    return port


def write_all(port: "serial.Serial", payload: bytes) -> None:
    written = port.write(payload)
    if written != len(payload):
        raise RuntimeError(f"串口写入不完整：{written}/{len(payload)} B")
    port.flush()


def read_ack(port: "serial.Serial", timeout: float) -> None:
    first = read_exact(port, 1, timeout)
    if first == b"E":
        tail = read_exact(port, 3, timeout)
        raise RuntimeError(f"FPGA 返回错误码 0x{tail[0]:02x}")
    tail = read_exact(port, 2, timeout)
    frame = first + tail
    if frame != b"K\r\n":
        raise RuntimeError(f"ACK 错误：{frame!r}")


def read_info(port: "serial.Serial", timeout: float) -> str:
    port.reset_input_buffer()
    write_all(port, b"I")
    line = port.readline().decode("ascii", errors="replace").strip()
    if line != EXPECTED_INFO:
        raise RuntimeError(f"INFO 不匹配：{line!r}")
    return line


def read_status(port: "serial.Serial", timeout: float) -> dict[str, bool]:
    port.reset_input_buffer()
    write_all(port, b"S")
    frame = read_exact(port, 4, timeout)
    if frame[:1] == b"E":
        raise RuntimeError(f"FPGA 返回错误码 0x{frame[1]:02x}")
    if frame[0] != ord("S") or frame[2:] != b"\r\n":
        raise RuntimeError(f"STATUS 帧错误：{frame!r}")
    flags = frame[1]
    return {
        "ddr_init_done": bool(flags & 0x01),
        "configured": bool(flags & 0x02),
        "loaded": bool(flags & 0x04),
        "result_valid": bool(flags & 0x08),
        "core_busy": bool(flags & 0x10),
        "trace_error": bool(flags & 0x20),
        "protocol_error": bool(flags & 0x40),
        "source_q28": bool(flags & 0x80),
    }


def configure_case(
    port: "serial.Serial",
    case: RuntimeQuantizerValidationCase,
    timeout: float,
) -> None:
    port.reset_input_buffer()
    write_all(port, b"C" + build_config_payload(case))
    read_ack(port, timeout)


def load_case(
    port: "serial.Serial",
    case: RuntimeQuantizerValidationCase,
    timeout: float,
) -> None:
    port.reset_input_buffer()
    write_all(port, b"L" + build_upload_payload(case))
    read_ack(port, timeout)


def read_result(
    port: "serial.Serial",
    case: RuntimeQuantizerValidationCase,
    timeout: float,
) -> bytes:
    port.reset_input_buffer()
    write_all(port, b"G")
    first = read_exact(port, 1, timeout)
    if first == b"E":
        tail = read_exact(port, 3, timeout)
        raise RuntimeError(f"FPGA 返回错误码 0x{tail[0]:02x}")
    if first != b"R":
        raise RuntimeError(f"结果前缀错误：{first!r}")
    return read_exact(port, case.result_bytes, timeout)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def compare_result(case: RuntimeQuantizerValidationCase, actual: bytes) -> None:
    expected = expected_result_payload(case)
    if len(actual) != len(expected):
        raise AssertionError(f"结果长度错误：{len(actual)} != {len(expected)}")
    if actual == expected:
        return

    actual_header = RESULT_HEADER_STRUCT.unpack_from(actual, 0)
    expected_header = RESULT_HEADER_STRUCT.unpack_from(expected, 0)
    header_diffs = [
        f"{name}: actual={a!r}, expected={e!r}"
        for name, a, e in zip(HEADER_FIELDS, actual_header, expected_header)
        if a != e
    ]
    if header_diffs:
        raise AssertionError("metadata 不匹配：\n  " + "\n  ".join(header_diffs))

    offset = RESULT_HEADER_STRUCT.size
    actual_activation = np.frombuffer(
        actual[offset : offset + case.activation_bytes], dtype=np.int8
    )
    expected_activation = case.activation_int8
    mismatch = np.flatnonzero(actual_activation != expected_activation)
    if mismatch.size:
        index = int(mismatch[0])
        raise AssertionError(
            f"INT8 首个不匹配 index={index}: "
            f"actual={int(actual_activation[index])}, "
            f"expected={int(expected_activation[index])}"
        )

    offset += case.activation_bytes
    actual_scales = np.frombuffer(actual[offset:], dtype="<u4").reshape(
        case.rows, case.padded_groups
    )
    expected_scales = case.combined_scale_q28_padded
    positions = np.argwhere(actual_scales != expected_scales)
    if positions.size:
        row, group = map(int, positions[0])
        raise AssertionError(
            f"UQ4.28 首个不匹配 row={row}, group={group}: "
            f"actual=0x{int(actual_scales[row, group]):08x}, "
            f"expected=0x{int(expected_scales[row, group]):08x}"
        )

    raise AssertionError(
        "结果字节不同但未定位到字段："
        f"actual_sha256={_sha256(actual)}, expected_sha256={_sha256(expected)}"
    )


def run_case(
    port: "serial.Serial",
    case: RuntimeQuantizerValidationCase,
    timeout: float,
    label: str | None = None,
) -> float:
    start = time.perf_counter()
    configure_case(port, case, timeout)
    load_case(port, case, timeout)
    actual = read_result(port, case, timeout)
    compare_result(case, actual)
    elapsed = time.perf_counter() - start
    prefix = label or case.name
    print(
        f"PASS {prefix}: source={'Q28' if case.source_q28 else 'Q6.10'}, "
        f"upload={case.upload_bytes} B, result={case.result_bytes} B, "
        f"elapsed={elapsed:.3f}s, sha256={_sha256(actual)}"
    )
    return elapsed


def resolve_case(
    cases: list[RuntimeQuantizerValidationCase],
    value: str,
) -> RuntimeQuantizerValidationCase:
    if value.isdigit():
        index = int(value)
        if 0 <= index < len(cases):
            return cases[index]
    for case in cases:
        if case.name == value:
            return case
    raise SystemExit(f"未知矩阵 {value!r}，可选：{', '.join(MATRIX_NAMES)} 或 0..6")


def print_ports() -> None:
    require_pyserial()
    ports = list(list_ports.comports())
    if not ports:
        print("未发现串口")
        return
    for item in ports:
        print(f"{item.device}: {item.description} [{item.hwid}]")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="G2 运行时量化 FPGA 逐位验证")
    parser.add_argument("--port", default=DEFAULT_PORT)
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("ports")
    sub.add_parser("info")
    sub.add_parser("status")
    case_parser = sub.add_parser("case")
    case_parser.add_argument("matrix")
    sub.add_parser("all")
    stress = sub.add_parser("stress")
    stress.add_argument("matrix")
    stress.add_argument("--rounds", type=int, default=300)
    stress.add_argument("--seed", type=int, default=DEFAULT_STRESS_SEED)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "ports":
        print_ports()
        return 0

    cases = build_fixed_validation_cases(args.image)
    validate_manifest(cases)
    with open_port(args.port, args.baud) as port:
        if args.command == "info":
            print(read_info(port, args.timeout))
            return 0
        if args.command == "status":
            print(read_status(port, args.timeout))
            return 0

        info = read_info(port, args.timeout)
        status = read_status(port, args.timeout)
        if not status["ddr_init_done"]:
            raise RuntimeError(f"DDR3 尚未初始化：{status}")
        print(f"INFO={info}; STATUS={status}")

        if args.command == "case":
            run_case(port, resolve_case(cases, args.matrix), args.timeout)
        elif args.command == "all":
            total_start = time.perf_counter()
            for case in cases:
                run_case(port, case, args.timeout)
            print(
                f"固定真实矩阵：{len(cases)}/{len(cases)} PASS, "
                f"total={time.perf_counter() - total_start:.3f}s"
            )
        elif args.command == "stress":
            if args.rounds <= 0:
                raise SystemExit("--rounds 必须大于 0")
            base = resolve_case(cases, args.matrix)
            rng = np.random.default_rng(args.seed)
            total_start = time.perf_counter()
            for iteration in range(args.rounds):
                case = random_source_case(base, rng, iteration)
                run_case(
                    port,
                    case,
                    args.timeout,
                    label=f"{base.name} stress {iteration + 1}/{args.rounds}",
                )
            print(
                f"压力测试：{args.rounds}/{args.rounds} PASS, "
                f"matrix={base.name}, seed={args.seed}, "
                f"total={time.perf_counter() - total_start:.3f}s"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
