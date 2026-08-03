#!/usr/bin/env python3
"""PGL50H G2 完整 layer0 Transformer Block 真实板卡逐位验证工具。

协议对应 ``transformer_block_host_ctrl.v``：

- ``I``：固件信息；
- ``S``：状态、阶段和错误码；
- ``C``：layer/query/window/count；
- ``W`` / ``R``：任意 256-bit 对齐 DDR3 区域写入/回读；
- ``P``：提交本次动态载荷；
- ``G``：从同一 block hidden 连贯执行 22 个阶段。

常驻参数由 ``transformer_block_g2_payload`` 从真实 P50 和既有定点参考自动生成，
中间结果及最终输出均与同一 ``TransformerBlockCase`` 逐字节比较。
"""

from __future__ import annotations

import argparse
import hashlib
import struct
import sys
import time
from pathlib import Path
from typing import Iterable, Sequence

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

from transformer_block_g2_payload import (  # noqa: E402
    DDRUpload,
    build_dynamic_uploads,
    build_layer_parameter_uploads,
    build_resident_uploads,
    build_stress_case,
)
from transformer_block_reference import (  # noqa: E402
    DEFAULT_IMAGE,
    TransformerBlockCase,
    build_fixed_real_cases,
    load_context,
    scratch_regions,
)

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

EXPECTED_INFO = "PANGU50K G2 BLOCK V1"
DEFAULT_PORT = "COM20"
DEFAULT_BAUD = 115200
DEFAULT_TIMEOUT = 30.0
CONFIG_STRUCT = struct.Struct("<4H")
TRANSFER_HEADER = struct.Struct("<II")
COMPLETION_STRUCT = struct.Struct("<BBBBI2s")
BEAT_BYTES = 32

STAGE_NAMES = {
    0x00: "IDLE",
    0x01: "INPUT_RMS",
    0x02: "QKV_QUANT",
    0x03: "Q_LINEAR",
    0x04: "K_LINEAR",
    0x05: "V_LINEAR",
    0x06: "ROPE",
    0x07: "KV_WRITE",
    0x08: "ATTENTION_SCORE",
    0x09: "SOFTMAX",
    0x0A: "ATTENTION_OUTPUT",
    0x0B: "OPROJ_QUANT",
    0x0C: "OPROJ_LINEAR",
    0x0D: "RESIDUAL1",
    0x0E: "POST_RMS",
    0x0F: "GATE_UP_QUANT",
    0x10: "GATE_LINEAR",
    0x11: "UP_LINEAR",
    0x12: "SILU",
    0x13: "SILU_UP_MUL",
    0x14: "DOWN_QUANT",
    0x15: "DOWN_LINEAR",
    0x16: "RESIDUAL2",
    0x17: "DONE",
    0x1F: "ERROR",
}


def require_pyserial() -> None:
    if serial is None:
        raise SystemExit("缺少 pyserial，请运行：python -m pip install pyserial")


def transfer_timeout(size: int, baud: int, base: float) -> float:
    # 8N1 每字节约 10 bit；再为 FPGA DDR3 状态机与系统调度保留 2 倍裕量。
    serial_seconds = (max(size, 1) * 10.0) / max(baud, 1)
    return max(base, serial_seconds * 3.0 + 10.0)


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


def write_all(port: "serial.Serial", payload: bytes) -> None:
    view = memoryview(payload)
    offset = 0
    while offset < len(view):
        written = port.write(view[offset : offset + 65536])
        if written is None or written <= 0:
            raise RuntimeError(f"串口写入停止于 {offset}/{len(payload)} B")
        offset += written
    port.flush()


def open_port(name: str, baud: int) -> "serial.Serial":
    require_pyserial()
    port = serial.Serial(
        name,
        baud,
        timeout=0.25,
        write_timeout=1800.0,
    )
    port.reset_input_buffer()
    port.reset_output_buffer()
    return port


def _read_error_tail(port: "serial.Serial", timeout: float) -> None:
    tail = read_exact(port, 3, timeout)
    if tail[1:] != b"\r\n":
        raise RuntimeError(f"FPGA ERROR 帧格式错误：{b'E' + tail!r}")
    raise RuntimeError(f"FPGA 返回错误码 0x{tail[0]:02x}")


def read_ack(port: "serial.Serial", timeout: float) -> None:
    first = read_exact(port, 1, timeout)
    if first == b"E":
        _read_error_tail(port, timeout)
    frame = first + read_exact(port, 2, timeout)
    if frame != b"K\r\n":
        raise RuntimeError(f"ACK 错误：{frame!r}")


def read_info(port: "serial.Serial", timeout: float) -> str:
    port.reset_input_buffer()
    write_all(port, b"I")
    line = port.readline().decode("ascii", errors="replace").strip()
    if line != EXPECTED_INFO:
        raise RuntimeError(f"INFO 不匹配：{line!r}")
    return line


def read_status(port: "serial.Serial", timeout: float) -> dict[str, object]:
    port.reset_input_buffer()
    write_all(port, b"S")
    first = read_exact(port, 1, timeout)
    if first == b"E":
        _read_error_tail(port, timeout)
    frame = first + read_exact(port, 5, timeout)
    if frame[0] != ord("S") or frame[4:] != b"\r\n":
        raise RuntimeError(f"STATUS 帧错误：{frame!r}")
    flags, stage, error_code = frame[1], frame[2], frame[3]
    return {
        "ddr_init_done": bool(flags & 0x01),
        "configured": bool(flags & 0x02),
        "any_write_seen": bool(flags & 0x04),
        "payload_committed": bool(flags & 0x08),
        "result_valid": bool(flags & 0x10),
        "block_busy": bool(flags & 0x20),
        "block_error": bool(flags & 0x40),
        "protocol_error": bool(flags & 0x80),
        "stage": stage,
        "stage_name": STAGE_NAMES.get(stage, f"UNKNOWN_0x{stage:02x}"),
        "error_code": error_code,
    }


def build_config_payload(
    case: TransformerBlockCase,
    *,
    layer_index: int = 0,
) -> bytes:
    if not 0 <= int(layer_index) < 28:
        raise ValueError("layer_index 必须在 0..27")
    return CONFIG_STRUCT.pack(
        int(layer_index),
        case.query_position,
        case.window_start,
        case.count,
    )


def configure_case(
    port: "serial.Serial",
    case: TransformerBlockCase,
    timeout: float,
    *,
    layer_index: int = 0,
) -> None:
    payload = build_config_payload(case, layer_index=layer_index)
    port.reset_input_buffer()
    write_all(port, b"C" + payload)
    read_ack(port, timeout)


def write_region(
    port: "serial.Serial",
    upload: DDRUpload,
    *,
    baud: int,
    timeout: float,
) -> float:
    if len(upload.payload) % BEAT_BYTES:
        raise ValueError(f"{upload.name} 长度未按 {BEAT_BYTES} B 对齐")
    header = TRANSFER_HEADER.pack(upload.controller_address, len(upload.payload))
    port.reset_input_buffer()
    started = time.perf_counter()
    write_all(port, b"W" + header + upload.payload)
    read_ack(port, transfer_timeout(len(upload.payload), baud, timeout))
    return time.perf_counter() - started


def read_region(
    port: "serial.Serial",
    controller_address: int,
    length: int,
    *,
    baud: int,
    timeout: float,
) -> bytes:
    if length <= 0 or length % BEAT_BYTES:
        raise ValueError("read length 必须为正且按 32 B 对齐")
    port.reset_input_buffer()
    write_all(port, b"R" + TRANSFER_HEADER.pack(controller_address, length))
    resolved_timeout = transfer_timeout(length, baud, timeout)
    first = read_exact(port, 1, resolved_timeout)
    if first == b"E":
        _read_error_tail(port, resolved_timeout)
    if first != b"R":
        raise RuntimeError(f"READ 前缀错误：{first!r}")
    returned_length = struct.unpack("<I", read_exact(port, 4, resolved_timeout))[0]
    if returned_length != length:
        raise RuntimeError(f"READ 长度错误：{returned_length} != {length}")
    return read_exact(port, length, resolved_timeout)


def commit_payload(port: "serial.Serial", timeout: float) -> None:
    port.reset_input_buffer()
    write_all(port, b"P")
    read_ack(port, timeout)


def execute_block(port: "serial.Serial", timeout: float) -> dict[str, object]:
    port.reset_input_buffer()
    write_all(port, b"G")
    first = read_exact(port, 1, timeout)
    if first == b"E":
        _read_error_tail(port, timeout)
    frame = first + read_exact(port, COMPLETION_STRUCT.size - 1, timeout)
    prefix, success, error_code, stage, cycles, suffix = COMPLETION_STRUCT.unpack(frame)
    if prefix != ord("D") or suffix != b"\r\n":
        raise RuntimeError(f"完成帧错误：{frame!r}")
    result = {
        "success": bool(success),
        "error_code": error_code,
        "stage": stage,
        "stage_name": STAGE_NAMES.get(stage, f"UNKNOWN_0x{stage:02x}"),
        "cycles": cycles,
        "seconds_at_100mhz": cycles / 100_000_000.0,
    }
    if not success:
        raise RuntimeError(f"完整 Block 执行失败：{result}")
    return result


def upload_many(
    port: "serial.Serial",
    uploads: Sequence[DDRUpload],
    *,
    baud: int,
    timeout: float,
    verify: bool,
) -> float:
    total_start = time.perf_counter()
    total_bytes = sum(len(item.payload) for item in uploads)
    for index, upload in enumerate(uploads, start=1):
        elapsed = write_region(port, upload, baud=baud, timeout=timeout)
        print(
            f"WRITE {index}/{len(uploads)} {upload.name}: "
            f"addr=0x{upload.controller_address:07x}, {len(upload.payload)} B, "
            f"{elapsed:.3f}s, sha256={upload.sha256}"
        )
        if verify:
            actual = read_region(
                port,
                upload.controller_address,
                len(upload.payload),
                baud=baud,
                timeout=timeout,
            )
            if actual != upload.payload:
                raise AssertionError(
                    f"{upload.name} DDR3 回读不一致："
                    f"actual={hashlib.sha256(actual).hexdigest()}, "
                    f"expected={upload.sha256}"
                )
    elapsed_total = time.perf_counter() - total_start
    print(
        f"UPLOAD PASS: {len(uploads)} transactions, {total_bytes} B, "
        f"elapsed={elapsed_total:.3f}s, verify={verify}"
    )
    return elapsed_total


def _expected_tensors(case: TransformerBlockCase) -> tuple[tuple[str, np.ndarray, str], ...]:
    return (
        ("input_norm_q10", case.input_norm_q10, "<i2"),
        ("q_q28", case.current_q_q28, "<i8"),
        ("k_q28", case.current_k_q28, "<i8"),
        ("q_rope_q28", case.current_q_rope_q28, "<i8"),
        ("k_rope_q28", case.current_k_rope_q28, "<i8"),
        ("v_q28", case.current_v_q28, "<i8"),
        ("scores_q28", case.scores_q28, "<i8"),
        ("probabilities_q31", case.probabilities_q31, "<u4"),
        ("attention_concat_q28", case.attention_concat_q28, "<i8"),
        ("oproj_q28", case.oproj_q28, "<i8"),
        ("attention_residual_q10", case.first_residual_q10, "<i2"),
        ("post_attention_norm_q10", case.post_attention_norm_q10, "<i2"),
        ("gate_q28", case.gate_q28, "<i8"),
        ("up_q28", case.up_q28, "<i8"),
        ("silu_gate_q10", case.silu_gate_q10, "<i2"),
        ("silu_up_q28", case.silu_up_q28, "<i8"),
        ("down_proj_q28", case.down_proj_q28, "<i8"),
        ("block_output_q10", case.output_q10, "<i2"),
    )


def compare_tensor_bytes(
    name: str,
    actual: bytes,
    expected_array: np.ndarray,
    dtype: str,
) -> None:
    expected = np.asarray(expected_array, dtype=dtype).tobytes(order="C")
    if actual == expected:
        return
    actual_values = np.frombuffer(actual, dtype=dtype)
    expected_values = np.frombuffer(expected, dtype=dtype)
    mismatch = np.flatnonzero(actual_values != expected_values)
    if mismatch.size:
        index = int(mismatch[0])
        raise AssertionError(
            f"{name} 首个不匹配 index={index}: "
            f"actual={int(actual_values[index])}, expected={int(expected_values[index])}; "
            f"actual_sha256={hashlib.sha256(actual).hexdigest()}, "
            f"expected_sha256={hashlib.sha256(expected).hexdigest()}"
        )
    raise AssertionError(f"{name} 字节不同但未定位到元素")


def verify_case_tensors(
    port: "serial.Serial",
    case: TransformerBlockCase,
    *,
    baud: int,
    timeout: float,
    final_only: bool,
) -> None:
    regions = {region.name: region for region in scratch_regions()}
    tensors = _expected_tensors(case)
    if final_only:
        tensors = tensors[-1:]
    for index, (name, expected, dtype) in enumerate(tensors, start=1):
        raw_expected = np.asarray(expected, dtype=dtype).tobytes(order="C")
        if len(raw_expected) % BEAT_BYTES:
            raise AssertionError(f"{name} 期望长度未按 32 B 对齐")
        region = regions[name]
        actual = read_region(
            port,
            region.controller_address,
            len(raw_expected),
            baud=baud,
            timeout=timeout,
        )
        compare_tensor_bytes(name, actual, expected, dtype)
        print(
            f"COMPARE {index}/{len(tensors)} {name}: {len(actual)} B, "
            f"sha256={hashlib.sha256(actual).hexdigest()} PASS"
        )


def run_fixed_case(
    port: "serial.Serial",
    case: TransformerBlockCase,
    *,
    baud: int,
    timeout: float,
    final_only: bool,
    layer_index: int = 0,
) -> float:
    configure_case(port, case, timeout, layer_index=layer_index)
    upload_many(
        port,
        build_dynamic_uploads(case, layer_index=layer_index),
        baud=baud,
        timeout=timeout,
        verify=False,
    )
    commit_payload(port, timeout)
    started = time.perf_counter()
    completion = execute_block(port, max(timeout, 120.0))
    elapsed = time.perf_counter() - started
    print(f"EXECUTE {case.label}: {completion}, wall={elapsed:.3f}s")
    verify_case_tensors(
        port,
        case,
        baud=baud,
        timeout=timeout,
        final_only=final_only,
    )
    print(
        f"CASE PASS query/count={case.query_position}/{case.count}, "
        f"output_sha256={hashlib.sha256(np.asarray(case.output_q10, dtype='<i2').tobytes()).hexdigest()}"
    )
    return elapsed


def resolve_case(cases: Sequence[TransformerBlockCase], value: str) -> TransformerBlockCase:
    if value.isdigit():
        index = int(value)
        if 0 <= index < len(cases):
            return cases[index]
    for case in cases:
        if str(case.query_position) == value:
            return case
    raise SystemExit("固定用例可选 index=0..3 或 query=0/1/5/15")


def print_ports() -> None:
    require_pyserial()
    ports = list(list_ports.comports())
    if not ports:
        print("未发现串口")
        return
    for item in ports:
        print(f"{item.device}: {item.description} [{item.hwid}]")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="G2 完整 layer0 Block FPGA 逐位验证")
    parser.add_argument("--port", default=DEFAULT_PORT)
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("ports")
    sub.add_parser("info")
    sub.add_parser("status")

    resident = sub.add_parser("resident")
    resident.add_argument("--verify", action="store_true")

    layer_params = sub.add_parser(
        "layer-params",
        help="把指定真实 layer0..23 的 19 笔参数换入当前 slot A",
    )
    layer_params.add_argument("layer", type=int)
    layer_params.add_argument("--verify", action="store_true")

    case_parser = sub.add_parser("case")
    case_parser.add_argument("case")
    case_parser.add_argument("--load-resident", action="store_true")
    case_parser.add_argument("--verify-resident", action="store_true")
    case_parser.add_argument("--final-only", action="store_true")

    all_parser = sub.add_parser("all")
    all_parser.add_argument("--skip-resident", action="store_true")
    all_parser.add_argument("--verify-resident", action="store_true")
    all_parser.add_argument("--final-only", action="store_true")

    stress = sub.add_parser("stress")
    stress.add_argument("--rounds", type=int, default=100)
    stress.add_argument("--seed", type=int, default=20260820)
    stress.add_argument("--start-index", type=int, default=0)
    stress.add_argument("--skip-resident", action="store_true")
    stress.add_argument("--verify-resident", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "ports":
        print_ports()
        return 0

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

        if args.command == "resident":
            upload_many(
                port,
                build_resident_uploads(image_path=args.image),
                baud=args.baud,
                timeout=args.timeout,
                verify=args.verify,
            )
            return 0

        if args.command == "layer-params":
            upload_many(
                port,
                build_layer_parameter_uploads(
                    args.layer,
                    image_path=args.image,
                ),
                baud=args.baud,
                timeout=args.timeout,
                verify=args.verify,
            )
            print(f"LAYER PARAMS PASS: layer={args.layer}, slot=A")
            return 0

        cases = build_fixed_real_cases(image_path=args.image)
        if args.command == "case":
            if args.load_resident:
                upload_many(
                    port,
                    build_resident_uploads(image_path=args.image),
                    baud=args.baud,
                    timeout=args.timeout,
                    verify=args.verify_resident,
                )
            run_fixed_case(
                port,
                resolve_case(cases, args.case),
                baud=args.baud,
                timeout=args.timeout,
                final_only=args.final_only,
            )
            return 0

        if args.command == "all":
            if not args.skip_resident:
                upload_many(
                    port,
                    build_resident_uploads(image_path=args.image),
                    baud=args.baud,
                    timeout=args.timeout,
                    verify=args.verify_resident,
                )
            started = time.perf_counter()
            for case in cases:
                run_fixed_case(
                    port,
                    case,
                    baud=args.baud,
                    timeout=args.timeout,
                    final_only=args.final_only,
                )
            print(
                f"完整 Block 固定用例：{len(cases)}/{len(cases)} PASS, "
                f"total={time.perf_counter() - started:.3f}s"
            )
            return 0

        if args.command == "stress":
            if args.rounds <= 0:
                raise SystemExit("--rounds 必须大于 0")
            if args.start_index < 0:
                raise SystemExit("--start-index 不能为负")
            context = load_context(args.image)
            if not args.skip_resident:
                upload_many(
                    port,
                    build_resident_uploads(context),
                    baud=args.baud,
                    timeout=args.timeout,
                    verify=args.verify_resident,
                )
            started = time.perf_counter()
            for local_index in range(args.rounds):
                global_index = args.start_index + local_index
                case = build_stress_case(
                    context,
                    seed=args.seed,
                    index=global_index,
                )
                run_fixed_case(
                    port,
                    case,
                    baud=args.baud,
                    timeout=args.timeout,
                    final_only=True,
                )
                print(
                    f"STRESS {local_index + 1}/{args.rounds} PASS: "
                    f"global_index={global_index}, seed={args.seed}, "
                    f"query/window/count={case.query_position}/"
                    f"{case.window_start}/{case.count}"
                )
            print(
                f"完整 Block 随机/边界：{args.rounds}/{args.rounds} PASS, "
                f"seed={args.seed}, start_index={args.start_index}, "
                f"total={time.perf_counter() - started:.3f}s"
            )
            return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
