#!/usr/bin/env python3
"""PGL50H layer0 MLP down_proj 软件自检与真实上板验证工具。"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import serial  # type: ignore
    from serial.tools import list_ports  # type: ignore
except ImportError:  # pragma: no cover - 仅无串口依赖环境
    serial = None
    list_ports = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_TOOLS = PROJECT_ROOT / "model_tools"
if str(MODEL_TOOLS) not in sys.path:
    sys.path.insert(0, str(MODEL_TOOLS))

from mlp_down_proj_reference import (  # noqa: E402
    DEFAULT_IMAGE,
    DEFAULT_MANIFEST,
    DEFAULT_STRESS_SEED,
    RESULT_BYTES,
    UPLOAD_BYTES,
    DownProjectionCase,
    build_fixed_real_cases,
    build_upload_payload,
    case_from_source_q28,
    load_down_projection_model,
    make_random_source_q28,
    software_stress,
    validate_manifest,
    verify_upload_payload,
)
from p50_format import P50Image  # noqa: E402

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

EXPECTED_INFO = "PANGU50K MLP DOWN V1"
BAUDRATE = 115200


@dataclass(frozen=True)
class FpgaStatus:
    ddr_init_done: bool
    loaded: bool
    result_valid: bool
    core_busy: bool


def require_pyserial() -> None:
    if serial is None:
        raise SystemExit("缺少 pyserial，请运行：python -m pip install pyserial")


def sha256_q28(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<i8").tobytes(order="C")).hexdigest()


def read_exact(port: "serial.Serial", size: int, timeout: float = 300.0) -> bytes:
    deadline = time.monotonic() + timeout
    output = bytearray()
    while len(output) < size:
        chunk = port.read(size - len(output))
        if chunk:
            output.extend(chunk)
            continue
        if time.monotonic() >= deadline:
            raise TimeoutError(f"串口读取超时：收到 {len(output)}/{size} B")
    return bytes(output)


def raise_if_error_frame(frame: bytes) -> None:
    if frame[:1] == b"E":
        code = frame[1] if len(frame) > 1 else -1
        raise RuntimeError(f"FPGA 返回错误码 0x{code:02x}")


def read_ack(port: "serial.Serial") -> None:
    frame = read_exact(port, 3, timeout=300.0)
    raise_if_error_frame(frame)
    if frame != b"K\r\n":
        raise RuntimeError(f"载荷 ACK 错误：{frame!r}")


def open_port(name: str) -> "serial.Serial":
    require_pyserial()
    port = serial.Serial(
        name,
        BAUDRATE,
        bytesize=serial.EIGHTBITS,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        timeout=0.1,
        write_timeout=300.0,
    )
    port.reset_input_buffer()
    port.reset_output_buffer()
    return port


def show_ports() -> None:
    require_pyserial()
    ports = list(list_ports.comports())
    if not ports:
        print("未发现串口")
        return
    for item in ports:
        print(f"{item.device}: {item.description}")


def command_info(port: "serial.Serial") -> str:
    port.reset_input_buffer()
    port.write(b"I")
    frame = port.read_until(b"\n", 64)
    raise_if_error_frame(frame)
    text = frame.decode("ascii", errors="replace").strip()
    if text != EXPECTED_INFO:
        raise RuntimeError(f"固件标识错误：{text!r} != {EXPECTED_INFO!r}")
    print(text)
    return text


def command_status(port: "serial.Serial") -> FpgaStatus:
    port.reset_input_buffer()
    port.write(b"S")
    frame = read_exact(port, 4, timeout=5.0)
    raise_if_error_frame(frame)
    if frame[0:1] != b"S" or frame[2:] != b"\r\n":
        raise RuntimeError(f"状态帧格式错误：{frame!r}")
    flags = frame[1]
    status = FpgaStatus(
        ddr_init_done=bool(flags & 0x01),
        loaded=bool(flags & 0x02),
        result_valid=bool(flags & 0x04),
        core_busy=bool(flags & 0x08),
    )
    print(
        "DDR3初始化={} loaded={} result_valid={} core_busy={}".format(
            int(status.ddr_init_done),
            int(status.loaded),
            int(status.result_valid),
            int(status.core_busy),
        )
    )
    return status


def wait_until_ready(port: "serial.Serial", timeout: float = 20.0) -> None:
    deadline = time.monotonic() + timeout
    while True:
        status = command_status(port)
        if status.ddr_init_done and not status.core_busy:
            return
        if time.monotonic() >= deadline:
            raise TimeoutError("等待 DDR3 初始化/核心空闲超时")
        time.sleep(0.25)


def write_all(port: "serial.Serial", payload: bytes, chunk_size: int = 4096) -> None:
    view = memoryview(payload)
    sent = 0
    while sent < len(view):
        end = min(sent + chunk_size, len(view))
        count = port.write(view[sent:end])
        if count is None or count <= 0:
            raise RuntimeError(f"串口写入停滞：{sent}/{len(view)} B")
        sent += count
    port.flush()


def load_case(port: "serial.Serial", case: DownProjectionCase) -> float:
    payload = build_upload_payload(case)
    if len(payload) != UPLOAD_BYTES:
        raise AssertionError(f"载荷长度错误：{len(payload)} != {UPLOAD_BYTES}")
    port.reset_input_buffer()
    start = time.perf_counter()
    port.write(b"L")
    write_all(port, payload)
    read_ack(port)
    return time.perf_counter() - start


def run_loaded_case(port: "serial.Serial") -> tuple[np.ndarray, float]:
    port.reset_input_buffer()
    start = time.perf_counter()
    port.write(b"G")
    reply = read_exact(port, 1 + RESULT_BYTES, timeout=120.0)
    elapsed = time.perf_counter() - start
    raise_if_error_frame(reply)
    if reply[:1] != b"R":
        raise RuntimeError(f"结果前缀错误：{reply[:16]!r}")
    output = np.frombuffer(reply[1:], dtype="<i8").copy()
    return output, elapsed


def run_and_compare(
    port: "serial.Serial", case: DownProjectionCase
) -> tuple[np.ndarray, float, float]:
    load_seconds = load_case(port, case)
    fpga, run_seconds = run_loaded_case(port)
    if not np.array_equal(fpga, case.expected_q28):
        mismatch = np.flatnonzero(fpga != case.expected_q28)
        first = int(mismatch[0])
        raise AssertionError(
            f"down_proj 不一致：{mismatch.size}/896，首个 row={first}，"
            f"FPGA={int(fpga[first])}，Python={int(case.expected_q28[first])}"
        )
    return fpga, load_seconds, run_seconds


def command_fixed(
    port: "serial.Serial", image_path: Path, manifest_path: Path, case_index: int
) -> None:
    cases = build_fixed_real_cases(image_path=image_path)
    manifest = validate_manifest(cases, manifest_path)
    if not 0 <= case_index < len(cases):
        raise ValueError(f"case 必须位于 0..{len(cases)-1}")
    case = cases[case_index]
    expected_hash = manifest["cases"][case_index]["sha256"]["output_fixed_q28"]
    verify_upload_payload(case)
    fpga, load_seconds, run_seconds = run_and_compare(port, case)
    actual_hash = sha256_q28(fpga)
    if actual_hash != expected_hash:
        raise AssertionError(f"输出 SHA256 错误：{actual_hash} != {expected_hash}")
    print(
        f"固定 case={case_index} query={case.query_position} count={case.count}："
        f"896/896 PASS"
    )
    print(f"输出 SHA256：{actual_hash}")
    print(f"上传={load_seconds:.2f}s，计算+回读={run_seconds:.2f}s")


def command_stress(
    port: "serial.Serial",
    image_path: Path,
    rounds: int,
    seed: int,
    start_index: int,
) -> None:
    if rounds <= 0 or start_index < 0:
        raise ValueError("rounds 必须大于 0，start-index 必须非负")
    image = P50Image(image_path)
    image.validate()
    model = load_down_projection_model(image)
    rng = np.random.default_rng(seed)
    total_load = 0.0
    total_run = 0.0
    for index in range(start_index + rounds):
        source = make_random_source_q28(rng, index)
        if index < start_index:
            continue
        case = case_from_source_q28(
            model,
            source,
            label=f"FPGA down_proj stress index={index} mode={index % 8}",
        )
        _, load_seconds, run_seconds = run_and_compare(port, case)
        total_load += load_seconds
        total_run += run_seconds
        print(
            f"stress {index-start_index+1}/{rounds} global_index={index} "
            f"mode={index % 8} PASS，上传={load_seconds:.2f}s，运行={run_seconds:.2f}s"
        )
    print(
        f"真实 FPGA down_proj 压力 PASS：{rounds}/{rounds}，seed={seed}，"
        f"start_index={start_index}，上传合计={total_load:.2f}s，运行合计={total_run:.2f}s"
    )


def command_selftest(image_path: Path, manifest_path: Path, rounds: int, seed: int) -> None:
    start = time.perf_counter()
    cases = build_fixed_real_cases(image_path=image_path)
    manifest = validate_manifest(cases, manifest_path)
    for case in cases:
        verify_upload_payload(case)
    software_stress(image_path=image_path, rounds=rounds, seed=seed)
    print(f"固定真实清单：{len(cases)}/{len(cases)} PASS")
    print(f"上传载荷：{UPLOAD_BYTES} B，结果：{RESULT_BYTES} B")
    for item in manifest["cases"]:
        print(
            f"query={item['query_position']} count={item['count']} "
            f"output={item['sha256']['output_fixed_q28']}"
        )
    print(f"软件随机/边界：{rounds}/{rounds} PASS，seed={seed}")
    print(f"总耗时：{time.perf_counter()-start:.2f}s")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PGL50H layer0 MLP down_proj 上位机")
    parser.add_argument("--port", default="COM20", help="串口，例如 COM20")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("ports", help="列出串口")
    sub.add_parser("info", help="读取固件标识")
    sub.add_parser("status", help="读取固件状态")

    selftest = sub.add_parser("selftest", help="仅运行软件固定清单和随机/边界自检")
    selftest.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    selftest.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    selftest.add_argument("--rounds", type=int, default=1000)
    selftest.add_argument("--seed", type=int, default=DEFAULT_STRESS_SEED)

    fixed = sub.add_parser("fixed", help="运行一组真实固定 down_proj")
    fixed.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    fixed.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    fixed.add_argument("--case", type=int, default=0)

    stress = sub.add_parser("stress", help="运行随机/边界真实上板回归")
    stress.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    stress.add_argument("--rounds", type=int, default=1)
    stress.add_argument("--seed", type=int, default=DEFAULT_STRESS_SEED)
    stress.add_argument("--start-index", type=int, default=0)
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
                wait_until_ready(port)
                command_info(port)
                command_fixed(port, args.image, args.manifest, args.case)
            elif args.command == "stress":
                wait_until_ready(port)
                command_info(port)
                command_stress(
                    port,
                    args.image,
                    args.rounds,
                    args.seed,
                    args.start_index,
                )
            else:  # pragma: no cover
                raise AssertionError(args.command)
        return 0
    except (
        AssertionError,
        FileNotFoundError,
        KeyError,
        OSError,
        RuntimeError,
        TimeoutError,
        ValueError,
    ) as error:
        print(f"错误：{error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
