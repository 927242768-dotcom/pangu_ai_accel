#!/usr/bin/env python3
"""盘古 PGL50H layer0 MLP gate_proj/up_proj 双投影验证工具。"""

from __future__ import annotations

import argparse
import hashlib
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

from model_tools.mlp_gate_up_reference import (  # noqa: E402
    DEFAULT_IMAGE,
    DEFAULT_MANIFEST,
    RESULT_BYTES,
    UPLOAD_BYTES,
    GateUpCase,
    ProjectionCase,
    build_fixed_real_cases,
    build_projection_payload,
    case_from_post_attention_q10,
    load_gate_up_models,
    validate_manifest,
)
from model_tools.p50_format import P50Image  # noqa: E402
from model_tools.post_attention_layernorm_reference import (  # noqa: E402
    make_random_input_q10,
)

BAUD_RATE = 115200
DEFAULT_STRESS_SEED = 20260808
EXPECTED_INFO = "PANGU50K MLP GATEUP V1"

ERROR_MESSAGES = {
    0x01: "未知命令",
    0x02: "DDR3 尚未初始化完成",
    0x04: "尚未加载完整投影数据",
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


def sha256_q28(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<i8").tobytes(order="C")).hexdigest()


def read_exact(port: "serial.Serial", size: int, timeout: float = 120.0) -> bytes:
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
        write_timeout=300.0,
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
    if text != EXPECTED_INFO:
        raise RuntimeError(f"固件标识错误：{text!r} != {EXPECTED_INFO!r}")
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


def wait_until_ready(port: "serial.Serial", timeout: float = 20.0) -> None:
    deadline = time.monotonic() + timeout
    while True:
        status = command_status(port)
        if status.ddr_ready:
            return
        if time.monotonic() >= deadline:
            raise TimeoutError("等待 DDR3 初始化完成超时")
        time.sleep(0.25)


def write_all(port: "serial.Serial", payload: bytes, chunk_size: int = 4096) -> None:
    sent = 0
    next_report = 256 * 1024
    view = memoryview(payload)
    while sent < len(payload):
        end = min(sent + chunk_size, len(payload))
        count = port.write(view[sent:end])
        if not count:
            raise TimeoutError("串口写入没有前进")
        sent += count
        if sent >= next_report or sent == len(payload):
            print(f"上传进度：{sent}/{len(payload)} B ({sent * 100 / len(payload):.1f}%)")
            next_report += 256 * 1024
    port.flush()


def load_projection(port: "serial.Serial", projection: ProjectionCase) -> None:
    payload = build_projection_payload(projection)
    if len(payload) != UPLOAD_BYTES:
        raise AssertionError(f"载荷长度错误：{len(payload)} != {UPLOAD_BYTES}")
    port.write(b"L")
    write_all(port, payload)
    read_ack(port)


def run_loaded_projection(port: "serial.Serial") -> np.ndarray:
    port.write(b"G")
    reply = read_exact(port, 1 + RESULT_BYTES, timeout=180.0)
    raise_if_error_frame(reply)
    if reply[0:1] != b"R":
        raise RuntimeError(f"Q28 结果帧头错误：{reply[:16]!r}")
    return np.frombuffer(reply[1:], dtype="<i8").copy()


def run_and_compare(
    port: "serial.Serial", projection: ProjectionCase, *, label: str
) -> np.ndarray:
    load_projection(port, projection)
    fpga = run_loaded_projection(port)
    if not np.array_equal(fpga, projection.expected_q28):
        mismatch = np.flatnonzero(fpga != projection.expected_q28)
        first = int(mismatch[0])
        raise RuntimeError(
            f"{label} 不一致：首个错误行={first}，FPGA={int(fpga[first])}，"
            f"Python={int(projection.expected_q28[first])}，总错误数={mismatch.size}"
        )
    return fpga


def select_projection(case: GateUpCase, name: str) -> ProjectionCase:
    if name == "gate":
        return case.gate
    if name == "up":
        return case.up
    raise ValueError(f"未知投影：{name}")


def command_fixed(
    port: "serial.Serial",
    image_path: Path,
    manifest_path: Path,
    case_index: int,
    projection_name: str,
) -> None:
    cases = build_fixed_real_cases(image_path=image_path)
    validate_manifest(cases, manifest_path)
    if not 0 <= case_index < len(cases):
        raise ValueError(f"case 必须位于 0..{len(cases) - 1}")
    case = cases[case_index]
    projection = select_projection(case, projection_name)
    wait_until_ready(port)
    started = time.monotonic()
    fpga = run_and_compare(
        port,
        projection,
        label=f"case{case_index} {projection_name}_proj",
    )
    elapsed = time.monotonic() - started
    print(
        f"真实 layer0 MLP {projection_name}_proj case{case_index} "
        f"4864/4864 逐位一致：PASS"
    )
    print(f"query={case.query_position}，count={case.count}")
    print(f"输出 SHA256：{sha256_q28(fpga)}")
    print(f"前 8 行：{fpga[:8].tolist()}")
    print(f"后 8 行：{fpga[-8:].tolist()}")
    print(f"上传、计算与回读总耗时：{elapsed:.2f} 秒")


def command_stress(
    port: "serial.Serial",
    image_path: Path,
    projection_name: str,
    rounds: int,
    seed: int,
    start_index: int,
) -> None:
    if rounds <= 0:
        raise ValueError("rounds 必须大于 0")
    if start_index < 0:
        raise ValueError("start-index 不能为负数")
    image = P50Image(image_path)
    image.validate()
    gate_model, up_model = load_gate_up_models(image)
    rng = np.random.default_rng(seed)
    wait_until_ready(port)
    started = time.monotonic()
    for index in range(rounds):
        mode_index = start_index + index
        source = make_random_input_q10(rng, mode_index)
        case = case_from_post_attention_q10(
            gate_model,
            up_model,
            source,
            label=f"FPGA stress mode={mode_index} {index + 1}/{rounds}",
        )
        projection = select_projection(case, projection_name)
        run_and_compare(
            port,
            projection,
            label=(
                f"{projection_name}_proj stress mode={mode_index} "
                f"{index + 1}/{rounds}"
            ),
        )
        print(
            f"真实 FPGA {projection_name}_proj 随机/边界已通过 "
            f"{index + 1}/{rounds}，mode_index={mode_index}"
        )
    elapsed = time.monotonic() - started
    print(
        f"真实 FPGA {projection_name}_proj 随机/边界 PASS：{rounds}/{rounds}，"
        f"seed={seed}，start_index={start_index}，耗时 {elapsed:.2f} 秒"
    )


def command_selftest(
    image_path: Path, manifest_path: Path, rounds: int, seed: int
) -> None:
    cases = build_fixed_real_cases(image_path=image_path)
    manifest = validate_manifest(cases, manifest_path)
    print("MLP gate/up 四组真实固定清单和载荷往返：PASS")
    print(f"每路上传载荷：{UPLOAD_BYTES} B，结果：{RESULT_BYTES} B")
    for index, case in enumerate(cases):
        print(
            f"case{index} query={case.query_position} count={case.count} "
            f"gate={manifest['cases'][index]['gate']['sha256']['output_fixed_q28']} "
            f"up={manifest['cases'][index]['up']['sha256']['output_fixed_q28']}"
        )

    if rounds <= 0:
        return
    image = P50Image(image_path)
    image.validate()
    gate_model, up_model = load_gate_up_models(image)
    rng = np.random.default_rng(seed)
    for index in range(rounds):
        source = make_random_input_q10(rng, index)
        case_from_post_attention_q10(
            gate_model,
            up_model,
            source,
            label=f"selftest {index + 1}/{rounds}",
        )
    print(f"MLP gate/up 软件自检 PASS：{rounds}/{rounds}，seed={seed}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PGL50H layer0 MLP gate_proj/up_proj M4864K896 上位机"
    )
    parser.add_argument("--port", default="COM20")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("ports", help="列出串口")
    sub.add_parser("info", help="读取并校验固件标识")
    sub.add_parser("status", help="读取状态")

    fixed = sub.add_parser("fixed", help="运行一组真实固定投影")
    fixed.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    fixed.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    fixed.add_argument("--case", type=int, default=0)
    fixed.add_argument("--projection", choices=("gate", "up"), required=True)

    stress = sub.add_parser("stress", help="运行随机/边界真实上板回归")
    stress.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    stress.add_argument("--projection", choices=("gate", "up"), required=True)
    stress.add_argument("--rounds", type=int, default=1)
    stress.add_argument("--seed", type=int, default=DEFAULT_STRESS_SEED)
    stress.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="选择 make_random_input_q10 的起始模式索引",
    )

    selftest = sub.add_parser("selftest", help="只运行软件清单、载荷和随机自检")
    selftest.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    selftest.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    selftest.add_argument("--rounds", type=int, default=10)
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
                command_fixed(
                    port,
                    args.image,
                    args.manifest,
                    args.case,
                    args.projection,
                )
            elif args.command == "stress":
                command_stress(
                    port,
                    args.image,
                    args.projection,
                    args.rounds,
                    args.seed,
                    args.start_index,
                )
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
