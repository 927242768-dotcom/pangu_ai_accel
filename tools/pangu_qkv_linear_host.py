#!/usr/bin/env python3
"""盘古 PGL50H layer0 Q/K/V 真实 Linear 统一验证工具。"""

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

from model_tools.linear_quant_reference import (  # noqa: E402
    DEFAULT_ACTIVATION_SEED,
    make_deterministic_activation,
)
from model_tools.p50_format import P50Image  # noqa: E402
from model_tools.qkv_linear_reference import (  # noqa: E402
    DEFAULT_IMAGE,
    DEFAULT_MANIFEST,
    ProjectionCase,
    ProjectionModel,
    ProjectionSpec,
    build_qkv_cases,
    build_upload_payload,
    case_from_model,
    case_hashes,
    load_qkv_models,
    projection_sequence,
    reshape_heads,
    validate_gqa_layout,
    validate_manifest,
)

BAUD_RATE = 115200
DEFAULT_STRESS_SEED = 20260729

ERROR_MESSAGES = {
    0x01: "未知命令",
    0x02: "DDR3 尚未初始化完成",
    0x04: "尚未加载当前投影数据",
    0x05: "投影类型非法",
    0xFF: "FPGA 状态机异常",
}


class FpgaStatus:
    def __init__(self, flags: int):
        self.ddr_ready = bool(flags & 0x01)
        self.data_loaded = bool(flags & 0x02)
        self.result_valid = bool(flags & 0x04)
        self.core_busy = bool(flags & 0x08)
        self.projection_selector = (flags >> 4) & 0x03

    @property
    def projection_key(self) -> str:
        return {0: "q", 1: "k", 2: "v"}.get(self.projection_selector, "?")


def require_pyserial() -> None:
    if serial is None:
        raise SystemExit("缺少 pyserial，请运行：python -m pip install pyserial")


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
    status = FpgaStatus(reply[1])
    print(
        "DDR3初始化={}，数据已加载={}，结果有效={}，核心忙={}，当前投影={}".format(
            "是" if status.ddr_ready else "否",
            "是" if status.data_loaded else "否",
            "是" if status.result_valid else "否",
            "是" if status.core_busy else "否",
            status.projection_key.upper(),
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


def select_projection(port: "serial.Serial", spec: ProjectionSpec) -> None:
    port.write(spec.command)
    read_ack(port)


def write_all(port: "serial.Serial", payload: bytes, chunk_size: int = 4096) -> None:
    sent = 0
    next_report = 64 * 1024
    view = memoryview(payload)
    while sent < len(payload):
        end = min(sent + chunk_size, len(payload))
        count = port.write(view[sent:end])
        if not count:
            raise TimeoutError("串口写入没有前进")
        sent += count
        if sent >= next_report or sent == len(payload):
            print(f"上传进度：{sent}/{len(payload)} B ({sent * 100 / len(payload):.1f}%)")
            next_report += 64 * 1024
    port.flush()


def load_case(port: "serial.Serial", case: ProjectionCase) -> None:
    select_projection(port, case.spec)
    payload = build_upload_payload(case)
    port.write(b"L")
    write_all(port, payload)
    read_ack(port)


def run_loaded_case(port: "serial.Serial", spec: ProjectionSpec) -> np.ndarray:
    port.write(b"G")
    reply = read_exact(port, 1 + spec.result_bytes, timeout=120.0)
    raise_if_error_frame(reply)
    if reply[0:1] != b"R":
        raise RuntimeError(f"Q28 结果帧头错误：{reply[:16]!r}")
    return np.frombuffer(reply[1:], dtype="<i8").copy()


def run_and_compare(port: "serial.Serial", case: ProjectionCase) -> np.ndarray:
    load_case(port, case)
    fpga = run_loaded_case(port, case.spec)
    if not np.array_equal(fpga, case.expected_q28):
        mismatch = np.flatnonzero(fpga != case.expected_q28)
        first = int(mismatch[0])
        raise RuntimeError(
            f"{case.label} 不一致：首个错误行={first}，"
            f"FPGA={int(fpga[first])}，Python={int(case.expected_q28[first])}，"
            f"总错误数={mismatch.size}"
        )
    heads = reshape_heads(fpga, case.spec)
    if not np.array_equal(heads.reshape(-1), fpga):
        raise RuntimeError(f"{case.spec.key}_proj GQA head 布局还原失败")
    return fpga


def load_models(image_path: Path) -> tuple[P50Image, dict[str, ProjectionModel]]:
    image = P50Image(image_path)
    image.validate()
    return image, load_qkv_models(image)


def fixed_cases(
    models: dict[str, ProjectionModel], manifest_path: Path
) -> dict[str, ProjectionCase]:
    cases = build_qkv_cases(models, activation_seed=DEFAULT_ACTIVATION_SEED)
    validate_manifest(cases, manifest_path, DEFAULT_ACTIVATION_SEED)
    return cases


def command_fixed(
    port: "serial.Serial",
    image_path: Path,
    manifest_path: Path,
    projection: str,
) -> None:
    wait_until_ready(port)
    _, models = load_models(image_path)
    cases = fixed_cases(models, manifest_path)
    for spec in projection_sequence(projection):
        case = cases[spec.key]
        started = time.monotonic()
        fpga = run_and_compare(port, case)
        elapsed = time.monotonic() - started
        hashes = case_hashes(case)
        print(
            f"真实 layer0 {spec.key}_proj M{spec.rows}K896 全输出逐位一致：PASS"
        )
        print(f"输出 SHA256：{hashes['output_fixed_q28']}")
        print(f"head shape：{reshape_heads(fpga, spec).shape}")
        print(f"前 8 行：{fpga[:8].tolist()}")
        print(f"后 8 行：{fpga[-8:].tolist()}")
        print(f"上传、计算与回读总耗时：{elapsed:.2f} 秒")


def command_stress(
    port: "serial.Serial",
    image_path: Path,
    projection: str,
    rounds: int,
    seed: int,
) -> None:
    if rounds <= 0:
        raise ValueError("rounds 必须大于 0")
    wait_until_ready(port)
    _, models = load_models(image_path)
    selected = projection_sequence(projection)
    started = time.monotonic()
    for index in range(rounds):
        activation_seed = seed + index
        activation = make_deterministic_activation(896, seed=activation_seed)
        round_cases: dict[str, ProjectionCase] = {}
        for spec in selected:
            case = case_from_model(
                models[spec.key],
                activation_values=activation,
                activation_seed=activation_seed,
                label=f"{spec.key}_proj 随机 hidden state {index + 1}/{rounds}",
            )
            round_cases[spec.key] = case
            run_and_compare(port, case)
        if projection == "all":
            validate_gqa_layout(round_cases)
        print(
            f"真实 QKV 随机 hidden state 已通过 {index + 1}/{rounds}，"
            f"activation_seed={activation_seed}，projection={projection}"
        )
    elapsed = time.monotonic() - started
    print(
        f"QKV 真实 FPGA 随机回归 PASS：{rounds}/{rounds}，"
        f"projection={projection}，seed_start={seed}，耗时 {elapsed:.2f} 秒"
    )


def command_selftest(
    image_path: Path,
    manifest_path: Path,
    projection: str,
    rounds: int,
    seed: int,
) -> None:
    if rounds <= 0:
        raise ValueError("rounds 必须大于 0")
    _, models = load_models(image_path)
    selected = projection_sequence(projection)
    fixed = fixed_cases(models, manifest_path)
    print("Q/K/V 固定清单、载荷往返、独立 Q28 重算和 GQA 布局：PASS")
    for spec in selected:
        hashes = case_hashes(fixed[spec.key])
        print(
            f"{spec.key}_proj: M={spec.rows}，heads={spec.heads}，"
            f"upload={spec.upload_bytes} B，output_sha256={hashes['output_fixed_q28']}"
        )

    started = time.monotonic()
    for index in range(rounds):
        activation_seed = seed + index
        activation = make_deterministic_activation(896, seed=activation_seed)
        round_cases: dict[str, ProjectionCase] = {}
        for spec in selected:
            round_cases[spec.key] = case_from_model(
                models[spec.key],
                activation_values=activation,
                activation_seed=activation_seed,
            )
        if projection == "all":
            validate_gqa_layout(round_cases)
        if index == 0 or index + 1 == rounds or (index + 1) % 100 == 0:
            print(
                f"QKV 软件随机 hidden state 已通过 {index + 1}/{rounds}，"
                f"activation_seed={activation_seed}，projection={projection}"
            )
    elapsed = time.monotonic() - started
    print(
        f"QKV 软件参考压力测试 PASS：{rounds}/{rounds}，"
        f"projection={projection}，seed_start={seed}，耗时 {elapsed:.2f} 秒"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PGL50H layer0 Q/K/V 分组 Q28 Linear 统一上位机"
    )
    parser.add_argument("--port", default="COM20")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("ports", help="列出串口")
    sub.add_parser("info", help="读取固件信息")
    sub.add_parser("status", help="读取状态")

    fixed = sub.add_parser("fixed", help="运行真实 layer0 Q/K/V 固定向量")
    fixed.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    fixed.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    fixed.add_argument("--projection", choices=["q", "k", "v", "all"], default="all")

    stress = sub.add_parser("stress", help="运行真实随机 hidden state 上板回归")
    stress.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    stress.add_argument("--projection", choices=["q", "k", "v", "all"], default="all")
    stress.add_argument("--rounds", type=int, default=3)
    stress.add_argument("--seed", type=int, default=DEFAULT_STRESS_SEED)

    selftest = sub.add_parser("selftest", help="只运行 Q/K/V 软件参考与载荷自检")
    selftest.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    selftest.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    selftest.add_argument("--projection", choices=["q", "k", "v", "all"], default="all")
    selftest.add_argument("--rounds", type=int, default=100)
    selftest.add_argument("--seed", type=int, default=DEFAULT_STRESS_SEED)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.command == "ports":
            show_ports()
            return 0
        if args.command == "selftest":
            command_selftest(
                args.image,
                args.manifest,
                args.projection,
                args.rounds,
                args.seed,
            )
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
                    args.projection,
                )
            elif args.command == "stress":
                command_stress(
                    port,
                    args.image,
                    args.projection,
                    args.rounds,
                    args.seed,
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
