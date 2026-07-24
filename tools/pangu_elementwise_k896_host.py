#!/usr/bin/env python3
"""盘古 PGL50H K=896 signed Q6.10 元素级算子验证工具。"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
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

from model_tools.elementwise_fixed_reference import (  # noqa: E402
    DEFAULT_LENGTH,
    DEFAULT_SCALE_Q10,
    DEFAULT_SEED,
    OP_ELEMENTWISE_MUL,
    OP_FIXED_SCALE,
    OP_RESIDUAL_ADD,
    OP_SILU,
    Q_MAX,
    Q_MIN,
    build_silu_pwl_endpoints,
    compute_elementwise_reference,
    elementwise_mul_q10,
    fixed_scale_q10,
    make_deterministic_q10_vectors,
    residual_add_q10,
    result_manifest,
    silu_pwl_q10,
)

BAUD_RATE = 115200
K = DEFAULT_LENGTH
DATA_BYTES = K * 2
PWL_ENTRIES = 65
PWL_PADDED_ENTRIES = 80
PWL_BYTES = PWL_PADDED_ENTRIES * 2
RESULT_BYTES = DATA_BYTES
UPLOAD_BYTES = DATA_BYTES + DATA_BYTES + PWL_BYTES
DEFAULT_MANIFEST = PROJECT_ROOT / "model_tools/elementwise_k896_reference.json"

OP_NAMES = {
    OP_RESIDUAL_ADD: "residual_add",
    OP_FIXED_SCALE: "fixed_scale",
    OP_ELEMENTWISE_MUL: "elementwise_mul",
    OP_SILU: "silu_pwl64",
}

ERROR_MESSAGES = {
    0x01: "未知命令",
    0x02: "DDR3 尚未初始化完成",
    0x03: "操作模式非法",
    0x04: "尚未加载元素级数据",
    0x05: "尚未配置操作模式",
    0xFF: "FPGA 状态机异常",
}


@dataclass(frozen=True)
class FpgaStatus:
    ddr_ready: bool
    data_loaded: bool
    result_valid: bool
    core_busy: bool
    configured: bool


@dataclass(frozen=True)
class ElementwiseCase:
    vector_a_q10: np.ndarray
    vector_b_q10: np.ndarray
    pwl_endpoints_q10: np.ndarray
    scale_q10: int
    expected: dict[int, np.ndarray]
    label: str


def require_pyserial() -> None:
    if serial is None:
        raise SystemExit("缺少 pyserial，请运行：python -m pip install pyserial")


def sha256_array(array: np.ndarray, dtype: str | np.dtype) -> str:
    return hashlib.sha256(np.asarray(array, dtype=dtype).tobytes(order="C")).hexdigest()


def _require_shape(array: np.ndarray, shape: tuple[int, ...], label: str) -> None:
    if array.shape != shape:
        raise ValueError(f"{label} 形状错误：{array.shape}，预期 {shape}")


def validate_case(case: ElementwiseCase) -> None:
    _require_shape(np.asarray(case.vector_a_q10), (K,), "vector_a_q10")
    _require_shape(np.asarray(case.vector_b_q10), (K,), "vector_b_q10")
    _require_shape(np.asarray(case.pwl_endpoints_q10), (PWL_ENTRIES,), "pwl_endpoints_q10")
    if not Q_MIN <= case.scale_q10 <= Q_MAX:
        raise ValueError("scale_q10 超出 signed int16")
    for op_mode, expected in case.expected.items():
        if op_mode not in OP_NAMES:
            raise ValueError(f"未知操作模式：{op_mode}")
        _require_shape(np.asarray(expected), (K,), f"expected[{op_mode}]")


def make_fixed_case() -> ElementwiseCase:
    vector_a, vector_b = make_deterministic_q10_vectors(K, DEFAULT_SEED)
    result = compute_elementwise_reference(
        vector_a_q10=vector_a,
        vector_b_q10=vector_b,
        scale_q10=DEFAULT_SCALE_Q10,
        seed=DEFAULT_SEED,
    )
    case = ElementwiseCase(
        vector_a_q10=vector_a.astype(np.int16),
        vector_b_q10=vector_b.astype(np.int16),
        pwl_endpoints_q10=build_silu_pwl_endpoints().astype(np.int16),
        scale_q10=DEFAULT_SCALE_Q10,
        expected={
            OP_RESIDUAL_ADD: result.residual_q10.astype(np.int16),
            OP_FIXED_SCALE: result.fixed_scale_q10.astype(np.int16),
            OP_ELEMENTWISE_MUL: result.elementwise_mul_q10.astype(np.int16),
            OP_SILU: result.silu_pwl_q10.astype(np.int16),
        },
        label="E2 K896 固定边界向量",
    )
    validate_case(case)
    return case


def make_random_case(seed: int) -> ElementwiseCase:
    rng = np.random.default_rng(seed)
    vector_a = rng.integers(Q_MIN, Q_MAX + 1, size=K, dtype=np.int32).astype(np.int16)
    vector_b = rng.integers(Q_MIN, Q_MAX + 1, size=K, dtype=np.int32).astype(np.int16)
    scale_q10 = int(rng.integers(Q_MIN, Q_MAX + 1))
    residual, _ = residual_add_q10(vector_a, vector_b)
    scaled, _ = fixed_scale_q10(vector_a, scale_q10)
    multiplied, _ = elementwise_mul_q10(vector_a, vector_b)
    case = ElementwiseCase(
        vector_a_q10=vector_a,
        vector_b_q10=vector_b,
        pwl_endpoints_q10=build_silu_pwl_endpoints().astype(np.int16),
        scale_q10=scale_q10,
        expected={
            OP_RESIDUAL_ADD: residual,
            OP_FIXED_SCALE: scaled,
            OP_ELEMENTWISE_MUL: multiplied,
            OP_SILU: silu_pwl_q10(vector_a),
        },
        label=f"E2 随机向量 seed={seed}",
    )
    validate_case(case)
    return case


def build_upload_payload(case: ElementwiseCase) -> bytes:
    validate_case(case)
    padded_pwl = np.zeros(PWL_PADDED_ENTRIES, dtype="<i2")
    padded_pwl[:PWL_ENTRIES] = np.asarray(case.pwl_endpoints_q10, dtype="<i2")
    payload = (
        np.asarray(case.vector_a_q10, dtype="<i2").tobytes(order="C")
        + np.asarray(case.vector_b_q10, dtype="<i2").tobytes(order="C")
        + padded_pwl.tobytes(order="C")
    )
    if len(payload) != UPLOAD_BYTES:
        raise AssertionError(f"上传载荷长度错误：{len(payload)} != {UPLOAD_BYTES}")
    return payload


def verify_payload_roundtrip(case: ElementwiseCase) -> str:
    payload = build_upload_payload(case)
    a_end = DATA_BYTES
    b_end = a_end + DATA_BYTES
    vector_a = np.frombuffer(payload[:a_end], dtype="<i2").copy()
    vector_b = np.frombuffer(payload[a_end:b_end], dtype="<i2").copy()
    pwl = np.frombuffer(payload[b_end:], dtype="<i2").copy()
    if not np.array_equal(vector_a, case.vector_a_q10.astype(np.int16)):
        raise RuntimeError("input A 上传往返不一致")
    if not np.array_equal(vector_b, case.vector_b_q10.astype(np.int16)):
        raise RuntimeError("input B 上传往返不一致")
    if not np.array_equal(pwl[:PWL_ENTRIES], case.pwl_endpoints_q10.astype(np.int16)):
        raise RuntimeError("PWL 端点上传往返不一致")
    if np.any(pwl[PWL_ENTRIES:] != 0):
        raise RuntimeError("PWL 补齐区域不是 0")
    return hashlib.sha256(payload).hexdigest()


def validate_fixed_case(case: ElementwiseCase, manifest_path: Path) -> dict[str, str]:
    vector_a, vector_b = make_deterministic_q10_vectors(K, DEFAULT_SEED)
    result = compute_elementwise_reference(
        vector_a_q10=vector_a,
        vector_b_q10=vector_b,
        scale_q10=DEFAULT_SCALE_Q10,
        seed=DEFAULT_SEED,
    )
    generated_manifest = result_manifest(result)
    committed = json.loads(manifest_path.read_text(encoding="utf-8"))
    if generated_manifest != committed:
        raise RuntimeError("E2 固定向量与 JSON 清单不一致")
    return {
        "input_a_q6_10": sha256_array(case.vector_a_q10, "<i2"),
        "input_b_q6_10": sha256_array(case.vector_b_q10, "<i2"),
        "pwl65_q6_10": sha256_array(case.pwl_endpoints_q10, "<i2"),
        "residual_q6_10": sha256_array(case.expected[OP_RESIDUAL_ADD], "<i2"),
        "fixed_scale_q6_10": sha256_array(case.expected[OP_FIXED_SCALE], "<i2"),
        "elementwise_mul_q6_10": sha256_array(case.expected[OP_ELEMENTWISE_MUL], "<i2"),
        "silu_pwl_q6_10": sha256_array(case.expected[OP_SILU], "<i2"),
        "upload_payload": verify_payload_roundtrip(case),
    }


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
    reply = read_exact(port, 3, timeout=15.0)
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
        configured=bool(flags & 0x10),
    )
    print(
        "DDR3初始化={}，数据已加载={}，结果有效={}，计算核心忙={}，配置有效={}".format(
            "是" if status.ddr_ready else "否",
            "是" if status.data_loaded else "否",
            "是" if status.result_valid else "否",
            "是" if status.core_busy else "否",
            "是" if status.configured else "否",
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


def load_case(port: "serial.Serial", case: ElementwiseCase) -> None:
    payload = build_upload_payload(case)
    port.write(b"L")
    port.write(payload)
    port.flush()
    read_ack(port)


def configure_operation(port: "serial.Serial", op_mode: int, scale_q10: int) -> None:
    if op_mode not in OP_NAMES:
        raise ValueError(f"未知操作模式：{op_mode}")
    if not Q_MIN <= scale_q10 <= Q_MAX:
        raise ValueError("scale_q10 超出 signed int16")
    port.write(b"C" + struct.pack("<Bh", op_mode, scale_q10))
    port.flush()
    read_ack(port)


def run_loaded_operation(port: "serial.Serial") -> np.ndarray:
    port.write(b"G")
    reply = read_exact(port, 1 + RESULT_BYTES, timeout=30.0)
    raise_if_error_frame(reply)
    if reply[0:1] != b"R":
        raise RuntimeError(f"元素级结果帧头错误：{reply[:16]!r}")
    return np.frombuffer(reply[1:], dtype="<i2").copy()


def run_and_compare(port: "serial.Serial", case: ElementwiseCase, op_mode: int) -> np.ndarray:
    configure_operation(port, op_mode, case.scale_q10)
    fpga = run_loaded_operation(port)
    expected = case.expected[op_mode]
    if not np.array_equal(fpga, expected):
        mismatch = np.flatnonzero(fpga != expected)
        first = int(mismatch[0])
        raise RuntimeError(
            f"{case.label} / {OP_NAMES[op_mode]} 不一致：首个错误元素={first}，"
            f"FPGA={int(fpga[first])}，Python={int(expected[first])}，"
            f"总错误数={mismatch.size}"
        )
    return fpga


def command_fixed(port: "serial.Serial", manifest_path: Path) -> None:
    wait_until_ready(port)
    case = make_fixed_case()
    hashes = validate_fixed_case(case, manifest_path)
    started = time.monotonic()
    load_case(port, case)
    for op_mode in OP_NAMES:
        fpga = run_and_compare(port, case, op_mode)
        print(
            f"{OP_NAMES[op_mode]} 固定向量逐位一致：PASS，"
            f"SHA256={sha256_array(fpga, '<i2')}，前16项={fpga[:16].tolist()}"
        )
    elapsed = time.monotonic() - started
    print("E2 K=896 四种元素级操作固定向量全部通过。")
    print(json.dumps(hashes, ensure_ascii=False, indent=2))
    print(f"上传、四次计算与回读总耗时：{elapsed:.2f} 秒")


def command_stress(port: "serial.Serial", rounds: int, seed: int) -> None:
    if rounds <= 0:
        raise ValueError("rounds 必须大于 0")
    wait_until_ready(port)
    started = time.monotonic()
    for index in range(rounds):
        case_seed = seed + index
        case = make_random_case(case_seed)
        load_case(port, case)
        for op_mode in OP_NAMES:
            run_and_compare(port, case, op_mode)
        if index == 0 or index + 1 == rounds or (index + 1) % 10 == 0:
            print(f"E2 真实 FPGA 随机向量已通过 {index + 1}/{rounds}，seed={case_seed}")
    elapsed = time.monotonic() - started
    print(
        f"E2 真实 FPGA 四操作随机回归 PASS：{rounds}/{rounds}，"
        f"seed_start={seed}，耗时 {elapsed:.2f} 秒"
    )


def command_selftest(manifest_path: Path, rounds: int, seed: int) -> None:
    if rounds <= 0:
        raise ValueError("rounds 必须大于 0")
    started = time.monotonic()
    fixed = make_fixed_case()
    hashes = validate_fixed_case(fixed, manifest_path)
    print("E2 固定载荷、PWL 补齐、JSON 清单和四操作金标准：PASS")
    print(f"上传载荷：{UPLOAD_BYTES} B")
    print(json.dumps(hashes, ensure_ascii=False, indent=2))

    for index in range(rounds):
        case_seed = seed + index
        case = make_random_case(case_seed)
        verify_payload_roundtrip(case)
        for op_mode, expected in case.expected.items():
            if expected.dtype != np.int16 or expected.shape != (K,):
                raise RuntimeError(f"{OP_NAMES[op_mode]} 软件结果格式错误")
        if index == 0 or index + 1 == rounds or (index + 1) % 100 == 0:
            print(f"E2 软件随机向量已通过 {index + 1}/{rounds}，seed={case_seed}")
    elapsed = time.monotonic() - started
    print(
        f"E2 软件参考与载荷压力测试 PASS：{rounds}/{rounds}，"
        f"seed_start={seed}，耗时 {elapsed:.2f} 秒"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PGL50H K896 signed Q6.10 元素级算子上位机"
    )
    parser.add_argument("--port", default="COM20")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("ports", help="列出串口")
    sub.add_parser("info", help="读取固件信息")
    sub.add_parser("status", help="读取状态")

    fixed = sub.add_parser("fixed", help="运行四种操作固定边界向量")
    fixed.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)

    stress = sub.add_parser("stress", help="运行四种操作随机上板回归")
    stress.add_argument("--rounds", type=int, default=100)
    stress.add_argument("--seed", type=int, default=20260727)

    selftest = sub.add_parser("selftest", help="只运行载荷和软件金标准自检")
    selftest.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    selftest.add_argument("--rounds", type=int, default=1000)
    selftest.add_argument("--seed", type=int, default=20260727)
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
        struct.error,
    ) as error:
        print(f"错误：{error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
