#!/usr/bin/env python3
"""盘古 PGL50H layer0 input_layernorm K=896 定点验证工具。"""

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

from model_tools.p50_format import P50Image  # noqa: E402
from model_tools.rmsnorm_fixed_reference import (  # noqa: E402
    DEFAULT_GAMMA,
    DEFAULT_IMAGE,
    DEFAULT_INPUT_SEED,
    DEFAULT_LENGTH,
    LUT_ONLY_INDEX_BITS,
    build_rsqrt_lut,
    make_deterministic_input,
    reference_from_p50,
    result_manifest,
)

BAUD_RATE = 115200
K = DEFAULT_LENGTH
DATA_BYTES = K * 2
LUT_ENTRIES = 1 << LUT_ONLY_INDEX_BITS
LUT_BYTES = LUT_ENTRIES * 4
RESULT_BYTES = DATA_BYTES
UPLOAD_BYTES = DATA_BYTES + DATA_BYTES + LUT_BYTES
DEFAULT_IMAGE_PATH = PROJECT_ROOT / DEFAULT_IMAGE
DEFAULT_MANIFEST = PROJECT_ROOT / "model_tools/rmsnorm_layer0_reference.json"

ERROR_MESSAGES = {
    0x01: "未知命令",
    0x02: "DDR3 尚未初始化完成",
    0x04: "尚未加载 RMSNorm 数据",
    0xFF: "FPGA 状态机异常",
}


@dataclass(frozen=True)
class FpgaStatus:
    ddr_ready: bool
    data_loaded: bool
    result_valid: bool
    core_busy: bool


@dataclass(frozen=True)
class RMSNormCase:
    activation_q10: np.ndarray
    gamma_q10: np.ndarray
    lut_q20: np.ndarray
    expected_lut_q10: np.ndarray
    expected_exact_q10: np.ndarray
    sum_squares: int
    variance_q20: int
    lut_rsqrt_q20: int
    label: str


def require_pyserial() -> None:
    if serial is None:
        raise SystemExit("缺少 pyserial，请运行：python -m pip install pyserial")


def sha256_array(array: np.ndarray, dtype: str | np.dtype) -> str:
    return hashlib.sha256(np.asarray(array, dtype=dtype).tobytes(order="C")).hexdigest()


def _require_shape(array: np.ndarray, shape: tuple[int, ...], label: str) -> None:
    if array.shape != shape:
        raise ValueError(f"{label} 形状错误：{array.shape}，预期 {shape}")


def validate_case(case: RMSNormCase) -> None:
    _require_shape(np.asarray(case.activation_q10), (K,), "activation_q10")
    _require_shape(np.asarray(case.gamma_q10), (K,), "gamma_q10")
    _require_shape(np.asarray(case.lut_q20), (LUT_ENTRIES,), "lut_q20")
    _require_shape(np.asarray(case.expected_lut_q10), (K,), "expected_lut_q10")
    _require_shape(np.asarray(case.expected_exact_q10), (K,), "expected_exact_q10")
    if np.asarray(case.activation_q10).dtype.kind != "i":
        raise ValueError("activation_q10 必须是有符号整数")
    if np.asarray(case.gamma_q10).dtype.kind != "i":
        raise ValueError("gamma_q10 必须是有符号整数")
    if np.asarray(case.lut_q20).dtype.kind != "u":
        raise ValueError("lut_q20 必须是无符号整数")
    if not 0 <= case.sum_squares < (1 << 40):
        raise ValueError("sum_squares 超出 40 位")
    if not 0 < case.variance_q20 < (1 << 40):
        raise ValueError("variance_q20 超出范围")
    if not 0 < case.lut_rsqrt_q20 <= 0xFFFFFFFF:
        raise ValueError("lut_rsqrt_q20 超出 uint32")


def make_stress_input(seed: int) -> np.ndarray:
    """生成范围随 seed 变化、最大绝对值不超过 8 的确定性输入。"""

    base = make_deterministic_input(K, seed=seed).astype(np.float64) / 4.0
    state = (int(seed) * 1664525 + 1013904223) & 0xFFFFFFFF
    exponent = -6.0 + 9.0 * (((state >> 8) & 0xFFFF) / 65535.0)
    amplitude = 2.0**exponent
    return (base * amplitude).astype(np.float32)


def case_from_image(
    image: P50Image,
    *,
    input_seed: int = DEFAULT_INPUT_SEED,
    fixed: bool = False,
    label: str | None = None,
) -> RMSNormCase:
    activation = (
        make_deterministic_input(K, input_seed)
        if fixed
        else make_stress_input(input_seed)
    )
    result = reference_from_p50(
        image,
        activation_values=activation,
        gamma_name=DEFAULT_GAMMA,
    )
    if result.activation.clipped_count:
        raise RuntimeError("RMSNorm 输入 Q6.10 发生饱和")
    if result.gamma.clipped_count:
        raise RuntimeError("真实 gamma Q6.10 发生饱和")
    if result.lut_output_saturated_count:
        raise RuntimeError("RMSNorm LUT 输出 Q6.10 发生饱和")
    case = RMSNormCase(
        activation_q10=result.activation.quantized.astype(np.int16),
        gamma_q10=result.gamma.quantized.astype(np.int16),
        lut_q20=build_rsqrt_lut(LUT_ONLY_INDEX_BITS).astype(np.uint32),
        expected_lut_q10=result.output_lut_q10.astype(np.int16),
        expected_exact_q10=result.output_exact_q10.astype(np.int16),
        sum_squares=result.sum_squares,
        variance_q20=result.variance_q20,
        lut_rsqrt_q20=result.lut_rsqrt_q20,
        label=label or f"layer0 RMSNorm seed={input_seed}",
    )
    validate_case(case)
    return case


def build_upload_payload(case: RMSNormCase) -> bytes:
    validate_case(case)
    payload = (
        np.asarray(case.activation_q10, dtype="<i2").tobytes(order="C")
        + np.asarray(case.gamma_q10, dtype="<i2").tobytes(order="C")
        + np.asarray(case.lut_q20, dtype="<u4").tobytes(order="C")
    )
    if len(payload) != UPLOAD_BYTES:
        raise AssertionError(f"上传载荷长度错误：{len(payload)} != {UPLOAD_BYTES}")
    return payload


def verify_payload_roundtrip(case: RMSNormCase) -> str:
    payload = build_upload_payload(case)
    activation_end = DATA_BYTES
    gamma_end = activation_end + DATA_BYTES
    activation = np.frombuffer(payload[:activation_end], dtype="<i2").copy()
    gamma = np.frombuffer(payload[activation_end:gamma_end], dtype="<i2").copy()
    lut = np.frombuffer(payload[gamma_end:], dtype="<u4").copy()
    if not np.array_equal(activation, case.activation_q10.astype(np.int16)):
        raise RuntimeError("activation_q10 上传往返不一致")
    if not np.array_equal(gamma, case.gamma_q10.astype(np.int16)):
        raise RuntimeError("gamma_q10 上传往返不一致")
    if not np.array_equal(lut, case.lut_q20.astype(np.uint32)):
        raise RuntimeError("rsqrt LUT 上传往返不一致")
    return hashlib.sha256(payload).hexdigest()


def validate_fixed_case(
    image: P50Image, case: RMSNormCase, manifest_path: Path
) -> dict[str, str]:
    fixed_result = reference_from_p50(
        image,
        activation_values=make_deterministic_input(K, DEFAULT_INPUT_SEED),
        gamma_name=DEFAULT_GAMMA,
    )
    generated_manifest = result_manifest(fixed_result)
    committed = json.loads(manifest_path.read_text(encoding="utf-8"))
    if generated_manifest != committed:
        raise RuntimeError("RMSNorm 固定向量与 JSON 清单不一致")
    payload_hash = verify_payload_roundtrip(case)
    return {
        "activation_q6_10": sha256_array(case.activation_q10, "<i2"),
        "gamma_q6_10": sha256_array(case.gamma_q10, "<i2"),
        "lut256_uq12_20": sha256_array(case.lut_q20, "<u4"),
        "output_lut_q6_10": sha256_array(case.expected_lut_q10, "<i2"),
        "upload_payload": payload_hash,
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


def load_case(port: "serial.Serial", case: RMSNormCase) -> None:
    payload = build_upload_payload(case)
    port.write(b"L")
    port.write(payload)
    port.flush()
    read_ack(port)


def run_loaded_case(port: "serial.Serial") -> np.ndarray:
    port.write(b"G")
    reply = read_exact(port, 1 + RESULT_BYTES, timeout=30.0)
    raise_if_error_frame(reply)
    if reply[0:1] != b"R":
        raise RuntimeError(f"RMSNorm 结果帧头错误：{reply[:16]!r}")
    return np.frombuffer(reply[1:], dtype="<i2").copy()


def run_and_compare(port: "serial.Serial", case: RMSNormCase) -> np.ndarray:
    load_case(port, case)
    fpga = run_loaded_case(port)
    if not np.array_equal(fpga, case.expected_lut_q10):
        mismatch = np.flatnonzero(fpga != case.expected_lut_q10)
        first = int(mismatch[0])
        raise RuntimeError(
            f"{case.label} 不一致：首个错误元素={first}，"
            f"FPGA={int(fpga[first])}，Python={int(case.expected_lut_q10[first])}，"
            f"总错误数={mismatch.size}"
        )
    return fpga


def command_fixed(port: "serial.Serial", image_path: Path, manifest_path: Path) -> None:
    wait_until_ready(port)
    image = P50Image(image_path)
    image.validate()
    case = case_from_image(image, input_seed=DEFAULT_INPUT_SEED, fixed=True)
    hashes = validate_fixed_case(image, case, manifest_path)
    started = time.monotonic()
    fpga = run_and_compare(port, case)
    elapsed = time.monotonic() - started
    exact_delta = np.max(
        np.abs(fpga.astype(np.int32) - case.expected_exact_q10.astype(np.int32))
    )
    print("真实 layer0 input_layernorm K=896 与 LUT256 金标准逐位一致：PASS")
    print(f"sum_squares={case.sum_squares}")
    print(f"variance_q20={case.variance_q20}")
    print(f"lut_rsqrt_q20={case.lut_rsqrt_q20}")
    print(f"输出 SHA256：{hashes['output_lut_q6_10']}")
    print(f"前 16 项：{fpga[:16].tolist()}")
    print(f"相对精确 rsqrt 路径最大差值：{int(exact_delta)} Q10 LSB")
    print(f"上传、计算与回读总耗时：{elapsed:.2f} 秒")


def command_stress(
    port: "serial.Serial", image_path: Path, rounds: int, seed: int
) -> None:
    if rounds <= 0:
        raise ValueError("rounds 必须大于 0")
    wait_until_ready(port)
    image = P50Image(image_path)
    image.validate()
    started = time.monotonic()
    for index in range(rounds):
        input_seed = seed + index
        case = case_from_image(
            image,
            input_seed=input_seed,
            fixed=False,
            label=f"RMSNorm 随机输入 {index + 1}/{rounds}",
        )
        run_and_compare(port, case)
        print(f"真实 RMSNorm 随机输入已通过 {index + 1}/{rounds}，seed={input_seed}")
    elapsed = time.monotonic() - started
    print(
        f"RMSNorm 真实 FPGA 随机回归 PASS：{rounds}/{rounds}，"
        f"seed_start={seed}，耗时 {elapsed:.2f} 秒"
    )


def command_selftest(
    image_path: Path,
    manifest_path: Path,
    rounds: int,
    seed: int,
) -> None:
    if rounds <= 0:
        raise ValueError("rounds 必须大于 0")
    image = P50Image(image_path)
    image.validate()
    started = time.monotonic()

    fixed = case_from_image(image, input_seed=DEFAULT_INPUT_SEED, fixed=True)
    hashes = validate_fixed_case(image, fixed, manifest_path)
    max_lut_exact_lsb = int(
        np.max(
            np.abs(
                fixed.expected_lut_q10.astype(np.int32)
                - fixed.expected_exact_q10.astype(np.int32)
            )
        )
    )
    print("RMSNorm 固定载荷打包/解包和真实 gamma 金标准：PASS")
    print(f"上传载荷：{UPLOAD_BYTES} B")
    print(f"sum_squares={fixed.sum_squares}")
    print(f"variance_q20={fixed.variance_q20}")
    print(f"lut_rsqrt_q20={fixed.lut_rsqrt_q20}")
    print(f"LUT 与精确路径最大差值：{max_lut_exact_lsb} Q10 LSB")
    print("固定向量哈希：")
    print(json.dumps(hashes, ensure_ascii=False, indent=2))

    max_random_lsb = 0
    for index in range(rounds):
        input_seed = seed + index
        case = case_from_image(image, input_seed=input_seed, fixed=False)
        verify_payload_roundtrip(case)
        delta = int(
            np.max(
                np.abs(
                    case.expected_lut_q10.astype(np.int32)
                    - case.expected_exact_q10.astype(np.int32)
                )
            )
        )
        max_random_lsb = max(max_random_lsb, delta)
        if index == 0 or index + 1 == rounds or (index + 1) % 100 == 0:
            print(f"RMSNorm 软件随机输入已通过 {index + 1}/{rounds}，seed={input_seed}")

    elapsed = time.monotonic() - started
    print(
        f"RMSNorm 软件参考压力测试 PASS：{rounds}/{rounds}，"
        f"seed_start={seed}，LUT最大偏差={max_random_lsb} Q10 LSB，"
        f"耗时 {elapsed:.2f} 秒"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PGL50H layer0 input_layernorm K896 定点 RMSNorm 上位机"
    )
    parser.add_argument("--port", default="COM20")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("ports", help="列出串口")
    sub.add_parser("info", help="读取固件信息")
    sub.add_parser("status", help="读取状态")

    fixed = sub.add_parser("fixed", help="运行真实 layer0 RMSNorm 固定向量")
    fixed.add_argument("--image", type=Path, default=DEFAULT_IMAGE_PATH)
    fixed.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)

    stress = sub.add_parser("stress", help="运行真实 RMSNorm 随机输入上板回归")
    stress.add_argument("--image", type=Path, default=DEFAULT_IMAGE_PATH)
    stress.add_argument("--rounds", type=int, default=100)
    stress.add_argument("--seed", type=int, default=20260726)

    selftest = sub.add_parser("selftest", help="只运行载荷和软件金标准自检")
    selftest.add_argument("--image", type=Path, default=DEFAULT_IMAGE_PATH)
    selftest.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    selftest.add_argument("--rounds", type=int, default=1000)
    selftest.add_argument("--seed", type=int, default=20260726)
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
                command_stress(port, args.image, args.rounds, args.seed)
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
