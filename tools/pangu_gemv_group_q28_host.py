#!/usr/bin/env python3
"""盘古 PGL50H 真实模型分组 UQ4.28 GEMV 验证工具。"""

from __future__ import annotations

import argparse
import json
import random
import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

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
    DEFAULT_BIAS,
    DEFAULT_WEIGHT,
    make_deterministic_activation,
    reference_from_p50,
)
from model_tools.p50_format import P50Image  # noqa: E402

BAUD_RATE = 115200
M = 4
K = 896
GROUP_SIZE = 64
GROUPS = K // GROUP_SIZE
ACTIVATION_BYTES = K
WEIGHT_ROW_BYTES = K // 2
WEIGHT_BYTES = M * WEIGHT_ROW_BYTES
SCALE_ROW_BYTES = 64
SCALE_BYTES = M * SCALE_ROW_BYTES
BIAS_BYTES = M * 8
UPLOAD_BYTES = ACTIVATION_BYTES + WEIGHT_BYTES + SCALE_BYTES + BIAS_BYTES
DEFAULT_IMAGE = PROJECT_ROOT / "model_output/yanbo_qwen25_0.5b_int4.p50"
DEFAULT_MANIFEST = PROJECT_ROOT / "model_tools/q_proj_m4k896_reference.json"
EXPECTED_FIXED_Q28 = np.asarray(
    [207253689, -173360554, 287606739, -223225713], dtype=np.int64
)

ERROR_MESSAGES = {
    0x01: "未知命令",
    0x02: "DDR3 尚未初始化完成",
    0x04: "尚未加载输入、权重、scale 和 bias",
    0xFF: "FPGA 状态机异常",
}


@dataclass(frozen=True)
class FpgaStatus:
    ddr_ready: bool
    data_loaded: bool
    result_valid: bool
    core_busy: bool


@dataclass(frozen=True)
class Q28Case:
    activation: np.ndarray
    weights: np.ndarray
    scales_q28: np.ndarray
    bias_q28: np.ndarray
    expected_q28: np.ndarray
    label: str


def require_pyserial() -> None:
    if serial is None:
        raise SystemExit("缺少 pyserial，请运行：python -m pip install pyserial")


def _require_shape(array: np.ndarray, shape: tuple[int, ...], label: str) -> None:
    if array.shape != shape:
        raise ValueError(f"{label} 形状错误：{array.shape}，预期 {shape}")


def validate_case(case: Q28Case) -> None:
    activation = np.asarray(case.activation)
    weights = np.asarray(case.weights)
    scales = np.asarray(case.scales_q28)
    bias = np.asarray(case.bias_q28)
    expected = np.asarray(case.expected_q28)

    _require_shape(activation, (K,), "activation")
    _require_shape(weights, (M, K), "weights")
    _require_shape(scales, (M, GROUPS), "scales_q28")
    _require_shape(bias, (M,), "bias_q28")
    _require_shape(expected, (M,), "expected_q28")

    if np.any(activation.astype(np.int16) < -127) or np.any(
        activation.astype(np.int16) > 127
    ):
        raise ValueError("activation 必须位于 [-127,127]")
    if np.any(weights.astype(np.int16) < -7) or np.any(
        weights.astype(np.int16) > 7
    ):
        raise ValueError("真实模型 INT4 权重必须位于 [-7,7]")
    if np.any(scales.astype(object) < 0) or np.any(
        scales.astype(object) > 0xFFFFFFFF
    ):
        raise ValueError("UQ4.28 scale 必须位于 uint32 范围")
    for value in np.concatenate((bias.reshape(-1), expected.reshape(-1))):
        if not -(1 << 63) <= int(value) <= (1 << 63) - 1:
            raise ValueError("bias/output 超出 signed int64")


def compute_q28_reference(
    activation: Sequence[int],
    weights: Sequence[Sequence[int]],
    scales_q28: Sequence[Sequence[int]],
    bias_q28: Sequence[int],
) -> np.ndarray:
    """按 FPGA 精确定义计算 signed int64 Q28 输出。"""

    if len(activation) != K or len(weights) != M or len(scales_q28) != M:
        raise ValueError("输入必须为固定 M=4、K=896、groups=14")
    outputs: list[int] = []
    for row in range(M):
        if len(weights[row]) != K or len(scales_q28[row]) != GROUPS:
            raise ValueError("权重或 scale 行长度错误")
        total = int(bias_q28[row])
        for group in range(GROUPS):
            begin = group * GROUP_SIZE
            end = begin + GROUP_SIZE
            group_acc = sum(
                int(activation[index]) * int(weights[row][index])
                for index in range(begin, end)
            )
            if not -(1 << 31) <= group_acc <= (1 << 31) - 1:
                raise OverflowError("分组点积超出 signed int32")
            total += group_acc * int(scales_q28[row][group])
        if not -(1 << 63) <= total <= (1 << 63) - 1:
            raise OverflowError(f"第 {row} 行 Q28 累加超出 signed int64")
        outputs.append(total)
    return np.asarray(outputs, dtype=np.int64)


def pack_int4_matrix(weights: np.ndarray) -> bytes:
    values = np.asarray(weights, dtype=np.int8)
    _require_shape(values, (M, K), "weights")
    if np.any(values < -7) or np.any(values > 7):
        raise ValueError("INT4 权重必须位于 [-7,7]")
    nibble = np.bitwise_and(values.astype(np.int16), 0x0F)
    packed = np.bitwise_or(nibble[:, 0::2], np.left_shift(nibble[:, 1::2], 4))
    return packed.astype(np.uint8).tobytes(order="C")


def unpack_int4_matrix(payload: bytes) -> np.ndarray:
    if len(payload) != WEIGHT_BYTES:
        raise ValueError(f"packed 权重必须为 {WEIGHT_BYTES} 字节")
    packed = np.frombuffer(payload, dtype=np.uint8).reshape(M, WEIGHT_ROW_BYTES)
    output = np.empty((M, K), dtype=np.int8)
    low = np.bitwise_and(packed, 0x0F).astype(np.int8)
    high = np.right_shift(packed, 4).astype(np.int8)
    low[low >= 8] -= 16
    high[high >= 8] -= 16
    output[:, 0::2] = low
    output[:, 1::2] = high
    return output


def build_upload_payload(case: Q28Case) -> bytes:
    validate_case(case)
    activation = np.asarray(case.activation, dtype=np.int8).tobytes(order="C")
    weights = pack_int4_matrix(np.asarray(case.weights, dtype=np.int8))

    scale_rows = bytearray()
    scales = np.asarray(case.scales_q28, dtype="<u4")
    for row in range(M):
        raw = scales[row].tobytes(order="C")
        scale_rows.extend(raw)
        scale_rows.extend(bytes(SCALE_ROW_BYTES - len(raw)))

    bias = np.asarray(case.bias_q28, dtype="<i8").tobytes(order="C")
    payload = activation + weights + bytes(scale_rows) + bias
    if len(payload) != UPLOAD_BYTES:
        raise AssertionError(f"上传载荷长度错误：{len(payload)} != {UPLOAD_BYTES}")
    return payload


def verify_payload_roundtrip(case: Q28Case) -> None:
    payload = build_upload_payload(case)
    activation_end = ACTIVATION_BYTES
    weight_end = activation_end + WEIGHT_BYTES
    scale_end = weight_end + SCALE_BYTES

    unpacked_activation = np.frombuffer(
        payload[:activation_end], dtype=np.int8
    ).copy()
    unpacked_weights = unpack_int4_matrix(payload[activation_end:weight_end])

    unpacked_scales = np.empty((M, GROUPS), dtype=np.uint32)
    scale_area = payload[weight_end:scale_end]
    for row in range(M):
        begin = row * SCALE_ROW_BYTES
        unpacked_scales[row] = np.frombuffer(
            scale_area[begin : begin + GROUPS * 4], dtype="<u4"
        )
    unpacked_bias = np.frombuffer(payload[scale_end:], dtype="<i8").copy()

    if not np.array_equal(unpacked_activation, case.activation.astype(np.int8)):
        raise RuntimeError("activation 上传往返不一致")
    if not np.array_equal(unpacked_weights, case.weights.astype(np.int8)):
        raise RuntimeError("packed INT4 上传往返不一致")
    if not np.array_equal(unpacked_scales, case.scales_q28.astype(np.uint32)):
        raise RuntimeError("UQ4.28 scale 上传往返不一致")
    if not np.array_equal(unpacked_bias, case.bias_q28.astype(np.int64)):
        raise RuntimeError("bias_q28 上传往返不一致")


def load_fixed_real_case(
    image_path: Path = DEFAULT_IMAGE, manifest_path: Path = DEFAULT_MANIFEST
) -> Q28Case:
    image = P50Image(image_path)
    image.validate()
    activation_float = make_deterministic_activation(K, seed=DEFAULT_ACTIVATION_SEED)
    result = reference_from_p50(
        image,
        weight_name=DEFAULT_WEIGHT,
        bias_name=DEFAULT_BIAS,
        row_start=0,
        row_count=M,
        column_start=0,
        column_count=K,
        activation_values=activation_float,
    )
    case = Q28Case(
        activation=result.activation.quantized.astype(np.int8),
        weights=result.weight_quantized.astype(np.int8),
        scales_q28=result.combined_scale_q28.astype(np.uint32),
        bias_q28=result.bias_q28.astype(np.int64),
        expected_q28=result.output_fixed_q28.astype(np.int64),
        label="layer0 q_proj M4K896 固定真实向量",
    )
    validate_case(case)
    reference = compute_q28_reference(
        case.activation, case.weights, case.scales_q28, case.bias_q28
    )
    if not np.array_equal(reference, case.expected_q28):
        raise RuntimeError("重新计算的 Q28 结果与 linear_quant_reference 不一致")
    if not np.array_equal(case.expected_q28, EXPECTED_FIXED_Q28):
        raise RuntimeError(
            f"固定向量发生变化：{case.expected_q28.tolist()} != "
            f"{EXPECTED_FIXED_Q28.tolist()}"
        )
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_expected = np.asarray(
            manifest["expected"]["output_fixed_q28"], dtype=np.int64
        )
        if not np.array_equal(manifest_expected, case.expected_q28):
            raise RuntimeError("固定向量与 JSON 清单不一致")
    verify_payload_roundtrip(case)
    return case


def make_synthetic_case() -> Q28Case:
    activation = np.asarray(
        [((index * 29 + 11) % 255) - 127 for index in range(K)], dtype=np.int8
    )
    weights = np.asarray(
        [
            [((row * 5 + index * 3 + 2) % 15) - 7 for index in range(K)]
            for row in range(M)
        ],
        dtype=np.int8,
    )
    scale_pattern = (
        0,
        1,
        0x00010001,
        0x7FFFFFFF,
        0x80000000,
        0xFFFFFFFF,
        123456789,
    )
    scales = np.asarray(
        [
            [scale_pattern[(row * GROUPS + group) % len(scale_pattern)] for group in range(GROUPS)]
            for row in range(M)
        ],
        dtype=np.uint32,
    )
    bias = np.asarray([0, -123456789, 1 << 45, -(1 << 45)], dtype=np.int64)
    expected = compute_q28_reference(activation, weights, scales, bias)
    return Q28Case(activation, weights, scales, bias, expected, "合成边界向量")


def make_random_case(rng: random.Random, index: int) -> Q28Case:
    activation = np.fromiter(
        (rng.randint(-127, 127) for _ in range(K)), dtype=np.int8, count=K
    )
    weights = np.fromiter(
        (rng.randint(-7, 7) for _ in range(M * K)),
        dtype=np.int8,
        count=M * K,
    ).reshape(M, K)
    scales = np.fromiter(
        (rng.getrandbits(32) for _ in range(M * GROUPS)),
        dtype=np.uint32,
        count=M * GROUPS,
    ).reshape(M, GROUPS)
    bias = np.fromiter(
        (rng.randint(-(1 << 48), (1 << 48)) for _ in range(M)),
        dtype=np.int64,
        count=M,
    )
    expected = compute_q28_reference(activation, weights, scales, bias)
    return Q28Case(activation, weights, scales, bias, expected, f"随机向量 {index}")


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
    reply = port.read_until(b"\n", 16)
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


def load_case(port: "serial.Serial", case: Q28Case) -> None:
    payload = build_upload_payload(case)
    port.write(b"L" + payload)
    read_ack(port)


def run_loaded_case(port: "serial.Serial") -> np.ndarray:
    port.write(b"G")
    reply = read_exact(port, 1 + M * 8, timeout=30.0)
    raise_if_error_frame(reply)
    if reply[0:1] != b"R":
        raise RuntimeError(f"Q28 结果帧头错误：{reply[:16]!r}")
    return np.asarray(struct.unpack("<4q", reply[1:]), dtype=np.int64)


def run_and_compare(port: "serial.Serial", case: Q28Case) -> np.ndarray:
    load_case(port, case)
    fpga = run_loaded_case(port)
    if not np.array_equal(fpga, case.expected_q28):
        raise RuntimeError(
            f"{case.label} 不一致：\nFPGA={fpga.tolist()}\n"
            f"Python={case.expected_q28.tolist()}"
        )
    return fpga


def command_fixed(port: "serial.Serial", image: Path, manifest: Path) -> None:
    wait_until_ready(port)
    case = load_fixed_real_case(image, manifest)
    fpga = run_and_compare(port, case)
    print("真实 layer0 q_proj M4K896 固定向量逐位一致：PASS")
    print(f"Q28 输出：{fpga.tolist()}")
    print(f"反量化输出：{(fpga.astype(np.float64) / (1 << 28)).tolist()}")


def command_stress(port: "serial.Serial", rounds: int, seed: int) -> None:
    if rounds <= 0:
        raise ValueError("rounds 必须大于 0")
    wait_until_ready(port)
    rng = random.Random(seed)
    started = time.monotonic()
    boundary = make_synthetic_case()
    run_and_compare(port, boundary)
    print("含 scale bit31/uint32 最大值的合成边界向量：PASS")
    for index in range(1, rounds + 1):
        case = make_random_case(rng, index)
        run_and_compare(port, case)
        if index == 1 or index == rounds or index % 100 == 0:
            print(f"随机分组 Q28 已通过 {index}/{rounds} 轮")
    elapsed = time.monotonic() - started
    print(
        f"随机分组缩放真实 FPGA 压力测试 PASS：{rounds}/{rounds}，"
        f"seed={seed}，耗时 {elapsed:.2f} 秒"
    )


def command_selftest(rounds: int, seed: int, include_real: bool) -> None:
    if rounds <= 0:
        raise ValueError("rounds 必须大于 0")
    boundary = make_synthetic_case()
    verify_payload_roundtrip(boundary)
    recomputed = compute_q28_reference(
        boundary.activation, boundary.weights, boundary.scales_q28, boundary.bias_q28
    )
    if not np.array_equal(recomputed, boundary.expected_q28):
        raise RuntimeError("合成边界向量参考不一致")

    rng = random.Random(seed)
    started = time.monotonic()
    for index in range(1, rounds + 1):
        case = make_random_case(rng, index)
        verify_payload_roundtrip(case)
        recomputed = compute_q28_reference(
            case.activation, case.weights, case.scales_q28, case.bias_q28
        )
        if not np.array_equal(recomputed, case.expected_q28):
            raise RuntimeError(f"第 {index} 轮软件参考不一致")
        if index == 1 or index == rounds or index % 100 == 0:
            print(f"软件自检已通过 {index}/{rounds} 轮")

    if include_real:
        fixed = load_fixed_real_case()
        print(f"真实固定向量：{fixed.expected_q28.tolist()}，PASS")

    elapsed = time.monotonic() - started
    print(
        f"Q28 载荷/参考软件自检 PASS：{rounds}/{rounds}，"
        f"seed={seed}，耗时 {elapsed:.2f} 秒"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PGL50H 固定 M4K896 分组 UQ4.28 GEMV 上位机"
    )
    parser.add_argument("--port", default="COM20")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("ports", help="列出串口")
    sub.add_parser("info", help="读取固件信息")
    sub.add_parser("status", help="读取状态")

    fixed = sub.add_parser("fixed", help="运行真实 layer0 q_proj 固定向量")
    fixed.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    fixed.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)

    stress = sub.add_parser("stress", help="运行随机分组 scale 上板压力测试")
    stress.add_argument("--rounds", type=int, default=1000)
    stress.add_argument("--seed", type=int, default=20260724)

    selftest = sub.add_parser("selftest", help="只运行 Python 载荷与金标准自检")
    selftest.add_argument("--rounds", type=int, default=1000)
    selftest.add_argument("--seed", type=int, default=20260724)
    selftest.add_argument("--include-real", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.command == "ports":
            show_ports()
            return 0
        if args.command == "selftest":
            command_selftest(args.rounds, args.seed, args.include_real)
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
    except (FileNotFoundError, KeyError, ValueError, OverflowError, RuntimeError, TimeoutError) as error:
        print(f"错误：{error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
