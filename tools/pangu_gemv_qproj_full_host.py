#!/usr/bin/env python3
"""盘古 PGL50H layer0 q_proj 完整真实 Linear 层验证工具。"""

from __future__ import annotations

import argparse
import hashlib
import json
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
    compute_groupwise_linear_reference,
    make_deterministic_activation,
    pack_int4_low_nibble_first,
)
from model_tools.p50_format import P50Image  # noqa: E402

BAUD_RATE = 115200
M = 896
K = 896
GROUP_SIZE = 64
GROUPS = K // GROUP_SIZE
ACTIVATION_BYTES = K
WEIGHT_ROW_BYTES = K // 2
WEIGHT_BYTES = M * WEIGHT_ROW_BYTES
SCALE_ROW_BYTES = 64
SCALE_BYTES = M * SCALE_ROW_BYTES
BIAS_ROW_BYTES = 32
BIAS_BYTES = M * BIAS_ROW_BYTES
RESULT_BYTES = M * 8
UPLOAD_BYTES = ACTIVATION_BYTES + WEIGHT_BYTES + SCALE_BYTES + BIAS_BYTES
DEFAULT_IMAGE = PROJECT_ROOT / "model_output/yanbo_qwen25_0.5b_int4.p50"
DEFAULT_MANIFEST = PROJECT_ROOT / "model_tools/q_proj_full_reference.json"
EXPECTED_FIRST4_Q28 = np.asarray(
    [207253689, -173360554, 287606739, -223225713], dtype=np.int64
)
EXPECTED_FIXED_OUTPUT_SHA256 = (
    "ea1f04bf4ff313dad07025ff35e66a088f13afd28d817422b89bb135f63525a0"
)

ERROR_MESSAGES = {
    0x01: "未知命令",
    0x02: "DDR3 尚未初始化完成",
    0x04: "尚未加载完整层数据",
    0xFF: "FPGA 状态机异常",
}


@dataclass(frozen=True)
class FpgaStatus:
    ddr_ready: bool
    data_loaded: bool
    result_valid: bool
    core_busy: bool


@dataclass(frozen=True)
class FullLayerCase:
    activation: np.ndarray
    weights: np.ndarray
    scales_q28: np.ndarray
    bias_q28: np.ndarray
    expected_q28: np.ndarray
    activation_scale: float
    label: str


@dataclass(frozen=True)
class FullLayerModel:
    weights: np.ndarray
    weight_scales: np.ndarray
    bias: np.ndarray


def require_pyserial() -> None:
    if serial is None:
        raise SystemExit("缺少 pyserial，请运行：python -m pip install pyserial")


def _require_shape(array: np.ndarray, shape: tuple[int, ...], label: str) -> None:
    if array.shape != shape:
        raise ValueError(f"{label} 形状错误：{array.shape}，预期 {shape}")


def sha256_array(array: np.ndarray, dtype: str | np.dtype) -> str:
    return hashlib.sha256(np.asarray(array, dtype=dtype).tobytes(order="C")).hexdigest()


def validate_case(case: FullLayerCase) -> None:
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
    weights: np.ndarray,
    scales_q28: np.ndarray,
    bias_q28: Sequence[int],
) -> np.ndarray:
    """按 FPGA 精确定义独立重算 896 行 signed int64 Q28 输出。"""

    act = np.asarray(activation, dtype=np.int8)
    weight_values = np.asarray(weights, dtype=np.int8)
    scales = np.asarray(scales_q28, dtype=np.uint32)
    bias = np.asarray(bias_q28, dtype=np.int64)
    _require_shape(act, (K,), "activation")
    _require_shape(weight_values, (M, K), "weights")
    _require_shape(scales, (M, GROUPS), "scales_q28")
    _require_shape(bias, (M,), "bias_q28")

    grouped_weights = weight_values.astype(np.int32).reshape(M, GROUPS, GROUP_SIZE)
    grouped_activation = act.astype(np.int32).reshape(GROUPS, GROUP_SIZE)
    accumulators = np.sum(
        grouped_weights * grouped_activation[np.newaxis, :, :],
        axis=2,
        dtype=np.int64,
    )
    if np.any(accumulators < np.iinfo(np.int32).min) or np.any(
        accumulators > np.iinfo(np.int32).max
    ):
        raise OverflowError("分组点积超出 signed int32")

    outputs: list[int] = []
    for row in range(M):
        total = int(bias[row])
        for group in range(GROUPS):
            total += int(accumulators[row, group]) * int(scales[row, group])
        if not -(1 << 63) <= total <= (1 << 63) - 1:
            raise OverflowError(f"第 {row} 行 Q28 累加超出 signed int64")
        outputs.append(total)
    return np.asarray(outputs, dtype=np.int64)


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


def build_upload_payload(case: FullLayerCase) -> bytes:
    validate_case(case)
    activation = np.asarray(case.activation, dtype=np.int8).tobytes(order="C")
    weights = pack_int4_low_nibble_first(
        np.asarray(case.weights, dtype=np.int8)
    ).astype(np.uint8).tobytes(order="C")

    scale_rows = np.zeros((M, SCALE_ROW_BYTES // 4), dtype="<u4")
    scale_rows[:, :GROUPS] = np.asarray(case.scales_q28, dtype="<u4")

    bias_rows = np.zeros((M, BIAS_ROW_BYTES // 8), dtype="<i8")
    bias_rows[:, 0] = np.asarray(case.bias_q28, dtype="<i8")

    payload = (
        activation
        + weights
        + scale_rows.tobytes(order="C")
        + bias_rows.tobytes(order="C")
    )
    if len(payload) != UPLOAD_BYTES:
        raise AssertionError(f"上传载荷长度错误：{len(payload)} != {UPLOAD_BYTES}")
    return payload


def verify_payload_roundtrip(case: FullLayerCase) -> str:
    payload = build_upload_payload(case)
    activation_end = ACTIVATION_BYTES
    weight_end = activation_end + WEIGHT_BYTES
    scale_end = weight_end + SCALE_BYTES

    unpacked_activation = np.frombuffer(
        payload[:activation_end], dtype=np.int8
    ).copy()
    unpacked_weights = unpack_int4_matrix(payload[activation_end:weight_end])
    scale_rows = np.frombuffer(
        payload[weight_end:scale_end], dtype="<u4"
    ).reshape(M, SCALE_ROW_BYTES // 4)
    bias_rows = np.frombuffer(payload[scale_end:], dtype="<i8").reshape(
        M, BIAS_ROW_BYTES // 8
    )

    if not np.array_equal(unpacked_activation, case.activation.astype(np.int8)):
        raise RuntimeError("activation 上传往返不一致")
    if not np.array_equal(unpacked_weights, case.weights.astype(np.int8)):
        raise RuntimeError("packed INT4 上传往返不一致")
    if not np.array_equal(
        scale_rows[:, :GROUPS], case.scales_q28.astype(np.uint32)
    ):
        raise RuntimeError("UQ4.28 scale 上传往返不一致")
    if np.any(scale_rows[:, GROUPS:] != 0):
        raise RuntimeError("scale 行补齐区域必须为 0")
    if not np.array_equal(bias_rows[:, 0], case.bias_q28.astype(np.int64)):
        raise RuntimeError("bias_q28 上传往返不一致")
    if np.any(bias_rows[:, 1:] != 0):
        raise RuntimeError("bias 行补齐区域必须为 0")
    return hashlib.sha256(payload).hexdigest()


def load_full_layer_model(image: P50Image) -> FullLayerModel:
    block = image.extract_block(DEFAULT_WEIGHT, 0, M, 0, K)
    if block.quantized is None or block.scales is None:
        raise RuntimeError("q_proj 权重不是分组 INT4 张量")
    bias = image.read_float16_tensor(DEFAULT_BIAS).astype(np.float32).reshape(-1)
    _require_shape(bias, (M,), "bias")
    return FullLayerModel(
        weights=block.quantized.astype(np.int8),
        weight_scales=block.scales.astype(np.float32),
        bias=bias,
    )


def case_from_model(
    model: FullLayerModel,
    activation_seed: int = DEFAULT_ACTIVATION_SEED,
    label: str | None = None,
) -> FullLayerCase:
    activation_float = make_deterministic_activation(K, seed=activation_seed)
    result = compute_groupwise_linear_reference(
        weight_quantized=model.weights,
        weight_scales=model.weight_scales,
        activation_values=activation_float,
        bias=model.bias,
        group_size=GROUP_SIZE,
        weight_name=DEFAULT_WEIGHT,
        bias_name=DEFAULT_BIAS,
    )
    if result.combined_scale_saturated_count:
        raise RuntimeError("真实完整层 combined scale 出现 UQ4.28 饱和")
    case = FullLayerCase(
        activation=result.activation.quantized.astype(np.int8),
        weights=model.weights,
        scales_q28=result.combined_scale_q28.astype(np.uint32),
        bias_q28=result.bias_q28.astype(np.int64),
        expected_q28=result.output_fixed_q28.astype(np.int64),
        activation_scale=float(result.activation.scale),
        label=label or f"layer0 q_proj 完整层 seed={activation_seed}",
    )
    validate_case(case)
    recomputed = compute_q28_reference(
        case.activation, case.weights, case.scales_q28, case.bias_q28
    )
    if not np.array_equal(recomputed, case.expected_q28):
        raise RuntimeError("独立 Q28 重算与 linear_quant_reference 不一致")
    return case


def load_real_case(
    image: P50Image,
    activation_seed: int = DEFAULT_ACTIVATION_SEED,
    label: str | None = None,
) -> FullLayerCase:
    return case_from_model(
        load_full_layer_model(image), activation_seed=activation_seed, label=label
    )


def validate_fixed_case(case: FullLayerCase, manifest_path: Path) -> dict[str, str]:
    if not np.array_equal(case.expected_q28[:4], EXPECTED_FIRST4_Q28):
        raise RuntimeError("完整层前 4 行与已验证 M4K896 基线不一致")
    output_hash = sha256_array(case.expected_q28, "<i8")
    if output_hash != EXPECTED_FIXED_OUTPUT_SHA256:
        raise RuntimeError(
            f"完整层固定输出哈希变化：{output_hash} != {EXPECTED_FIXED_OUTPUT_SHA256}"
        )
    payload_hash = verify_payload_roundtrip(case)
    packed_weights = pack_int4_low_nibble_first(case.weights)
    hashes = {
        "activation_int8": sha256_array(case.activation, np.int8),
        "packed_weight_int4": sha256_array(packed_weights, np.uint8),
        "combined_scale_uq4_28": sha256_array(case.scales_q28, "<u4"),
        "bias_q28": sha256_array(case.bias_q28, "<i8"),
        "output_fixed_q28": output_hash,
        "upload_payload": payload_hash,
    }
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("sha256") != hashes:
            raise RuntimeError("完整层固定向量与 JSON 清单哈希不一致")
    return hashes


def read_exact(port: "serial.Serial", size: int, timeout: float = 60.0) -> bytes:
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
        write_timeout=120.0,
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


def load_case(port: "serial.Serial", case: FullLayerCase) -> None:
    payload = build_upload_payload(case)
    port.write(b"L")
    write_all(port, payload)
    read_ack(port)


def run_loaded_case(port: "serial.Serial") -> np.ndarray:
    port.write(b"G")
    reply = read_exact(port, 1 + RESULT_BYTES, timeout=60.0)
    raise_if_error_frame(reply)
    if reply[0:1] != b"R":
        raise RuntimeError(f"Q28 结果帧头错误：{reply[:16]!r}")
    return np.frombuffer(reply[1:], dtype="<i8").copy()


def run_and_compare(port: "serial.Serial", case: FullLayerCase) -> np.ndarray:
    load_case(port, case)
    fpga = run_loaded_case(port)
    if not np.array_equal(fpga, case.expected_q28):
        mismatch = np.flatnonzero(fpga != case.expected_q28)
        first = int(mismatch[0])
        raise RuntimeError(
            f"{case.label} 不一致：首个错误行={first}，"
            f"FPGA={int(fpga[first])}，Python={int(case.expected_q28[first])}，"
            f"总错误数={mismatch.size}"
        )
    return fpga


def command_fixed(
    port: "serial.Serial", image_path: Path, manifest_path: Path
) -> None:
    wait_until_ready(port)
    image = P50Image(image_path)
    image.validate()
    model = load_full_layer_model(image)
    case = case_from_model(model)
    hashes = validate_fixed_case(case, manifest_path)
    started = time.monotonic()
    fpga = run_and_compare(port, case)
    elapsed = time.monotonic() - started
    print("真实 layer0 q_proj M896K896 完整层逐位一致：PASS")
    print(f"输出 SHA256：{hashes['output_fixed_q28']}")
    print(f"前 8 行：{fpga[:8].tolist()}")
    print(f"后 8 行：{fpga[-8:].tolist()}")
    print(f"上传、计算与回读总耗时：{elapsed:.2f} 秒")


def command_stress(
    port: "serial.Serial", image_path: Path, rounds: int, seed: int
) -> None:
    if rounds <= 0:
        raise ValueError("rounds 必须大于 0")
    wait_until_ready(port)
    image = P50Image(image_path)
    image.validate()
    model = load_full_layer_model(image)
    started = time.monotonic()
    for index in range(rounds):
        activation_seed = seed + index
        case = case_from_model(
            model,
            activation_seed=activation_seed,
            label=f"完整 q_proj 随机激活 {index + 1}/{rounds}",
        )
        run_and_compare(port, case)
        print(
            f"真实完整层随机激活已通过 {index + 1}/{rounds}，"
            f"activation_seed={activation_seed}"
        )
    elapsed = time.monotonic() - started
    print(
        f"完整 q_proj 真实 FPGA 随机激活回归 PASS：{rounds}/{rounds}，"
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
    model = load_full_layer_model(image)
    started = time.monotonic()

    fixed = case_from_model(model)
    hashes = validate_fixed_case(fixed, manifest_path)
    print("完整层固定载荷打包/解包和独立 Q28 重算：PASS")
    print(f"上传载荷：{UPLOAD_BYTES} B")
    print(f"输出 SHA256：{hashes['output_fixed_q28']}")
    print(f"上传载荷 SHA256：{hashes['upload_payload']}")
    print(f"前 8 行：{fixed.expected_q28[:8].tolist()}")
    print(f"后 8 行：{fixed.expected_q28[-8:].tolist()}")
    print("固定向量哈希：")
    print(json.dumps(hashes, ensure_ascii=False, indent=2))

    for index in range(rounds):
        activation_seed = seed + index
        case = case_from_model(model, activation_seed=activation_seed)
        if index == 0 or index + 1 == rounds or (index + 1) % 10 == 0:
            print(
                f"完整层软件随机激活已通过 {index + 1}/{rounds}，"
                f"activation_seed={activation_seed}"
            )

    elapsed = time.monotonic() - started
    print(
        f"完整 q_proj 软件参考压力测试 PASS：{rounds}/{rounds}，"
        f"seed_start={seed}，耗时 {elapsed:.2f} 秒"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PGL50H layer0 q_proj 完整 M896K896 分组 Q28 Linear 上位机"
    )
    parser.add_argument("--port", default="COM20")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("ports", help="列出串口")
    sub.add_parser("info", help="读取固件信息")
    sub.add_parser("status", help="读取状态")

    fixed = sub.add_parser("fixed", help="运行真实 layer0 q_proj 完整层固定向量")
    fixed.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    fixed.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)

    stress = sub.add_parser("stress", help="运行真实完整层随机激活上板回归")
    stress.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    stress.add_argument("--rounds", type=int, default=3)
    stress.add_argument("--seed", type=int, default=20260725)

    selftest = sub.add_parser("selftest", help="只运行完整层载荷和软件金标准自检")
    selftest.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    selftest.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    selftest.add_argument("--rounds", type=int, default=100)
    selftest.add_argument("--seed", type=int, default=20260725)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.command == "ports":
            show_ports()
            return 0
        if args.command == "selftest":
            command_selftest(
                args.image, args.manifest, args.rounds, args.seed
            )
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
