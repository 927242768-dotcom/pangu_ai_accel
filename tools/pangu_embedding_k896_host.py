#!/usr/bin/env python3
"""盘古 PGL50H 真实 tied Embedding K=896 验证工具。"""

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

from model_tools.embedding_fixed_reference import (  # noqa: E402
    DEFAULT_FIXED_TOKEN_IDS,
    DEFAULT_IMAGE,
    DEFAULT_RANDOM_SEED,
    DEFAULT_TENSOR,
    EMBEDDING_DIM,
    GROUPS_PER_ROW,
    Q10_FACTOR,
    ROW_SLOT_BYTES,
    VOCAB_SIZE,
    EmbeddingReferenceResult,
    build_manifest,
    compute_embedding_reference,
    embedding_slot_ctrl_addr,
    make_random_token_ids,
    pack_embedding_payload,
    unpack_embedding_payload,
)
from model_tools.p50_format import P50Image  # noqa: E402

BAUD_RATE = 115200
RESULT_BYTES = EMBEDDING_DIM * 2
DEFAULT_MANIFEST = PROJECT_ROOT / "model_tools/embedding_k896_reference.json"

ERROR_MESSAGES = {
    0x01: "未知命令",
    0x02: "DDR3 尚未初始化完成",
    0x03: "Token ID 越界",
    0x04: "尚未配置 Token ID",
    0x05: "尚未加载该 Token 的 Embedding 行",
    0xFF: "FPGA 状态机异常",
}


@dataclass(frozen=True)
class FpgaStatus:
    ddr_ready: bool
    row_loaded: bool
    result_valid: bool
    core_busy: bool
    configured: bool


def require_pyserial() -> None:
    if serial is None:
        raise SystemExit("缺少 pyserial，请运行：python -m pip install pyserial")


def sha256_array(array: np.ndarray, dtype: str | np.dtype) -> str:
    return hashlib.sha256(np.asarray(array, dtype=dtype).tobytes(order="C")).hexdigest()


def reference_from_image(
    image: P50Image, token_id: int, tensor_name: str = DEFAULT_TENSOR
) -> EmbeddingReferenceResult:
    quantized, scales, _ = image.read_int4_row(tensor_name, int(token_id))
    return compute_embedding_reference(
        token_id=int(token_id),
        quantized_int4=quantized,
        scales_fp16=scales,
        tensor_name=tensor_name,
        vocab_size=VOCAB_SIZE,
    )


def verify_payload_roundtrip(result: EmbeddingReferenceResult) -> str:
    payload = pack_embedding_payload(result)
    quantized, scales_q28, padding = unpack_embedding_payload(payload)
    if not np.array_equal(quantized, result.quantized_int4):
        raise RuntimeError("Embedding INT4 载荷往返不一致")
    if not np.array_equal(scales_q28, result.scales_q28):
        raise RuntimeError("Embedding Q28 scale 载荷往返不一致")
    if padding != b"\x00" * 8:
        raise RuntimeError("Embedding 行槽补齐区域不是 0")
    return hashlib.sha256(payload).hexdigest()


def validate_manifest(image_path: Path, manifest_path: Path) -> None:
    generated = build_manifest(image_path, DEFAULT_FIXED_TOKEN_IDS)
    committed = json.loads(manifest_path.read_text(encoding="utf-8"))
    if generated != committed:
        raise RuntimeError("E3 固定 Token 清单与真实 P50 当前结果不一致")


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
        row_loaded=bool(flags & 0x02),
        result_valid=bool(flags & 0x04),
        core_busy=bool(flags & 0x08),
        configured=bool(flags & 0x10),
    )
    print(
        "DDR3初始化={}，行已加载={}，结果有效={}，计算核心忙={}，Token配置={}".format(
            "是" if status.ddr_ready else "否",
            "是" if status.row_loaded else "否",
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


def configure_token(port: "serial.Serial", token_id: int) -> None:
    if not 0 <= int(token_id) < VOCAB_SIZE:
        raise ValueError(f"Token ID 越界：{token_id}")
    port.write(b"C" + struct.pack("<I", int(token_id)))
    port.flush()
    read_ack(port)


def load_row(port: "serial.Serial", result: EmbeddingReferenceResult) -> None:
    payload = pack_embedding_payload(result)
    if len(payload) != ROW_SLOT_BYTES:
        raise AssertionError("Embedding 上传载荷不是 512 B")
    port.write(b"L")
    port.write(payload)
    port.flush()
    read_ack(port)


def run_loaded_embedding(port: "serial.Serial") -> np.ndarray:
    port.write(b"G")
    reply = read_exact(port, 1 + RESULT_BYTES, timeout=30.0)
    raise_if_error_frame(reply)
    if reply[0:1] != b"R":
        raise RuntimeError(f"Embedding 结果帧头错误：{reply[:16]!r}")
    return np.frombuffer(reply[1:], dtype="<i2").copy()


def run_and_compare(
    port: "serial.Serial", result: EmbeddingReferenceResult
) -> np.ndarray:
    configure_token(port, result.token_id)
    load_row(port, result)
    fpga = run_loaded_embedding(port)
    expected = result.fixed_q10
    if not np.array_equal(fpga, expected):
        mismatch = np.flatnonzero(fpga != expected)
        first = int(mismatch[0])
        raise RuntimeError(
            f"Token ID {result.token_id} Embedding 不一致：index={first}，"
            f"FPGA={int(fpga[first])}，Python={int(expected[first])}，"
            f"总错误数={mismatch.size}"
        )
    return fpga


def command_fixed(port: "serial.Serial", image_path: Path, manifest_path: Path) -> None:
    wait_until_ready(port)
    validate_manifest(image_path, manifest_path)
    image = P50Image(image_path)
    started = time.monotonic()
    for token_id in DEFAULT_FIXED_TOKEN_IDS:
        result = reference_from_image(image, int(token_id))
        payload_hash = verify_payload_roundtrip(result)
        fpga = run_and_compare(port, result)
        print(
            f"Token ID {token_id} 固定真实 Embedding 逐位一致：PASS，"
            f"slot=0x{embedding_slot_ctrl_addr(int(token_id)):07x}，"
            f"output_SHA256={sha256_array(fpga, '<i2')}，"
            f"payload_SHA256={payload_hash}，前16项={fpga[:16].tolist()}"
        )
    elapsed = time.monotonic() - started
    print(
        f"E3 四个固定 Token ID 全部通过：{list(DEFAULT_FIXED_TOKEN_IDS)}，"
        f"上传、计算与回读总耗时 {elapsed:.2f} 秒"
    )


def command_stress(
    port: "serial.Serial", image_path: Path, rounds: int, seed: int
) -> None:
    if rounds <= 0:
        raise ValueError("rounds 必须大于 0")
    wait_until_ready(port)
    image = P50Image(image_path)
    token_ids = make_random_token_ids(rounds, seed)
    started = time.monotonic()
    for index, token_id_raw in enumerate(token_ids.tolist()):
        token_id = int(token_id_raw)
        result = reference_from_image(image, token_id)
        verify_payload_roundtrip(result)
        run_and_compare(port, result)
        if index == 0 or index + 1 == rounds or (index + 1) % 10 == 0:
            print(
                f"E3 真实 FPGA 随机 Token 已通过 {index + 1}/{rounds}，"
                f"token_id={token_id}"
            )
    elapsed = time.monotonic() - started
    print(
        f"E3 真实 FPGA 随机 Token 回归 PASS：{rounds}/{rounds}，"
        f"seed={seed}，耗时 {elapsed:.2f} 秒"
    )


def command_selftest(
    image_path: Path, manifest_path: Path, rounds: int, seed: int
) -> None:
    if rounds <= 0:
        raise ValueError("rounds 必须大于 0")
    validate_manifest(image_path, manifest_path)
    image = P50Image(image_path)
    token_ids = make_random_token_ids(rounds, seed)
    started = time.monotonic()
    maximum_error = 0.0
    for index, token_id_raw in enumerate(token_ids.tolist()):
        result = reference_from_image(image, int(token_id_raw))
        verify_payload_roundtrip(result)
        if not np.array_equal(result.fixed_q10, result.direct_q10):
            raise RuntimeError(f"Token {result.token_id} 固定路径与直接 Q10 不一致")
        maximum_error = max(
            maximum_error,
            float(np.max(np.abs(result.q10_quantization_error))),
        )
        if index == 0 or index + 1 == rounds or (index + 1) % 100 == 0:
            print(
                f"E3 真实 P50 软件 Token 已通过 {index + 1}/{rounds}，"
                f"token_id={result.token_id}"
            )
    elapsed = time.monotonic() - started
    print(
        f"E3 真实 P50 软件/载荷压力 PASS：{rounds}/{rounds}，seed={seed}，"
        f"最大量化误差={maximum_error:.9f}，耗时 {elapsed:.2f} 秒"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PGL50H 真实 tied Embedding K896 上位机")
    parser.add_argument("--port", default="COM20")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("ports", help="列出串口")
    sub.add_parser("info", help="读取固件信息")
    sub.add_parser("status", help="读取状态")

    fixed = sub.add_parser("fixed", help="运行四个固定真实 Token ID")
    fixed.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    fixed.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)

    stress = sub.add_parser("stress", help="运行随机真实 Token ID 上板回归")
    stress.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    stress.add_argument("--rounds", type=int, default=100)
    stress.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)

    selftest = sub.add_parser("selftest", help="只运行真实 P50 软件和载荷自检")
    selftest.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    selftest.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    selftest.add_argument("--rounds", type=int, default=1000)
    selftest.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
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
        IndexError,
        KeyError,
        ValueError,
        RuntimeError,
        TimeoutError,
        struct.error,
    ) as error:
        print(f"错误：{error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
