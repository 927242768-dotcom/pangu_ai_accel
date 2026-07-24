#!/usr/bin/env python3
"""盘古 PGL50H F3 KV Cache 定点闭环验证工具。"""

from __future__ import annotations

import argparse
import hashlib
import struct
import sys
import time
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

from model_tools.kv_cache_reference import (  # noqa: E402
    DEFAULT_FIXED_SLOTS,
    DEFAULT_IMAGE,
    DEFAULT_MANIFEST,
    DEFAULT_STRESS_SEED,
    MAX_CONTEXT,
    MAX_READ_TOKENS,
    NUM_LAYERS,
    TOKEN_SLOT_BYTES,
    KVTokenCase,
    build_fixed_real_cases,
    build_token_payload,
    decode_token_payload,
    make_deterministic_token,
    sha256_bytes,
    software_stress,
    validate_manifest,
)

BAUD_RATE = 115200
DEFAULT_BOARD_TOKENS = 300

ERROR_MESSAGES = {
    0x01: "未知命令",
    0x02: "DDR3 尚未初始化完成",
    0x03: "尚未配置 layer/position",
    0x04: "layer 或起始 position 配置非法",
    0x05: "上下文已满，禁止继续写入",
    0x06: "历史读取范围非法",
    0xFF: "FPGA 状态机异常",
}


class FpgaProtocolError(RuntimeError):
    def __init__(self, code: int):
        self.code = int(code)
        super().__init__(
            f"FPGA 返回错误 0x{self.code:02X}："
            f"{ERROR_MESSAGES.get(self.code, '未知错误')}"
        )


class FpgaStatus:
    def __init__(
        self,
        flags: int,
        layer: int,
        start_position: int,
        current_position: int,
        written_count: int,
    ):
        self.ddr_ready = bool(flags & 0x01)
        self.configured = bool(flags & 0x02)
        self.write_valid = bool(flags & 0x04)
        self.read_valid = bool(flags & 0x08)
        self.context_full = bool(flags & 0x20)
        self.protocol_error = bool(flags & 0x40)
        self.layer = layer
        self.start_position = start_position
        self.current_position = current_position
        self.written_count = written_count


def require_pyserial() -> None:
    if serial is None:
        raise SystemExit("缺少 pyserial，请运行：python -m pip install pyserial")


def read_exact(port: "serial.Serial", size: int, timeout: float = 180.0) -> bytes:
    data = bytearray()
    deadline = time.monotonic() + timeout
    while len(data) < size:
        chunk = port.read(size - len(data))
        if chunk:
            data.extend(chunk)
        elif time.monotonic() >= deadline:
            raise TimeoutError(f"串口超时：期望 {size} 字节，只收到 {len(data)} 字节")
    return bytes(data)


def read_frame_prefix(port: "serial.Serial", expected: bytes) -> bytes:
    first = read_exact(port, 1, timeout=30.0)
    if first == b"E":
        rest = read_exact(port, 3, timeout=5.0)
        if rest[-2:] != b"\r\n":
            raise RuntimeError(f"错误帧格式错误：{first + rest!r}")
        raise FpgaProtocolError(rest[0])
    if first != expected:
        raise RuntimeError(f"FPGA 帧头错误：收到 {first!r}，预期 {expected!r}")
    return first


def read_ack(port: "serial.Serial") -> None:
    read_frame_prefix(port, b"K")
    tail = read_exact(port, 2, timeout=5.0)
    if tail != b"\r\n":
        raise RuntimeError(f"FPGA 确认帧错误：{b'K' + tail!r}")


def write_all(port: "serial.Serial", payload: bytes, chunk_size: int = 4096) -> None:
    sent = 0
    view = memoryview(payload)
    while sent < len(payload):
        end = min(sent + chunk_size, len(payload))
        count = port.write(view[sent:end])
        if not count:
            raise TimeoutError("串口写入没有前进")
        sent += count
    port.flush()


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
    if reply.startswith(b"E") and len(reply) >= 2:
        raise FpgaProtocolError(reply[1])
    text = reply.decode("ascii", errors="replace").strip()
    print(text)
    return text


def command_status(port: "serial.Serial", *, quiet: bool = False) -> FpgaStatus:
    port.write(b"S")
    read_frame_prefix(port, b"S")
    body = read_exact(port, 10, timeout=10.0)
    if body[-2:] != b"\r\n":
        raise RuntimeError(f"状态帧格式错误：{b'S' + body!r}")
    status = FpgaStatus(
        flags=body[0],
        layer=body[1],
        start_position=int.from_bytes(body[2:4], "little"),
        current_position=int.from_bytes(body[4:6], "little"),
        written_count=int.from_bytes(body[6:8], "little"),
    )
    if not quiet:
        print(
            "DDR3={}，已配置={}，写有效={}，读有效={}，上下文满={}，协议错误={}，"
            "layer={}，start={}，current={}，written={}".format(
                "是" if status.ddr_ready else "否",
                "是" if status.configured else "否",
                "是" if status.write_valid else "否",
                "是" if status.read_valid else "否",
                "是" if status.context_full else "否",
                "是" if status.protocol_error else "否",
                status.layer,
                status.start_position,
                status.current_position,
                status.written_count,
            )
        )
    return status


def wait_until_ready(port: "serial.Serial", timeout: float = 20.0) -> None:
    deadline = time.monotonic() + timeout
    while True:
        status = command_status(port, quiet=True)
        if status.ddr_ready:
            return
        if time.monotonic() >= deadline:
            raise TimeoutError("等待 DDR3 初始化完成超时")
        time.sleep(0.25)


def configure(port: "serial.Serial", layer: int, start_position: int) -> None:
    if not 0 <= layer < NUM_LAYERS:
        raise ValueError(f"layer 必须位于 0..{NUM_LAYERS - 1}")
    if not 0 <= start_position < MAX_CONTEXT:
        raise ValueError(f"start_position 必须位于 0..{MAX_CONTEXT - 1}")
    port.write(b"C" + struct.pack("<BH", layer, start_position))
    read_ack(port)


def reset_cursor(port: "serial.Serial") -> None:
    port.write(b"Z")
    read_ack(port)


def write_token(port: "serial.Serial", payload: bytes) -> tuple[int, int]:
    if len(payload) != TOKEN_SLOT_BYTES:
        raise ValueError(f"token 载荷必须为 {TOKEN_SLOT_BYTES} B")
    write_all(port, b"W" + payload)
    read_frame_prefix(port, b"K")
    body = read_exact(port, 5, timeout=30.0)
    if body[-2:] != b"\r\n":
        raise RuntimeError(f"写确认帧格式错误：{b'K' + body!r}")
    return body[0], int.from_bytes(body[1:3], "little")


def expect_write_error(port: "serial.Serial", expected_code: int) -> None:
    port.write(b"W")
    try:
        read_frame_prefix(port, b"K")
    except FpgaProtocolError as error:
        if error.code != expected_code:
            raise RuntimeError(
                f"错误码不一致：收到 0x{error.code:02X}，预期 0x{expected_code:02X}"
            ) from error
        return
    raise RuntimeError("预期写入被拒绝，但 FPGA 未返回错误")


def read_history(port: "serial.Serial", start_position: int, count: int) -> tuple[int, int, list[bytes]]:
    if not 0 <= start_position < MAX_CONTEXT:
        raise ValueError("start_position 越界")
    if not 1 <= count <= MAX_READ_TOKENS:
        raise ValueError(f"count 必须位于 1..{MAX_READ_TOKENS}")
    if start_position + count > MAX_CONTEXT:
        raise ValueError("读取范围超出硬件上下文")

    port.write(b"R" + struct.pack("<HB", start_position, count))
    read_frame_prefix(port, b"D")
    header = read_exact(port, 4, timeout=10.0)
    layer = header[0]
    returned_start = int.from_bytes(header[1:3], "little")
    returned_count = header[3]
    if returned_start != start_position or returned_count != count:
        raise RuntimeError(
            f"历史读取头错误：start={returned_start}/{start_position}，"
            f"count={returned_count}/{count}"
        )
    raw = read_exact(port, count * TOKEN_SLOT_BYTES, timeout=240.0)
    tokens = [
        raw[index * TOKEN_SLOT_BYTES : (index + 1) * TOKEN_SLOT_BYTES]
        for index in range(count)
    ]
    return layer, returned_start, tokens


def compare_payload(actual: bytes, expected: bytes, label: str) -> None:
    if actual == expected:
        return
    actual_values = np.frombuffer(actual, dtype="<i8")
    expected_values = np.frombuffer(expected, dtype="<i8")
    mismatch = np.flatnonzero(actual_values != expected_values)
    index = int(mismatch[0])
    kind = "K" if index < 128 else "V"
    vector_index = index if index < 128 else index - 128
    head, dim = divmod(vector_index, 64)
    raise RuntimeError(
        f"{label} 不一致：{kind}[head={head},dim={dim}]，"
        f"FPGA={int(actual_values[index])}，Python={int(expected_values[index])}，"
        f"总错误数={mismatch.size}"
    )


def write_cases_and_readback(port: "serial.Serial", cases: Sequence[KVTokenCase]) -> None:
    if not cases:
        raise ValueError("cases 不能为空")
    layer = cases[0].layer
    start = cases[0].position
    expected_positions = list(range(start, start + len(cases)))
    if any(case.layer != layer for case in cases):
        raise ValueError("同一批 cases 必须位于同一 layer")
    if [case.position for case in cases] != expected_positions:
        raise ValueError("同一批 cases 必须为连续 position")

    configure(port, layer, start)
    payloads = []
    for case in cases:
        payload = case.payload
        ack_layer, ack_position = write_token(port, payload)
        if ack_layer != layer or ack_position != case.position:
            raise RuntimeError(
                f"写确认地址错误：layer={ack_layer}/{layer}，"
                f"position={ack_position}/{case.position}"
            )
        payloads.append(payload)

    read_layer, _, returned = read_history(port, start, len(cases))
    if read_layer != layer:
        raise RuntimeError(f"历史读取 layer 错误：{read_layer} != {layer}")
    for index, (actual, expected) in enumerate(zip(returned, payloads)):
        compare_payload(actual, expected, f"layer={layer}, position={start + index}")


def command_fixed(port: "serial.Serial", image_path: Path, manifest_path: Path) -> None:
    wait_until_ready(port)
    cases = build_fixed_real_cases(DEFAULT_FIXED_SLOTS, image_path=image_path)
    manifest = validate_manifest(cases, manifest_path)

    started = time.monotonic()
    write_cases_and_readback(port, cases[:2])
    print("真实 layer0 position=0..1 连续写入、自动推进与历史顺序读取：PASS")

    for case in cases[2:]:
        write_cases_and_readback(port, (case,))
        print(
            f"真实 K/V layer={case.layer}, position={case.position} 逐位一致：PASS，"
            f"SHA256={sha256_bytes(case.payload)}"
        )

    status = command_status(port, quiet=True)
    if not status.context_full or status.current_position != MAX_CONTEXT:
        raise RuntimeError(
            f"末槽状态错误：full={status.context_full}, current={status.current_position}"
        )
    expect_write_error(port, 0x05)
    print("layer27 最后槽结束于 1 GiB，下一 token 写入被正确拒绝：PASS")

    if manifest["layout"]["kv_end_bytes"] != 1 << 30:
        raise RuntimeError("固定清单的 KV 结束地址不是 1 GiB")
    print(f"固定真实 K/V 全部通过，耗时 {time.monotonic() - started:.2f} 秒")


def command_isolation(port: "serial.Serial", seed: int) -> None:
    """同一 position 写入不同层，再跨配置回读，验证层间不覆盖。"""

    wait_until_ready(port)
    position = 4096
    layer_a, layer_b = 3, 17
    k_a, v_a = make_deterministic_token(seed)
    k_b, v_b = make_deterministic_token(seed + 1)
    payload_a = build_token_payload(k_a, v_a)
    payload_b = build_token_payload(k_b, v_b)

    configure(port, layer_a, position)
    write_token(port, payload_a)
    configure(port, layer_b, position)
    write_token(port, payload_b)

    configure(port, layer_b, 0)
    read_layer_b, _, tokens_b = read_history(port, position, 1)
    compare_payload(tokens_b[0], payload_b, "layer B 隔离回读")
    configure(port, layer_a, 0)
    read_layer_a, _, tokens_a = read_history(port, position, 1)
    compare_payload(tokens_a[0], payload_a, "layer A 隔离回读")
    if read_layer_a != layer_a or read_layer_b != layer_b or payload_a == payload_b:
        raise RuntimeError("层间隔离元数据错误")
    print(
        f"层间防覆盖 PASS：layer {layer_a}/{layer_b}，position={position}，"
        f"A={hashlib.sha256(payload_a).hexdigest()}，"
        f"B={hashlib.sha256(payload_b).hexdigest()}"
    )


def command_stress(port: "serial.Serial", tokens: int, seed: int) -> None:
    if tokens <= 0:
        raise ValueError("tokens 必须大于 0")
    wait_until_ready(port)
    rng = np.random.default_rng(seed)
    passed = 0
    batch_index = 0
    remembered: tuple[int, int, bytes] | None = None
    started = time.monotonic()

    while passed < tokens:
        count = min(
            int(rng.integers(1, MAX_READ_TOKENS + 1)),
            tokens - passed,
        )
        layer = int(rng.integers(0, NUM_LAYERS))
        start = int(rng.integers(0, MAX_CONTEXT - count + 1))
        configure(port, layer, start)

        expected_payloads: list[bytes] = []
        for offset in range(count):
            token_seed = seed ^ ((batch_index + 1) << 20) ^ (offset << 8) ^ layer ^ start
            k, v = make_deterministic_token(token_seed)
            payload = build_token_payload(k, v)
            ack_layer, ack_position = write_token(port, payload)
            if ack_layer != layer or ack_position != start + offset:
                raise RuntimeError("随机压力写确认 layer/position 错误")
            expected_payloads.append(payload)

        read_layer, _, actual_payloads = read_history(port, start, count)
        if read_layer != layer:
            raise RuntimeError("随机压力读取 layer 错误")
        for offset, (actual, expected) in enumerate(
            zip(actual_payloads, expected_payloads)
        ):
            compare_payload(
                actual,
                expected,
                f"stress layer={layer}, position={start + offset}",
            )

        if remembered is not None and batch_index % 8 == 0:
            old_layer, old_position, old_payload = remembered
            configure(port, old_layer, 0)
            returned_layer, _, old_readback = read_history(port, old_position, 1)
            if returned_layer != old_layer:
                raise RuntimeError("跨批回读 layer 错误")
            compare_payload(old_readback[0], old_payload, "跨批层间防覆盖")

        remembered = (layer, start + count - 1, expected_payloads[-1])
        passed += count
        batch_index += 1
        print(
            f"随机 KV Cache 已通过 {passed}/{tokens} token，"
            f"本批 layer={layer}, start={start}, count={count}"
        )

    print(
        f"KV Cache 真实 FPGA 随机层/位置回归 PASS：{tokens}/{tokens} token，"
        f"seed={seed}，耗时 {time.monotonic() - started:.2f} 秒"
    )


def command_selftest(
    image_path: Path,
    manifest_path: Path,
    rounds: int,
    seed: int,
) -> None:
    cases = build_fixed_real_cases(DEFAULT_FIXED_SLOTS, image_path=image_path)
    manifest = validate_manifest(cases, manifest_path)
    software_stress(rounds=rounds, seed=seed)
    print("F3 地址容量、首尾边界、连续布局、真实清单和载荷往返：PASS")
    print(manifest["layout"]["address_formula_ctrl"])
    print(
        f"KV 区域：{manifest['layout']['kv_total_bytes'] // (1 << 20)} MiB，"
        f"上下文上限：{manifest['layout']['max_context']} token"
    )
    print(f"KV Cache 软件随机压力 PASS：{rounds}/{rounds}，seed={seed}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PGL50H F3 KV Cache 上位机")
    parser.add_argument("--port", default="COM20")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("ports", help="列出串口")
    sub.add_parser("info", help="读取固件信息")
    sub.add_parser("status", help="读取状态")

    fixed = sub.add_parser("fixed", help="运行真实 K/V 固定与末地址边界测试")
    fixed.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    fixed.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)

    isolation = sub.add_parser("isolation", help="验证不同层同位置互不覆盖")
    isolation.add_argument("--seed", type=int, default=DEFAULT_STRESS_SEED)

    stress = sub.add_parser("stress", help="随机层/位置写入和历史读取回归")
    stress.add_argument("--tokens", type=int, default=DEFAULT_BOARD_TOKENS)
    stress.add_argument("--seed", type=int, default=DEFAULT_STRESS_SEED)

    selftest = sub.add_parser("selftest", help="只运行软件参考和载荷自检")
    selftest.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    selftest.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    selftest.add_argument("--rounds", type=int, default=1000)
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
                command_fixed(port, args.image, args.manifest)
            elif args.command == "isolation":
                command_isolation(port, args.seed)
            elif args.command == "stress":
                command_stress(port, args.tokens, args.seed)
            else:  # pragma: no cover
                raise AssertionError(args.command)
        return 0
    except (
        FileNotFoundError,
        OSError,
        RuntimeError,
        TimeoutError,
        ValueError,
    ) as error:
        print(f"错误：{error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
