#!/usr/bin/env python3
"""PGL50H H3 真实 24 层参数换入与顺序执行上位机。

固件协议在 G2 ``I/S/C/W/R/P/G`` 基础上新增：

- ``L``：读回当前 ``layer/query/window/count``；
- ``M``：DDR3 内部 32 B beat copy，用于层末 1792 B hidden 交接。

第一版严格使用 H2 slot A，每层先配置并读回、再上传 19 笔参数、提交、执行，
最后将 ``block_output_q10`` 复制回 ``block_hidden_q10``。当前不包含最终 RMSNorm、
LM Head、logits 或采样。115200 UART 仅适合正确性验证。
"""

from __future__ import annotations

import argparse
import hashlib
import struct
import sys
import time
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_TOOLS = PROJECT_ROOT / "model_tools"
if str(MODEL_TOOLS) not in sys.path:
    sys.path.insert(0, str(MODEL_TOOLS))

from full_model_layer_sequence import (  # noqa: E402
    ACTIVE_LAYER_COUNT,
    build_common_runtime_uploads,
    build_layer_sequence_manifest,
    build_layer_uploads,
    hidden_copy_contract,
    validate_execution_window,
)
from transformer_block_g2_payload import build_dynamic_uploads  # noqa: E402
from transformer_block_reference import (  # noqa: E402
    DEFAULT_IMAGE,
    build_fixed_real_cases,
    scratch_regions,
)
try:
    from .pangu_transformer_block_host import (  # type: ignore[attr-defined] # noqa: E402
        DEFAULT_BAUD,
        DEFAULT_PORT,
        DEFAULT_TIMEOUT,
        commit_payload,
        execute_block,
        open_port,
        print_ports,
        read_ack,
        read_exact,
        read_info,
        read_region,
        read_status,
        transfer_timeout,
        upload_many,
        write_all,
    )
except ImportError:
    from pangu_transformer_block_host import (  # noqa: E402
        DEFAULT_BAUD,
        DEFAULT_PORT,
        DEFAULT_TIMEOUT,
        commit_payload,
        execute_block,
        open_port,
        print_ports,
        read_ack,
        read_exact,
        read_info,
        read_region,
        read_status,
        transfer_timeout,
        upload_many,
        write_all,
    )

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

EXPECTED_INFO = "PANGU50K H3 LAYER V1"
CONFIG_STRUCT = struct.Struct("<4H")
COPY_HEADER = struct.Struct("<III")
CONFIG_FRAME = struct.Struct("<c4H2s")
BEAT_BYTES = 32


def configure_layer(
    port: object,
    *,
    layer: int,
    query_position: int,
    window_start: int,
    count: int,
    timeout: float,
) -> None:
    if not 0 <= layer < ACTIVE_LAYER_COUNT:
        raise ValueError("layer 必须为 0..23")
    validate_execution_window(query_position, window_start, count)
    payload = CONFIG_STRUCT.pack(layer, query_position, window_start, count)
    port.reset_input_buffer()
    write_all(port, b"C" + payload)
    read_ack(port, timeout)


def read_configuration(port: object, timeout: float) -> dict[str, int]:
    port.reset_input_buffer()
    write_all(port, b"L")
    frame = read_exact(port, CONFIG_FRAME.size, timeout)
    prefix, layer, query, window, count, suffix = CONFIG_FRAME.unpack(frame)
    if prefix != b"L" or suffix != b"\r\n":
        raise RuntimeError(f"配置读回帧错误：{frame!r}")
    return {
        "layer": layer,
        "query_position": query,
        "window_start": window,
        "count": count,
    }


def verify_configuration(
    actual: dict[str, int],
    *,
    layer: int,
    query_position: int,
    window_start: int,
    count: int,
) -> None:
    expected = {
        "layer": layer,
        "query_position": query_position,
        "window_start": window_start,
        "count": count,
    }
    if actual != expected:
        raise AssertionError(f"配置读回不一致：actual={actual}, expected={expected}")


def copy_region(
    port: object,
    *,
    source_controller_address: int,
    destination_controller_address: int,
    length: int,
    baud: int,
    timeout: float,
) -> float:
    if source_controller_address < 0 or destination_controller_address < 0:
        raise ValueError("copy 地址不能为负")
    if source_controller_address % 8 or destination_controller_address % 8:
        raise ValueError("copy controller 地址必须按 8 word/32 B 对齐")
    if length <= 0 or length % BEAT_BYTES:
        raise ValueError("copy length 必须为正且按 32 B 对齐")
    source_end = source_controller_address + length // 4
    destination_end = destination_controller_address + length // 4
    if not (
        source_end <= destination_controller_address
        or destination_end <= source_controller_address
    ):
        raise ValueError("copy 源/目标范围不能重叠")
    port.reset_input_buffer()
    started = time.perf_counter()
    write_all(
        port,
        b"M"
        + COPY_HEADER.pack(
            source_controller_address,
            destination_controller_address,
            length,
        ),
    )
    read_ack(port, transfer_timeout(length, baud, timeout))
    return time.perf_counter() - started


def read_block_output_hash(
    port: object,
    *,
    baud: int,
    timeout: float,
) -> str:
    regions = {item.name: item for item in scratch_regions()}
    output = regions["block_output_q10"]
    payload = read_region(
        port,
        output.controller_address,
        output.size_bytes,
        baud=baud,
        timeout=timeout,
    )
    return hashlib.sha256(payload).hexdigest()


def _check_idle_status(port: object, timeout: float) -> dict[str, object]:
    status = read_status(port, timeout)
    if not status["ddr_init_done"]:
        raise RuntimeError(f"DDR3 尚未初始化：{status}")
    if status["block_busy"] or status["block_error"] or status["protocol_error"]:
        raise RuntimeError(f"H3 状态异常：{status}")
    return status


def run_layer(
    port: object,
    *,
    layer: int,
    query_position: int,
    window_start: int,
    count: int,
    image: Path,
    baud: int,
    timeout: float,
    verify_load: bool,
    copy_output: bool,
    read_output: bool,
) -> dict[str, object]:
    # C 会清除本轮 any_write/commit 标志，因此必须先配置，再上传参数。
    configure_layer(
        port,
        layer=layer,
        query_position=query_position,
        window_start=window_start,
        count=count,
        timeout=timeout,
    )
    config = read_configuration(port, timeout)
    verify_configuration(
        config,
        layer=layer,
        query_position=query_position,
        window_start=window_start,
        count=count,
    )

    uploads = build_layer_uploads(layer, image_path=image)
    upload_seconds = upload_many(
        port,
        uploads,
        baud=baud,
        timeout=timeout,
        verify=verify_load,
    )
    commit_payload(port, timeout)
    execution = execute_block(port, timeout)
    output_sha256 = (
        read_block_output_hash(port, baud=baud, timeout=timeout)
        if read_output
        else None
    )

    copy_seconds = 0.0
    if copy_output:
        copy = hidden_copy_contract()
        copy_seconds = copy_region(
            port,
            source_controller_address=int(copy["source_controller_address"]),
            destination_controller_address=int(copy["destination_controller_address"]),
            length=int(copy["length"]),
            baud=baud,
            timeout=timeout,
        )

    status = _check_idle_status(port, timeout)
    result = {
        "layer": layer,
        "configuration": config,
        "upload_seconds": upload_seconds,
        "execution": execution,
        "output_sha256": output_sha256,
        "copy_output_to_input": copy_output,
        "copy_seconds": copy_seconds,
        "status": status,
    }
    print(
        f"LAYER {layer} PASS: stage={execution['stage_name']}, "
        f"cycles={execution['cycles']}, copy={copy_output}, "
        f"output_sha256={output_sha256 or 'not-read'}"
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="H3 真实 24 层参数换入与顺序执行")
    parser.add_argument("--port", default=DEFAULT_PORT)
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("ports")
    sub.add_parser("info")
    sub.add_parser("status")
    sub.add_parser("config")
    sub.add_parser("dry-run")

    tables = sub.add_parser("tables")
    tables.add_argument("--verify", action="store_true")

    prepare = sub.add_parser("prepare-query0")
    prepare.add_argument("--verify", action="store_true")

    load = sub.add_parser("load-layer")
    load.add_argument("layer", type=int)
    load.add_argument("--query", type=int, default=0)
    load.add_argument("--window", type=int, default=0)
    load.add_argument("--count", type=int, default=1)
    load.add_argument("--verify", action="store_true")
    load.add_argument("--commit", action="store_true")

    sub.add_parser("copy-hidden")

    run = sub.add_parser("run-layer")
    run.add_argument("layer", type=int)
    run.add_argument("--query", type=int, default=0)
    run.add_argument("--window", type=int, default=0)
    run.add_argument("--count", type=int, default=1)
    run.add_argument("--verify-load", action="store_true")
    run.add_argument("--copy-output", action="store_true")
    run.add_argument("--read-output", action="store_true")

    sequence = sub.add_parser("run-sequence")
    sequence.add_argument("--start-layer", type=int, default=0)
    sequence.add_argument("--end-layer", type=int, default=23)
    sequence.add_argument("--query", type=int, default=0)
    sequence.add_argument("--window", type=int, default=0)
    sequence.add_argument("--count", type=int, default=1)
    sequence.add_argument("--load-tables", action="store_true")
    sequence.add_argument("--prepare-query0", action="store_true")
    sequence.add_argument("--verify-load", action="store_true")
    sequence.add_argument("--read-each-output", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "ports":
        print_ports()
        return 0
    if args.command == "dry-run":
        manifest = build_layer_sequence_manifest(image_path=args.image)
        print(
            f"H3 dry-run: layers={manifest['start_layer']}..{manifest['end_layer']}, "
            f"uploads={manifest['total_upload_transactions']}, "
            f"bytes={manifest['total_upload_bytes']}, "
            f"hidden_copies={manifest['hidden_copy_count']}"
        )
        return 0

    with open_port(args.port, args.baud) as port:
        if args.command == "info":
            print(read_info(port, args.timeout))
            return 0
        if args.command == "status":
            print(read_status(port, args.timeout))
            return 0
        if args.command == "config":
            print(read_configuration(port, args.timeout))
            return 0

        info = read_info(port, args.timeout)
        if info != EXPECTED_INFO:
            raise RuntimeError(f"固件标识错误：{info!r} != {EXPECTED_INFO!r}")
        status = _check_idle_status(port, args.timeout)
        print(f"INFO={info}; STATUS={status}")

        if args.command == "tables":
            upload_many(
                port,
                build_common_runtime_uploads(args.image),
                baud=args.baud,
                timeout=args.timeout,
                verify=args.verify,
            )
            return 0

        if args.command == "prepare-query0":
            case = build_fixed_real_cases(image_path=args.image)[0]
            if (case.query_position, case.window_start, case.count) != (0, 0, 1):
                raise AssertionError("固定 query0 用例契约异常")
            upload_many(
                port,
                build_dynamic_uploads(case),
                baud=args.baud,
                timeout=args.timeout,
                verify=args.verify,
            )
            return 0

        if args.command == "load-layer":
            configure_layer(
                port,
                layer=args.layer,
                query_position=args.query,
                window_start=args.window,
                count=args.count,
                timeout=args.timeout,
            )
            config = read_configuration(port, args.timeout)
            verify_configuration(
                config,
                layer=args.layer,
                query_position=args.query,
                window_start=args.window,
                count=args.count,
            )
            upload_many(
                port,
                build_layer_uploads(args.layer, image_path=args.image),
                baud=args.baud,
                timeout=args.timeout,
                verify=args.verify,
            )
            if args.commit:
                commit_payload(port, args.timeout)
            print(f"LAYER {args.layer} LOAD PASS: config={config}, commit={args.commit}")
            return 0

        if args.command == "copy-hidden":
            copy = hidden_copy_contract()
            elapsed = copy_region(
                port,
                source_controller_address=int(copy["source_controller_address"]),
                destination_controller_address=int(copy["destination_controller_address"]),
                length=int(copy["length"]),
                baud=args.baud,
                timeout=args.timeout,
            )
            print(f"HIDDEN COPY PASS: {copy['length']} B, elapsed={elapsed:.6f}s")
            return 0

        if args.command == "run-layer":
            run_layer(
                port,
                layer=args.layer,
                query_position=args.query,
                window_start=args.window,
                count=args.count,
                image=args.image,
                baud=args.baud,
                timeout=args.timeout,
                verify_load=args.verify_load,
                copy_output=args.copy_output,
                read_output=args.read_output,
            )
            return 0

        if args.command == "run-sequence":
            manifest = build_layer_sequence_manifest(
                query_position=args.query,
                window_start=args.window,
                count=args.count,
                start_layer=args.start_layer,
                end_layer=args.end_layer,
                image_path=args.image,
            )
            if args.prepare_query0:
                if (args.query, args.window, args.count) != (0, 0, 1):
                    raise SystemExit("--prepare-query0 只允许 query/window/count=0/0/1")
                case = build_fixed_real_cases(image_path=args.image)[0]
                upload_many(
                    port,
                    build_dynamic_uploads(case),
                    baud=args.baud,
                    timeout=args.timeout,
                    verify=False,
                )
            if args.load_tables:
                upload_many(
                    port,
                    build_common_runtime_uploads(args.image),
                    baud=args.baud,
                    timeout=args.timeout,
                    verify=args.verify_load,
                )

            started = time.perf_counter()
            for layer in range(args.start_layer, args.end_layer + 1):
                run_layer(
                    port,
                    layer=layer,
                    query_position=args.query,
                    window_start=args.window,
                    count=args.count,
                    image=args.image,
                    baud=args.baud,
                    timeout=args.timeout,
                    verify_load=args.verify_load,
                    copy_output=layer != args.end_layer,
                    read_output=args.read_each_output or layer == args.end_layer,
                )
            print(
                f"H3 SEQUENCE PASS: layers={args.start_layer}..{args.end_layer}, "
                f"hidden_copies={manifest['hidden_copy_count']}, "
                f"elapsed={time.perf_counter() - started:.3f}s"
            )
            return 0

    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
