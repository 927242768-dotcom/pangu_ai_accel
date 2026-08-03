#!/usr/bin/env python3
"""阶段 H3：真实 24 层参数换入、配置、执行与 hidden 交接契约。

本模块不重新定义 G2 算术，只把 H1/H2 已冻结的信息转为可直接交给
UART host 的逐层事务：

1. 按 H2 的 19 笔模板从真实 P50 读取当前层参数；
2. packed INT4/FP16 scale 原样复制；
3. RMS gamma 转为 signed Q6.10；
4. Q/K/V bias 转为 signed Q28，并按每输出行 32 B 展开；
5. 配置真实 layer0..23，执行 G2 Block；
6. 除最后一层外，将 ``block_output_q10`` 复制回 ``block_hidden_q10``。

当前只建立层间控制与主机换层正确性基线；最终 RMSNorm、LM Head 和 logits
不属于本模块。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

try:
    from .full_model_memory_plan import DEFAULT_IMAGE, build_full_model_memory_plan
    from .transformer_block_g2_payload import (
        BEAT_BYTES,
        DDRUpload,
        TransformerBlockPayloadError,
        build_layer_parameter_uploads,
        build_resident_uploads,
    )
except ImportError:
    from full_model_memory_plan import DEFAULT_IMAGE, build_full_model_memory_plan
    from transformer_block_g2_payload import (
        BEAT_BYTES,
        DDRUpload,
        TransformerBlockPayloadError,
        build_layer_parameter_uploads,
        build_resident_uploads,
    )

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

DEFAULT_REFERENCE = Path(__file__).with_name("full_model_layer_sequence_reference.json")
SCHEMA = "pangu50k.full_model_layer_sequence.v1"
CONTROLLER_WORD_BYTES = 4
ACTIVE_LAYER_COUNT = 24
MAX_CONTEXT = 16_384
COMMON_RUNTIME_NAMES = (
    "rms_lut_uq12_20",
    "softmax_exp_lut_q31",
    "silu_pwl_q10",
)


class FullModelLayerSequenceError(ValueError):
    """表示 H3 层参数或层间事务不满足冻结契约。"""


def build_common_runtime_uploads(
    image_path: str | Path = DEFAULT_IMAGE,
) -> list[DDRUpload]:
    """返回所有层共用、每次上电只需上传一次的三个查表区域。"""

    resident = build_resident_uploads(image_path=Path(image_path))
    by_name = {item.name: item for item in resident}
    uploads = [by_name[name] for name in COMMON_RUNTIME_NAMES]
    if len(uploads) != 3 or not all(item.persistent for item in uploads):
        raise FullModelLayerSequenceError("公共 runtime 查表集合异常")
    return uploads


def build_layer_uploads(
    layer_index: int,
    *,
    slot: str = "A",
    image_path: str | Path = DEFAULT_IMAGE,
    plan: dict[str, Any] | None = None,
) -> list[DDRUpload]:
    """从真实 P50 构造一个活动层的 19 笔 DDR3 上传事务。"""

    resolved_plan = build_full_model_memory_plan(image_path) if plan is None else plan
    if slot != "A":
        raise FullModelLayerSequenceError(
            "H3 第一版只允许 slot A；slot B 预取尚未进入顺序执行基线"
        )
    try:
        canonical = build_layer_parameter_uploads(
            layer_index,
            image_path=Path(image_path),
        )
    except TransformerBlockPayloadError as error:
        raise FullModelLayerSequenceError(str(error)) from error
    uploads = [
        DDRUpload(
            name=f"layer{layer_index}_{item.name}",
            controller_address=item.controller_address,
            payload=item.payload,
            persistent=False,
        )
        for item in canonical
    ]

    if len(uploads) != 19:
        raise FullModelLayerSequenceError(f"layer{layer_index} 上传事务数量错误")
    if sum(len(item.payload) for item in uploads) != int(
        resolved_plan["layer_streaming"]["destination_bytes_per_layer"]
    ):
        raise FullModelLayerSequenceError(f"layer{layer_index} 上传总长度错误")
    return uploads


def hidden_copy_contract(plan: dict[str, Any] | None = None) -> dict[str, int | str]:
    """返回层末 ``pong -> ping`` 的 FPGA DDR3 内复制命令参数。"""

    resolved = build_full_model_memory_plan() if plan is None else plan
    handoff = resolved["runtime"]["hidden_handoff"]
    source = int(handoff["pong_byte_address"])
    destination = int(handoff["ping_byte_address"])
    length = int(handoff["size_bytes"])
    if any(value % BEAT_BYTES for value in (source, destination, length)):
        raise FullModelLayerSequenceError("hidden copy 未按 32 B 对齐")
    if not (source + length <= destination or destination + length <= source):
        raise FullModelLayerSequenceError("hidden copy 源/目标范围重叠")
    return {
        "command": "M",
        "source_byte_address": source,
        "source_controller_address": source // CONTROLLER_WORD_BYTES,
        "destination_byte_address": destination,
        "destination_controller_address": destination // CONTROLLER_WORD_BYTES,
        "length": length,
        "beats": length // BEAT_BYTES,
    }


def validate_execution_window(query_position: int, window_start: int, count: int) -> None:
    if not 1 <= count <= 16:
        raise FullModelLayerSequenceError("count 必须为 1..16")
    if not 0 <= window_start < MAX_CONTEXT or not 0 <= query_position < MAX_CONTEXT:
        raise FullModelLayerSequenceError("query/window 超出 16384 硬件上下文")
    if query_position != window_start + count - 1:
        raise FullModelLayerSequenceError("query_position 必须等于 window_start + count - 1")


def _upload_set_sha256(uploads: Sequence[DDRUpload]) -> str:
    digest = hashlib.sha256()
    for upload in uploads:
        encoded_name = upload.name.encode("utf-8")
        digest.update(struct.pack("<I", len(encoded_name)))
        digest.update(encoded_name)
        digest.update(struct.pack("<II", upload.controller_address, len(upload.payload)))
        digest.update(bytes.fromhex(upload.sha256))
    return digest.hexdigest()


def build_layer_sequence_manifest(
    *,
    query_position: int = 0,
    window_start: int = 0,
    count: int = 1,
    start_layer: int = 0,
    end_layer: int = ACTIVE_LAYER_COUNT - 1,
    slot: str = "A",
    image_path: str | Path = DEFAULT_IMAGE,
) -> dict[str, Any]:
    """构造 H3 第一版的层间执行清单，不包含最终 Norm/LM Head。"""

    validate_execution_window(query_position, window_start, count)
    if not 0 <= start_layer <= end_layer < ACTIVE_LAYER_COUNT:
        raise FullModelLayerSequenceError("layer 范围必须位于 0..23")
    plan = build_full_model_memory_plan(image_path)
    layers: list[dict[str, Any]] = []
    for layer_index in range(start_layer, end_layer + 1):
        uploads = build_layer_uploads(
            layer_index,
            slot=slot,
            image_path=image_path,
            plan=plan,
        )
        layers.append(
            {
                "layer_index": layer_index,
                "slot": slot,
                "upload_transactions": len(uploads),
                "upload_bytes": sum(len(item.payload) for item in uploads),
                "upload_set_sha256": _upload_set_sha256(uploads),
                "configure": {
                    "layer": layer_index,
                    "query_position": query_position,
                    "window_start": window_start,
                    "count": count,
                },
                "commit_required": True,
                "execute_required": True,
                "copy_output_to_input": layer_index != end_layer,
            }
        )
    copy = hidden_copy_contract(plan)
    layer_count = len(layers)
    return {
        "schema": SCHEMA,
        "mode": "slot_a_sequential_layer_reload_with_hidden_copy",
        "active_model_layers": ACTIVE_LAYER_COUNT,
        "start_layer": start_layer,
        "end_layer": end_layer,
        "layer_count": layer_count,
        "slot": slot,
        "window": {
            "query_position": query_position,
            "window_start": window_start,
            "count": count,
        },
        "common_runtime_uploads": list(COMMON_RUNTIME_NAMES),
        "per_layer_upload_transactions": 19,
        "per_layer_upload_bytes": int(
            plan["layer_streaming"]["destination_bytes_per_layer"]
        ),
        "total_upload_transactions": layer_count * 19,
        "total_upload_bytes": layer_count
        * int(plan["layer_streaming"]["destination_bytes_per_layer"]),
        "hidden_copy": copy,
        "hidden_copy_count": max(0, layer_count - 1),
        "hidden_copy_total_bytes": max(0, layer_count - 1) * int(copy["length"]),
        "layers": layers,
        "not_included": ["final_rmsnorm", "lm_head", "logits", "sampling"],
    }


def reference_snapshot(
    image_path: str | Path = DEFAULT_IMAGE,
) -> dict[str, Any]:
    manifest = build_layer_sequence_manifest(image_path=image_path)
    return {
        "schema": manifest["schema"],
        "mode": manifest["mode"],
        "active_model_layers": manifest["active_model_layers"],
        "layer_range": [manifest["start_layer"], manifest["end_layer"]],
        "per_layer_upload_transactions": manifest["per_layer_upload_transactions"],
        "per_layer_upload_bytes": manifest["per_layer_upload_bytes"],
        "total_upload_transactions": manifest["total_upload_transactions"],
        "total_upload_bytes": manifest["total_upload_bytes"],
        "hidden_copy": manifest["hidden_copy"],
        "hidden_copy_count": manifest["hidden_copy_count"],
        "hidden_copy_total_bytes": manifest["hidden_copy_total_bytes"],
        "common_runtime_uploads": manifest["common_runtime_uploads"],
        "layer_upload_set_sha256": [
            item["upload_set_sha256"] for item in manifest["layers"]
        ],
        "not_included": manifest["not_included"],
    }


def load_reference(path: str | Path = DEFAULT_REFERENCE) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify_reference(
    image_path: str | Path = DEFAULT_IMAGE,
    reference_path: str | Path = DEFAULT_REFERENCE,
) -> None:
    actual = reference_snapshot(image_path)
    expected = load_reference(reference_path)
    if actual != expected:
        raise FullModelLayerSequenceError("H3 层间事务冻结清单不一致")


def _summary(manifest: dict[str, Any]) -> str:
    copy = manifest["hidden_copy"]
    return "\n".join(
        (
            f"schema={manifest['schema']}",
            f"layers={manifest['start_layer']}..{manifest['end_layer']} "
            f"({manifest['layer_count']})",
            f"per_layer={manifest['per_layer_upload_transactions']} transactions, "
            f"{manifest['per_layer_upload_bytes']} B",
            f"all_layers={manifest['total_upload_transactions']} transactions, "
            f"{manifest['total_upload_bytes']} B",
            f"hidden_copy={manifest['hidden_copy_count']} x {copy['length']} B, "
            f"src=0x{copy['source_byte_address']:08x}, "
            f"dst=0x{copy['destination_byte_address']:08x}",
            "final RMSNorm/LM Head/logits: not included",
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="H3 真实 24 层换层与 hidden 交接契约")
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("summary")
    sub.add_parser("verify")
    dump = sub.add_parser("dump")
    dump.add_argument("--output", type=Path)
    layer = sub.add_parser("layer")
    layer.add_argument("layer_index", type=int)
    layer.add_argument("--slot", choices=("A", "B"), default="A")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "summary":
        print(_summary(build_layer_sequence_manifest(image_path=args.image)))
        return 0
    if args.command == "verify":
        verify_reference(args.image, args.reference)
        print("H3 层间换层清单验证通过：24 层、456 笔上传、23 次 hidden copy")
        return 0
    if args.command == "dump":
        payload = json.dumps(reference_snapshot(args.image), ensure_ascii=False, indent=2) + "\n"
        if args.output is None:
            print(payload, end="")
        else:
            args.output.write_text(payload, encoding="utf-8")
        return 0
    if args.command == "layer":
        uploads = build_layer_uploads(args.layer_index, slot=args.slot, image_path=args.image)
        for upload in uploads:
            print(
                f"{upload.name}: addr=0x{upload.controller_address:07x}, "
                f"bytes={len(upload.payload)}, sha256={upload.sha256}"
            )
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
