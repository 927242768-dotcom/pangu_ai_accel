#!/usr/bin/env python3
"""阶段 H2：真实 24 层模型的参数换层与 1 GiB DDR3 布局。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable

try:
    from .kv_cache_reference import (
        AXI_BEAT_BYTES,
        DDR_BYTES,
        KV_BASE_BYTES,
        LAYER_STRIDE_BYTES,
        MAX_CONTEXT,
        TOKEN_SLOT_BYTES,
        VECTOR_BYTES,
    )
    from .model_layer_descriptor import (
        DEFAULT_EXTERNAL_METADATA,
        DEFAULT_IMAGE,
        build_model_layer_descriptor,
    )
    from .p50_format import align_up
    from .transformer_block_reference import parameter_regions, scratch_regions
except ImportError:
    from kv_cache_reference import (
        AXI_BEAT_BYTES,
        DDR_BYTES,
        KV_BASE_BYTES,
        LAYER_STRIDE_BYTES,
        MAX_CONTEXT,
        TOKEN_SLOT_BYTES,
        VECTOR_BYTES,
    )
    from model_layer_descriptor import (
        DEFAULT_EXTERNAL_METADATA,
        DEFAULT_IMAGE,
        build_model_layer_descriptor,
    )
    from p50_format import align_up
    from transformer_block_reference import parameter_regions, scratch_regions

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

DEFAULT_REFERENCE = Path(__file__).with_name("full_model_memory_plan_reference.json")
SCHEMA = "pangu50k.full_model_memory_plan.v1"
CTRL_WORD_BYTES = 4
PAGE_BYTES = 4096
UART_BAUD = 115200
UART_BITS_PER_BYTE = 10
RUNTIME_BASE = 0x0000_0000
RUNTIME_SIZE = 0x0100_0000
SLOT_A_BASE = 0x0100_0000
SLOT_B_BASE = 0x0200_0000
SLOT_SIZE = 0x0100_0000
LOW_FREE_BASE = 0x0300_0000


class FullModelMemoryPlanError(ValueError):
    """表示阶段 H2 内存或传输契约不合法。"""


def _region(name: str, start: int, size: int, purpose: str, state: str) -> dict[str, Any]:
    if start < 0 or size <= 0 or start % AXI_BEAT_BYTES or size % AXI_BEAT_BYTES:
        raise FullModelMemoryPlanError(f"{name} 地址或长度非法")
    end = start + size
    return {
        "name": name,
        "byte_start": start,
        "byte_end": end,
        "size_bytes": size,
        "controller_start": start // CTRL_WORD_BYTES,
        "controller_end": end // CTRL_WORD_BYTES,
        "purpose": purpose,
        "state": state,
    }


def _existing(item: Any) -> dict[str, Any]:
    return {
        "name": item.name,
        "byte_start": int(item.byte_address),
        "byte_end": int(item.end_byte_address),
        "size_bytes": int(item.size_bytes),
        "controller_start": int(item.controller_address),
        "data_format": item.data_format,
        "lifetime": item.lifetime,
    }


def _check_cover(regions: list[dict[str, Any]]) -> None:
    ordered = sorted(regions, key=lambda item: item["byte_start"])
    if ordered[0]["byte_start"] != 0:
        raise FullModelMemoryPlanError("DDR3 分区未从 0 开始")
    for left, right in zip(ordered, ordered[1:]):
        if left["byte_end"] != right["byte_start"]:
            raise FullModelMemoryPlanError(
                f"DDR3 分区不连续：{left['name']} -> {right['name']}"
            )
    if ordered[-1]["byte_end"] != DDR_BYTES:
        raise FullModelMemoryPlanError("DDR3 分区未结束于 1 GiB")


TRANSFER_SPECS: tuple[tuple[str, str, str, str], ...] = (
    ("input_layernorm_weight", "data", "input_rms_gamma_q10", "fp16_to_q6_10"),
    ("q_proj_weight", "data", "q_weight_int4", "copy_int4"),
    ("q_proj_weight", "scale", "q_weight_scale_fp16", "copy_fp16_scale"),
    ("q_proj_bias", "data", "q_bias_q28", "fp16_bias_to_q28_row32"),
    ("k_proj_weight", "data", "k_weight_int4", "copy_int4"),
    ("k_proj_weight", "scale", "k_weight_scale_fp16", "copy_fp16_scale"),
    ("k_proj_bias", "data", "k_bias_q28", "fp16_bias_to_q28_row32"),
    ("v_proj_weight", "data", "v_weight_int4", "copy_int4"),
    ("v_proj_weight", "scale", "v_weight_scale_fp16", "copy_fp16_scale"),
    ("v_proj_bias", "data", "v_bias_q28", "fp16_bias_to_q28_row32"),
    ("o_proj_weight", "data", "oproj_weight_int4", "copy_int4"),
    ("o_proj_weight", "scale", "oproj_weight_scale_fp16", "copy_fp16_scale"),
    ("post_attention_layernorm_weight", "data", "post_rms_gamma_q10", "fp16_to_q6_10"),
    ("gate_proj_weight", "data", "gate_weight_int4", "copy_int4"),
    ("gate_proj_weight", "scale", "gate_weight_scale_fp16", "copy_fp16_scale"),
    ("up_proj_weight", "data", "up_weight_int4", "copy_int4"),
    ("up_proj_weight", "scale", "up_weight_scale_fp16", "copy_fp16_scale"),
    ("down_proj_weight", "data", "down_weight_int4", "copy_int4"),
    ("down_proj_weight", "scale", "down_weight_scale_fp16", "copy_fp16_scale"),
)


def _transfer_template(descriptor: dict[str, Any]) -> list[dict[str, Any]]:
    roles = {item["role"]: item for item in descriptor["tensor_templates"]}
    scratch = {item.name: item for item in scratch_regions()}
    params = {item.name: item for item in parameter_regions()}
    result: list[dict[str, Any]] = []
    for role, component, destination_name, transform in TRANSFER_SPECS:
        source = roles[role]
        if component == "data":
            source_offset = int(source["data_offset_in_layer"])
            source_nbytes = int(source["data_nbytes"])
        else:
            source_offset = int(source["scale_offset_in_layer"])
            source_nbytes = int(source["scale_nbytes"])
        destination = scratch.get(destination_name) or params.get(destination_name)
        if destination is None:
            raise FullModelMemoryPlanError(f"未知目标区域：{destination_name}")
        destination_nbytes = int(destination.size_bytes)
        if transform.startswith("copy_") and source_nbytes != destination_nbytes:
            raise FullModelMemoryPlanError(f"{role}/{component} 直拷长度不一致")
        if transform == "fp16_to_q6_10" and source_nbytes != destination_nbytes:
            raise FullModelMemoryPlanError(f"{role} gamma 长度不一致")
        if transform == "fp16_bias_to_q28_row32":
            rows = int(source["shape"][0])
            if source_nbytes != rows * 2 or destination_nbytes != rows * 32:
                raise FullModelMemoryPlanError(f"{role} bias 行布局不一致")
        item: dict[str, Any] = {
            "source_role": role,
            "source_component": component,
            "source_offset_in_layer": source_offset,
            "source_nbytes": source_nbytes,
            "destination_region": destination_name,
            "destination_byte_address_slot_a": int(destination.byte_address),
            "destination_nbytes": destination_nbytes,
            "transform": transform,
        }
        if destination_name in params:
            item["destination_offset_in_slot"] = (
                int(destination.byte_address) - SLOT_A_BASE
            )
        result.append(item)
    if len(result) != 19:
        raise FullModelMemoryPlanError(f"层事务数量错误：{len(result)}")
    return result


def _global_layout(descriptor: dict[str, Any], base: int) -> dict[str, Any]:
    by_role = {item["role"]: item for item in descriptor["global_tensors"]}
    embedding = by_role["embedding_and_tied_lm_head_weight"]
    final_norm = by_role["final_norm_weight"]
    vocab = int(descriptor["model"]["vocab_size"])
    groups = int(embedding["groups_per_row"])

    weight_start = base
    weight_end = weight_start + int(embedding["data_nbytes"])
    scale_start = align_up(weight_end, PAGE_BYTES)
    scale_end = scale_start + int(embedding["scale_nbytes"])
    norm_start = align_up(scale_end, PAGE_BYTES)
    norm_end = norm_start + int(final_norm["data_nbytes"])
    combined_start = align_up(norm_end, PAGE_BYTES)
    combined_size = vocab * groups * 4
    combined_end = combined_start + combined_size
    logits_start = align_up(combined_end, PAGE_BYTES)
    logits_size = vocab * 8
    logits_end = logits_start + logits_size
    end = align_up(logits_end, PAGE_BYTES)

    regions = [
        _region(
            "embedding_lm_head_int4",
            weight_start,
            int(embedding["data_nbytes"]),
            "tied Embedding/LM Head packed INT4",
            "session_resident",
        ),
        _region(
            "embedding_lm_head_scale_fp16",
            scale_start,
            int(embedding["scale_nbytes"]),
            "tied Embedding/LM Head raw FP16 scale",
            "session_resident",
        ),
        _region(
            "final_norm_gamma_q10",
            norm_start,
            int(final_norm["data_nbytes"]),
            "最终 RMSNorm gamma",
            "session_resident",
        ),
        _region(
            "lm_head_combined_scale_uq4_28",
            combined_start,
            combined_size,
            "LM Head 每 token combined scale",
            "regenerated_per_token",
        ),
        _region(
            "lm_head_logits_q28",
            logits_start,
            logits_size,
            "完整 vocab signed int64 Q28 logits",
            "per_token",
        ),
    ]
    return {
        "byte_start": base,
        "byte_end": end,
        "size_bytes": end - base,
        "regions": regions,
        "source": {
            "embedding_data_offset": int(embedding["data_offset"]),
            "embedding_data_nbytes": int(embedding["data_nbytes"]),
            "embedding_scale_offset": int(embedding["scale_offset"]),
            "embedding_scale_nbytes": int(embedding["scale_nbytes"]),
            "final_norm_offset": int(final_norm["data_offset"]),
            "final_norm_nbytes": int(final_norm["data_nbytes"]),
        },
    }


def build_full_model_memory_plan(
    image_path: str | Path = DEFAULT_IMAGE,
    external_metadata_path: str | Path | None = DEFAULT_EXTERNAL_METADATA,
) -> dict[str, Any]:
    descriptor = build_model_layer_descriptor(image_path, external_metadata_path)
    active_layers = int(descriptor["model"]["num_hidden_layers"])
    kv_size = active_layers * LAYER_STRIDE_BYTES
    kv_end = KV_BASE_BYTES + kv_size
    global_layout = _global_layout(descriptor, kv_end)
    global_end = int(global_layout["byte_end"])
    if global_end > DDR_BYTES:
        raise FullModelMemoryPlanError("全局区越过 1 GiB")

    partitions = [
        _region(
            "runtime_control_scratch",
            RUNTIME_BASE,
            RUNTIME_SIZE,
            "G2 中间张量、执行载荷、表和层 gamma",
            "mixed_runtime",
        ),
        _region(
            "layer_parameter_slot_a",
            SLOT_A_BASE,
            SLOT_SIZE,
            "当前层参数与运行时 combined scale",
            "reload_each_layer",
        ),
        _region(
            "layer_parameter_slot_b",
            SLOT_B_BASE,
            SLOT_SIZE,
            "后续高速接口预取下一层",
            "reserved_prefetch",
        ),
        _region(
            "low_free_reserve",
            LOW_FREE_BASE,
            KV_BASE_BYTES - LOW_FREE_BASE,
            "层调度、DMA 和传输 staging 扩展",
            "free_reserve",
        ),
        _region(
            "kv_cache_active_24_layers",
            KV_BASE_BYTES,
            kv_size,
            "24 个真实层的 16384-token K/V Cache",
            "session_kv",
        ),
        _region(
            "global_resident_and_lm_head",
            kv_end,
            global_end - kv_end,
            "tied Embedding/LM Head、最终 Norm 与 logits",
            "mixed_global",
        ),
        _region(
            "high_free_reserve",
            global_end,
            DDR_BYTES - global_end,
            "top-k、采样、调试和后续扩展",
            "free_reserve",
        ),
    ]
    _check_cover(partitions)

    runtime = [_existing(item) for item in scratch_regions()]
    parameters = [_existing(item) for item in parameter_regions()]
    for item in runtime:
        if not 0 <= item["byte_start"] < item["byte_end"] <= RUNTIME_SIZE:
            raise FullModelMemoryPlanError(f"scratch 越界：{item['name']}")
    for item in parameters:
        if not SLOT_A_BASE <= item["byte_start"] < item["byte_end"] <= SLOT_A_BASE + SLOT_SIZE:
            raise FullModelMemoryPlanError(f"slot A 参数越界：{item['name']}")

    transfer_template = _transfer_template(descriptor)
    source_bytes = sum(item["source_nbytes"] for item in transfer_template)
    destination_bytes = sum(item["destination_nbytes"] for item in transfer_template)
    parameter_by_name = {item["name"]: item for item in parameters}
    generated_names = (
        "q_scale_uq4_28",
        "k_scale_uq4_28",
        "v_scale_uq4_28",
        "oproj_scale_uq4_28",
        "gate_scale_uq4_28",
        "up_scale_uq4_28",
        "down_scale_uq4_28",
    )
    runtime_by_name = {item["name"]: item for item in runtime}
    ping = runtime_by_name["block_hidden_q10"]
    pong = runtime_by_name["block_output_q10"]
    slot_used_end = max(item["byte_end"] for item in parameters)

    global_source_bytes = (
        global_layout["source"]["embedding_data_nbytes"]
        + global_layout["source"]["embedding_scale_nbytes"]
        + global_layout["source"]["final_norm_nbytes"]
    )
    per_layer_uart_seconds = destination_bytes * UART_BITS_PER_BYTE / UART_BAUD
    all_layer_uart_seconds = per_layer_uart_seconds * active_layers

    return {
        "schema": SCHEMA,
        "model": {
            "active_layers": active_layers,
            "hardware_layer_capacity": descriptor["hardware_contract"]["layer_capacity"],
            "hidden_size": descriptor["model"]["hidden_size"],
            "vocab_size": descriptor["model"]["vocab_size"],
            "model_max_positions": descriptor["model"]["max_position_embeddings"],
            "hardware_max_context": MAX_CONTEXT,
            "p50_image_size_bytes": descriptor["image"]["size_bytes"],
        },
        "decision": {
            "scheme": "hybrid_global_resident_layer_reload",
            "all_parameters_resident_with_16k_kv": False,
            "global_resident": [
                "embedding_lm_head_int4",
                "embedding_lm_head_scale_fp16",
                "final_norm_gamma_q10",
            ],
            "layer_policy": "按 layer0..23 顺序重载到 slot A",
            "slot_b_policy": "仅预留；当前 G2 无参数基址选择",
            "shared_tables_policy": "RMS/Softmax/SiLU 会话常驻，RoPE 每 query 更新",
            "combined_scale_policy": "七组 combined scale 均由 FPGA 每次调用重建",
        },
        "ddr": {
            "total_bytes": DDR_BYTES,
            "partitions": partitions,
        },
        "runtime": {
            "existing_regions": runtime,
            "hidden_handoff": {
                "ping_region": ping["name"],
                "ping_byte_address": ping["byte_start"],
                "pong_region": pong["name"],
                "pong_byte_address": pong["byte_start"],
                "size_bytes": ping["size_bytes"],
                "current_mode": "layer_end_copy_pong_to_ping",
                "copy_bytes_per_layer": ping["size_bytes"],
                "future_mode": "地址参数化后直接 ping-pong",
                "rtl_status": "尚未实现层间选择；G2 固定地址保持不变",
            },
        },
        "layer_parameter_slots": {
            "slot_size_bytes": SLOT_SIZE,
            "slot_a_base": SLOT_A_BASE,
            "slot_b_base": SLOT_B_BASE,
            "slot_a_used_span_bytes": slot_used_end - SLOT_A_BASE,
            "slot_a_free_bytes": SLOT_A_BASE + SLOT_SIZE - slot_used_end,
            "existing_slot_a_regions": parameters,
            "fpga_generated_combined_scale_regions": [
                parameter_by_name[name] for name in generated_names
            ],
        },
        "kv_cache": {
            "byte_start": KV_BASE_BYTES,
            "byte_end": kv_end,
            "size_bytes": kv_size,
            "size_mib": kv_size // (1 << 20),
            "active_layers": active_layers,
            "layer_stride_bytes": LAYER_STRIDE_BYTES,
            "max_context": MAX_CONTEXT,
            "token_slot_bytes": TOKEN_SLOT_BYTES,
            "k_vector_bytes": VECTOR_BYTES,
            "v_vector_bytes": VECTOR_BYTES,
            "unused_capacity_layer_slots": descriptor["hardware_contract"]["unused_layer_slots"],
        },
        "global_layout": global_layout,
        "layer_streaming": {
            "transfer_count_per_layer": len(transfer_template),
            "source_bytes_per_layer": source_bytes,
            "destination_bytes_per_layer": destination_bytes,
            "source_layer_bases": descriptor["layer_layout"]["layer_bases"],
            "transfer_template": transfer_template,
            "load_sequence": [
                "等待 Block idle",
                "转换并上传当前层 19 笔参数",
                "设置真实 layer0..23",
                "执行完整 G2 Block 并写本层 KV",
                "把 1792 B 输出交给下一层输入",
            ],
        },
        "transport": {
            "current_uart_baud": UART_BAUD,
            "uart_bits_per_byte": UART_BITS_PER_BYTE,
            "global_source_bytes": global_source_bytes,
            "global_uart_seconds": global_source_bytes * UART_BITS_PER_BYTE / UART_BAUD,
            "per_layer_destination_bytes": destination_bytes,
            "per_layer_uart_seconds": per_layer_uart_seconds,
            "all_layers_reload_uart_seconds_per_token": all_layer_uart_seconds,
            "all_layers_reload_uart_hours_per_token": all_layer_uart_seconds / 3600.0,
            "validation_only": True,
            "usable_inference_requires_faster_transport": True,
        },
    }


def expand_layer_transfer_plan(
    plan: dict[str, Any], layer_index: int, *, slot: str = "A"
) -> list[dict[str, Any]]:
    bases = plan["layer_streaming"]["source_layer_bases"]
    if not 0 <= layer_index < len(bases):
        raise IndexError(f"layer_index={layer_index} 越界")
    normalized = slot.upper()
    if normalized not in {"A", "B"}:
        raise ValueError("slot 必须是 A 或 B")
    delta = 0 if normalized == "A" else SLOT_B_BASE - SLOT_A_BASE
    source_base = int(bases[layer_index])
    result: list[dict[str, Any]] = []
    for template in plan["layer_streaming"]["transfer_template"]:
        destination = int(template["destination_byte_address_slot_a"])
        if "destination_offset_in_slot" in template:
            destination += delta
        item = dict(template)
        item.update(
            {
                "layer_index": layer_index,
                "slot": normalized,
                "source_byte_offset": source_base
                + int(template["source_offset_in_layer"]),
                "destination_byte_address": destination,
                "destination_controller_address": destination // CTRL_WORD_BYTES,
            }
        )
        result.append(item)
    return result


def reference_snapshot(plan: dict[str, Any]) -> dict[str, Any]:
    canonical = json.dumps(
        plan, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return {
        "schema": "pangu50k.full_model_memory_plan.reference.v1",
        "plan_schema": plan["schema"],
        "plan_sha256": hashlib.sha256(canonical).hexdigest(),
        "active_layers": plan["model"]["active_layers"],
        "hardware_layer_capacity": plan["model"]["hardware_layer_capacity"],
        "hardware_max_context": plan["model"]["hardware_max_context"],
        "scheme": plan["decision"]["scheme"],
        "partitions": [
            {
                "name": item["name"],
                "byte_start": item["byte_start"],
                "byte_end": item["byte_end"],
            }
            for item in plan["ddr"]["partitions"]
        ],
        "layer_transfer_count": plan["layer_streaming"]["transfer_count_per_layer"],
        "source_bytes_per_layer": plan["layer_streaming"]["source_bytes_per_layer"],
        "destination_bytes_per_layer": plan["layer_streaming"]["destination_bytes_per_layer"],
        "kv_size_bytes": plan["kv_cache"]["size_bytes"],
        "global_layout_end": plan["global_layout"]["byte_end"],
        "high_free_bytes": plan["ddr"]["partitions"][-1]["size_bytes"],
        "hidden_copy_bytes": plan["runtime"]["hidden_handoff"]["copy_bytes_per_layer"],
        "uart_hours_per_token": plan["transport"]["all_layers_reload_uart_hours_per_token"],
    }


def load_reference(path: str | Path = DEFAULT_REFERENCE) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise FullModelMemoryPlanError("H2 reference 顶层必须是对象")
    return value


def verify_reference(
    image_path: str | Path = DEFAULT_IMAGE,
    external_metadata_path: str | Path | None = DEFAULT_EXTERNAL_METADATA,
    reference_path: str | Path = DEFAULT_REFERENCE,
) -> dict[str, Any]:
    generated = build_full_model_memory_plan(image_path, external_metadata_path)
    if reference_snapshot(generated) != load_reference(reference_path):
        raise FullModelMemoryPlanError("冻结的 H2 内存方案与当前模型不一致")
    return generated


def _print_summary(plan: dict[str, Any]) -> None:
    print(
        f"schema={plan['schema']}, scheme={plan['decision']['scheme']}, "
        f"layers={plan['model']['active_layers']}/"
        f"{plan['model']['hardware_layer_capacity']}"
    )
    for item in plan["ddr"]["partitions"]:
        print(
            f"{item['name']}: 0x{item['byte_start']:08x}.."
            f"0x{item['byte_end'] - 1:08x}, "
            f"{item['size_bytes'] / (1 << 20):.3f} MiB"
        )
    stream = plan["layer_streaming"]
    transport = plan["transport"]
    print(
        f"per_layer={stream['transfer_count_per_layer']} transactions, "
        f"source={stream['source_bytes_per_layer']} B, "
        f"destination={stream['destination_bytes_per_layer']} B"
    )
    print(
        f"UART115200={transport['per_layer_uart_seconds']:.3f}s/layer, "
        f"{transport['all_layers_reload_uart_hours_per_token']:.3f}h/token; "
        "仅用于正确性验证"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument(
        "--external-metadata", type=Path, default=DEFAULT_EXTERNAL_METADATA
    )
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("summary")
    sub.add_parser("dump")
    sub.add_parser("verify")
    layer = sub.add_parser("layer")
    layer.add_argument("index", type=int)
    layer.add_argument("--slot", choices=("A", "B", "a", "b"), default="A")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    plan = build_full_model_memory_plan(args.image, args.external_metadata)
    if args.command == "summary":
        _print_summary(plan)
    elif args.command == "dump":
        print(json.dumps(plan, ensure_ascii=False, indent=2))
    elif args.command == "verify":
        if reference_snapshot(plan) != load_reference(args.reference):
            raise FullModelMemoryPlanError("冻结的 H2 内存方案不匹配")
        print("H2 内存方案验证通过：24 层按层换入，768 MiB KV，全局参数顶部常驻")
    elif args.command == "layer":
        print(
            json.dumps(
                expand_layer_transfer_plan(plan, args.index, slot=args.slot),
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        raise AssertionError(args.command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
