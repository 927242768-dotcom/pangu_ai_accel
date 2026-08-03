#!/usr/bin/env python3
"""阶段 H：从真实 P50 镜像建立完整模型层描述表。

描述表区分两个容易混淆的数量：

- ``model.num_hidden_layers`` 是真实模型需要执行的层数；
- ``hardware_contract.layer_capacity`` 是当前 KV/控制地址契约可容纳的层数。

当前 Qwen2.5-0.5B P50 镜像实际包含 24 层，而现有硬件地址契约预留 28 层。
本模块使用“层模板 + 层基址”紧凑描述 24×12 个层内张量，并可按层展开为
包含绝对主机文件偏移的完整清单。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable

try:
    from .p50_format import P50Image
    from .kv_cache_reference import MAX_CONTEXT as HARDWARE_MAX_CONTEXT
    from .kv_cache_reference import NUM_LAYERS as HARDWARE_LAYER_CAPACITY
except ImportError:
    from p50_format import P50Image
    from kv_cache_reference import MAX_CONTEXT as HARDWARE_MAX_CONTEXT
    from kv_cache_reference import NUM_LAYERS as HARDWARE_LAYER_CAPACITY

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE = PROJECT_ROOT / "model_output/yanbo_qwen25_0.5b_int4.p50"
DEFAULT_EXTERNAL_METADATA = PROJECT_ROOT / "model_output/yanbo_qwen25_0.5b_int4.json"
DEFAULT_REFERENCE = Path(__file__).with_name("model_layer_descriptor_reference.json")

SCHEMA = "pangu50k.model_layer_descriptor.v1"

# 顺序按一个 Transformer Block 的参数消费顺序冻结。
LAYER_TENSOR_ROLES: tuple[tuple[str, str], ...] = (
    ("input_layernorm_weight", "input_layernorm.weight"),
    ("q_proj_weight", "self_attn.q_proj.weight"),
    ("q_proj_bias", "self_attn.q_proj.bias"),
    ("k_proj_weight", "self_attn.k_proj.weight"),
    ("k_proj_bias", "self_attn.k_proj.bias"),
    ("v_proj_weight", "self_attn.v_proj.weight"),
    ("v_proj_bias", "self_attn.v_proj.bias"),
    ("o_proj_weight", "self_attn.o_proj.weight"),
    ("post_attention_layernorm_weight", "post_attention_layernorm.weight"),
    ("gate_proj_weight", "mlp.gate_proj.weight"),
    ("up_proj_weight", "mlp.up_proj.weight"),
    ("down_proj_weight", "mlp.down_proj.weight"),
)


class ModelLayerDescriptorError(ValueError):
    """表示模型层目录与冻结的 Qwen2.5 Block 结构不一致。"""


def _relative_project_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _entry_end(entry: dict[str, Any]) -> int:
    end = int(entry["data_offset"]) + int(entry["data_nbytes"])
    if "scale_offset" in entry:
        end = max(end, int(entry["scale_offset"]) + int(entry["scale_nbytes"]))
    return end


def _global_tensor(role: str, entry: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "role": role,
        "name": str(entry["name"]),
        "shape": [int(item) for item in entry["shape"]],
        "source_dtype": str(entry["source_dtype"]),
        "storage": str(entry["storage"]),
        "data_offset": int(entry["data_offset"]),
        "data_nbytes": int(entry["data_nbytes"]),
    }
    for key in ("scale_offset", "scale_nbytes", "padded_columns", "groups_per_row"):
        if key in entry:
            result[key] = int(entry[key])
    return result


def _layer_template(
    role: str,
    suffix: str,
    entry: dict[str, Any],
    layer_base: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "role": role,
        "name_suffix": suffix,
        "shape": [int(item) for item in entry["shape"]],
        "source_dtype": str(entry["source_dtype"]),
        "storage": str(entry["storage"]),
        "data_offset_in_layer": int(entry["data_offset"]) - layer_base,
        "data_nbytes": int(entry["data_nbytes"]),
    }
    if "scale_offset" in entry:
        result.update(
            {
                "scale_offset_in_layer": int(entry["scale_offset"]) - layer_base,
                "scale_nbytes": int(entry["scale_nbytes"]),
                "padded_columns": int(entry["padded_columns"]),
                "groups_per_row": int(entry["groups_per_row"]),
            }
        )
    return result


def _expected_layer_names(layer_index: int) -> list[str]:
    prefix = f"model.layers.{layer_index}."
    return [prefix + suffix for _, suffix in LAYER_TENSOR_ROLES]


def _validate_template_entry(
    *,
    layer_index: int,
    layer_base: int,
    template: dict[str, Any],
    entry: dict[str, Any],
) -> None:
    expected_name = f"model.layers.{layer_index}.{template['name_suffix']}"
    if entry["name"] != expected_name:
        raise ModelLayerDescriptorError(
            f"layer{layer_index} 张量名不匹配：{entry['name']} != {expected_name}"
        )

    exact_fields = ("shape", "source_dtype", "storage", "data_nbytes")
    for key in exact_fields:
        if entry[key] != template[key]:
            raise ModelLayerDescriptorError(
                f"{expected_name} {key} 不匹配：{entry[key]!r} != {template[key]!r}"
            )

    expected_data_offset = layer_base + int(template["data_offset_in_layer"])
    if int(entry["data_offset"]) != expected_data_offset:
        raise ModelLayerDescriptorError(
            f"{expected_name} data_offset={entry['data_offset']} != {expected_data_offset}"
        )

    optional = ("scale_nbytes", "padded_columns", "groups_per_row")
    for key in optional:
        if key in template:
            if int(entry[key]) != int(template[key]):
                raise ModelLayerDescriptorError(
                    f"{expected_name} {key}={entry[key]} != {template[key]}"
                )
        elif key in entry:
            raise ModelLayerDescriptorError(f"{expected_name} 不应包含 {key}")

    if "scale_offset_in_layer" in template:
        expected_scale_offset = layer_base + int(template["scale_offset_in_layer"])
        if int(entry["scale_offset"]) != expected_scale_offset:
            raise ModelLayerDescriptorError(
                f"{expected_name} scale_offset={entry['scale_offset']} != "
                f"{expected_scale_offset}"
            )
    elif "scale_offset" in entry:
        raise ModelLayerDescriptorError(f"{expected_name} 不应包含 scale_offset")


def build_model_layer_descriptor(
    image_path: str | Path = DEFAULT_IMAGE,
    external_metadata_path: str | Path | None = DEFAULT_EXTERNAL_METADATA,
) -> dict[str, Any]:
    """读取真实 P50 并构造经过完整一致性检查的紧凑层描述表。"""

    image = P50Image(image_path)
    external = None if external_metadata_path is None else Path(external_metadata_path)
    image.validate(external)

    model = image.metadata.get("model")
    if not isinstance(model, dict):
        raise ModelLayerDescriptorError("P50 metadata 缺少 model 对象")
    layer_count = int(model.get("num_hidden_layers", 0))
    if layer_count <= 0:
        raise ModelLayerDescriptorError(f"num_hidden_layers 非法：{layer_count}")
    if layer_count > HARDWARE_LAYER_CAPACITY:
        raise ModelLayerDescriptorError(
            f"模型层数 {layer_count} 超过硬件容量 {HARDWARE_LAYER_CAPACITY}"
        )

    layer_entries: dict[int, dict[str, dict[str, Any]]] = {}
    for layer_index in range(layer_count):
        names = _expected_layer_names(layer_index)
        entries = {name: image.tensor(name) for name in names}
        prefix = f"model.layers.{layer_index}."
        actual_names = set(image.tensor_names(prefix))
        if actual_names != set(names):
            missing = sorted(set(names) - actual_names)
            extra = sorted(actual_names - set(names))
            raise ModelLayerDescriptorError(
                f"layer{layer_index} 张量集合异常：missing={missing}, extra={extra}"
            )
        layer_entries[layer_index] = entries

    actual_layer_ids = sorted(
        {
            int(name.split(".")[2])
            for name in image.tensor_names()
            if name.startswith("model.layers.")
        }
    )
    expected_layer_ids = list(range(layer_count))
    if actual_layer_ids != expected_layer_ids:
        raise ModelLayerDescriptorError(
            f"层号不连续：{actual_layer_ids} != {expected_layer_ids}"
        )

    layer_bases = [
        int(layer_entries[index][f"model.layers.{index}.input_layernorm.weight"]["data_offset"])
        for index in range(layer_count)
    ]
    strides = [right - left for left, right in zip(layer_bases, layer_bases[1:])]
    if not strides or len(set(strides)) != 1:
        raise ModelLayerDescriptorError(f"层基址步长不唯一：{strides}")
    layer_stride = strides[0]

    layer0_base = layer_bases[0]
    templates = [
        _layer_template(
            role,
            suffix,
            layer_entries[0][f"model.layers.0.{suffix}"],
            layer0_base,
        )
        for role, suffix in LAYER_TENSOR_ROLES
    ]

    for layer_index, layer_base in enumerate(layer_bases):
        entries = layer_entries[layer_index]
        for template in templates:
            name = f"model.layers.{layer_index}.{template['name_suffix']}"
            _validate_template_entry(
                layer_index=layer_index,
                layer_base=layer_base,
                template=template,
                entry=entries[name],
            )

    layer_ends = [
        max(_entry_end(entry) for entry in layer_entries[index].values())
        for index in range(layer_count)
    ]
    layer_spans = [end - base for base, end in zip(layer_bases, layer_ends)]
    if len(set(layer_spans)) != 1:
        raise ModelLayerDescriptorError(f"层有效跨度不唯一：{layer_spans}")
    layer_span = layer_spans[0]
    if layer_span > layer_stride:
        raise ModelLayerDescriptorError(
            f"层有效跨度 {layer_span} 超过层步长 {layer_stride}"
        )

    global_names = {
        "model.embed_tokens.weight",
        "model.norm.weight",
    }
    actual_global_names = {
        name for name in image.tensor_names() if not name.startswith("model.layers.")
    }
    if actual_global_names != global_names:
        raise ModelLayerDescriptorError(
            f"全局张量集合异常：{sorted(actual_global_names)}"
        )

    if not bool(model.get("tie_word_embeddings")):
        raise ModelLayerDescriptorError("当前描述器要求 tied embedding/LM Head")

    descriptor: dict[str, Any] = {
        "schema": SCHEMA,
        "image": {
            "path": _relative_project_path(Path(image_path)),
            "external_metadata_path": (
                None if external is None else _relative_project_path(external)
            ),
            "size_bytes": int(image.file_size),
            "data_offset": int(image.header.data_offset),
            "tensor_count": int(image.header.tensor_count),
            "group_size": int(image.header.group_size),
            "metadata_sha256": _canonical_sha256(image.metadata),
            "flags": {
                "lora_merged": image.header.lora_merged,
                "tied_embedding": image.header.tied_embedding,
            },
        },
        "architecture": str(image.metadata["architecture"]),
        "model": dict(model),
        "hardware_contract": {
            "layer_capacity": int(HARDWARE_LAYER_CAPACITY),
            "active_layer_count": layer_count,
            "unused_layer_slots": int(HARDWARE_LAYER_CAPACITY - layer_count),
            "max_context": int(HARDWARE_MAX_CONTEXT),
            "model_max_position_embeddings": int(model["max_position_embeddings"]),
            "context_limited_by_hardware": (
                int(HARDWARE_MAX_CONTEXT) < int(model["max_position_embeddings"])
            ),
        },
        "quantization": dict(image.metadata["quantization"]),
        "tensor_counts": {
            "global": 2,
            "per_layer": len(templates),
            "all_layers": layer_count * len(templates),
            "total": int(image.header.tensor_count),
        },
        "global_tensors": [
            _global_tensor(
                "embedding_and_tied_lm_head_weight",
                image.tensor("model.embed_tokens.weight"),
            ),
            _global_tensor("final_norm_weight", image.tensor("model.norm.weight")),
        ],
        "lm_head": {
            "separate_tensor_present": False,
            "tied_to": "model.embed_tokens.weight",
        },
        "layer_layout": {
            "layer_count": layer_count,
            "tensor_count_per_layer": len(templates),
            "first_layer_base_offset": layer_bases[0],
            "layer_stride_bytes": layer_stride,
            "layer_payload_span_bytes": layer_span,
            "layer_alignment_gap_bytes": layer_stride - layer_span,
            "layer_bases": layer_bases,
        },
        "tensor_templates": templates,
    }

    if descriptor["tensor_counts"]["global"] + descriptor["tensor_counts"]["all_layers"] != descriptor["tensor_counts"]["total"]:
        raise ModelLayerDescriptorError("全局+层内张量数量与 P50 header 不一致")
    return descriptor


def expand_layer_tensors(
    descriptor: dict[str, Any], layer_index: int
) -> list[dict[str, Any]]:
    """把紧凑描述中的一层展开为带绝对主机文件偏移的 12 个张量。"""

    layout = descriptor["layer_layout"]
    layer_count = int(layout["layer_count"])
    if not 0 <= layer_index < layer_count:
        raise IndexError(f"layer_index={layer_index} 越界，有效范围 0..{layer_count - 1}")
    layer_base = int(layout["layer_bases"][layer_index])
    result: list[dict[str, Any]] = []
    for template in descriptor["tensor_templates"]:
        item: dict[str, Any] = {
            "role": template["role"],
            "name": f"model.layers.{layer_index}.{template['name_suffix']}",
            "shape": list(template["shape"]),
            "source_dtype": template["source_dtype"],
            "storage": template["storage"],
            "data_offset": layer_base + int(template["data_offset_in_layer"]),
            "data_nbytes": int(template["data_nbytes"]),
        }
        if "scale_offset_in_layer" in template:
            item.update(
                {
                    "scale_offset": layer_base
                    + int(template["scale_offset_in_layer"]),
                    "scale_nbytes": int(template["scale_nbytes"]),
                    "padded_columns": int(template["padded_columns"]),
                    "groups_per_row": int(template["groups_per_row"]),
                }
            )
        result.append(item)
    return result


def expand_all_layers(descriptor: dict[str, Any]) -> list[dict[str, Any]]:
    """展开全部活动层，供主机权重加载器或后续微码生成器直接消费。"""

    return [
        {
            "layer_index": layer_index,
            "base_offset": int(descriptor["layer_layout"]["layer_bases"][layer_index]),
            "tensors": expand_layer_tensors(descriptor, layer_index),
        }
        for layer_index in range(int(descriptor["layer_layout"]["layer_count"]))
    ]


def load_reference(path: str | Path = DEFAULT_REFERENCE) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ModelLayerDescriptorError("描述表 reference 顶层必须是对象")
    return value


def verify_reference(
    image_path: str | Path = DEFAULT_IMAGE,
    external_metadata_path: str | Path | None = DEFAULT_EXTERNAL_METADATA,
    reference_path: str | Path = DEFAULT_REFERENCE,
) -> dict[str, Any]:
    generated = build_model_layer_descriptor(image_path, external_metadata_path)
    reference = load_reference(reference_path)
    if generated != reference:
        raise ModelLayerDescriptorError(
            "冻结的模型层描述表与真实 P50 不一致；请先审查模型或 schema 变化"
        )
    return generated


def _print_summary(descriptor: dict[str, Any]) -> None:
    model = descriptor["model"]
    hardware = descriptor["hardware_contract"]
    layout = descriptor["layer_layout"]
    print(f"schema={descriptor['schema']}")
    print(
        f"model_layers={model['num_hidden_layers']}, "
        f"hardware_layer_capacity={hardware['layer_capacity']}, "
        f"unused_layer_slots={hardware['unused_layer_slots']}"
    )
    print(
        f"model_max_positions={model['max_position_embeddings']}, "
        f"hardware_max_context={hardware['max_context']}"
    )
    print(
        f"layer_tensors={layout['tensor_count_per_layer']}, "
        f"layer_stride={layout['layer_stride_bytes']}, "
        f"layer_span={layout['layer_payload_span_bytes']}, "
        f"layer_gap={layout['layer_alignment_gap_bytes']}"
    )
    print(
        f"first_layer_base={layout['layer_bases'][0]}, "
        f"last_layer_base={layout['layer_bases'][-1]}"
    )
    for template in descriptor["tensor_templates"]:
        scale = (
            f", scale_offset_in_layer={template['scale_offset_in_layer']}, "
            f"groups={template['groups_per_row']}"
            if "scale_offset_in_layer" in template
            else ""
        )
        print(
            f"{template['role']}: shape={template['shape']}, "
            f"storage={template['storage']}, "
            f"data_offset_in_layer={template['data_offset_in_layer']}"
            f"{scale}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument(
        "--external-metadata", type=Path, default=DEFAULT_EXTERNAL_METADATA
    )
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("summary", help="打印模型层描述摘要")
    subparsers.add_parser("verify", help="与冻结 reference 逐项比较")
    subparsers.add_parser("dump", help="输出紧凑 JSON 描述表")
    layer = subparsers.add_parser("layer", help="展开指定层的 12 个张量")
    layer.add_argument("index", type=int)
    subparsers.add_parser("all-layers", help="展开全部活动层")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    descriptor = build_model_layer_descriptor(args.image, args.external_metadata)
    if args.command == "summary":
        _print_summary(descriptor)
    elif args.command == "verify":
        reference = load_reference(args.reference)
        if descriptor != reference:
            raise ModelLayerDescriptorError(
                "冻结的模型层描述表与真实 P50 不一致"
            )
        print(
            "H1 模型层描述表验证通过："
            f"{descriptor['layer_layout']['layer_count']} 层，"
            f"{descriptor['tensor_counts']['all_layers']} 个层内张量"
        )
    elif args.command == "dump":
        print(json.dumps(descriptor, ensure_ascii=False, indent=2))
    elif args.command == "layer":
        print(
            json.dumps(
                expand_layer_tensors(descriptor, args.index),
                ensure_ascii=False,
                indent=2,
            )
        )
    elif args.command == "all-layers":
        print(json.dumps(expand_all_layers(descriptor), ensure_ascii=False, indent=2))
    else:
        raise AssertionError(args.command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
