#!/usr/bin/env python3
"""将 Qwen2.5-0.5B + LoRA 导出为盘古50K FPGA 友好的 INT4 模型镜像。

镜像设计目标：
1. 所有二维权重按输出行、输入维分组做对称 INT4 量化；
2. 每组保存一个 FP16 scale，组内两个 INT4 打包为一个字节；
3. 一维 bias / RMSNorm 权重保存为 FP16；
4. 先把 q_proj / v_proj 的 LoRA 增量合并到基础权重；
5. 输出一个带固定头、JSON 索引和 4 KiB 对齐数据区的 .p50 文件。

该工具只负责模型格式转换，不代表当前 V1 位流已经实现完整 Qwen 推理。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Iterator, Sequence

import numpy as np
import torch
from safetensors import safe_open

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

MAGIC = b"P50Q4V1\0"
FORMAT_VERSION = 1
HEADER_SIZE = 4096
DATA_ALIGNMENT = 4096
DEFAULT_GROUP_SIZE = 64
CHUNK_ROWS = 256

# 头部字段：magic, version, header_size, metadata_size, data_offset,
# tensor_count, group_size, flags, reserved。
HEADER_STRUCT = struct.Struct("<8sIIQQIIII")
FLAG_LORA_MERGED = 1 << 0
FLAG_TIED_EMBEDDING = 1 << 1


@dataclass(frozen=True)
class TensorPlan:
    name: str
    shape: tuple[int, ...]
    source_dtype: str
    storage: str
    packed_nbytes: int
    scale_nbytes: int
    padded_columns: int
    groups_per_row: int

    @property
    def total_nbytes(self) -> int:
        return self.packed_nbytes + self.scale_nbytes


def align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def human_bytes(value: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    number = float(value)
    for unit in units:
        if number < 1024.0 or unit == units[-1]:
            return f"{number:.2f} {unit}"
        number /= 1024.0
    return f"{value} B"


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def tensor_execution_key(name: str) -> tuple[int, int, str]:
    """让镜像中的张量尽量按实际推理顺序排列。"""
    if name == "model.embed_tokens.weight":
        return (0, 0, name)
    if name.startswith("model.layers."):
        parts = name.split(".")
        layer = int(parts[2])
        suffix = ".".join(parts[3:])
        order = {
            "input_layernorm.weight": 0,
            "self_attn.q_proj.weight": 1,
            "self_attn.q_proj.bias": 2,
            "self_attn.k_proj.weight": 3,
            "self_attn.k_proj.bias": 4,
            "self_attn.v_proj.weight": 5,
            "self_attn.v_proj.bias": 6,
            "self_attn.o_proj.weight": 7,
            "post_attention_layernorm.weight": 8,
            "mlp.gate_proj.weight": 9,
            "mlp.up_proj.weight": 10,
            "mlp.down_proj.weight": 11,
        }.get(suffix, 99)
        return (1 + layer, order, name)
    if name == "model.norm.weight":
        return (1000, 0, name)
    return (2000, 0, name)


def build_lora_map(adapter_path: Path, adapter_config_path: Path) -> tuple[dict[str, tuple[torch.Tensor, torch.Tensor, float]], dict[str, Any]]:
    config = load_json(adapter_config_path)
    rank = int(config["r"])
    alpha = float(config["lora_alpha"])
    scaling = alpha / rank

    tensors: dict[str, torch.Tensor] = {}
    with safe_open(str(adapter_path), framework="pt", device="cpu") as handle:
        for key in handle.keys():
            tensors[key] = handle.get_tensor(key).float()

    pairs: dict[str, tuple[torch.Tensor, torch.Tensor, float]] = {}
    suffix_a = ".lora_A.weight"
    for key, a_tensor in tensors.items():
        if not key.endswith(suffix_a):
            continue
        b_key = key[: -len(suffix_a)] + ".lora_B.weight"
        if b_key not in tensors:
            raise ValueError(f"LoRA 缺少配对张量：{b_key}")

        base_prefix = key[: -len(suffix_a)]
        if base_prefix.startswith("base_model.model."):
            base_prefix = base_prefix[len("base_model.model.") :]
        base_key = base_prefix + ".weight"
        pairs[base_key] = (a_tensor, tensors[b_key], scaling)

    return pairs, config


def make_plan(name: str, shape: Sequence[int], source_dtype: str, group_size: int) -> TensorPlan:
    dimensions = tuple(int(item) for item in shape)
    if len(dimensions) == 2:
        rows, columns = dimensions
        padded_columns = align_up(columns, group_size)
        groups_per_row = padded_columns // group_size
        packed_nbytes = rows * padded_columns // 2
        scale_nbytes = rows * groups_per_row * 2
        return TensorPlan(
            name=name,
            shape=dimensions,
            source_dtype=source_dtype,
            storage="int4_groupwise_symmetric",
            packed_nbytes=packed_nbytes,
            scale_nbytes=scale_nbytes,
            padded_columns=padded_columns,
            groups_per_row=groups_per_row,
        )

    element_count = math.prod(dimensions)
    return TensorPlan(
        name=name,
        shape=dimensions,
        source_dtype=source_dtype,
        storage="float16",
        packed_nbytes=element_count * 2,
        scale_nbytes=0,
        padded_columns=0,
        groups_per_row=0,
    )


def inspect_model(model_path: Path, group_size: int) -> tuple[list[TensorPlan], int, int]:
    plans: list[TensorPlan] = []
    source_bytes = 0
    parameters = 0
    with safe_open(str(model_path), framework="pt", device="cpu") as handle:
        names = sorted(handle.keys(), key=tensor_execution_key)
        for name in names:
            tensor_slice = handle.get_slice(name)
            shape = tuple(tensor_slice.get_shape())
            dtype = str(tensor_slice.get_dtype())
            plan = make_plan(name, shape, dtype, group_size)
            plans.append(plan)
            count = math.prod(shape)
            parameters += count
            # 当前基础模型为 BF16；这里按实际源 dtype 做常见字节估算。
            source_bytes += count * (2 if dtype in {"BF16", "F16"} else 4)
    return plans, parameters, source_bytes


def estimate_image_layout(plans: Sequence[TensorPlan], metadata_size_guess: int = 256 * 1024) -> tuple[int, int]:
    data_offset = align_up(HEADER_SIZE + metadata_size_guess, DATA_ALIGNMENT)
    cursor = data_offset
    for plan in plans:
        cursor = align_up(cursor, DATA_ALIGNMENT)
        cursor += plan.packed_nbytes
        if plan.scale_nbytes:
            cursor = align_up(cursor, 64)
            cursor += plan.scale_nbytes
    return data_offset, cursor


def print_analysis(
    plans: Sequence[TensorPlan],
    config: dict[str, Any],
    parameters: int,
    source_bytes: int,
    group_size: int,
) -> None:
    int4_bytes = sum(plan.packed_nbytes for plan in plans if plan.storage.startswith("int4"))
    scale_bytes = sum(plan.scale_nbytes for plan in plans)
    fp16_bytes = sum(plan.packed_nbytes for plan in plans if plan.storage == "float16")
    _, estimated_total = estimate_image_layout(plans)

    hidden_size = int(config["hidden_size"])
    layers = int(config["num_hidden_layers"])
    kv_heads = int(config["num_key_value_heads"])
    attention_heads = int(config["num_attention_heads"])
    head_dim = hidden_size // attention_heads
    kv_values_per_token = layers * kv_heads * head_dim * 2

    print("\n=== Qwen2.5 FPGA 导出分析 ===")
    print(f"参数量：{parameters:,}")
    print(f"源 BF16 权重估算：{human_bytes(source_bytes)}")
    print(f"INT4 权重：{human_bytes(int4_bytes)}")
    print(f"FP16 分组尺度：{human_bytes(scale_bytes)}（group={group_size}）")
    print(f"FP16 bias / norm：{human_bytes(fp16_bytes)}")
    print(f"含对齐和索引的镜像估算：约 {human_bytes(estimated_total)}")
    print(f"KV Cache 每 token：INT8 {human_bytes(kv_values_per_token)}，FP16 {human_bytes(kv_values_per_token * 2)}")
    print(f"512 token KV Cache：INT8 {human_bytes(kv_values_per_token * 512)}，FP16 {human_bytes(kv_values_per_token * 1024)}")
    print("板载 DDR3：1.00 GiB；容量上可容纳 INT4 模型和短上下文，但计算吞吐与完整控制逻辑仍是主要瓶颈。")


def merge_lora_if_needed(
    name: str,
    tensor: torch.Tensor,
    lora_map: dict[str, tuple[torch.Tensor, torch.Tensor, float]],
) -> torch.Tensor:
    if name not in lora_map:
        return tensor.float()
    a_tensor, b_tensor, scaling = lora_map[name]
    if tuple(tensor.shape) != (b_tensor.shape[0], a_tensor.shape[1]):
        raise ValueError(
            f"LoRA 与基础权重形状不一致：{name}, base={tuple(tensor.shape)}, "
            f"A={tuple(a_tensor.shape)}, B={tuple(b_tensor.shape)}"
        )
    return tensor.float() + torch.matmul(b_tensor, a_tensor) * scaling


def quantize_int4_groupwise(tensor: torch.Tensor, group_size: int) -> tuple[bytes, bytes, int]:
    """返回 packed INT4、FP16 scales、填充后的输入列数。"""
    if tensor.ndim != 2:
        raise ValueError("INT4 量化只支持二维张量")
    rows, columns = tensor.shape
    padded_columns = align_up(columns, group_size)
    groups = padded_columns // group_size

    packed_parts: list[bytes] = []
    scale_parts: list[bytes] = []

    for start in range(0, rows, CHUNK_ROWS):
        end = min(start + CHUNK_ROWS, rows)
        chunk = tensor[start:end].contiguous().float()
        if padded_columns != columns:
            chunk = torch.nn.functional.pad(chunk, (0, padded_columns - columns))
        grouped = chunk.view(end - start, groups, group_size)
        maximum = grouped.abs().amax(dim=2)
        scales = maximum / 7.0
        scales = torch.where(scales > 0, scales, torch.ones_like(scales))
        quantized = torch.round(grouped / scales.unsqueeze(-1)).clamp_(-7, 7).to(torch.int8)
        flat = quantized.view(end - start, padded_columns)
        low = torch.bitwise_and(flat[:, 0::2].to(torch.int16), 0x0F)
        high = torch.bitwise_left_shift(
            torch.bitwise_and(flat[:, 1::2].to(torch.int16), 0x0F), 4
        )
        packed = torch.bitwise_or(low, high).to(torch.uint8).contiguous()
        packed_parts.append(packed.numpy().tobytes(order="C"))
        scale_parts.append(scales.to(torch.float16).contiguous().numpy().tobytes(order="C"))

    return b"".join(packed_parts), b"".join(scale_parts), padded_columns


def pad_file(handle: BinaryIO, alignment: int) -> None:
    position = handle.tell()
    target = align_up(position, alignment)
    if target > position:
        handle.write(b"\0" * (target - position))


def build_metadata_base(
    config: dict[str, Any],
    adapter_config: dict[str, Any],
    group_size: int,
    model_path: Path,
    adapter_path: Path,
) -> dict[str, Any]:
    return {
        "format": "pangu50k-qwen-int4",
        "format_version": FORMAT_VERSION,
        "model_type": config.get("model_type"),
        "architecture": config.get("architectures", [None])[0],
        "source_model": str(model_path.resolve()),
        "source_adapter": str(adapter_path.resolve()),
        "quantization": {
            "weight_bits": 4,
            "scheme": "symmetric_per_row_group",
            "group_size": group_size,
            "range": [-7, 7],
            "packed_order": "low_nibble_first",
            "scale_dtype": "float16",
        },
        "lora": {
            "merged": True,
            "rank": adapter_config.get("r"),
            "alpha": adapter_config.get("lora_alpha"),
            "targets": adapter_config.get("target_modules", []),
        },
        "model": {
            "hidden_size": config["hidden_size"],
            "intermediate_size": config["intermediate_size"],
            "num_hidden_layers": config["num_hidden_layers"],
            "num_attention_heads": config["num_attention_heads"],
            "num_key_value_heads": config["num_key_value_heads"],
            "vocab_size": config["vocab_size"],
            "rms_norm_eps": config["rms_norm_eps"],
            "rope_theta": config["rope_theta"],
            "max_position_embeddings": config["max_position_embeddings"],
            "tie_word_embeddings": config.get("tie_word_embeddings", False),
        },
        "tensor_order": "execution_order",
        "tensors": [],
    }


def export_image(
    model_path: Path,
    adapter_path: Path,
    config_path: Path,
    adapter_config_path: Path,
    output_path: Path,
    metadata_path: Path,
    group_size: int,
) -> None:
    config = load_json(config_path)
    lora_map, adapter_config = build_lora_map(adapter_path, adapter_config_path)
    metadata = build_metadata_base(
        config, adapter_config, group_size, model_path, adapter_path
    )

    # 先留出固定头和较充裕的 JSON 索引区域，避免完成量化后搬移大数据区。
    metadata_reserve = 512 * 1024
    data_offset = align_up(HEADER_SIZE + metadata_reserve, DATA_ALIGNMENT)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    start_time = time.monotonic()

    with safe_open(str(model_path), framework="pt", device="cpu") as model, output_path.open("w+b") as output:
        output.write(b"\0" * data_offset)
        names = sorted(model.keys(), key=tensor_execution_key)

        for index, name in enumerate(names, start=1):
            source = model.get_tensor(name)
            tensor = merge_lora_if_needed(name, source, lora_map)
            pad_file(output, DATA_ALIGNMENT)
            data_position = output.tell()

            entry: dict[str, Any] = {
                "name": name,
                "shape": list(tensor.shape),
                "source_dtype": str(source.dtype).replace("torch.", ""),
            }

            if tensor.ndim == 2:
                packed, scales, padded_columns = quantize_int4_groupwise(tensor, group_size)
                output.write(packed)
                scale_position = align_up(output.tell(), 64)
                output.write(b"\0" * (scale_position - output.tell()))
                output.write(scales)
                entry.update(
                    {
                        "storage": "int4_groupwise_symmetric",
                        "data_offset": data_position,
                        "data_nbytes": len(packed),
                        "scale_offset": scale_position,
                        "scale_nbytes": len(scales),
                        "padded_columns": padded_columns,
                        "groups_per_row": padded_columns // group_size,
                    }
                )
            else:
                fp16 = tensor.to(torch.float16).contiguous().numpy().tobytes(order="C")
                output.write(fp16)
                entry.update(
                    {
                        "storage": "float16",
                        "data_offset": data_position,
                        "data_nbytes": len(fp16),
                    }
                )

            metadata["tensors"].append(entry)
            elapsed = time.monotonic() - start_time
            print(
                f"[{index:3d}/{len(names):3d}] {name:64s} "
                f"{human_bytes(output.tell())}  {elapsed:.1f}s",
                flush=True,
            )
            del tensor, source

        final_size = output.tell()
        metadata["image_size"] = final_size
        metadata["data_offset"] = data_offset
        metadata_bytes = json.dumps(
            metadata, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
        if len(metadata_bytes) > metadata_reserve:
            raise RuntimeError(
                f"JSON 索引超过预留区：{len(metadata_bytes)} > {metadata_reserve}"
            )

        flags = FLAG_LORA_MERGED
        if config.get("tie_word_embeddings", False):
            flags |= FLAG_TIED_EMBEDDING
        header = HEADER_STRUCT.pack(
            MAGIC,
            FORMAT_VERSION,
            HEADER_SIZE,
            len(metadata_bytes),
            data_offset,
            len(metadata["tensors"]),
            group_size,
            flags,
            0,
        )
        output.seek(0)
        output.write(header)
        output.write(b"\0" * (HEADER_SIZE - len(header)))
        output.write(metadata_bytes)
        output.flush()
        os.fsync(output.fileno())

    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)

    print("\n导出完成：")
    print(f"镜像：{output_path}（{human_bytes(output_path.stat().st_size)}）")
    print(f"索引：{metadata_path}")
    print(f"耗时：{time.monotonic() - start_time:.1f} 秒")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Qwen2.5-0.5B + LoRA -> 盘古50K INT4 模型镜像"
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(r"D:\LLM\models\Qwen2.5-0.5B-Instruct"),
    )
    parser.add_argument(
        "--adapter-dir",
        type=Path,
        default=Path(r"D:\LLM\adapters\qwen2.5-0.5b-chat-lora"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("model_output/yanbo_qwen25_0.5b_int4.p50"),
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("model_output/yanbo_qwen25_0.5b_int4.json"),
    )
    parser.add_argument("--group-size", type=int, default=DEFAULT_GROUP_SIZE)
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="只计算精确参数量和内存预算，不生成大文件",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.group_size <= 0 or args.group_size % 2:
        raise SystemExit("group-size 必须是正偶数")

    model_path = args.model_dir / "model.safetensors"
    config_path = args.model_dir / "config.json"
    adapter_path = args.adapter_dir / "adapter_model.safetensors"
    adapter_config_path = args.adapter_dir / "adapter_config.json"
    for path in (model_path, config_path, adapter_path, adapter_config_path):
        if not path.is_file():
            raise SystemExit(f"缺少文件：{path}")

    config = load_json(config_path)
    plans, parameters, source_bytes = inspect_model(model_path, args.group_size)
    print_analysis(plans, config, parameters, source_bytes, args.group_size)
    if args.analyze_only:
        return 0

    export_image(
        model_path=model_path,
        adapter_path=adapter_path,
        config_path=config_path,
        adapter_config_path=adapter_config_path,
        output_path=args.output,
        metadata_path=args.metadata,
        group_size=args.group_size,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n用户中断。", file=sys.stderr)
        raise SystemExit(130)
