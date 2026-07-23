#!/usr/bin/env python3
"""校验盘古50K .p50 模型镜像的结构和抽样量化结果。"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors import safe_open

from export_qwen25_fpga import (
    HEADER_SIZE,
    HEADER_STRUCT,
    MAGIC,
    build_lora_map,
)

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_header_and_metadata(path: Path) -> tuple[dict[str, int | bytes], dict[str, Any]]:
    with path.open("rb") as handle:
        raw = handle.read(HEADER_STRUCT.size)
        if len(raw) != HEADER_STRUCT.size:
            raise ValueError("文件过短，无法读取固定头")
        fields = HEADER_STRUCT.unpack(raw)
        header = {
            "magic": fields[0],
            "version": fields[1],
            "header_size": fields[2],
            "metadata_size": fields[3],
            "data_offset": fields[4],
            "tensor_count": fields[5],
            "group_size": fields[6],
            "flags": fields[7],
            "reserved": fields[8],
        }
        if header["magic"] != MAGIC:
            raise ValueError(f"魔数错误：{header['magic']!r}")
        if header["header_size"] != HEADER_SIZE:
            raise ValueError(f"头部大小异常：{header['header_size']}")
        handle.seek(int(header["header_size"]))
        metadata_raw = handle.read(int(header["metadata_size"]))
        metadata = json.loads(metadata_raw.decode("utf-8"))
    return header, metadata


def validate_ranges(path: Path, header: dict[str, int | bytes], metadata: dict[str, Any]) -> None:
    file_size = path.stat().st_size
    tensors = metadata["tensors"]
    if len(tensors) != header["tensor_count"]:
        raise ValueError(
            f"张量数量不一致：header={header['tensor_count']} metadata={len(tensors)}"
        )
    if metadata.get("image_size") != file_size:
        raise ValueError(
            f"镜像大小不一致：metadata={metadata.get('image_size')} actual={file_size}"
        )
    if metadata.get("data_offset") != header["data_offset"]:
        raise ValueError("data_offset 在固定头和 JSON 中不一致")

    ranges: list[tuple[int, int, str]] = []
    for tensor in tensors:
        name = tensor["name"]
        data_start = int(tensor["data_offset"])
        data_end = data_start + int(tensor["data_nbytes"])
        if data_start < int(header["data_offset"]) or data_end > file_size:
            raise ValueError(f"数据越界：{name}")
        ranges.append((data_start, data_end, f"{name}:data"))

        if "scale_offset" in tensor:
            scale_start = int(tensor["scale_offset"])
            scale_end = scale_start + int(tensor["scale_nbytes"])
            if scale_start < int(header["data_offset"]) or scale_end > file_size:
                raise ValueError(f"scale 越界：{name}")
            ranges.append((scale_start, scale_end, f"{name}:scale"))

    ranges.sort()
    for previous, current in zip(ranges, ranges[1:]):
        if current[0] < previous[1]:
            raise ValueError(f"数据区重叠：{previous[2]} 与 {current[2]}")


def decode_group(
    image_path: Path,
    entry: dict[str, Any],
    row: int,
    group: int,
    group_size: int,
) -> tuple[np.ndarray, float]:
    padded_columns = int(entry["padded_columns"])
    groups_per_row = int(entry["groups_per_row"])
    row_bytes = padded_columns // 2
    packed_offset = int(entry["data_offset"]) + row * row_bytes + group * (group_size // 2)
    scale_offset = int(entry["scale_offset"]) + (row * groups_per_row + group) * 2

    with image_path.open("rb") as handle:
        handle.seek(packed_offset)
        packed = np.frombuffer(handle.read(group_size // 2), dtype=np.uint8)
        handle.seek(scale_offset)
        scale = float(np.frombuffer(handle.read(2), dtype=np.float16)[0])

    values = np.empty(group_size, dtype=np.int8)
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    values[0::2] = np.where(low >= 8, low.astype(np.int16) - 16, low).astype(np.int8)
    values[1::2] = np.where(high >= 8, high.astype(np.int16) - 16, high).astype(np.int8)
    return values.astype(np.float32) * scale, scale


def merged_reference_group(
    model: Any,
    name: str,
    row: int,
    start_column: int,
    group_size: int,
    lora_map: dict[str, tuple[torch.Tensor, torch.Tensor, float]],
) -> np.ndarray:
    tensor = model.get_tensor(name).float()
    columns = tensor.shape[1]
    end = min(start_column + group_size, columns)
    result = torch.zeros(group_size, dtype=torch.float32)
    result[: end - start_column] = tensor[row, start_column:end]

    if name in lora_map:
        a_tensor, b_tensor, scaling = lora_map[name]
        delta = torch.matmul(b_tensor[row], a_tensor[:, start_column:end]) * scaling
        result[: end - start_column] += delta
    return result.numpy()


def verify_quantization(
    image_path: Path,
    metadata: dict[str, Any],
    model_path: Path,
    adapter_path: Path,
    adapter_config_path: Path,
) -> None:
    entries = {item["name"]: item for item in metadata["tensors"]}
    group_size = int(metadata["quantization"]["group_size"])
    lora_map, _ = build_lora_map(adapter_path, adapter_config_path)
    samples = [
        ("model.layers.0.self_attn.q_proj.weight", 0, 0),
        ("model.layers.7.self_attn.v_proj.weight", 31, 3),
        ("model.layers.23.mlp.gate_proj.weight", 1024, 5),
        ("model.layers.12.mlp.down_proj.weight", 400, 20),
    ]

    print("\n抽样反量化误差：")
    with safe_open(str(model_path), framework="pt", device="cpu") as model:
        for name, row, group in samples:
            entry = entries[name]
            decoded, scale = decode_group(image_path, entry, row, group, group_size)
            reference = merged_reference_group(
                model,
                name,
                row,
                group * group_size,
                group_size,
                lora_map,
            )
            error = np.abs(decoded - reference)
            maximum = float(error.max())
            mean = float(error.mean())
            limit = scale / 2.0 + 5e-4
            status = "PASS" if maximum <= limit else "FAIL"
            print(
                f"{status}  {name} row={row} group={group}  "
                f"max={maximum:.6f} mean={mean:.6f} scale={scale:.6f}"
            )
            if status != "PASS":
                raise ValueError(f"量化误差超过理论舍入上限：{name}")


def main() -> int:
    parser = argparse.ArgumentParser(description="校验盘古50K .p50 模型镜像")
    parser.add_argument(
        "--image",
        type=Path,
        default=Path("model_output/yanbo_qwen25_0.5b_int4.p50"),
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
    args = parser.parse_args()

    header, metadata = read_header_and_metadata(args.image)
    validate_ranges(args.image, header, metadata)
    print("固定头与 JSON 索引：PASS")
    print(f"张量数量：{header['tensor_count']}")
    print(f"文件大小：{args.image.stat().st_size:,} 字节")
    print(f"SHA256：{sha256_file(args.image)}")

    verify_quantization(
        args.image,
        metadata,
        args.model_dir / "model.safetensors",
        args.adapter_dir / "adapter_model.safetensors",
        args.adapter_dir / "adapter_config.json",
    )
    print("\n镜像校验：全部 PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
