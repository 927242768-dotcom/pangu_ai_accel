#!/usr/bin/env python3
"""校验盘古50K .p50 模型镜像的结构和抽样量化结果。"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors import safe_open

from export_qwen25_fpga import build_lora_map
from p50_format import P50Image

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


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
        "--metadata",
        type=Path,
        default=Path("model_output/yanbo_qwen25_0.5b_int4.json"),
        help="与镜像内嵌目录逐字段比较的外部 JSON 元数据",
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

    image = P50Image(args.image)
    report = image.validate(args.metadata)
    metadata = image.metadata
    print("固定头、内嵌 JSON 索引与派生布局：PASS")
    print("外部 JSON 与镜像内嵌 JSON：逐字段完全一致")
    print(f"张量数量：{report.tensor_count}")
    print(f"文件大小：{report.image_size:,} 字节")
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
