#!/usr/bin/env python3
"""查看、校验并提取盘古 50K ``.p50`` 模型镜像中的张量。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .p50_format import ExtractedBlock, P50FormatError, P50Image
except ImportError:
    from p50_format import ExtractedBlock, P50FormatError, P50Image

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

DEFAULT_IMAGE = Path("model_output/yanbo_qwen25_0.5b_int4.p50")
DEFAULT_METADATA = Path("model_output/yanbo_qwen25_0.5b_int4.json")


def human_bytes(value: int) -> str:
    units = ("B", "KiB", "MiB", "GiB")
    number = float(value)
    for unit in units:
        if number < 1024.0 or unit == units[-1]:
            return f"{number:.2f} {unit}"
        number /= 1024.0
    return f"{value} B"


def shape_text(entry: dict[str, Any]) -> str:
    return "x".join(str(item) for item in entry["shape"])


def print_summary(image: P50Image) -> None:
    header = image.header
    quantization = image.metadata["quantization"]
    storage_counts: dict[str, int] = {}
    for entry in image.tensors:
        storage = str(entry["storage"])
        storage_counts[storage] = storage_counts.get(storage, 0) + 1

    print("=== P50 镜像摘要 ===")
    print(f"文件：{image.path}")
    print(f"大小：{image.file_size:,} 字节（{human_bytes(image.file_size)}）")
    print(f"magic：{header.magic!r}")
    print(f"版本：{header.version}")
    print(f"固定头：{header.header_size} 字节")
    print(f"内嵌 JSON：{header.metadata_size:,} 字节")
    print(f"数据区起点：{header.data_offset:,}")
    print(f"张量数量：{header.tensor_count}")
    print(f"存储类型：{storage_counts}")
    print(
        "INT4："
        f"{quantization['scheme']}，group={quantization['group_size']}，"
        f"范围={quantization['range']}，{quantization['packed_order']}，"
        f"scale={quantization['scale_dtype']}，zero_point=0（对称量化）"
    )
    print(f"LoRA 已合并：{header.lora_merged}")
    print(f"Embedding/LM Head 共享：{header.tied_embedding}")


def command_verify(args: argparse.Namespace) -> int:
    image = P50Image(args.image)
    report = image.validate(args.metadata)
    print("P50 固定头、内嵌目录、形状、偏移和长度：PASS")
    if report.external_metadata_checked:
        print("外部 JSON 与镜像内嵌 JSON：逐字段完全一致")
    print(f"张量：{report.tensor_count}（INT4={report.int4_tensor_count}, FP16={report.float16_tensor_count}）")
    print(f"INT4/FP16 数据：{human_bytes(report.data_bytes)}")
    print(f"FP16 scales：{human_bytes(report.scale_bytes)}")
    print(f"镜像大小：{report.image_size:,} 字节")
    return 0


def command_summary(args: argparse.Namespace) -> int:
    image = P50Image(args.image)
    image.validate(args.metadata if args.check_metadata else None)
    print_summary(image)
    return 0


def command_list(args: argparse.Namespace) -> int:
    image = P50Image(args.image)
    image.validate()
    names = image.tensor_names(args.contains)
    if args.limit is not None:
        names = names[: args.limit]
    print("序号  形状              存储类型                       数据偏移      数据字节  张量名")
    for index, name in enumerate(names):
        entry = image.tensor(name)
        print(
            f"{index:4d}  {shape_text(entry):17s}  {entry['storage']:29s}  "
            f"{int(entry['data_offset']):10d}  {int(entry['data_nbytes']):10d}  {name}"
        )
    print(f"共 {len(names)} 个匹配张量")
    return 0


def command_describe(args: argparse.Namespace) -> int:
    image = P50Image(args.image)
    image.validate()
    entry = image.tensor(args.tensor)
    print(json.dumps(entry, ensure_ascii=False, indent=2))
    if entry["storage"] == "int4_groupwise_symmetric":
        print("zero_point：0（对称量化，无独立 zero point 数据区）")
        print(
            "存储顺序：按输出行 row-major；每行按输入列分组；"
            "每字节低半字节在前、高半字节在后；scales 按 [row, group] row-major。"
        )
    else:
        print("存储顺序：连续 C-order FP16。")
    return 0


def _preview(array: np.ndarray, count: int = 16) -> str:
    flat = array.reshape(-1)
    shown = flat[:count]
    suffix = " ..." if flat.size > count else ""
    return np.array2string(shown, separator=", ", max_line_width=160) + suffix


def save_block(block: ExtractedBlock, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "values": block.values,
        "row_start": np.asarray(block.row_start, dtype=np.int64),
        "row_count": np.asarray(block.row_count, dtype=np.int64),
        "column_start": np.asarray(block.column_start, dtype=np.int64),
        "column_count": np.asarray(block.column_count, dtype=np.int64),
        "tensor_name": np.asarray(block.tensor_name),
        "storage": np.asarray(block.storage),
    }
    if block.quantized is not None:
        payload["quantized"] = block.quantized
    if block.scales is not None:
        payload["scales"] = block.scales
    if block.scale_group_start is not None:
        payload["scale_group_start"] = np.asarray(
            block.scale_group_start, dtype=np.int64
        )
    np.savez_compressed(output, **payload)


def print_extraction(block: ExtractedBlock, output: Path | None) -> None:
    print(f"张量：{block.tensor_name}")
    print(f"存储：{block.storage}")
    print(
        f"范围：rows [{block.row_start}, {block.row_start + block.row_count})，"
        f"columns [{block.column_start}, {block.column_start + block.column_count})"
    )
    print(f"values：shape={block.values.shape} dtype={block.values.dtype}")
    print(f"values 预览：{_preview(block.values)}")
    if block.quantized is not None:
        print(
            f"quantized：shape={block.quantized.shape} dtype={block.quantized.dtype}，"
            f"预览={_preview(block.quantized)}"
        )
    if block.scales is not None:
        group_end = block.scale_group_start + block.scales.shape[1]
        print(
            f"scales：groups [{block.scale_group_start}, {group_end})，"
            f"shape={block.scales.shape} dtype={block.scales.dtype}，"
            f"预览={_preview(block.scales)}"
        )
    if output is not None:
        save_block(block, output)
        print(f"已保存：{output}")


def command_row(args: argparse.Namespace) -> int:
    image = P50Image(args.image)
    image.validate()
    block = image.extract_row(args.tensor, args.row)
    print_extraction(block, args.output)
    return 0


def command_block(args: argparse.Namespace) -> int:
    image = P50Image(args.image)
    image.validate()
    block = image.extract_block(
        args.tensor,
        args.row_start,
        args.row_count,
        args.column_start,
        args.column_count,
    )
    print_extraction(block, args.output)
    return 0


def add_image_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="盘古 50K .p50 模型镜像解析、校验与张量提取工具"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    verify = subparsers.add_parser("verify", help="全量校验镜像和外部 JSON")
    add_image_argument(verify)
    verify.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    verify.set_defaults(func=command_verify)

    summary = subparsers.add_parser("summary", help="显示镜像摘要")
    add_image_argument(summary)
    summary.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    summary.add_argument(
        "--check-metadata", action="store_true", help="同时校验外部 JSON"
    )
    summary.set_defaults(func=command_summary)

    list_parser = subparsers.add_parser("list", help="列出张量目录")
    add_image_argument(list_parser)
    list_parser.add_argument("--contains", help="只显示名称包含该字符串的张量")
    list_parser.add_argument("--limit", type=int)
    list_parser.set_defaults(func=command_list)

    describe = subparsers.add_parser("describe", help="查看单个张量的完整目录项")
    add_image_argument(describe)
    describe.add_argument("--tensor", required=True)
    describe.set_defaults(func=command_describe)

    row = subparsers.add_parser("row", help="按张量名提取任意一行")
    add_image_argument(row)
    row.add_argument("--tensor", required=True)
    row.add_argument("--row", type=int, required=True)
    row.add_argument("--output", type=Path, help="可选：保存为压缩 NPZ")
    row.set_defaults(func=command_row)

    block = subparsers.add_parser("block", help="按张量名提取任意二维数据块")
    add_image_argument(block)
    block.add_argument("--tensor", required=True)
    block.add_argument("--row-start", type=int, required=True)
    block.add_argument("--row-count", type=int, required=True)
    block.add_argument("--column-start", type=int, required=True)
    block.add_argument("--column-count", type=int, required=True)
    block.add_argument("--output", type=Path, help="可选：保存为压缩 NPZ")
    block.set_defaults(func=command_block)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        return int(args.func(args))
    except (P50FormatError, FileNotFoundError, KeyError, IndexError, json.JSONDecodeError) as error:
        print(f"错误：{error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
