#!/usr/bin/env python3
"""盘古 50K ``.p50`` 模型镜像的解析、校验与张量提取支持。

该模块只依赖 NumPy，不依赖 PyTorch 或 safetensors，便于后续上位机工具直接复用。
"""

from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

MAGIC = b"P50Q4V1\0"
FORMAT_VERSION = 1
HEADER_SIZE = 4096
DATA_ALIGNMENT = 4096
SCALE_ALIGNMENT = 64
HEADER_STRUCT = struct.Struct("<8sIIQQIIII")

FLAG_LORA_MERGED = 1 << 0
FLAG_TIED_EMBEDDING = 1 << 1
KNOWN_FLAGS = FLAG_LORA_MERGED | FLAG_TIED_EMBEDDING


class P50FormatError(ValueError):
    """表示镜像结构、元数据或提取参数不合法。"""


@dataclass(frozen=True)
class P50Header:
    magic: bytes
    version: int
    header_size: int
    metadata_size: int
    data_offset: int
    tensor_count: int
    group_size: int
    flags: int
    reserved: int

    @property
    def lora_merged(self) -> bool:
        return bool(self.flags & FLAG_LORA_MERGED)

    @property
    def tied_embedding(self) -> bool:
        return bool(self.flags & FLAG_TIED_EMBEDDING)


@dataclass(frozen=True)
class ValidationReport:
    tensor_count: int
    int4_tensor_count: int
    float16_tensor_count: int
    data_bytes: int
    scale_bytes: int
    image_size: int
    external_metadata_checked: bool


@dataclass(frozen=True)
class ExtractedBlock:
    tensor_name: str
    storage: str
    row_start: int
    row_count: int
    column_start: int
    column_count: int
    values: np.ndarray
    quantized: np.ndarray | None = None
    scales: np.ndarray | None = None
    scale_group_start: int | None = None


def align_up(value: int, alignment: int) -> int:
    if alignment <= 0:
        raise ValueError("alignment 必须为正数")
    return (value + alignment - 1) // alignment * alignment


def _product(shape: Iterable[int]) -> int:
    return math.prod(int(item) for item in shape)


def _first_difference(left: Any, right: Any, path: str = "$") -> str | None:
    """返回两个 JSON 值的第一处差异路径，完全一致时返回 ``None``。"""
    if type(left) is not type(right):
        return f"{path}: 类型不同 {type(left).__name__} != {type(right).__name__}"

    if isinstance(left, dict):
        left_keys = set(left)
        right_keys = set(right)
        if left_keys != right_keys:
            missing = sorted(left_keys - right_keys)
            extra = sorted(right_keys - left_keys)
            return f"{path}: 键集合不同，右侧缺少={missing}，右侧新增={extra}"
        for key in left:
            difference = _first_difference(left[key], right[key], f"{path}.{key}")
            if difference is not None:
                return difference
        return None

    if isinstance(left, list):
        if len(left) != len(right):
            return f"{path}: 列表长度不同 {len(left)} != {len(right)}"
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            difference = _first_difference(left_item, right_item, f"{path}[{index}]")
            if difference is not None:
                return difference
        return None

    if left != right:
        return f"{path}: 值不同 {left!r} != {right!r}"
    return None


class P50Image:
    """读取并操作一个 ``.p50`` 模型镜像。"""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"找不到 .p50 镜像：{self.path}")
        self.file_size = self.path.stat().st_size
        self.header, self.metadata = self._read_header_and_metadata()
        tensors = self.metadata.get("tensors")
        if not isinstance(tensors, list):
            raise P50FormatError("内嵌 JSON 缺少 tensors 列表")
        self.tensors: list[dict[str, Any]] = tensors
        self._tensor_map: dict[str, dict[str, Any]] = {}
        for entry in self.tensors:
            name = entry.get("name")
            if not isinstance(name, str) or not name:
                raise P50FormatError("张量目录中存在空名称或非字符串名称")
            if name in self._tensor_map:
                raise P50FormatError(f"张量名称重复：{name}")
            self._tensor_map[name] = entry

    def _read_header_and_metadata(self) -> tuple[P50Header, dict[str, Any]]:
        with self.path.open("rb") as handle:
            raw = handle.read(HEADER_STRUCT.size)
            if len(raw) != HEADER_STRUCT.size:
                raise P50FormatError("文件过短，无法读取固定头")
            fields = HEADER_STRUCT.unpack(raw)
            header = P50Header(
                magic=fields[0],
                version=int(fields[1]),
                header_size=int(fields[2]),
                metadata_size=int(fields[3]),
                data_offset=int(fields[4]),
                tensor_count=int(fields[5]),
                group_size=int(fields[6]),
                flags=int(fields[7]),
                reserved=int(fields[8]),
            )

            if header.magic != MAGIC:
                raise P50FormatError(f"魔数错误：{header.magic!r}")
            if header.version != FORMAT_VERSION:
                raise P50FormatError(
                    f"不支持的格式版本：{header.version}，当前工具仅支持 {FORMAT_VERSION}"
                )
            if header.header_size != HEADER_SIZE:
                raise P50FormatError(
                    f"固定头大小异常：{header.header_size}，预期 {HEADER_SIZE}"
                )
            if header.metadata_size <= 0:
                raise P50FormatError("metadata_size 必须大于 0")
            metadata_end = header.header_size + header.metadata_size
            if metadata_end > header.data_offset:
                raise P50FormatError(
                    f"JSON 索引越过数据区：metadata_end={metadata_end}, "
                    f"data_offset={header.data_offset}"
                )
            if header.data_offset > self.file_size:
                raise P50FormatError("data_offset 超出文件大小")

            handle.seek(header.header_size)
            metadata_raw = handle.read(header.metadata_size)
            if len(metadata_raw) != header.metadata_size:
                raise P50FormatError("JSON 索引读取不完整")
            try:
                metadata = json.loads(metadata_raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise P50FormatError(f"JSON 索引无法解析：{error}") from error
            if not isinstance(metadata, dict):
                raise P50FormatError("JSON 索引顶层必须是对象")
        return header, metadata

    def tensor_names(self, contains: str | None = None) -> list[str]:
        names = list(self._tensor_map)
        if contains is None:
            return names
        return [name for name in names if contains in name]

    def tensor(self, name: str) -> dict[str, Any]:
        try:
            return self._tensor_map[name]
        except KeyError as error:
            raise KeyError(f"镜像中不存在张量：{name}") from error

    def validate(self, external_metadata_path: str | Path | None = None) -> ValidationReport:
        """完整校验固定头、内嵌目录、派生长度、偏移范围与可选外部 JSON。"""
        header = self.header
        metadata = self.metadata

        if header.tensor_count != len(self.tensors):
            raise P50FormatError(
                f"张量数量不一致：header={header.tensor_count}, metadata={len(self.tensors)}"
            )
        if header.group_size <= 0 or header.group_size % 2:
            raise P50FormatError(f"group_size 必须为正偶数：{header.group_size}")
        if header.flags & ~KNOWN_FLAGS:
            raise P50FormatError(f"固定头包含未知 flags：0x{header.flags:08x}")
        if header.reserved != 0:
            raise P50FormatError(f"固定头 reserved 字段必须为 0：{header.reserved}")
        if header.data_offset % DATA_ALIGNMENT:
            raise P50FormatError(
                f"数据区起点未按 {DATA_ALIGNMENT} 字节对齐：{header.data_offset}"
            )

        if metadata.get("format") != "pangu50k-qwen-int4":
            raise P50FormatError(f"未知 format：{metadata.get('format')!r}")
        if metadata.get("format_version") != header.version:
            raise P50FormatError("format_version 在固定头和 JSON 中不一致")
        if metadata.get("image_size") != self.file_size:
            raise P50FormatError(
                f"镜像大小不一致：metadata={metadata.get('image_size')}, "
                f"actual={self.file_size}"
            )
        if metadata.get("data_offset") != header.data_offset:
            raise P50FormatError("data_offset 在固定头和 JSON 中不一致")

        quantization = metadata.get("quantization")
        if not isinstance(quantization, dict):
            raise P50FormatError("缺少 quantization 对象")
        expected_quantization = {
            "weight_bits": 4,
            "scheme": "symmetric_per_row_group",
            "group_size": header.group_size,
            "range": [-7, 7],
            "packed_order": "low_nibble_first",
            "scale_dtype": "float16",
        }
        for key, expected in expected_quantization.items():
            if quantization.get(key) != expected:
                raise P50FormatError(
                    f"quantization.{key} 异常：{quantization.get(key)!r} != {expected!r}"
                )

        ranges: list[tuple[int, int, str]] = []
        int4_count = 0
        float16_count = 0
        data_bytes = 0
        scale_bytes = 0

        for entry in self.tensors:
            name = entry["name"]
            shape_raw = entry.get("shape")
            if not isinstance(shape_raw, list) or not shape_raw:
                raise P50FormatError(f"张量 shape 非法：{name}")
            shape = tuple(int(item) for item in shape_raw)
            if any(item <= 0 for item in shape):
                raise P50FormatError(f"张量 shape 必须全部为正数：{name} {shape}")

            storage = entry.get("storage")
            data_offset = self._required_nonnegative_int(entry, "data_offset", name)
            data_nbytes = self._required_nonnegative_int(entry, "data_nbytes", name)
            if data_nbytes <= 0:
                raise P50FormatError(f"data_nbytes 必须大于 0：{name}")
            if data_offset % DATA_ALIGNMENT:
                raise P50FormatError(
                    f"张量数据未按 {DATA_ALIGNMENT} 字节对齐：{name} offset={data_offset}"
                )
            self._append_checked_range(
                ranges, data_offset, data_nbytes, f"{name}:data", header.data_offset
            )
            data_bytes += data_nbytes

            if storage == "int4_groupwise_symmetric":
                int4_count += 1
                if len(shape) != 2:
                    raise P50FormatError(f"INT4 张量必须是二维：{name} {shape}")
                rows, columns = shape
                padded_columns = self._required_nonnegative_int(
                    entry, "padded_columns", name
                )
                groups_per_row = self._required_nonnegative_int(
                    entry, "groups_per_row", name
                )
                expected_padded = align_up(columns, header.group_size)
                expected_groups = expected_padded // header.group_size
                expected_data_nbytes = rows * expected_padded // 2
                expected_scale_nbytes = rows * expected_groups * 2
                if padded_columns != expected_padded:
                    raise P50FormatError(
                        f"padded_columns 不匹配：{name} {padded_columns} != {expected_padded}"
                    )
                if groups_per_row != expected_groups:
                    raise P50FormatError(
                        f"groups_per_row 不匹配：{name} {groups_per_row} != {expected_groups}"
                    )
                if data_nbytes != expected_data_nbytes:
                    raise P50FormatError(
                        f"INT4 数据长度不匹配：{name} {data_nbytes} != {expected_data_nbytes}"
                    )

                scale_offset = self._required_nonnegative_int(entry, "scale_offset", name)
                scale_nbytes_entry = self._required_nonnegative_int(
                    entry, "scale_nbytes", name
                )
                if scale_offset % SCALE_ALIGNMENT:
                    raise P50FormatError(
                        f"scale 未按 {SCALE_ALIGNMENT} 字节对齐：{name} offset={scale_offset}"
                    )
                if scale_nbytes_entry != expected_scale_nbytes:
                    raise P50FormatError(
                        f"scale 长度不匹配：{name} {scale_nbytes_entry} != {expected_scale_nbytes}"
                    )
                self._append_checked_range(
                    ranges,
                    scale_offset,
                    scale_nbytes_entry,
                    f"{name}:scale",
                    header.data_offset,
                )
                scale_bytes += scale_nbytes_entry
            elif storage == "float16":
                float16_count += 1
                expected_data_nbytes = _product(shape) * 2
                if data_nbytes != expected_data_nbytes:
                    raise P50FormatError(
                        f"FP16 数据长度不匹配：{name} {data_nbytes} != {expected_data_nbytes}"
                    )
                for forbidden in (
                    "scale_offset",
                    "scale_nbytes",
                    "padded_columns",
                    "groups_per_row",
                ):
                    if forbidden in entry:
                        raise P50FormatError(f"FP16 张量不应包含 {forbidden}：{name}")
            else:
                raise P50FormatError(f"未知存储类型：{name} storage={storage!r}")

        ranges.sort(key=lambda item: (item[0], item[1]))
        for previous, current in zip(ranges, ranges[1:]):
            if current[0] < previous[1]:
                raise P50FormatError(
                    f"数据区重叠：{previous[2]} [{previous[0]}, {previous[1]}) 与 "
                    f"{current[2]} [{current[0]}, {current[1]})"
                )

        external_checked = False
        if external_metadata_path is not None:
            external_path = Path(external_metadata_path)
            if not external_path.is_file():
                raise FileNotFoundError(f"找不到外部 JSON 元数据：{external_path}")
            with external_path.open("r", encoding="utf-8") as handle:
                external_metadata = json.load(handle)
            difference = _first_difference(self.metadata, external_metadata)
            if difference is not None:
                raise P50FormatError(f"外部 JSON 与镜像内嵌目录不一致：{difference}")
            external_checked = True

        return ValidationReport(
            tensor_count=len(self.tensors),
            int4_tensor_count=int4_count,
            float16_tensor_count=float16_count,
            data_bytes=data_bytes,
            scale_bytes=scale_bytes,
            image_size=self.file_size,
            external_metadata_checked=external_checked,
        )

    @staticmethod
    def _required_nonnegative_int(entry: dict[str, Any], key: str, name: str) -> int:
        value = entry.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise P50FormatError(f"{key} 必须是非负整数：{name} value={value!r}")
        return value

    def _append_checked_range(
        self,
        ranges: list[tuple[int, int, str]],
        offset: int,
        nbytes: int,
        label: str,
        minimum_offset: int,
    ) -> None:
        end = offset + nbytes
        if offset < minimum_offset or end > self.file_size:
            raise P50FormatError(
                f"数据范围越界：{label} [{offset}, {end})，文件大小={self.file_size}"
            )
        ranges.append((offset, end, label))

    @staticmethod
    def _decode_packed_int4(packed: np.ndarray) -> np.ndarray:
        low = packed & 0x0F
        high = (packed >> 4) & 0x0F
        values = np.empty(packed.size * 2, dtype=np.int8)
        low_signed = np.where(low >= 8, low.astype(np.int16) - 16, low)
        high_signed = np.where(high >= 8, high.astype(np.int16) - 16, high)
        values[0::2] = low_signed.astype(np.int8)
        values[1::2] = high_signed.astype(np.int8)
        return values

    def read_int4_group(
        self, name: str, row: int, group: int
    ) -> tuple[np.ndarray, np.float16]:
        entry = self.tensor(name)
        if entry["storage"] != "int4_groupwise_symmetric":
            raise P50FormatError(f"张量不是 INT4：{name}")
        rows, _ = (int(item) for item in entry["shape"])
        groups_per_row = int(entry["groups_per_row"])
        if not 0 <= row < rows:
            raise IndexError(f"row 越界：{row}，有效范围 0..{rows - 1}")
        if not 0 <= group < groups_per_row:
            raise IndexError(
                f"group 越界：{group}，有效范围 0..{groups_per_row - 1}"
            )

        group_size = self.header.group_size
        row_bytes = int(entry["padded_columns"]) // 2
        packed_offset = (
            int(entry["data_offset"])
            + row * row_bytes
            + group * (group_size // 2)
        )
        scale_offset = int(entry["scale_offset"]) + (
            row * groups_per_row + group
        ) * 2
        with self.path.open("rb") as handle:
            handle.seek(packed_offset)
            packed_raw = handle.read(group_size // 2)
            handle.seek(scale_offset)
            scale_raw = handle.read(2)
        if len(packed_raw) != group_size // 2 or len(scale_raw) != 2:
            raise P50FormatError(f"读取 INT4 group 不完整：{name}")
        packed = np.frombuffer(packed_raw, dtype=np.uint8)
        quantized = self._decode_packed_int4(packed)
        scale = np.frombuffer(scale_raw, dtype="<f2")[0]
        return quantized, scale

    def read_int4_row(
        self, name: str, row: int, include_padding: bool = False
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """返回 ``(quantized, scales, dequantized)``。"""
        entry = self.tensor(name)
        if entry["storage"] != "int4_groupwise_symmetric":
            raise P50FormatError(f"张量不是 INT4：{name}")
        rows, columns = (int(item) for item in entry["shape"])
        if not 0 <= row < rows:
            raise IndexError(f"row 越界：{row}，有效范围 0..{rows - 1}")

        padded_columns = int(entry["padded_columns"])
        groups_per_row = int(entry["groups_per_row"])
        row_bytes = padded_columns // 2
        packed_offset = int(entry["data_offset"]) + row * row_bytes
        scale_offset = int(entry["scale_offset"]) + row * groups_per_row * 2
        with self.path.open("rb") as handle:
            handle.seek(packed_offset)
            packed_raw = handle.read(row_bytes)
            handle.seek(scale_offset)
            scale_raw = handle.read(groups_per_row * 2)
        if len(packed_raw) != row_bytes or len(scale_raw) != groups_per_row * 2:
            raise P50FormatError(f"读取 INT4 行不完整：{name} row={row}")

        quantized = self._decode_packed_int4(
            np.frombuffer(packed_raw, dtype=np.uint8)
        )
        scales = np.frombuffer(scale_raw, dtype="<f2").copy()
        expanded_scales = np.repeat(scales.astype(np.float32), self.header.group_size)
        dequantized = quantized.astype(np.float32) * expanded_scales
        if include_padding:
            return quantized, scales, dequantized
        return quantized[:columns], scales, dequantized[:columns]

    def read_float16_tensor(self, name: str) -> np.ndarray:
        entry = self.tensor(name)
        if entry["storage"] != "float16":
            raise P50FormatError(f"张量不是 FP16：{name}")
        nbytes = int(entry["data_nbytes"])
        with self.path.open("rb") as handle:
            handle.seek(int(entry["data_offset"]))
            raw = handle.read(nbytes)
        if len(raw) != nbytes:
            raise P50FormatError(f"读取 FP16 张量不完整：{name}")
        shape = tuple(int(item) for item in entry["shape"])
        return np.frombuffer(raw, dtype="<f2").copy().reshape(shape)

    def extract_row(self, name: str, row: int) -> ExtractedBlock:
        entry = self.tensor(name)
        shape = tuple(int(item) for item in entry["shape"])
        if entry["storage"] == "int4_groupwise_symmetric":
            quantized, scales, dequantized = self.read_int4_row(name, row)
            return ExtractedBlock(
                tensor_name=name,
                storage=entry["storage"],
                row_start=row,
                row_count=1,
                column_start=0,
                column_count=shape[1],
                values=dequantized[np.newaxis, :],
                quantized=quantized[np.newaxis, :],
                scales=scales[np.newaxis, :],
                scale_group_start=0,
            )

        values = self.read_float16_tensor(name)
        if values.ndim == 1:
            if row != 0:
                raise IndexError("一维 FP16 张量只支持 row=0")
            row_values = values[np.newaxis, :]
            column_count = values.shape[0]
        elif values.ndim == 2:
            if not 0 <= row < values.shape[0]:
                raise IndexError(
                    f"row 越界：{row}，有效范围 0..{values.shape[0] - 1}"
                )
            row_values = values[row : row + 1]
            column_count = values.shape[1]
        else:
            raise P50FormatError(f"当前行提取仅支持一维或二维 FP16 张量：{name}")
        return ExtractedBlock(
            tensor_name=name,
            storage=entry["storage"],
            row_start=row,
            row_count=1,
            column_start=0,
            column_count=column_count,
            values=row_values,
        )

    def extract_block(
        self,
        name: str,
        row_start: int,
        row_count: int,
        column_start: int,
        column_count: int,
    ) -> ExtractedBlock:
        entry = self.tensor(name)
        shape = tuple(int(item) for item in entry["shape"])
        if len(shape) != 2:
            raise P50FormatError(f"块提取只支持二维张量：{name} shape={shape}")
        rows, columns = shape
        self._validate_slice("row", row_start, row_count, rows)
        self._validate_slice("column", column_start, column_count, columns)
        row_end = row_start + row_count
        column_end = column_start + column_count

        if entry["storage"] == "int4_groupwise_symmetric":
            quantized_rows: list[np.ndarray] = []
            dequantized_rows: list[np.ndarray] = []
            scale_rows: list[np.ndarray] = []
            first_group = column_start // self.header.group_size
            last_group = (column_end - 1) // self.header.group_size
            for row in range(row_start, row_end):
                quantized, scales, dequantized = self.read_int4_row(name, row)
                quantized_rows.append(quantized[column_start:column_end])
                dequantized_rows.append(dequantized[column_start:column_end])
                scale_rows.append(scales[first_group : last_group + 1])
            return ExtractedBlock(
                tensor_name=name,
                storage=entry["storage"],
                row_start=row_start,
                row_count=row_count,
                column_start=column_start,
                column_count=column_count,
                values=np.stack(dequantized_rows),
                quantized=np.stack(quantized_rows),
                scales=np.stack(scale_rows),
                scale_group_start=first_group,
            )

        values = self.read_float16_tensor(name)
        return ExtractedBlock(
            tensor_name=name,
            storage=entry["storage"],
            row_start=row_start,
            row_count=row_count,
            column_start=column_start,
            column_count=column_count,
            values=values[row_start:row_end, column_start:column_end],
        )

    @staticmethod
    def _validate_slice(label: str, start: int, count: int, limit: int) -> None:
        if start < 0:
            raise IndexError(f"{label}_start 不能为负数：{start}")
        if count <= 0:
            raise IndexError(f"{label}_count 必须大于 0：{count}")
        if start + count > limit:
            raise IndexError(
                f"{label} 范围越界：start={start}, count={count}, limit={limit}"
            )
