#!/usr/bin/env python3
"""``p50_format`` 的独立小镜像单元测试。"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    from .p50_format import (
        DATA_ALIGNMENT,
        FORMAT_VERSION,
        HEADER_SIZE,
        HEADER_STRUCT,
        MAGIC,
        P50FormatError,
        P50Image,
    )
except ImportError:
    from p50_format import (
        DATA_ALIGNMENT,
        FORMAT_VERSION,
        HEADER_SIZE,
        HEADER_STRUCT,
        MAGIC,
        P50FormatError,
        P50Image,
    )


def pack_int4(values: list[int]) -> bytes:
    if len(values) % 2:
        raise ValueError("INT4 数量必须为偶数")
    result = bytearray()
    for low, high in zip(values[0::2], values[1::2]):
        result.append((low & 0x0F) | ((high & 0x0F) << 4))
    return bytes(result)


def build_test_image(directory: Path) -> tuple[Path, Path]:
    image_path = directory / "sample.p50"
    metadata_path = directory / "sample.json"
    group_size = 4
    data_offset = DATA_ALIGNMENT * 2
    weight_data_offset = data_offset
    weight_scale_offset = weight_data_offset + 64
    bias_data_offset = DATA_ALIGNMENT * 3
    image_size = bias_data_offset + 4

    metadata = {
        "format": "pangu50k-qwen-int4",
        "format_version": FORMAT_VERSION,
        "quantization": {
            "weight_bits": 4,
            "scheme": "symmetric_per_row_group",
            "group_size": group_size,
            "range": [-7, 7],
            "packed_order": "low_nibble_first",
            "scale_dtype": "float16",
        },
        "tensor_order": "execution_order",
        "tensors": [
            {
                "name": "linear.weight",
                "shape": [2, 3],
                "source_dtype": "bfloat16",
                "storage": "int4_groupwise_symmetric",
                "data_offset": weight_data_offset,
                "data_nbytes": 4,
                "scale_offset": weight_scale_offset,
                "scale_nbytes": 4,
                "padded_columns": 4,
                "groups_per_row": 1,
            },
            {
                "name": "norm.weight",
                "shape": [2],
                "source_dtype": "bfloat16",
                "storage": "float16",
                "data_offset": bias_data_offset,
                "data_nbytes": 4,
            },
        ],
        "image_size": image_size,
        "data_offset": data_offset,
    }
    metadata_raw = json.dumps(
        metadata, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    if HEADER_SIZE + len(metadata_raw) > data_offset:
        raise AssertionError("测试 JSON 超出预留区")

    header = HEADER_STRUCT.pack(
        MAGIC,
        FORMAT_VERSION,
        HEADER_SIZE,
        len(metadata_raw),
        data_offset,
        2,
        group_size,
        1,
        0,
    )
    image = bytearray(image_size)
    image[: len(header)] = header
    image[HEADER_SIZE : HEADER_SIZE + len(metadata_raw)] = metadata_raw
    image[weight_data_offset : weight_data_offset + 4] = (
        pack_int4([-7, -1, 0, 5]) + pack_int4([7, 2, -3, 0])
    )
    image[weight_scale_offset : weight_scale_offset + 4] = np.asarray(
        [0.5, 1.0], dtype="<f2"
    ).tobytes()
    image[bias_data_offset : bias_data_offset + 4] = np.asarray(
        [1.5, -2.0], dtype="<f2"
    ).tobytes()
    image_path.write_bytes(image)
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return image_path, metadata_path


class P50FormatTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        self.image_path, self.metadata_path = build_test_image(self.directory)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_validate_and_external_metadata(self) -> None:
        image = P50Image(self.image_path)
        report = image.validate(self.metadata_path)
        self.assertEqual(report.tensor_count, 2)
        self.assertEqual(report.int4_tensor_count, 1)
        self.assertEqual(report.float16_tensor_count, 1)
        self.assertTrue(report.external_metadata_checked)

    def test_decode_int4_row(self) -> None:
        image = P50Image(self.image_path)
        quantized, scales, values = image.read_int4_row("linear.weight", 0)
        np.testing.assert_array_equal(quantized, np.asarray([-7, -1, 0], dtype=np.int8))
        np.testing.assert_array_equal(scales, np.asarray([0.5], dtype=np.float16))
        np.testing.assert_allclose(values, np.asarray([-3.5, -0.5, 0.0], dtype=np.float32))

    def test_extract_cross_row_block(self) -> None:
        image = P50Image(self.image_path)
        block = image.extract_block("linear.weight", 0, 2, 1, 2)
        np.testing.assert_array_equal(
            block.quantized,
            np.asarray([[-1, 0], [2, -3]], dtype=np.int8),
        )
        np.testing.assert_allclose(
            block.values,
            np.asarray([[-0.5, 0.0], [2.0, -3.0]], dtype=np.float32),
        )

    def test_read_float16_tensor(self) -> None:
        image = P50Image(self.image_path)
        values = image.read_float16_tensor("norm.weight")
        np.testing.assert_array_equal(values, np.asarray([1.5, -2.0], dtype=np.float16))

    def test_external_metadata_difference_is_reported(self) -> None:
        metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        metadata["tensors"][0]["shape"] = [2, 4]
        bad_path = self.directory / "bad.json"
        bad_path.write_text(json.dumps(metadata), encoding="utf-8")
        with self.assertRaisesRegex(P50FormatError, r"tensors\[0\]\.shape\[1\]"):
            P50Image(self.image_path).validate(bad_path)


if __name__ == "__main__":
    unittest.main()
