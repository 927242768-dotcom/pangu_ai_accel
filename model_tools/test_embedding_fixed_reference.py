#!/usr/bin/env python3
"""``embedding_fixed_reference`` 的地址、载荷、RNE 与真实 P50 回归。"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

try:
    from .embedding_fixed_reference import (
        DEFAULT_FIXED_TOKEN_IDS,
        DEFAULT_IMAGE,
        DEFAULT_RANDOM_SEED,
        DEFAULT_TENSOR,
        EMBEDDING_DIM,
        GROUPS_PER_ROW,
        GROUP_SIZE,
        PACKED_ROW_BYTES,
        Q10_FACTOR,
        Q10_MAX,
        Q10_MIN,
        RESULT_CTRL_ADDR,
        ROW_SLOT_BEATS,
        ROW_SLOT_BYTES,
        ROW_SLOT_CTRL_STRIDE,
        SCALE_Q28_BYTES,
        VOCAB_SIZE,
        EmbeddingReferenceError,
        _round_shift_signed_array,
        build_manifest,
        compute_embedding_reference,
        embedding_slot_byte_offset,
        embedding_slot_ctrl_addr,
        load_embedding_reference,
        make_random_token_ids,
        pack_embedding_payload,
        unpack_embedding_payload,
    )
    from .p50_format import P50Image
except ImportError:
    from embedding_fixed_reference import (
        DEFAULT_FIXED_TOKEN_IDS,
        DEFAULT_IMAGE,
        DEFAULT_RANDOM_SEED,
        DEFAULT_TENSOR,
        EMBEDDING_DIM,
        GROUPS_PER_ROW,
        GROUP_SIZE,
        PACKED_ROW_BYTES,
        Q10_FACTOR,
        Q10_MAX,
        Q10_MIN,
        RESULT_CTRL_ADDR,
        ROW_SLOT_BEATS,
        ROW_SLOT_BYTES,
        ROW_SLOT_CTRL_STRIDE,
        SCALE_Q28_BYTES,
        VOCAB_SIZE,
        EmbeddingReferenceError,
        _round_shift_signed_array,
        build_manifest,
        compute_embedding_reference,
        embedding_slot_byte_offset,
        embedding_slot_ctrl_addr,
        load_embedding_reference,
        make_random_token_ids,
        pack_embedding_payload,
        unpack_embedding_payload,
    )
    from p50_format import P50Image


class EmbeddingFixedReferenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.image_path = Path(DEFAULT_IMAGE)
        cls.has_real_image = cls.image_path.is_file()

    def test_constants_and_slot_layout(self) -> None:
        self.assertEqual(EMBEDDING_DIM, 896)
        self.assertEqual(GROUP_SIZE, 64)
        self.assertEqual(GROUPS_PER_ROW, 14)
        self.assertEqual(PACKED_ROW_BYTES, 448)
        self.assertEqual(SCALE_Q28_BYTES, 56)
        self.assertEqual(ROW_SLOT_BYTES, 512)
        self.assertEqual(ROW_SLOT_BEATS, 16)
        self.assertEqual(ROW_SLOT_CTRL_STRIDE, 128)
        self.assertEqual(RESULT_CTRL_ADDR, 0x02000000)

    def test_slot_address_boundaries(self) -> None:
        self.assertEqual(embedding_slot_ctrl_addr(0), 0)
        self.assertEqual(embedding_slot_ctrl_addr(1), 128)
        self.assertEqual(embedding_slot_byte_offset(1), 512)
        self.assertEqual(
            embedding_slot_ctrl_addr(VOCAB_SIZE - 1),
            (VOCAB_SIZE - 1) * ROW_SLOT_CTRL_STRIDE,
        )
        self.assertEqual(
            embedding_slot_byte_offset(VOCAB_SIZE - 1),
            (VOCAB_SIZE - 1) * ROW_SLOT_BYTES,
        )
        self.assertLess(embedding_slot_ctrl_addr(VOCAB_SIZE - 1), RESULT_CTRL_ADDR)
        for invalid in (-1, VOCAB_SIZE, VOCAB_SIZE + 1):
            with self.assertRaises(EmbeddingReferenceError):
                embedding_slot_ctrl_addr(invalid)

    def test_signed_rne_shift_ties_and_sign(self) -> None:
        values = np.asarray([1, 3, 5, 7, -1, -3, -5, -7], dtype=np.int64)
        output = _round_shift_signed_array(values, 1)
        np.testing.assert_array_equal(
            output, np.asarray([0, 2, 2, 4, 0, -2, -2, -4], dtype=np.int64)
        )

    def test_synthetic_q10_ties_match_direct_quantization(self) -> None:
        quantized = np.zeros(EMBEDDING_DIM, dtype=np.int8)
        quantized[:6] = np.asarray([1, 3, 5, -1, -3, -5], dtype=np.int8)
        scales = np.full(GROUPS_PER_ROW, np.float16(2.0**-11), dtype=np.float16)
        result = compute_embedding_reference(
            token_id=0,
            quantized_int4=quantized,
            scales_fp16=scales,
        )
        np.testing.assert_array_equal(
            result.fixed_q10[:6],
            np.asarray([0, 2, 2, 0, -2, -2], dtype=np.int16),
        )
        np.testing.assert_array_equal(result.fixed_q10, result.direct_q10)
        self.assertEqual(result.saturated_count, 0)

    def test_synthetic_positive_and_negative_saturation(self) -> None:
        quantized = np.zeros(EMBEDDING_DIM, dtype=np.int8)
        quantized[0] = 7
        quantized[1] = -7
        scales = np.full(GROUPS_PER_ROW, np.float16(15.0), dtype=np.float16)
        result = compute_embedding_reference(
            token_id=1,
            quantized_int4=quantized,
            scales_fp16=scales,
        )
        self.assertEqual(int(result.fixed_q10[0]), Q10_MAX)
        self.assertEqual(int(result.fixed_q10[1]), Q10_MIN)
        self.assertEqual(result.saturated_count, 2)

    def test_payload_roundtrip_and_padding(self) -> None:
        quantized = ((np.arange(EMBEDDING_DIM, dtype=np.int16) % 15) - 7).astype(np.int8)
        scales = np.asarray(
            [np.float16((index + 1) / 1024.0) for index in range(GROUPS_PER_ROW)],
            dtype=np.float16,
        )
        result = compute_embedding_reference(
            token_id=2026,
            quantized_int4=quantized,
            scales_fp16=scales,
        )
        payload = pack_embedding_payload(result)
        self.assertEqual(len(payload), ROW_SLOT_BYTES)
        unpacked_q, unpacked_scales, padding = unpack_embedding_payload(payload)
        np.testing.assert_array_equal(unpacked_q, result.quantized_int4)
        np.testing.assert_array_equal(unpacked_scales, result.scales_q28)
        self.assertEqual(padding, b"\x00" * 8)

    def test_random_token_ids_are_reproducible_and_cover_boundaries(self) -> None:
        left = make_random_token_ids(1000, DEFAULT_RANDOM_SEED)
        right = make_random_token_ids(1000, DEFAULT_RANDOM_SEED)
        np.testing.assert_array_equal(left, right)
        np.testing.assert_array_equal(
            left[:4], np.asarray([0, 1, VOCAB_SIZE - 2, VOCAB_SIZE - 1], dtype=np.uint32)
        )
        self.assertTrue(np.all(left < VOCAB_SIZE))

    def test_real_manifest_matches_committed_json(self) -> None:
        if not self.has_real_image:
            self.skipTest(f"缺少真实 P50 镜像：{self.image_path}")
        generated = build_manifest(self.image_path, DEFAULT_FIXED_TOKEN_IDS)
        committed = json.loads(
            Path(__file__).with_name("embedding_k896_reference.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(generated, committed)

    def test_all_real_embedding_scales_are_exact_uq4_28(self) -> None:
        if not self.has_real_image:
            self.skipTest(f"缺少真实 P50 镜像：{self.image_path}")
        image = P50Image(self.image_path)
        entry = image.tensor(DEFAULT_TENSOR)
        scales = np.memmap(
            self.image_path,
            dtype="<f2",
            mode="r",
            offset=int(entry["scale_offset"]),
            shape=(int(entry["scale_nbytes"]) // 2,),
        ).astype(np.float64)
        q28 = np.rint(scales * (1 << 28))
        self.assertTrue(np.all(np.isfinite(scales)))
        self.assertTrue(np.all(scales > 0.0))
        self.assertTrue(np.all(q28 >= 0.0))
        self.assertTrue(np.all(q28 <= np.iinfo(np.uint32).max))
        np.testing.assert_array_equal(q28 / (1 << 28), scales)

    def test_real_fixed_tokens_are_bit_exact_and_within_half_lsb(self) -> None:
        if not self.has_real_image:
            self.skipTest(f"缺少真实 P50 镜像：{self.image_path}")
        for token_id in DEFAULT_FIXED_TOKEN_IDS:
            result = load_embedding_reference(self.image_path, token_id)
            np.testing.assert_array_equal(result.fixed_q10, result.direct_q10)
            self.assertEqual(result.saturated_count, 0)
            self.assertLessEqual(
                float(np.max(np.abs(result.q10_quantization_error))),
                0.5 / Q10_FACTOR,
            )

    def test_real_random_1000_token_rows(self) -> None:
        if not self.has_real_image:
            self.skipTest(f"缺少真实 P50 镜像：{self.image_path}")
        token_ids = make_random_token_ids(1000, DEFAULT_RANDOM_SEED)
        for token_id in token_ids.tolist():
            result = load_embedding_reference(self.image_path, int(token_id))
            np.testing.assert_array_equal(result.fixed_q10, result.direct_q10)
            self.assertEqual(result.saturated_count, 0)
            self.assertLessEqual(
                float(np.max(np.abs(result.q10_quantization_error))),
                0.5 / Q10_FACTOR,
            )
            payload = pack_embedding_payload(result)
            unpacked_q, unpacked_scales, padding = unpack_embedding_payload(payload)
            np.testing.assert_array_equal(unpacked_q, result.quantized_int4)
            np.testing.assert_array_equal(unpacked_scales, result.scales_q28)
            self.assertEqual(padding, b"\x00" * 8)


if __name__ == "__main__":
    unittest.main()
