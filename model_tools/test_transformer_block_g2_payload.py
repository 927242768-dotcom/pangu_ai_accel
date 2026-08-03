#!/usr/bin/env python3
"""G2 完整 Block DDR3 参数/动态载荷测试。"""

from __future__ import annotations

import unittest

import numpy as np

try:
    from .runtime_linear_quant_reference import (
        quantize_q10_and_build_scales,
        quantize_q28_and_build_scales,
    )
    from .runtime_quantizer_validation import build_fixed_validation_cases
    from .transformer_block_g2_payload import (
        BEAT_BYTES,
        build_dynamic_uploads,
        build_resident_uploads,
        build_stress_case,
        uploads_manifest,
    )
    from .transformer_block_reference import (
        build_fixed_real_cases,
        kv_slot_byte_addresses,
        load_context,
        sha256_array,
    )
except ImportError:
    from runtime_linear_quant_reference import (
        quantize_q10_and_build_scales,
        quantize_q28_and_build_scales,
    )
    from runtime_quantizer_validation import build_fixed_validation_cases
    from transformer_block_g2_payload import (
        BEAT_BYTES,
        build_dynamic_uploads,
        build_resident_uploads,
        build_stress_case,
        uploads_manifest,
    )
    from transformer_block_reference import (
        build_fixed_real_cases,
        kv_slot_byte_addresses,
        load_context,
        sha256_array,
    )


class TransformerBlockG2PayloadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.context = load_context()
        cls.resident = build_resident_uploads(cls.context)
        cls.cases = build_fixed_real_cases()

    def test_resident_upload_count_size_and_alignment(self) -> None:
        self.assertEqual(len(self.resident), 22)
        self.assertEqual(sum(len(item.payload) for item in self.resident), 7_964_352)
        self.assertTrue(all(item.persistent for item in self.resident))
        self.assertTrue(all(len(item.payload) % BEAT_BYTES == 0 for item in self.resident))
        self.assertTrue(all(item.controller_address % 8 == 0 for item in self.resident))
        self.assertEqual(len({item.name for item in self.resident}), len(self.resident))

    def test_resident_contains_only_raw_parameters_not_combined_scales(self) -> None:
        names = {item.name for item in self.resident}
        required = {
            "input_rms_gamma_q10",
            "post_rms_gamma_q10",
            "rms_lut_uq12_20",
            "softmax_exp_lut_q31",
            "silu_pwl_q10",
            "q_weight_int4",
            "k_weight_int4",
            "v_weight_int4",
            "oproj_weight_int4",
            "gate_weight_int4",
            "up_weight_int4",
            "down_weight_int4",
            "q_weight_scale_fp16",
            "k_weight_scale_fp16",
            "v_weight_scale_fp16",
            "oproj_weight_scale_fp16",
            "gate_weight_scale_fp16",
            "up_weight_scale_fp16",
            "down_weight_scale_fp16",
            "q_bias_q28",
            "k_bias_q28",
            "v_bias_q28",
        }
        self.assertEqual(names, required)
        self.assertFalse(any("scale_uq4_28" in name for name in names))

    def test_bias_uploads_use_one_padded_256_bit_row_per_output(self) -> None:
        by_name = {item.name: item for item in self.resident}
        for key, rows in (("q", 896), ("k", 128), ("v", 128)):
            padded = np.frombuffer(
                by_name[f"{key}_bias_q28"].payload, dtype="<i8"
            ).reshape(rows, 4)
            self.assertEqual(padded.shape, (rows, 4))
            np.testing.assert_array_equal(padded[:, 1:], 0)

    def test_dynamic_uploads_match_each_fixed_case(self) -> None:
        self.assertEqual([case.count for case in self.cases], [1, 2, 6, 16])
        for case in self.cases:
            uploads = build_dynamic_uploads(case)
            self.assertEqual(len(uploads), 2 + 2 * (case.count - 1))
            self.assertTrue(all(not item.persistent for item in uploads))
            self.assertTrue(all(len(item.payload) % BEAT_BYTES == 0 for item in uploads))
            by_name = {item.name: item for item in uploads}
            hidden = np.frombuffer(
                by_name["block_hidden_q10"].payload, dtype="<i2"
            ).copy()
            np.testing.assert_array_equal(hidden, case.block_input_q10)
            for index in range(case.count - 1):
                position = case.window_start + index
                k_addr, v_addr = kv_slot_byte_addresses(0, position)
                k = by_name[f"kv_history_k_position_{position}"]
                v = by_name[f"kv_history_v_position_{position}"]
                self.assertEqual(k.byte_address, k_addr)
                self.assertEqual(v.byte_address, v_addr)
                np.testing.assert_array_equal(
                    np.frombuffer(k.payload, dtype="<i8").reshape(2, 64),
                    case.history_k_q28[index],
                )
                np.testing.assert_array_equal(
                    np.frombuffer(v.payload, dtype="<i8").reshape(2, 64),
                    case.history_v_q28[index],
                )

    def test_fixed_output_hashes_remain_g1_verified_values(self) -> None:
        self.assertEqual(
            [sha256_array(case.output_q10, "<i2") for case in self.cases],
            [
                "630952eaf6fe179639773f2b60d1e9e990f380b0e698fa3d493dd7c279c96104",
                "1cd96d92e43203abda26cace446c2664a0cbaa1fad29a8658a0d94941fbfbea7",
                "b2365afdc2857c543628c9d15fd829005ede98aa99d35a8c4f973f61b35ed9dc",
                "c164aab5251afc3954b8689a826a1241d1c7d6757adef5c5a232e127b59a4032",
            ],
        )

    def test_stress_cases_cover_boundaries_and_are_deterministic(self) -> None:
        cases = [
            build_stress_case(self.context, seed=20260820, index=index)
            for index in range(8)
        ]
        self.assertEqual(
            [(case.query_position, case.window_start, case.count) for case in cases[:4]],
            [(0, 0, 1), (1, 0, 2), (15, 0, 16), (16383, 16368, 16)],
        )
        replay = build_stress_case(self.context, seed=20260820, index=7)
        self.assertEqual(replay.query_position, cases[7].query_position)
        self.assertEqual(replay.window_start, cases[7].window_start)
        np.testing.assert_array_equal(replay.output_q10, cases[7].output_q10)
        for case in cases[4:]:
            self.assertGreaterEqual(case.count, 1)
            self.assertLessEqual(case.count, 16)
            self.assertEqual(case.query_position - case.window_start + 1, case.count)

    def test_q10_shift18_is_bit_exact_through_q28_quantizer(self) -> None:
        fixed = [case for case in build_fixed_validation_cases() if not case.source_q28]
        self.assertEqual([case.name for case in fixed], [
            "q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"
        ])
        for case in fixed:
            scales = case.raw_scale_fp16_bits.view(np.float16).astype(np.float32)
            scales = scales.reshape(case.rows, case.groups)
            q10 = quantize_q10_and_build_scales(
                case.source_values, scales, verify_numpy=False
            )
            q28 = quantize_q28_and_build_scales(
                case.source_values.astype(np.int64) << 18,
                scales,
                verify_numpy=False,
            )
            np.testing.assert_array_equal(q10.activation_int8, q28.activation_int8)
            np.testing.assert_array_equal(
                q10.combined_scale_q28, q28.combined_scale_q28
            )
            self.assertEqual(q10.max_abs_float32_bits, q28.max_abs_float32_bits)

        rng = np.random.default_rng(20260820)
        boundary = np.asarray(
            ([0, 1, -1, 2047, -2047, 4094, -4094, 32767, -32768] * 100)[:896],
            dtype=np.int16,
        )
        vectors = [
            np.zeros(896, dtype=np.int16),
            boundary,
            *[
                rng.integers(-32768, 32768, size=896, dtype=np.int16)
                for _ in range(128)
            ],
        ]
        scales = np.asarray(
            [[2.0**-14, 0.001, 0.125, 1.0, 65504.0]], dtype=np.float16
        ).astype(np.float32)
        for vector in vectors:
            q10 = quantize_q10_and_build_scales(vector, scales, verify_numpy=False)
            q28 = quantize_q28_and_build_scales(
                vector.astype(np.int64) << 18, scales, verify_numpy=False
            )
            np.testing.assert_array_equal(q10.activation_int8, q28.activation_int8)
            np.testing.assert_array_equal(
                q10.combined_scale_q28, q28.combined_scale_q28
            )
            self.assertEqual(q10.max_abs_float32_bits, q28.max_abs_float32_bits)

    def test_manifest_addresses_and_hashes_are_serializable(self) -> None:
        manifest = uploads_manifest(self.resident)
        self.assertEqual(len(manifest), 22)
        self.assertTrue(all(item["controller_address"].startswith("0x") for item in manifest))
        self.assertTrue(all(len(item["sha256"]) == 64 for item in manifest))


if __name__ == "__main__":
    unittest.main()
