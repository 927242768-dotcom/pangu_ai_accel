#!/usr/bin/env python3
"""G1 layer0 MLP ``down_proj`` 软件参考测试。"""

from __future__ import annotations

import unittest

import numpy as np

try:
    from .mlp_down_proj_reference import (
        ACTIVATION_BYTES,
        BIAS_BYTES,
        DEFAULT_IMAGE,
        GROUPS,
        K,
        M,
        MAX_OUTPUT_MAGNITUDE,
        RESULT_BYTES,
        SCALE_BYTES,
        SCALE_ROW_BYTES,
        UPLOAD_BYTES,
        WEIGHT_BYTES,
        build_fixed_real_cases,
        build_upload_payload,
        case_from_source_q28,
        fixed_manifest,
        load_down_projection_model,
        q28_to_float32,
        verify_upload_payload,
    )
    from .mlp_silu_up_mul_reference import build_fixed_real_cases as build_mul_fixed_cases
    from .p50_format import P50Image
except ImportError:
    from mlp_down_proj_reference import (
        ACTIVATION_BYTES,
        BIAS_BYTES,
        DEFAULT_IMAGE,
        GROUPS,
        K,
        M,
        MAX_OUTPUT_MAGNITUDE,
        RESULT_BYTES,
        SCALE_BYTES,
        SCALE_ROW_BYTES,
        UPLOAD_BYTES,
        WEIGHT_BYTES,
        build_fixed_real_cases,
        build_upload_payload,
        case_from_source_q28,
        fixed_manifest,
        load_down_projection_model,
        q28_to_float32,
        verify_upload_payload,
    )
    from mlp_silu_up_mul_reference import build_fixed_real_cases as build_mul_fixed_cases
    from p50_format import P50Image


class MLPDownProjReferenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.image = P50Image(DEFAULT_IMAGE)
        cls.image.validate()
        cls.model = load_down_projection_model(cls.image)
        cls.source_cases = build_mul_fixed_cases()
        cls.cases = build_fixed_real_cases()

    def test_real_tensor_and_sources(self) -> None:
        self.assertEqual(self.model.weights.shape, (M, K))
        self.assertEqual(self.model.weight_scales.shape, (M, GROUPS))
        self.assertEqual(len(self.cases), 4)
        for source, case in zip(self.source_cases, self.cases, strict=True):
            self.assertTrue(np.array_equal(case.source_q28, source.output_q28))
            self.assertEqual(case.activation_int8.shape, (K,))
            self.assertEqual(case.scales_q28.shape, (M, GROUPS))
            self.assertEqual(case.expected_q28.shape, (M,))
            self.assertFalse(np.any(case.bias_q28))

    def test_q28_to_float32_definition(self) -> None:
        source = np.zeros(K, dtype=np.int64)
        source[:8] = [0, 1 << 28, -(1 << 28), 1 << 27, -(1 << 27), 1, -1, 3 << 28]
        actual = q28_to_float32(source)
        expected = np.asarray([0.0, 1.0, -1.0, 0.5, -0.5, 2.0**-28, -(2.0**-28), 3.0], dtype=np.float32)
        self.assertTrue(np.array_equal(actual[:8], expected))

    def test_payload_layout_and_roundtrip(self) -> None:
        self.assertEqual(ACTIVATION_BYTES, 4_864)
        self.assertEqual(WEIGHT_BYTES, 2_179_072)
        self.assertEqual(SCALE_ROW_BYTES, 320)
        self.assertEqual(SCALE_BYTES, 286_720)
        self.assertEqual(BIAS_BYTES, 28_672)
        self.assertEqual(UPLOAD_BYTES, 2_499_328)
        self.assertEqual(RESULT_BYTES, 7_168)
        payload = build_upload_payload(self.cases[0])
        self.assertEqual(len(payload), UPLOAD_BYTES)
        self.assertEqual(len(verify_upload_payload(self.cases[0])), 64)

    def test_zero_input_is_exact_zero(self) -> None:
        case = case_from_source_q28(
            self.model,
            np.zeros(K, dtype=np.int64),
            label="zero down_proj input",
        )
        self.assertFalse(np.any(case.activation_int8))
        self.assertFalse(np.any(case.expected_q28))

    def test_int64_bound_is_proven_safe(self) -> None:
        self.assertLessEqual(MAX_OUTPUT_MAGNITUDE, np.iinfo(np.int64).max)

    def test_extreme_input_has_defined_scale_saturation(self) -> None:
        source = np.empty(K, dtype=np.int64)
        source[0::2] = np.iinfo(np.int64).min
        source[1::2] = np.iinfo(np.int64).max
        case = case_from_source_q28(self.model, source, label="extreme Q28")
        self.assertGreater(case.linear_result.combined_scale_saturated_count, 0)
        self.assertTrue(np.all(case.scales_q28 <= np.iinfo(np.uint32).max))
        self.assertEqual(case.expected_q28.dtype, np.int64)

    def test_manifest_contains_four_real_cases(self) -> None:
        manifest = fixed_manifest(self.cases)
        self.assertEqual(manifest["operator"], "qwen2_layer0_mlp_down_projection")
        self.assertEqual(len(manifest["cases"]), 4)
        self.assertTrue(manifest["definition"]["int64_safe"])
        self.assertEqual(manifest["definition"]["groups_per_row"], 76)
        self.assertEqual(manifest["definition"]["upload_bytes"], UPLOAD_BYTES)


if __name__ == "__main__":
    unittest.main()
