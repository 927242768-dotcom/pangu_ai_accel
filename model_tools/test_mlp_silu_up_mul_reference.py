#!/usr/bin/env python3
"""G1 layer0 MLP ``SiLU(gate) × up`` 软件参考测试。"""

from __future__ import annotations

import unittest

import numpy as np

try:
    from .mlp_gate_up_reference import build_fixed_real_cases as build_gate_up_fixed_cases
    from .mlp_silu_reference import case_from_gate_q28
    from .mlp_silu_up_mul_reference import (
        FULL_PRODUCT_BITS,
        M,
        RESULT_BYTES,
        SILU_BYTES,
        UPLOAD_BYTES,
        UP_BYTES,
        build_fixed_real_cases,
        build_upload_payload,
        case_from_inputs,
        fixed_manifest,
        multiply_q10_q28_to_q28,
        multiply_scalar_q10_q28_to_q28,
        verify_upload_payload,
    )
except ImportError:
    from mlp_gate_up_reference import build_fixed_real_cases as build_gate_up_fixed_cases
    from mlp_silu_reference import case_from_gate_q28
    from mlp_silu_up_mul_reference import (
        FULL_PRODUCT_BITS,
        M,
        RESULT_BYTES,
        SILU_BYTES,
        UPLOAD_BYTES,
        UP_BYTES,
        build_fixed_real_cases,
        build_upload_payload,
        case_from_inputs,
        fixed_manifest,
        multiply_q10_q28_to_q28,
        multiply_scalar_q10_q28_to_q28,
        verify_upload_payload,
    )


class MLPSiLUUpMulReferenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.gate_up_cases = build_gate_up_fixed_cases()
        cls.cases = build_fixed_real_cases()

    def test_real_shapes_sources_and_ranges(self) -> None:
        self.assertEqual(len(self.cases), 4)
        self.assertEqual(len(self.gate_up_cases), 4)
        for gate_up, case in zip(self.gate_up_cases, self.cases, strict=True):
            silu = case_from_gate_q28(gate_up.gate.expected_q28, label="source")
            self.assertEqual(case.silu_q10.shape, (M,))
            self.assertEqual(case.up_q28.shape, (M,))
            self.assertEqual(case.output_q28.shape, (M,))
            self.assertTrue(np.array_equal(case.silu_q10, silu.output_pwl_q10))
            self.assertTrue(np.array_equal(case.up_q28, gate_up.up.expected_q28))
            self.assertEqual(case.saturated_count, 0)

    def test_positive_and_negative_rne_ties(self) -> None:
        # product/2^10 分别为 0.5、1.5、2.5、-0.5、-1.5、-2.5。
        vectors = [
            (1, 512, 0),
            (1, 1536, 2),
            (1, 2560, 2),
            (-1, 512, 0),
            (-1, 1536, -2),
            (-1, 2560, -2),
            (1, -1536, -2),
            (-1, -1536, 2),
        ]
        for silu, up, expected in vectors:
            actual, saturated = multiply_scalar_q10_q28_to_q28(silu, up)
            self.assertEqual(actual, expected)
            self.assertFalse(saturated)

    def test_full_width_extremes_and_saturation(self) -> None:
        self.assertEqual(FULL_PRODUCT_BITS, 80)
        positive, pos_sat = multiply_scalar_q10_q28_to_q28(
            32767, np.iinfo(np.int64).max
        )
        negative, neg_sat = multiply_scalar_q10_q28_to_q28(
            -32768, np.iinfo(np.int64).max
        )
        min_times_min, min_sat = multiply_scalar_q10_q28_to_q28(
            -32768, np.iinfo(np.int64).min
        )
        self.assertEqual(positive, np.iinfo(np.int64).max)
        self.assertEqual(negative, np.iinfo(np.int64).min)
        self.assertEqual(min_times_min, np.iinfo(np.int64).max)
        self.assertTrue(pos_sat)
        self.assertTrue(neg_sat)
        self.assertTrue(min_sat)

    def test_exact_simple_values(self) -> None:
        silu = np.zeros(M, dtype=np.int16)
        up = np.zeros(M, dtype=np.int64)
        silu[:6] = [1024, -1024, 512, -512, 1, -1]
        up[:6] = [1 << 28, 1 << 28, 2 << 28, 2 << 28, 1024, 1024]
        output, saturated = multiply_q10_q28_to_q28(silu, up)
        self.assertEqual(saturated, 0)
        self.assertEqual(
            output[:6].tolist(),
            [1 << 28, -(1 << 28), 1 << 28, -(1 << 28), 1, -1],
        )

    def test_payload_layout_and_roundtrip(self) -> None:
        self.assertEqual(SILU_BYTES, 9_728)
        self.assertEqual(UP_BYTES, 38_912)
        self.assertEqual(UPLOAD_BYTES, 48_640)
        self.assertEqual(RESULT_BYTES, 38_912)
        payload = build_upload_payload(self.cases[0])
        self.assertEqual(len(payload), UPLOAD_BYTES)
        self.assertEqual(len(verify_upload_payload(self.cases[0])), 64)

    def test_zero_multipliers(self) -> None:
        silu = np.zeros(M, dtype=np.int16)
        up = np.full(M, np.iinfo(np.int64).max, dtype=np.int64)
        case = case_from_inputs(silu, up, label="zero silu")
        self.assertFalse(np.any(case.output_q28))
        silu.fill(1)
        up.fill(0)
        case = case_from_inputs(silu, up, label="zero up")
        self.assertFalse(np.any(case.output_q28))

    def test_manifest_contains_four_real_cases(self) -> None:
        manifest = fixed_manifest(self.cases)
        self.assertEqual(manifest["operator"], "qwen2_layer0_mlp_silu_gate_times_up")
        self.assertEqual(len(manifest["cases"]), 4)
        self.assertEqual(manifest["definition"]["upload_bytes"], UPLOAD_BYTES)
        self.assertEqual(manifest["definition"]["result_bytes"], RESULT_BYTES)
        self.assertEqual(manifest["definition"]["full_product"], "signed 80 bit Q38")


if __name__ == "__main__":
    unittest.main()
