#!/usr/bin/env python3
"""G1 layer0 MLP ``SiLU(gate)`` 软件参考测试。"""

from __future__ import annotations

import unittest

import numpy as np

try:
    from .elementwise_fixed_reference import build_silu_pwl_endpoints, silu_pwl_q10
    from .mlp_silu_reference import (
        INPUT_BYTES,
        M,
        PWL_BYTES,
        RESULT_BYTES,
        UPLOAD_BYTES,
        build_fixed_real_cases,
        build_upload_payload,
        case_from_gate_q28,
        fixed_manifest,
        rne_q28_to_q10,
        verify_upload_payload,
    )
except ImportError:
    from elementwise_fixed_reference import build_silu_pwl_endpoints, silu_pwl_q10
    from mlp_silu_reference import (
        INPUT_BYTES,
        M,
        PWL_BYTES,
        RESULT_BYTES,
        UPLOAD_BYTES,
        build_fixed_real_cases,
        build_upload_payload,
        case_from_gate_q28,
        fixed_manifest,
        rne_q28_to_q10,
        verify_upload_payload,
    )


class MLPSiLUReferenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cases = build_fixed_real_cases()

    def test_real_shapes_and_ranges(self) -> None:
        self.assertEqual(len(self.cases), 4)
        for case in self.cases:
            self.assertEqual(case.gate_q28.shape, (M,))
            self.assertEqual(case.gate_q10.shape, (M,))
            self.assertEqual(case.output_pwl_q10.shape, (M,))
            self.assertEqual(case.output_exact_q10.shape, (M,))
            self.assertEqual(case.rescale_saturated_count, 0)
            self.assertLessEqual(case.pwl_max_abs_error_lsb, 4)
            self.assertLess(int(np.max(np.abs(case.gate_q10.astype(np.int32)))), 8192)

    def test_rne_positive_and_negative_ties(self) -> None:
        half = 1 << 17
        step = 1 << 18
        values = np.asarray(
            [
                0,
                half,
                step + half,
                2 * step + half,
                -half,
                -(step + half),
                -(2 * step + half),
            ],
            dtype=np.int64,
        )
        output, saturated = rne_q28_to_q10(values)
        self.assertEqual(saturated, 0)
        self.assertEqual(output.tolist(), [0, 0, 2, 2, 0, -2, -2])

    def test_int64_extremes_saturate(self) -> None:
        values = np.asarray(
            [np.iinfo(np.int64).min, np.iinfo(np.int64).max], dtype=np.int64
        )
        output, saturated = rne_q28_to_q10(values)
        self.assertEqual(output.tolist(), [-32768, 32767])
        self.assertEqual(saturated, 2)

    def test_reuses_verified_e2_pwl(self) -> None:
        for case in self.cases:
            expected = silu_pwl_q10(case.gate_q10)
            self.assertTrue(np.array_equal(expected, case.output_pwl_q10))
        endpoints = build_silu_pwl_endpoints()
        self.assertEqual(endpoints.shape, (65,))

    def test_payload_layout_and_roundtrip(self) -> None:
        self.assertEqual(INPUT_BYTES, 38_912)
        self.assertEqual(PWL_BYTES, 160)
        self.assertEqual(UPLOAD_BYTES, 39_072)
        self.assertEqual(RESULT_BYTES, 9_728)
        payload = build_upload_payload(self.cases[0])
        self.assertEqual(len(payload), UPLOAD_BYTES)
        self.assertEqual(len(verify_upload_payload(self.cases[0])), 64)

    def test_zero_and_tail_rules(self) -> None:
        gate_q10 = np.zeros(M, dtype=np.int64)
        gate_q10[:6] = np.asarray([-9000, -8192, -8191, 8191, 8192, 9000])
        gate_q28 = gate_q10 * np.int64(1 << 18)
        case = case_from_gate_q28(gate_q28, label="tail rules")
        self.assertEqual(case.output_pwl_q10[0], 0)
        self.assertEqual(case.output_pwl_q10[1], -3)
        self.assertEqual(case.output_pwl_q10[4], 8192)
        self.assertEqual(case.output_pwl_q10[5], 9000)

    def test_manifest_contains_four_real_cases(self) -> None:
        manifest = fixed_manifest(self.cases)
        self.assertEqual(manifest["operator"], "qwen2_layer0_mlp_silu_gate")
        self.assertEqual(len(manifest["cases"]), 4)
        self.assertEqual(manifest["definition"]["upload_bytes"], UPLOAD_BYTES)
        self.assertEqual(manifest["definition"]["result_bytes"], RESULT_BYTES)


if __name__ == "__main__":
    unittest.main()
