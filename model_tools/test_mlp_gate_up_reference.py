#!/usr/bin/env python3
"""G1 layer0 MLP gate_proj/up_proj 双投影软件参考测试。"""

from __future__ import annotations

import unittest

import numpy as np

try:
    from .mlp_gate_up_reference import (
        BIAS_BYTES,
        GROUPS,
        K,
        M,
        RESULT_BYTES,
        SCALE_BYTES,
        UPLOAD_BYTES,
        WEIGHT_BYTES,
        build_fixed_real_cases,
        build_projection_payload,
        compute_q28_reference,
        q10_to_float32,
        verify_projection_payload,
    )
except ImportError:
    from mlp_gate_up_reference import (
        BIAS_BYTES,
        GROUPS,
        K,
        M,
        RESULT_BYTES,
        SCALE_BYTES,
        UPLOAD_BYTES,
        WEIGHT_BYTES,
        build_fixed_real_cases,
        build_projection_payload,
        compute_q28_reference,
        q10_to_float32,
        verify_projection_payload,
    )


class MLPGateUpReferenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cases = build_fixed_real_cases()

    def test_real_shapes_and_shared_activation(self) -> None:
        self.assertEqual(len(self.cases), 4)
        for case in self.cases:
            self.assertEqual(case.source_post_attention_q10.shape, (K,))
            self.assertEqual(case.gate.weights.shape, (M, K))
            self.assertEqual(case.up.weights.shape, (M, K))
            self.assertEqual(case.gate.scales_q28.shape, (M, GROUPS))
            self.assertEqual(case.up.scales_q28.shape, (M, GROUPS))
            self.assertEqual(case.gate.expected_q28.shape, (M,))
            self.assertEqual(case.up.expected_q28.shape, (M,))
            self.assertTrue(
                np.array_equal(case.gate.activation_int8, case.up.activation_int8)
            )
            self.assertEqual(case.gate.activation_scale, case.up.activation_scale)

    def test_q10_conversion_is_exact(self) -> None:
        source = self.cases[0].source_post_attention_q10
        restored = np.rint(q10_to_float32(source) * (1 << 10)).astype(np.int16)
        self.assertTrue(np.array_equal(restored, source))

    def test_bias_absent_and_zero(self) -> None:
        for case in self.cases:
            self.assertFalse(np.any(case.gate.bias_q28))
            self.assertFalse(np.any(case.up.bias_q28))

    def test_independent_q28_recompute(self) -> None:
        for case in self.cases[:1]:
            gate = compute_q28_reference(
                case.gate.activation_int8,
                case.gate.weights,
                case.gate.scales_q28,
            )
            up = compute_q28_reference(
                case.up.activation_int8,
                case.up.weights,
                case.up.scales_q28,
            )
            self.assertTrue(np.array_equal(gate, case.gate.expected_q28))
            self.assertTrue(np.array_equal(up, case.up.expected_q28))

    def test_payload_layout_and_roundtrip(self) -> None:
        self.assertEqual(UPLOAD_BYTES, 2_646_912)
        self.assertEqual(RESULT_BYTES, 38_912)
        self.assertEqual(WEIGHT_BYTES, 2_179_072)
        self.assertEqual(SCALE_BYTES, 311_296)
        self.assertEqual(BIAS_BYTES, 155_648)
        for projection in (self.cases[0].gate, self.cases[0].up):
            payload = build_projection_payload(projection)
            self.assertEqual(len(payload), UPLOAD_BYTES)
            self.assertEqual(len(verify_projection_payload(projection)), 64)

    def test_zero_input_zero_output(self) -> None:
        case = self.cases[0]
        zero_activation = np.zeros(K, dtype=np.int8)
        gate = compute_q28_reference(
            zero_activation, case.gate.weights, case.gate.scales_q28
        )
        up = compute_q28_reference(
            zero_activation, case.up.weights, case.up.scales_q28
        )
        self.assertFalse(np.any(gate))
        self.assertFalse(np.any(up))


if __name__ == "__main__":
    unittest.main()
