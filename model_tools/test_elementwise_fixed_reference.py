#!/usr/bin/env python3
"""``elementwise_fixed_reference`` 的定点算术、SiLU 近似和压力测试。"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

try:
    from .elementwise_fixed_reference import (
        DEFAULT_LENGTH,
        DEFAULT_SCALE_Q10,
        DEFAULT_SEED,
        Q_FACTOR,
        Q_MAX,
        Q_MIN,
        SILU_PWL_SEGMENTS,
        _round_shift_signed_scalar,
        build_silu_lut_midpoint,
        build_silu_pwl_endpoints,
        compute_elementwise_reference,
        elementwise_mul_q10,
        fixed_scale_q10,
        make_deterministic_q10_vectors,
        quantize_q6_10,
        residual_add_q10,
        result_manifest,
        silu_exact_q10,
        silu_lut_q10,
        silu_pwl_q10,
        silu_scheme_metrics,
    )
except ImportError:
    from elementwise_fixed_reference import (
        DEFAULT_LENGTH,
        DEFAULT_SCALE_Q10,
        DEFAULT_SEED,
        Q_FACTOR,
        Q_MAX,
        Q_MIN,
        SILU_PWL_SEGMENTS,
        _round_shift_signed_scalar,
        build_silu_lut_midpoint,
        build_silu_pwl_endpoints,
        compute_elementwise_reference,
        elementwise_mul_q10,
        fixed_scale_q10,
        make_deterministic_q10_vectors,
        quantize_q6_10,
        residual_add_q10,
        result_manifest,
        silu_exact_q10,
        silu_lut_q10,
        silu_pwl_q10,
        silu_scheme_metrics,
    )


def scalar_saturate(value: int) -> int:
    return min(max(int(value), Q_MIN), Q_MAX)


class ElementwiseFixedReferenceTest(unittest.TestCase):
    def test_signed_rne_shift_handles_ties_and_sign(self) -> None:
        self.assertEqual(_round_shift_signed_scalar(1, 1), 0)
        self.assertEqual(_round_shift_signed_scalar(3, 1), 2)
        self.assertEqual(_round_shift_signed_scalar(5, 1), 2)
        self.assertEqual(_round_shift_signed_scalar(7, 1), 4)
        self.assertEqual(_round_shift_signed_scalar(-1, 1), 0)
        self.assertEqual(_round_shift_signed_scalar(-3, 1), -2)
        self.assertEqual(_round_shift_signed_scalar(-5, 1), -2)
        self.assertEqual(_round_shift_signed_scalar(-7, 1), -4)

    def test_q6_10_quantization_uses_rne_and_saturation(self) -> None:
        result = quantize_q6_10(
            np.asarray(
                [
                    0.5 / Q_FACTOR,
                    1.5 / Q_FACTOR,
                    2.5 / Q_FACTOR,
                    -0.5 / Q_FACTOR,
                    -1.5 / Q_FACTOR,
                    32.0,
                    -33.0,
                ]
            )
        )
        np.testing.assert_array_equal(
            result.quantized,
            np.asarray([0, 2, 2, 0, -2, 32767, -32768], dtype=np.int16),
        )
        self.assertEqual(result.clipped_count, 2)

    def test_residual_add_saturates_both_directions(self) -> None:
        a = np.asarray([32767, -32768, 1000, -1000], dtype=np.int16)
        b = np.asarray([1, -1, 2000, -2000], dtype=np.int16)
        output, saturated = residual_add_q10(a, b)
        np.testing.assert_array_equal(
            output, np.asarray([32767, -32768, 3000, -3000], dtype=np.int16)
        )
        self.assertEqual(saturated, 2)

    def test_fixed_scale_rne_and_saturation(self) -> None:
        vector = np.asarray([1, 3, 5, -1, -3, -5, 32767, -32768], dtype=np.int16)
        output, saturated = fixed_scale_q10(vector, 512)
        np.testing.assert_array_equal(
            output, np.asarray([0, 2, 2, 0, -2, -2, 16384, -16384], dtype=np.int16)
        )
        self.assertEqual(saturated, 0)

        output2, saturated2 = fixed_scale_q10(
            np.asarray([32767, -32768], dtype=np.int16), 32767
        )
        np.testing.assert_array_equal(
            output2, np.asarray([32767, -32768], dtype=np.int16)
        )
        self.assertEqual(saturated2, 2)

    def test_elementwise_mul_matches_scalar_formula(self) -> None:
        a = np.asarray([1, 3, 5, -1, -3, -5, 32767, -32768], dtype=np.int16)
        b = np.asarray([512, 512, 512, 512, 512, 512, 32767, -32768], dtype=np.int16)
        output, saturated = elementwise_mul_q10(a, b)
        expected = []
        expected_saturated = 0
        for left, right in zip(a.tolist(), b.tolist()):
            rounded = _round_shift_signed_scalar(left * right, 10)
            clipped = scalar_saturate(rounded)
            expected_saturated += int(clipped != rounded)
            expected.append(clipped)
        np.testing.assert_array_equal(output, np.asarray(expected, dtype=np.int16))
        self.assertEqual(saturated, expected_saturated)

    def test_silu_tables_have_expected_shapes(self) -> None:
        lut = build_silu_lut_midpoint()
        endpoints = build_silu_pwl_endpoints()
        self.assertEqual(lut.shape, (2048,))
        self.assertEqual(endpoints.shape, (SILU_PWL_SEGMENTS + 1,))
        self.assertEqual(int(endpoints[0]), -3)
        self.assertEqual(int(endpoints[-1]), 8189)

    def test_silu_tail_rules_and_key_points(self) -> None:
        inputs = np.asarray(
            [-32768, -8193, -8192, -1024, 0, 1024, 8191, 8192, 32767],
            dtype=np.int16,
        )
        exact = silu_exact_q10(inputs)
        lut = silu_lut_q10(inputs)
        pwl = silu_pwl_q10(inputs)
        self.assertEqual(int(lut[0]), 0)
        self.assertEqual(int(pwl[0]), 0)
        self.assertEqual(int(lut[1]), 0)
        self.assertEqual(int(pwl[1]), 0)
        self.assertEqual(int(lut[-2]), 8192)
        self.assertEqual(int(pwl[-2]), 8192)
        self.assertEqual(int(lut[-1]), 32767)
        self.assertEqual(int(pwl[-1]), 32767)
        self.assertLessEqual(int(np.max(np.abs(lut.astype(np.int32) - exact.astype(np.int32)))), 5)
        self.assertLessEqual(int(np.max(np.abs(pwl.astype(np.int32) - exact.astype(np.int32)))), 4)

    def test_fixed_vector_covers_silu_top_segment(self) -> None:
        vector_a, _ = make_deterministic_q10_vectors(DEFAULT_LENGTH, DEFAULT_SEED)
        top_segment = vector_a[
            (vector_a.astype(np.int32) >= 63 * 256 - 8192)
            & (vector_a.astype(np.int32) < 8192)
        ]
        self.assertGreaterEqual(top_segment.size, 3)
        self.assertTrue(np.any(((top_segment.astype(np.int32) + 8192) & 0xFF) != 0))

    def test_silu_full_int16_domain_metrics(self) -> None:
        metrics = {metric.name: metric for metric in silu_scheme_metrics()}
        self.assertEqual(metrics["lut2048_midpoint"].max_abs_error_lsb, 5)
        self.assertEqual(metrics["pwl64_endpoints"].max_abs_error_lsb, 4)
        self.assertEqual(metrics["lut2048_midpoint"].estimated_table_bits, 32768)
        self.assertEqual(metrics["pwl64_endpoints"].estimated_table_bits, 1040)
        self.assertLess(
            metrics["pwl64_endpoints"].mean_abs_error_lsb,
            metrics["lut2048_midpoint"].mean_abs_error_lsb,
        )

    def test_fixed_manifest_matches_committed_json(self) -> None:
        vector_a, vector_b = make_deterministic_q10_vectors(DEFAULT_LENGTH, DEFAULT_SEED)
        result = compute_elementwise_reference(
            vector_a_q10=vector_a,
            vector_b_q10=vector_b,
            scale_q10=DEFAULT_SCALE_Q10,
            seed=DEFAULT_SEED,
        )
        manifest = result_manifest(result)
        committed = json.loads(
            Path(__file__).with_name("elementwise_k896_reference.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(manifest, committed)

    def test_random_software_stress_1000_vectors(self) -> None:
        rng = np.random.default_rng(20260727)
        for _ in range(1000):
            length = int(rng.integers(1, 897))
            a = rng.integers(Q_MIN, Q_MAX + 1, size=length, dtype=np.int32).astype(np.int16)
            b = rng.integers(Q_MIN, Q_MAX + 1, size=length, dtype=np.int32).astype(np.int16)
            scale = int(rng.integers(Q_MIN, Q_MAX + 1))

            residual, residual_saturated = residual_add_q10(a, b)
            scaled, scaled_saturated = fixed_scale_q10(a, scale)
            multiplied, multiplied_saturated = elementwise_mul_q10(a, b)

            expected_residual = []
            expected_scaled = []
            expected_multiplied = []
            expected_residual_saturated = 0
            expected_scaled_saturated = 0
            expected_multiplied_saturated = 0
            for left, right in zip(a.tolist(), b.tolist()):
                raw_residual = left + right
                clipped_residual = scalar_saturate(raw_residual)
                expected_residual_saturated += int(raw_residual != clipped_residual)
                expected_residual.append(clipped_residual)

                raw_scaled = _round_shift_signed_scalar(left * scale, 10)
                clipped_scaled = scalar_saturate(raw_scaled)
                expected_scaled_saturated += int(raw_scaled != clipped_scaled)
                expected_scaled.append(clipped_scaled)

                raw_multiplied = _round_shift_signed_scalar(left * right, 10)
                clipped_multiplied = scalar_saturate(raw_multiplied)
                expected_multiplied_saturated += int(raw_multiplied != clipped_multiplied)
                expected_multiplied.append(clipped_multiplied)

            np.testing.assert_array_equal(
                residual, np.asarray(expected_residual, dtype=np.int16)
            )
            np.testing.assert_array_equal(
                scaled, np.asarray(expected_scaled, dtype=np.int16)
            )
            np.testing.assert_array_equal(
                multiplied, np.asarray(expected_multiplied, dtype=np.int16)
            )
            self.assertEqual(residual_saturated, expected_residual_saturated)
            self.assertEqual(scaled_saturated, expected_scaled_saturated)
            self.assertEqual(multiplied_saturated, expected_multiplied_saturated)

            exact = silu_exact_q10(a).astype(np.int32)
            lut = silu_lut_q10(a).astype(np.int32)
            pwl = silu_pwl_q10(a).astype(np.int32)
            self.assertLessEqual(int(np.max(np.abs(lut - exact))), 5)
            self.assertLessEqual(int(np.max(np.abs(pwl - exact))), 4)


if __name__ == "__main__":
    unittest.main()
