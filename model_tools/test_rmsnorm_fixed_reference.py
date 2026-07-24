#!/usr/bin/env python3
"""``rmsnorm_fixed_reference`` 的定点算术、真实 gamma 与压力测试。"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

try:
    from .p50_format import P50Image
    from .rmsnorm_fixed_reference import (
        ACTIVATION_FACTOR,
        DEFAULT_EPSILON,
        DEFAULT_GAMMA,
        DEFAULT_IMAGE,
        DEFAULT_INPUT_SEED,
        DEFAULT_LENGTH,
        GAMMA_FACTOR,
        OUTPUT_FACTOR,
        RSQRT_FACTOR,
        VARIANCE_FACTOR,
        _round_div_unsigned,
        _round_shift_signed_scalar,
        build_rsqrt_lut,
        compute_rmsnorm_reference,
        make_deterministic_input,
        quantize_activation_q6_10,
        quantize_epsilon_q20,
        quantize_gamma_q6_10,
        reference_from_p50,
        result_manifest,
        rsqrt_exact_q20,
        rsqrt_lut_q20,
        rsqrt_newton_q20,
    )
except ImportError:
    from p50_format import P50Image
    from rmsnorm_fixed_reference import (
        ACTIVATION_FACTOR,
        DEFAULT_EPSILON,
        DEFAULT_GAMMA,
        DEFAULT_IMAGE,
        DEFAULT_INPUT_SEED,
        DEFAULT_LENGTH,
        GAMMA_FACTOR,
        OUTPUT_FACTOR,
        RSQRT_FACTOR,
        VARIANCE_FACTOR,
        _round_div_unsigned,
        _round_shift_signed_scalar,
        build_rsqrt_lut,
        compute_rmsnorm_reference,
        make_deterministic_input,
        quantize_activation_q6_10,
        quantize_epsilon_q20,
        quantize_gamma_q6_10,
        reference_from_p50,
        result_manifest,
        rsqrt_exact_q20,
        rsqrt_lut_q20,
        rsqrt_newton_q20,
    )


class RMSNormFixedReferenceTest(unittest.TestCase):
    def test_signed_rne_shift_handles_ties_and_sign(self) -> None:
        self.assertEqual(_round_shift_signed_scalar(1, 1), 0)
        self.assertEqual(_round_shift_signed_scalar(3, 1), 2)
        self.assertEqual(_round_shift_signed_scalar(5, 1), 2)
        self.assertEqual(_round_shift_signed_scalar(-1, 1), 0)
        self.assertEqual(_round_shift_signed_scalar(-3, 1), -2)
        self.assertEqual(_round_shift_signed_scalar(-5, 1), -2)

    def test_unsigned_rne_division_handles_ties(self) -> None:
        self.assertEqual(_round_div_unsigned(1, 2), 0)
        self.assertEqual(_round_div_unsigned(3, 2), 2)
        self.assertEqual(_round_div_unsigned(5, 2), 2)
        self.assertEqual(_round_div_unsigned(7, 2), 4)

    def test_q6_10_quantization_uses_rne_and_saturation(self) -> None:
        activation = quantize_activation_q6_10(
            np.asarray([0.5 / ACTIVATION_FACTOR, 1.5 / ACTIVATION_FACTOR, 32.0])
        )
        np.testing.assert_array_equal(
            activation.quantized, np.asarray([0, 2, 32767], dtype=np.int16)
        )
        self.assertEqual(activation.clipped_count, 1)

        gamma = quantize_gamma_q6_10(
            np.asarray([-0.5 / GAMMA_FACTOR, -1.5 / GAMMA_FACTOR])
        )
        np.testing.assert_array_equal(
            gamma.quantized, np.asarray([0, -2], dtype=np.int16)
        )

    def test_epsilon_q20_is_nonzero_and_close(self) -> None:
        epsilon_q20 = quantize_epsilon_q20(DEFAULT_EPSILON)
        self.assertEqual(epsilon_q20, 1)
        self.assertLess(
            abs(epsilon_q20 / VARIANCE_FACTOR - DEFAULT_EPSILON), 5e-8
        )

    def test_rsqrt_lut_and_newton_cover_wide_binary_range(self) -> None:
        for exponent in range(-19, 11):
            for mantissa in (1.0, 1.125, 1.5, 1.875):
                value = mantissa * (2.0**exponent)
                variance_q20 = max(1, int(np.rint(value * VARIANCE_FACTOR)))
                exact = rsqrt_exact_q20(variance_q20)
                lut = rsqrt_lut_q20(variance_q20)
                nr = rsqrt_newton_q20(variance_q20)
                exact_float = exact / RSQRT_FACTOR
                self.assertLessEqual(abs(lut / RSQRT_FACTOR - exact_float), 1.5e-3 * exact_float + 2 / RSQRT_FACTOR)
                self.assertLessEqual(abs(nr / RSQRT_FACTOR - exact_float), 2.0e-4 * exact_float + 2 / RSQRT_FACTOR)

    def test_zero_vector_outputs_exact_zero(self) -> None:
        activation = np.zeros(DEFAULT_LENGTH, dtype=np.float32)
        gamma = np.linspace(-0.5, 0.75, DEFAULT_LENGTH, dtype=np.float32)
        result = compute_rmsnorm_reference(
            activation_values=activation,
            gamma_values=gamma,
            epsilon=DEFAULT_EPSILON,
        )
        np.testing.assert_array_equal(
            result.output_exact_q10, np.zeros(DEFAULT_LENGTH, dtype=np.int16)
        )
        np.testing.assert_array_equal(result.output_lut_q10, result.output_exact_q10)
        np.testing.assert_array_equal(result.output_nr_q10, result.output_exact_q10)
        self.assertEqual(result.sum_squares, 0)
        self.assertEqual(result.variance_q20, 1)

    def test_small_direct_vector_matches_quantized_formula(self) -> None:
        activation = np.asarray([1.0, -2.0, 3.0, -4.0], dtype=np.float32)
        gamma = np.asarray([0.5, 0.25, -0.75, 1.0], dtype=np.float32)
        result = compute_rmsnorm_reference(
            activation_values=activation,
            gamma_values=gamma,
            epsilon=DEFAULT_EPSILON,
        )
        x = result.activation.quantized.astype(np.float64) / ACTIVATION_FACTOR
        g = result.gamma.quantized.astype(np.float64) / GAMMA_FACTOR
        variance = result.variance_q20 / VARIANCE_FACTOR
        expected = x * (1.0 / np.sqrt(variance)) * g
        np.testing.assert_allclose(
            result.output_quantized_float,
            expected,
            rtol=0.0,
            atol=1e-12,
        )
        self.assertLessEqual(
            float(np.max(np.abs(result.output_exact_float - expected))),
            1.5 / OUTPUT_FACTOR,
        )

    def test_lut_shapes_and_monotonicity(self) -> None:
        lut256 = build_rsqrt_lut(8)
        lut32 = build_rsqrt_lut(5)
        self.assertEqual(lut256.shape, (256,))
        self.assertEqual(lut32.shape, (32,))
        self.assertTrue(np.all(np.diff(lut256.astype(np.int64)) < 0))
        self.assertTrue(np.all(np.diff(lut32.astype(np.int64)) < 0))

    @unittest.skipUnless(Path(DEFAULT_IMAGE).is_file(), "本地没有真实 .p50 镜像")
    def test_real_layer0_fixed_manifest(self) -> None:
        image = P50Image(DEFAULT_IMAGE)
        image.validate()
        activation = make_deterministic_input(DEFAULT_LENGTH, DEFAULT_INPUT_SEED)
        result = reference_from_p50(
            image,
            activation_values=activation,
            gamma_name=DEFAULT_GAMMA,
        )
        self.assertEqual(result.length, DEFAULT_LENGTH)
        self.assertEqual(result.activation.clipped_count, 0)
        self.assertEqual(result.gamma.clipped_count, 0)
        self.assertEqual(result.exact_output_saturated_count, 0)
        self.assertLessEqual(
            float(np.max(np.abs(result.output_lut_float - result.output_exact_float))),
            1.0 / OUTPUT_FACTOR,
        )
        self.assertLessEqual(
            float(np.max(np.abs(result.output_nr_float - result.output_exact_float))),
            1.0 / OUTPUT_FACTOR,
        )
        manifest = result_manifest(result)
        committed = json.loads(
            Path(__file__).with_name("rmsnorm_layer0_reference.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(manifest, committed)

    @unittest.skipUnless(Path(DEFAULT_IMAGE).is_file(), "本地没有真实 .p50 镜像")
    def test_random_software_stress_1000_vectors(self) -> None:
        image = P50Image(DEFAULT_IMAGE)
        image.validate()
        gamma = image.read_float16_tensor(DEFAULT_GAMMA).astype(np.float32).reshape(-1)
        rng = np.random.default_rng(20260726)
        max_lut_lsb = 0
        max_nr_lsb = 0
        for _ in range(1000):
            amplitude = float(2.0 ** rng.uniform(-6.0, 3.0))
            activation = rng.uniform(-amplitude, amplitude, size=DEFAULT_LENGTH).astype(
                np.float32
            )
            result = compute_rmsnorm_reference(
                activation_values=activation,
                gamma_values=gamma,
                epsilon=DEFAULT_EPSILON,
                gamma_name=DEFAULT_GAMMA,
            )
            self.assertEqual(result.activation.clipped_count, 0)
            self.assertEqual(result.gamma.clipped_count, 0)
            self.assertEqual(result.exact_output_saturated_count, 0)
            lut_lsb = int(
                np.max(
                    np.abs(
                        result.output_lut_q10.astype(np.int32)
                        - result.output_exact_q10.astype(np.int32)
                    )
                )
            )
            nr_lsb = int(
                np.max(
                    np.abs(
                        result.output_nr_q10.astype(np.int32)
                        - result.output_exact_q10.astype(np.int32)
                    )
                )
            )
            max_lut_lsb = max(max_lut_lsb, lut_lsb)
            max_nr_lsb = max(max_nr_lsb, nr_lsb)
        self.assertLessEqual(max_lut_lsb, 3)
        self.assertLessEqual(max_nr_lsb, 1)


if __name__ == "__main__":
    unittest.main()
