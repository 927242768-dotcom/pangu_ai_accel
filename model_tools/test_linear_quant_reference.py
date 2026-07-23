#!/usr/bin/env python3
"""``linear_quant_reference`` 的纯软件与真实 P50 集成测试。"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

try:
    from .linear_quant_reference import (
        COMBINED_SCALE_FACTOR,
        COMBINED_SCALE_QMAX,
        DEFAULT_ACTIVATION_SEED,
        DEFAULT_BIAS,
        DEFAULT_IMAGE,
        DEFAULT_WEIGHT,
        compute_groupwise_linear_reference,
        make_deterministic_activation,
        pack_int4_low_nibble_first,
        quantize_activation_int8,
        quantize_uq4_28,
        reference_from_p50,
        result_manifest,
    )
    from .p50_format import P50Image
except ImportError:
    from linear_quant_reference import (
        COMBINED_SCALE_FACTOR,
        COMBINED_SCALE_QMAX,
        DEFAULT_ACTIVATION_SEED,
        DEFAULT_BIAS,
        DEFAULT_IMAGE,
        DEFAULT_WEIGHT,
        compute_groupwise_linear_reference,
        make_deterministic_activation,
        pack_int4_low_nibble_first,
        quantize_activation_int8,
        quantize_uq4_28,
        reference_from_p50,
        result_manifest,
    )
    from p50_format import P50Image


class LinearQuantReferenceTest(unittest.TestCase):
    def test_activation_quantization_uses_symmetric_int8_and_rne(self) -> None:
        values = np.asarray([127.0, 2.5, 1.5, -1.5, -2.5, -127.0], dtype=np.float32)
        result = quantize_activation_int8(values)
        self.assertEqual(result.scale, 1.0)
        self.assertEqual(result.clipped_count, 0)
        np.testing.assert_array_equal(
            result.quantized,
            np.asarray([127, 2, 2, -2, -2, -127], dtype=np.int8),
        )

    def test_zero_activation_is_exact(self) -> None:
        result = quantize_activation_int8(np.zeros(16, dtype=np.float32))
        self.assertEqual(result.scale, 1.0)
        np.testing.assert_array_equal(result.quantized, np.zeros(16, dtype=np.int8))
        np.testing.assert_array_equal(result.dequantized, np.zeros(16, dtype=np.float32))

    def test_uq4_28_rne_and_saturation(self) -> None:
        lsb = 1.0 / COMBINED_SCALE_FACTOR
        quantized, saturated = quantize_uq4_28(
            np.asarray([0.0, 0.5 * lsb, 1.5 * lsb, 16.0], dtype=np.float64)
        )
        np.testing.assert_array_equal(
            quantized,
            np.asarray([0, 0, 2, COMBINED_SCALE_QMAX], dtype=np.uint32),
        )
        self.assertEqual(saturated, 1)

    def test_groupwise_reference_matches_direct_float_path(self) -> None:
        weights = np.asarray(
            [
                [1, -2, 3, -4, 5, -6, 7, -7],
                [-7, 6, -5, 4, -3, 2, -1, 0],
            ],
            dtype=np.int8,
        )
        scales = np.asarray([[0.5, 0.25], [0.125, 0.75]], dtype=np.float32)
        activation = np.asarray([1, -2, 3, -4, 5, -6, 7, 127], dtype=np.float32)
        bias = np.asarray([0.25, -0.5], dtype=np.float32)
        result = compute_groupwise_linear_reference(
            weight_quantized=weights,
            weight_scales=scales,
            activation_values=activation,
            bias=bias,
            group_size=4,
        )

        expanded_scales = np.repeat(scales, 4, axis=1)
        expected = np.sum(
            weights.astype(np.float64)
            * expanded_scales.astype(np.float64)
            * activation[np.newaxis, :].astype(np.float64),
            axis=1,
        ) + bias
        np.testing.assert_allclose(result.output_p50_float, expected, rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(
            result.output_quantized_float, expected, rtol=0.0, atol=1e-12
        )
        self.assertEqual(result.combined_scale_saturated_count, 0)
        self.assertTrue(
            np.all(np.abs(result.fixed_error) <= result.fixed_error_bound + 1e-12)
        )

    def test_random_software_stress_1000_cases(self) -> None:
        rng = np.random.default_rng(20260723)
        for _ in range(1000):
            rows = int(rng.integers(1, 9))
            groups = int(rng.integers(1, 15))
            columns = groups * 64
            weights = rng.integers(-7, 8, size=(rows, columns), dtype=np.int8)
            scales = rng.uniform(4e-4, 0.25, size=(rows, groups)).astype(np.float32)
            activation = rng.uniform(-4.0, 4.0, size=columns).astype(np.float32)
            bias = rng.uniform(-0.5, 0.5, size=rows).astype(np.float32)
            result = compute_groupwise_linear_reference(
                weight_quantized=weights,
                weight_scales=scales,
                activation_values=activation,
                bias=bias,
                group_size=64,
            )
            expanded_scales = np.repeat(scales, 64, axis=1)
            activation_quantized_float64 = (
                result.activation.quantized.astype(np.float64) * result.activation.scale
            )
            expected_quantized = np.sum(
                weights.astype(np.float64)
                * expanded_scales.astype(np.float64)
                * activation_quantized_float64[np.newaxis, :],
                axis=1,
            ) + bias
            np.testing.assert_allclose(
                result.output_quantized_float,
                expected_quantized,
                rtol=0.0,
                atol=1e-10,
            )
            self.assertEqual(result.combined_scale_saturated_count, 0)
            self.assertTrue(
                np.all(np.abs(result.fixed_error) <= result.fixed_error_bound + 1e-12)
            )

    def test_packed_int4_low_nibble_first(self) -> None:
        values = np.asarray([[-7, -1, 0, 7]], dtype=np.int8)
        packed = pack_int4_low_nibble_first(values)
        np.testing.assert_array_equal(packed, np.asarray([[0xF9, 0x70]], dtype=np.uint8))

    def test_deterministic_activation_is_reproducible(self) -> None:
        first = make_deterministic_activation(896, DEFAULT_ACTIVATION_SEED)
        second = make_deterministic_activation(896, DEFAULT_ACTIVATION_SEED)
        np.testing.assert_array_equal(first, second)
        self.assertGreater(float(np.max(np.abs(first))), 3.9)
        self.assertLess(float(np.max(np.abs(first))), 4.0)

    @unittest.skipUnless(Path(DEFAULT_IMAGE).is_file(), "本地没有真实 .p50 镜像")
    def test_real_q_proj_m4k896_reference(self) -> None:
        image = P50Image(DEFAULT_IMAGE)
        image.validate()
        activation = make_deterministic_activation(896, DEFAULT_ACTIVATION_SEED)
        result = reference_from_p50(
            image,
            weight_name=DEFAULT_WEIGHT,
            bias_name=DEFAULT_BIAS,
            row_start=0,
            row_count=4,
            column_start=0,
            column_count=896,
            activation_values=activation,
        )
        self.assertEqual(result.group_accumulators.shape, (4, 14))
        self.assertEqual(result.combined_scale_q28.shape, (4, 14))
        self.assertEqual(result.combined_scale_saturated_count, 0)
        self.assertTrue(
            np.all(np.abs(result.fixed_error) <= result.fixed_error_bound + 1e-12)
        )
        manifest = result_manifest(result)
        committed_manifest_path = Path(__file__).with_name(
            "q_proj_m4k896_reference.json"
        )
        committed = json.loads(committed_manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["slice"], committed["slice"])
        self.assertEqual(
            manifest["activation_format"]["scale"],
            committed["activation_format"]["scale"],
        )
        self.assertEqual(manifest["expected"], committed["expected"])
        self.assertEqual(manifest["sha256"], committed["sha256"])


if __name__ == "__main__":
    unittest.main()
