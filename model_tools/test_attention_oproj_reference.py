#!/usr/bin/env python3
"""``attention_oproj_reference`` 的输入量化、真实 O_proj 和固定清单测试。"""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

try:
    from .attention_oproj_reference import (
        DEFAULT_IMAGE,
        DEFAULT_MANIFEST,
        GROUPS,
        K,
        M,
        AttentionOProjReferenceError,
        attention_q28_to_float32,
        build_fixed_real_cases,
        case_from_attention_q28,
        compute_q28_reference,
        load_oproj_model,
        make_random_attention_q28,
        software_stress,
        validate_manifest,
    )
    from .p50_format import P50Image
except ImportError:
    from attention_oproj_reference import (
        DEFAULT_IMAGE,
        DEFAULT_MANIFEST,
        GROUPS,
        K,
        M,
        AttentionOProjReferenceError,
        attention_q28_to_float32,
        build_fixed_real_cases,
        case_from_attention_q28,
        compute_q28_reference,
        load_oproj_model,
        make_random_attention_q28,
        software_stress,
        validate_manifest,
    )
    from p50_format import P50Image


class AttentionOProjReferenceTest(unittest.TestCase):
    def test_q28_to_float32_shape(self) -> None:
        values = np.arange(K, dtype=np.int64) - K // 2
        converted = attention_q28_to_float32(values)
        self.assertEqual(converted.shape, (K,))
        self.assertEqual(converted.dtype, np.float32)
        self.assertAlmostEqual(float(converted[K // 2 + 1]), 1.0 / (1 << 28))
        with self.assertRaises(AttentionOProjReferenceError):
            attention_q28_to_float32(values[:-1])

    def test_independent_q28_small_synthetic(self) -> None:
        activation = np.zeros(K, dtype=np.int8)
        activation[:64] = 1
        weights = np.zeros((M, K), dtype=np.int8)
        weights[:, :64] = 2
        scales = np.ones((M, GROUPS), dtype=np.uint32)
        output = compute_q28_reference(activation, weights, scales)
        np.testing.assert_array_equal(output, 128)

    @unittest.skipUnless(Path(DEFAULT_IMAGE).is_file(), "本地没有真实 .p50 镜像")
    def test_zero_input_strict_zero_output(self) -> None:
        image = P50Image(DEFAULT_IMAGE)
        image.validate()
        model = load_oproj_model(image)
        case = case_from_attention_q28(
            model,
            np.zeros(K, dtype=np.int64),
            label="zero",
        )
        np.testing.assert_array_equal(case.activation_int8, 0)
        np.testing.assert_array_equal(case.bias_q28, 0)
        np.testing.assert_array_equal(case.expected_q28, 0)
        self.assertEqual(case.activation_scale, 1.0)

    @unittest.skipUnless(Path(DEFAULT_IMAGE).is_file(), "本地没有真实 .p50 镜像")
    def test_real_model_shape_and_no_bias(self) -> None:
        image = P50Image(DEFAULT_IMAGE)
        image.validate()
        model = load_oproj_model(image)
        self.assertEqual(model.weights.shape, (M, K))
        self.assertEqual(model.weight_scales.shape, (M, GROUPS))
        self.assertTrue(np.all(model.weight_scales > 0.0))

    @unittest.skipUnless(Path(DEFAULT_IMAGE).is_file(), "本地没有真实 .p50 镜像")
    def test_real_fixed_manifest(self) -> None:
        cases = build_fixed_real_cases()
        self.assertEqual(len(cases), 4)
        for case in cases:
            self.assertEqual(case.source_attention_q28.shape, (K,))
            self.assertEqual(case.activation_int8.shape, (K,))
            self.assertEqual(case.scales_q28.shape, (M, GROUPS))
            self.assertEqual(case.expected_q28.shape, (M,))
            self.assertEqual(case.linear_result.combined_scale_saturated_count, 0)
            np.testing.assert_array_equal(
                compute_q28_reference(
                    case.activation_int8,
                    case.weights,
                    case.scales_q28,
                    case.bias_q28,
                ),
                case.expected_q28,
            )
        committed = validate_manifest(cases, DEFAULT_MANIFEST)
        self.assertEqual(
            committed["definition"]["weight_tensor"],
            "model.layers.0.self_attn.o_proj.weight",
        )
        self.assertEqual(committed["definition"]["bias"], "absent in P50; bias_q28 is all zero")

    @unittest.skipUnless(Path(DEFAULT_IMAGE).is_file(), "本地没有真实 .p50 镜像")
    def test_small_random_stress(self) -> None:
        software_stress(rounds=8, seed=20260805)

    def test_random_generator_modes(self) -> None:
        rng = np.random.default_rng(20260805)
        generated = [make_random_attention_q28(rng, index) for index in range(8)]
        self.assertTrue(np.all(generated[0] == 0))
        self.assertTrue(np.all(generated[1] == generated[1][0]))
        self.assertEqual(int(generated[2][0]), 16 << 28)
        self.assertEqual(int(generated[2][1]), -(16 << 28))
        for values in generated:
            self.assertEqual(values.shape, (K,))
            self.assertEqual(values.dtype, np.int64)


if __name__ == "__main__":
    unittest.main()
