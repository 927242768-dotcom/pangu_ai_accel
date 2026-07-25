#!/usr/bin/env python3
"""G2 运行时激活量化与 combined scale 精确整数规格测试。"""

from __future__ import annotations

import unittest

import numpy as np

try:
    from .runtime_linear_quant_reference import (
        q28_int64_to_binary32_bits_exact,
        quantize_q10_activation_exact,
        quantize_q10_and_build_scales,
        quantize_q28_activation_exact,
        quantize_q28_and_build_scales,
        round_div_rne_nonnegative,
    )
    from .transformer_block_reference import build_case, load_context
except ImportError:
    from runtime_linear_quant_reference import (
        q28_int64_to_binary32_bits_exact,
        quantize_q10_activation_exact,
        quantize_q10_and_build_scales,
        quantize_q28_activation_exact,
        quantize_q28_and_build_scales,
        round_div_rne_nonnegative,
    )
    from transformer_block_reference import build_case, load_context


class RuntimeLinearQuantReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.context = load_context()
        cls.case = build_case(cls.context, query_position=0, window_start=0)

    def test_round_div_rne_ties_to_even(self) -> None:
        self.assertEqual(round_div_rne_nonnegative(1, 2), 0)
        self.assertEqual(round_div_rne_nonnegative(3, 2), 2)
        self.assertEqual(round_div_rne_nonnegative(5, 2), 2)
        self.assertEqual(round_div_rne_nonnegative(7, 2), 4)

    def test_q10_activation_zero_and_signed_extremes(self) -> None:
        zeros, maximum = quantize_q10_activation_exact(np.zeros(16, dtype=np.int16))
        np.testing.assert_array_equal(zeros, np.zeros(16, dtype=np.int8))
        self.assertEqual(maximum, 0)
        source = np.asarray([-32768, -16384, -1, 0, 1, 16384, 32767], dtype=np.int16)
        quantized, maximum = quantize_q10_activation_exact(source)
        self.assertEqual(maximum, 32768)
        self.assertEqual(int(quantized[0]), -127)
        self.assertEqual(int(quantized[-1]), 127)
        self.assertEqual(int(quantized[3]), 0)

    def test_q28_double_rounding_bits_match_numpy(self) -> None:
        rng = np.random.default_rng(20260725)
        random_values = rng.integers(
            np.iinfo(np.int64).min,
            np.iinfo(np.int64).max,
            size=10000,
            dtype=np.int64,
        )
        edges = np.asarray(
            [
                np.iinfo(np.int64).min,
                np.iinfo(np.int64).min + 1,
                -(1 << 53) - 1,
                -(1 << 53),
                -1,
                0,
                1,
                (1 << 53) - 1,
                1 << 53,
                (1 << 53) + 1,
                np.iinfo(np.int64).max,
            ],
            dtype=np.int64,
        )
        source = np.concatenate([edges, random_values])
        exact = q28_int64_to_binary32_bits_exact(source)
        numpy_bits = (
            source.astype(np.float64) / float(1 << 28)
        ).astype(np.float32).view(np.uint32)
        np.testing.assert_array_equal(exact, numpy_bits)

    def test_q28_binary32_quantization_is_deterministic(self) -> None:
        source = np.asarray(
            [0, 1, -1, 1 << 27, -(1 << 27), (1 << 40) + 12345],
            dtype=np.int64,
        )
        first, maximum_first = quantize_q28_activation_exact(source)
        second, maximum_second = quantize_q28_activation_exact(source)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(maximum_first.view(np.uint32), maximum_second.view(np.uint32))
        self.assertEqual(int(first[-1]), 127)

    def test_real_qkv_q10_matches_existing_numpy_definition(self) -> None:
        for key in ("q", "k", "v"):
            model = self.context.attention.qkv_models[key]
            result = quantize_q10_and_build_scales(
                self.case.input_norm_q10,
                model.weight_scales,
            )
            self.assertEqual(result.activation_int8.shape, (896,))
            self.assertEqual(result.combined_scale_q28.shape, model.weight_scales.shape)
            self.assertEqual(result.saturated_scale_count, 0)

    def test_real_gate_up_q10_share_identical_activation(self) -> None:
        gate = quantize_q10_and_build_scales(
            self.case.post_attention_norm_q10,
            self.context.gate_model.weight_scales,
        )
        up = quantize_q10_and_build_scales(
            self.case.post_attention_norm_q10,
            self.context.up_model.weight_scales,
        )
        np.testing.assert_array_equal(gate.activation_int8, up.activation_int8)
        self.assertEqual(gate.activation_scale, up.activation_scale)
        self.assertEqual(gate.combined_scale_q28.shape, (4864, 14))
        self.assertEqual(up.combined_scale_q28.shape, (4864, 14))

    def test_real_oproj_and_down_q28_match_existing_numpy_definition(self) -> None:
        oproj = quantize_q28_and_build_scales(
            self.case.attention_concat_q28.reshape(-1),
            self.context.attention.oproj_model.weight_scales,
        )
        down = quantize_q28_and_build_scales(
            self.case.silu_up_q28,
            self.context.down_model.weight_scales,
        )
        self.assertEqual(oproj.activation_int8.shape, (896,))
        self.assertEqual(oproj.combined_scale_q28.shape, (896, 14))
        self.assertEqual(down.activation_int8.shape, (4864,))
        self.assertEqual(down.combined_scale_q28.shape, (896, 76))
        self.assertEqual(oproj.saturated_scale_count, 0)
        self.assertEqual(down.saturated_scale_count, 0)


if __name__ == "__main__":
    unittest.main()
