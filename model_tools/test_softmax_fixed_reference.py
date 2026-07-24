#!/usr/bin/env python3
"""``softmax_fixed_reference`` 的 mask、LUT、倒数、归一化和真实 F4 测试。"""

from __future__ import annotations

import math
import unittest
from pathlib import Path

import numpy as np

try:
    from .softmax_fixed_reference import (
        DEFAULT_FLOAT_TOLERANCE,
        DEFAULT_IMAGE,
        DEFAULT_MANIFEST,
        EXP_INTERVALS,
        EXP_LUT_ENTRIES,
        EXP_LUT_PADDED_BYTES,
        EXP_LUT_Q31,
        EXP_MIN_Q28,
        MASK_VALUE,
        MAX_TOKENS,
        PROB_BYTES,
        PROB_ONE,
        Q_HEADS,
        SoftmaxReferenceError,
        build_exp_lut_payload,
        build_fixed_real_cases,
        decode_exp_lut_payload,
        decode_probabilities,
        encode_probabilities,
        exp_pwl_q31,
        max_probability_error,
        round_div_rne_unsigned,
        round_shift_rne_unsigned,
        softmax_head_q31,
        softmax_scores_q31,
        software_stress,
        validate_manifest,
    )
except ImportError:
    from softmax_fixed_reference import (
        DEFAULT_FLOAT_TOLERANCE,
        DEFAULT_IMAGE,
        DEFAULT_MANIFEST,
        EXP_INTERVALS,
        EXP_LUT_ENTRIES,
        EXP_LUT_PADDED_BYTES,
        EXP_LUT_Q31,
        EXP_MIN_Q28,
        MASK_VALUE,
        MAX_TOKENS,
        PROB_BYTES,
        PROB_ONE,
        Q_HEADS,
        SoftmaxReferenceError,
        build_exp_lut_payload,
        build_fixed_real_cases,
        decode_exp_lut_payload,
        decode_probabilities,
        encode_probabilities,
        exp_pwl_q31,
        max_probability_error,
        round_div_rne_unsigned,
        round_shift_rne_unsigned,
        softmax_head_q31,
        softmax_scores_q31,
        software_stress,
        validate_manifest,
    )


class SoftmaxFixedReferenceTest(unittest.TestCase):
    def test_lut_shape_endpoints_and_monotonicity(self) -> None:
        self.assertEqual(EXP_LUT_ENTRIES, 513)
        self.assertEqual(EXP_INTERVALS, 512)
        self.assertEqual(EXP_LUT_Q31.shape, (513,))
        self.assertEqual(int(EXP_LUT_Q31[0]), PROB_ONE)
        self.assertEqual(
            int(EXP_LUT_Q31[-1]), round(math.exp(-16.0) * PROB_ONE)
        )
        self.assertTrue(np.all(EXP_LUT_Q31[1:] <= EXP_LUT_Q31[:-1]))

    def test_unsigned_rne_ties(self) -> None:
        self.assertEqual(round_shift_rne_unsigned(4, 3), 0)
        self.assertEqual(round_shift_rne_unsigned(12, 3), 2)
        self.assertEqual(round_shift_rne_unsigned(20, 3), 2)
        self.assertEqual(round_shift_rne_unsigned(28, 3), 4)
        self.assertEqual(round_div_rne_unsigned(1, 2), 0)
        self.assertEqual(round_div_rne_unsigned(3, 2), 2)
        self.assertEqual(round_div_rne_unsigned(5, 2), 2)
        self.assertEqual(round_div_rne_unsigned(7, 2), 4)

    def test_exp_boundaries(self) -> None:
        self.assertEqual(exp_pwl_q31(0), PROB_ONE)
        self.assertEqual(exp_pwl_q31(EXP_MIN_Q28), int(EXP_LUT_Q31[-1]))
        self.assertEqual(exp_pwl_q31(EXP_MIN_Q28 - 1), 0)
        for index in (0, 1, 31, 32, 255, 511, 512):
            difference = -(index << 23)
            self.assertEqual(exp_pwl_q31(difference), int(EXP_LUT_Q31[index]))

    def test_all_mask_and_single_valid(self) -> None:
        all_mask = np.full(MAX_TOKENS, MASK_VALUE, dtype=np.int64)
        probabilities, debug = softmax_head_q31(all_mask)
        np.testing.assert_array_equal(probabilities, 0)
        self.assertTrue(debug.all_masked)
        self.assertEqual(debug.sum_exp_q31, 0)
        self.assertEqual(debug.reciprocal_q31, 0)

        single = all_mask.copy()
        single[9] = -(3 << 28)
        probabilities, debug = softmax_head_q31(single)
        self.assertFalse(debug.all_masked)
        self.assertEqual(int(probabilities[9]), PROB_ONE)
        self.assertEqual(int(np.count_nonzero(probabilities)), 1)

    def test_equal_scores_are_uniform(self) -> None:
        scores = np.full(MAX_TOKENS, MASK_VALUE, dtype=np.int64)
        scores[:16] = 7 << 28
        probabilities, debug = softmax_head_q31(scores)
        expected = PROB_ONE // 16
        np.testing.assert_array_equal(probabilities, expected)
        self.assertEqual(debug.sum_exp_q31, 16 * PROB_ONE)
        self.assertEqual(int(np.sum(probabilities, dtype=np.uint64)), PROB_ONE)

    def test_mask_and_extreme_difference(self) -> None:
        scores = np.full(MAX_TOKENS, MASK_VALUE, dtype=np.int64)
        scores[0] = 4 << 28
        scores[1] = -(12 << 28)
        scores[2] = -(13 << 28) - 1
        probabilities, debug = softmax_head_q31(scores)
        self.assertEqual(int(debug.exp_q31[0]), PROB_ONE)
        self.assertEqual(int(debug.exp_q31[1]), int(EXP_LUT_Q31[-1]))
        self.assertEqual(int(debug.exp_q31[2]), 0)
        self.assertEqual(int(probabilities[3]), 0)
        self.assertLessEqual(max_probability_error(
            np.tile(scores, (Q_HEADS, 1)),
            np.tile(probabilities, (Q_HEADS, 1)),
        ), DEFAULT_FLOAT_TOLERANCE)

    def test_probability_and_lut_payload_roundtrip(self) -> None:
        rng = np.random.default_rng(20260803)
        probabilities = rng.integers(
            0, PROB_ONE + 1, size=(Q_HEADS, MAX_TOKENS), dtype=np.uint32
        )
        payload = encode_probabilities(probabilities)
        self.assertEqual(len(payload), PROB_BYTES)
        np.testing.assert_array_equal(decode_probabilities(payload), probabilities)

        lut_payload = build_exp_lut_payload()
        self.assertEqual(len(lut_payload), EXP_LUT_PADDED_BYTES)
        np.testing.assert_array_equal(decode_exp_lut_payload(lut_payload), EXP_LUT_Q31)
        broken = bytearray(lut_payload)
        broken[-1] = 1
        with self.assertRaises(SoftmaxReferenceError):
            decode_exp_lut_payload(bytes(broken))

    def test_matrix_shape_and_mask_zero(self) -> None:
        scores = np.full((Q_HEADS, MAX_TOKENS), MASK_VALUE, dtype=np.int64)
        scores[:, :4] = np.asarray([0, -(1 << 28), -(2 << 28), -(3 << 28)])
        probabilities, debug = softmax_scores_q31(scores)
        self.assertEqual(probabilities.shape, (Q_HEADS, MAX_TOKENS))
        self.assertEqual(len(debug), Q_HEADS)
        np.testing.assert_array_equal(probabilities[:, 4:], 0)
        self.assertLessEqual(
            max_probability_error(scores, probabilities), DEFAULT_FLOAT_TOLERANCE
        )

    def test_small_random_stress(self) -> None:
        worst_error = software_stress(rounds=50, seed=20260803)
        self.assertLessEqual(worst_error, DEFAULT_FLOAT_TOLERANCE)

    @unittest.skipUnless(Path(DEFAULT_IMAGE).is_file(), "本地没有真实 .p50 镜像")
    def test_real_manifest(self) -> None:
        cases = build_fixed_real_cases()
        self.assertEqual(len(cases), 4)
        for case in cases:
            self.assertEqual(case.scores_q28.shape, (Q_HEADS, MAX_TOKENS))
            self.assertEqual(case.expected_probs_q31.shape, (Q_HEADS, MAX_TOKENS))
            self.assertLessEqual(
                max_probability_error(case.scores_q28, case.expected_probs_q31),
                DEFAULT_FLOAT_TOLERANCE,
            )
        committed = validate_manifest(cases, DEFAULT_MANIFEST)
        self.assertEqual(committed["definition"]["output_format"], "unsigned UQ1.31 uint32")
        self.assertEqual(committed["definition"]["one"], PROB_ONE)


if __name__ == "__main__":
    unittest.main()
