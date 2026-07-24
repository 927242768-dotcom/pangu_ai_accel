#!/usr/bin/env python3
"""``attention_output_reference`` 的 GQA、Q59、RNE、饱和和真实固定测试。"""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

try:
    from .attention_output_reference import (
        DEFAULT_IMAGE,
        DEFAULT_MANIFEST,
        HEAD_DIM,
        INT64_MAX,
        INT64_MIN,
        KV_HEADS,
        MAX_TOKENS,
        OUTPUT_BYTES,
        OUTPUT_VALUES,
        PROB_ONE,
        Q_HEADS,
        V_BYTES,
        AttentionOutputReferenceError,
        attention_output_q28,
        build_fixed_real_cases,
        decode_attention_output,
        decode_v_vector,
        encode_attention_output,
        encode_v_vector,
        flatten_attention_heads,
        reshape_attention_heads,
        round_shift_rne_signed,
        software_stress,
        validate_manifest,
        weighted_sum_head_q28,
    )
except ImportError:
    from attention_output_reference import (
        DEFAULT_IMAGE,
        DEFAULT_MANIFEST,
        HEAD_DIM,
        INT64_MAX,
        INT64_MIN,
        KV_HEADS,
        MAX_TOKENS,
        OUTPUT_BYTES,
        OUTPUT_VALUES,
        PROB_ONE,
        Q_HEADS,
        V_BYTES,
        AttentionOutputReferenceError,
        attention_output_q28,
        build_fixed_real_cases,
        decode_attention_output,
        decode_v_vector,
        encode_attention_output,
        encode_v_vector,
        flatten_attention_heads,
        reshape_attention_heads,
        round_shift_rne_signed,
        software_stress,
        validate_manifest,
        weighted_sum_head_q28,
    )


class AttentionOutputReferenceTest(unittest.TestCase):
    def test_signed_rne_ties(self) -> None:
        self.assertEqual(round_shift_rne_signed(1, 1), 0)
        self.assertEqual(round_shift_rne_signed(3, 1), 2)
        self.assertEqual(round_shift_rne_signed(5, 1), 2)
        self.assertEqual(round_shift_rne_signed(7, 1), 4)
        self.assertEqual(round_shift_rne_signed(-1, 1), 0)
        self.assertEqual(round_shift_rne_signed(-3, 1), -2)
        self.assertEqual(round_shift_rne_signed(-5, 1), -2)
        self.assertEqual(round_shift_rne_signed(-7, 1), -4)
        with self.assertRaises(AttentionOutputReferenceError):
            round_shift_rne_signed(1, 0)

    def test_all_mask_outputs_zero(self) -> None:
        probabilities = np.zeros((Q_HEADS, MAX_TOKENS), dtype=np.uint32)
        history = np.full((4, KV_HEADS, HEAD_DIM), INT64_MAX, dtype=np.int64)
        output, debug = attention_output_q28(probabilities, history, count=4)
        np.testing.assert_array_equal(output, 0)
        self.assertEqual(debug.max_abs_accumulator_q59, 0)
        self.assertEqual(debug.saturated_values, 0)

    def test_single_token_exact_copy_and_gqa_mapping(self) -> None:
        probabilities = np.zeros((Q_HEADS, MAX_TOKENS), dtype=np.uint32)
        probabilities[:, 0] = PROB_ONE
        history = np.zeros((1, KV_HEADS, HEAD_DIM), dtype=np.int64)
        history[0, 0] = np.arange(HEAD_DIM, dtype=np.int64) - 32
        history[0, 1] = 1000 + np.arange(HEAD_DIM, dtype=np.int64)

        output, debug = attention_output_q28(probabilities, history, count=1)
        for head in range(7):
            np.testing.assert_array_equal(output[head], history[0, 0])
        for head in range(7, Q_HEADS):
            np.testing.assert_array_equal(output[head], history[0, 1])
        self.assertEqual(debug.saturated_values, 0)

    def test_full_window_uniform_identical_v(self) -> None:
        probabilities = np.full(
            (Q_HEADS, MAX_TOKENS), PROB_ONE // MAX_TOKENS, dtype=np.uint32
        )
        base = np.arange(KV_HEADS * HEAD_DIM, dtype=np.int64).reshape(
            KV_HEADS, HEAD_DIM
        )
        history = np.repeat(base[np.newaxis, :, :], MAX_TOKENS, axis=0)
        output, _ = attention_output_q28(
            probabilities, history, count=MAX_TOKENS
        )
        for head in range(Q_HEADS):
            np.testing.assert_array_equal(output[head], base[head // 7])

    def test_q59_rne_and_saturation(self) -> None:
        probabilities = np.zeros(MAX_TOKENS, dtype=np.uint32)
        probabilities[0] = 1
        history = np.zeros((1, HEAD_DIM), dtype=np.int64)
        history[0, 0] = 1 << 30
        history[0, 1] = 3 << 30
        history[0, 2] = -(1 << 30)
        history[0, 3] = -(3 << 30)
        output, _, saturated = weighted_sum_head_q28(probabilities, history)
        self.assertEqual(output[:4].tolist(), [0, 2, 0, -2])
        self.assertEqual(saturated, 0)

        probabilities[:] = PROB_ONE
        history = np.full((MAX_TOKENS, HEAD_DIM), INT64_MAX, dtype=np.int64)
        output, _, saturated = weighted_sum_head_q28(probabilities, history)
        np.testing.assert_array_equal(output, INT64_MAX)
        self.assertEqual(saturated, HEAD_DIM)

        history.fill(INT64_MIN)
        output, _, saturated = weighted_sum_head_q28(probabilities, history)
        np.testing.assert_array_equal(output, INT64_MIN)
        self.assertEqual(saturated, HEAD_DIM)

    def test_unused_probability_slots_must_be_zero(self) -> None:
        probabilities = np.zeros((Q_HEADS, MAX_TOKENS), dtype=np.uint32)
        probabilities[0, 4] = 1
        history = np.zeros((4, KV_HEADS, HEAD_DIM), dtype=np.int64)
        with self.assertRaises(AttentionOutputReferenceError):
            attention_output_q28(probabilities, history, count=4)

    def test_probability_range_validation(self) -> None:
        probabilities = np.zeros((Q_HEADS, MAX_TOKENS), dtype=np.uint64)
        probabilities[0, 0] = PROB_ONE + 1
        history = np.zeros((1, KV_HEADS, HEAD_DIM), dtype=np.int64)
        with self.assertRaises(AttentionOutputReferenceError):
            attention_output_q28(probabilities, history, count=1)

        signed = np.zeros((Q_HEADS, MAX_TOKENS), dtype=np.int64)
        signed[0, 0] = -1
        with self.assertRaises(AttentionOutputReferenceError):
            attention_output_q28(signed, history, count=1)

    def test_payload_and_concat_roundtrip(self) -> None:
        rng = np.random.default_rng(20260804)
        v = rng.integers(
            -(1 << 40), 1 << 40, size=(KV_HEADS, HEAD_DIM), dtype=np.int64
        )
        v_payload = encode_v_vector(v)
        self.assertEqual(len(v_payload), V_BYTES)
        np.testing.assert_array_equal(decode_v_vector(v_payload), v)

        heads = rng.integers(
            -(1 << 50), 1 << 50, size=(Q_HEADS, HEAD_DIM), dtype=np.int64
        )
        flat = flatten_attention_heads(heads)
        self.assertEqual(flat.shape, (OUTPUT_VALUES,))
        np.testing.assert_array_equal(reshape_attention_heads(flat), heads)
        payload = encode_attention_output(heads)
        self.assertEqual(len(payload), OUTPUT_BYTES)
        np.testing.assert_array_equal(decode_attention_output(payload), heads)

    def test_small_random_stress(self) -> None:
        saturated = software_stress(rounds=50, seed=20260804)
        self.assertGreaterEqual(saturated, 0)

    @unittest.skipUnless(Path(DEFAULT_IMAGE).is_file(), "本地没有真实 .p50 镜像")
    def test_real_manifest(self) -> None:
        cases = build_fixed_real_cases()
        self.assertEqual(len(cases), 4)
        self.assertEqual([case.count for case in cases], [1, 2, 6, 16])
        for case in cases:
            self.assertEqual(
                case.probabilities_q31.shape, (Q_HEADS, MAX_TOKENS)
            )
            self.assertEqual(
                case.v_history_q28.shape,
                (case.count, KV_HEADS, HEAD_DIM),
            )
            self.assertEqual(
                case.expected_heads_q28.shape, (Q_HEADS, HEAD_DIM)
            )
            self.assertEqual(case.debug.saturated_values, 0)
        committed = validate_manifest(cases, DEFAULT_MANIFEST)
        self.assertEqual(
            committed["definition"]["product_format"],
            "signed Q59 exact integer product",
        )
        self.assertEqual(
            committed["definition"]["output_flat_layout"],
            "head-major contiguous [896]",
        )


if __name__ == "__main__":
    unittest.main()
