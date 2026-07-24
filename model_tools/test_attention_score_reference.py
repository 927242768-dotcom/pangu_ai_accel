#!/usr/bin/env python3
"""``attention_score_reference`` 的定点、GQA、mask、载荷和真实模型测试。"""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

try:
    from .attention_score_reference import (
        DEFAULT_IMAGE,
        DEFAULT_MANIFEST,
        GQA_GROUP_SIZE,
        HEAD_DIM,
        KV_HEADS,
        MASK_VALUE,
        MAX_TOKENS,
        OUTPUT_SHIFT,
        Q_HEADS,
        SCORE_BYTES,
        AttentionScoreReferenceError,
        attention_scores_q28,
        build_fixed_real_cases,
        build_k_payload,
        build_q_payload,
        decode_k_payload,
        decode_q_payload,
        decode_scores,
        encode_scores,
        gqa_kv_head,
        round_shift_rne,
        scaled_dot_q28,
        software_stress,
        validate_manifest,
        validate_window,
    )
except ImportError:
    from attention_score_reference import (
        DEFAULT_IMAGE,
        DEFAULT_MANIFEST,
        GQA_GROUP_SIZE,
        HEAD_DIM,
        KV_HEADS,
        MASK_VALUE,
        MAX_TOKENS,
        OUTPUT_SHIFT,
        Q_HEADS,
        SCORE_BYTES,
        AttentionScoreReferenceError,
        attention_scores_q28,
        build_fixed_real_cases,
        build_k_payload,
        build_q_payload,
        decode_k_payload,
        decode_q_payload,
        decode_scores,
        encode_scores,
        gqa_kv_head,
        round_shift_rne,
        scaled_dot_q28,
        software_stress,
        validate_manifest,
        validate_window,
    )


class AttentionScoreReferenceTest(unittest.TestCase):
    def test_model_shape_and_gqa_mapping(self) -> None:
        self.assertEqual(Q_HEADS, 14)
        self.assertEqual(KV_HEADS, 2)
        self.assertEqual(HEAD_DIM, 64)
        self.assertEqual(GQA_GROUP_SIZE, 7)
        self.assertEqual([gqa_kv_head(index) for index in range(14)], [0] * 7 + [1] * 7)

    def test_rne_ties_positive_and_negative(self) -> None:
        shift = 3
        self.assertEqual(round_shift_rne(4, shift), 0)
        self.assertEqual(round_shift_rne(12, shift), 2)
        self.assertEqual(round_shift_rne(20, shift), 2)
        self.assertEqual(round_shift_rne(28, shift), 4)
        self.assertEqual(round_shift_rne(-4, shift), 0)
        self.assertEqual(round_shift_rne(-12, shift), -2)
        self.assertEqual(round_shift_rne(-20, shift), -2)
        self.assertEqual(round_shift_rne(-28, shift), -4)

    def test_scaled_dot_exact_power_of_two(self) -> None:
        q = np.zeros(HEAD_DIM, dtype=np.int64)
        k = np.zeros(HEAD_DIM, dtype=np.int64)
        q[0] = 1 << 28
        k[0] = 1 << 28
        self.assertEqual(OUTPUT_SHIFT, 31)
        self.assertEqual(scaled_dot_q28(q, k), 1 << 25)

    def test_causal_mask_and_fixed_output_slots(self) -> None:
        q = np.ones((Q_HEADS, HEAD_DIM), dtype=np.int64) << 28
        history = np.ones((4, KV_HEADS, HEAD_DIM), dtype=np.int64) << 28
        scores = attention_scores_q28(
            q,
            history,
            query_position=11,
            window_start=10,
            count=4,
        )
        expected_valid = 8 << 28
        np.testing.assert_array_equal(scores[:, :2], expected_valid)
        np.testing.assert_array_equal(scores[:, 2:4], MASK_VALUE)
        np.testing.assert_array_equal(scores[:, 4:], MASK_VALUE)

    def test_gqa_heads_use_expected_kv_head(self) -> None:
        q = np.zeros((Q_HEADS, HEAD_DIM), dtype=np.int64)
        q[:, 0] = 1 << 28
        history = np.zeros((1, KV_HEADS, HEAD_DIM), dtype=np.int64)
        history[0, 0, 0] = 2 << 28
        history[0, 1, 0] = 5 << 28
        scores = attention_scores_q28(
            q,
            history,
            query_position=0,
            window_start=0,
            count=1,
        )
        np.testing.assert_array_equal(scores[:7, 0], 1 << 26)
        np.testing.assert_array_equal(scores[7:, 0], 5 << 25)

    def test_payload_roundtrip(self) -> None:
        rng = np.random.default_rng(20260802)
        q = rng.integers(-(1 << 40), 1 << 40, size=(Q_HEADS, HEAD_DIM), dtype=np.int64)
        k = rng.integers(-(1 << 40), 1 << 40, size=(KV_HEADS, HEAD_DIM), dtype=np.int64)
        scores = rng.integers(-(1 << 40), 1 << 40, size=(Q_HEADS, MAX_TOKENS), dtype=np.int64)
        np.testing.assert_array_equal(decode_q_payload(build_q_payload(q)), q)
        np.testing.assert_array_equal(decode_k_payload(build_k_payload(k)), k)
        payload = encode_scores(scores)
        self.assertEqual(len(payload), SCORE_BYTES)
        np.testing.assert_array_equal(decode_scores(payload), scores)

    def test_invalid_window_and_head_raise(self) -> None:
        for query, start, count in ((-1, 0, 1), (0, -1, 1), (0, 0, 0), (0, 0, 17)):
            with self.assertRaises(AttentionScoreReferenceError):
                validate_window(query, start, count)
        with self.assertRaises(AttentionScoreReferenceError):
            gqa_kv_head(Q_HEADS)

    def test_small_random_stress(self) -> None:
        software_stress(rounds=25, seed=20260802)

    @unittest.skipUnless(Path(DEFAULT_IMAGE).is_file(), "本地没有真实 .p50 镜像")
    def test_real_manifest(self) -> None:
        cases = build_fixed_real_cases()
        self.assertEqual(len(cases), 4)
        for case in cases:
            self.assertEqual(case.q_q28.shape, (Q_HEADS, HEAD_DIM))
            self.assertEqual(case.k_history_q28.shape, (case.count, KV_HEADS, HEAD_DIM))
            self.assertEqual(case.expected_scores_q28.shape, (Q_HEADS, MAX_TOKENS))
        committed = validate_manifest(cases, DEFAULT_MANIFEST)
        self.assertEqual(committed["definition"]["output_layout"], "head-major [14,16]")
        self.assertEqual(committed["definition"]["mask_value"], MASK_VALUE)


if __name__ == "__main__":
    unittest.main()
