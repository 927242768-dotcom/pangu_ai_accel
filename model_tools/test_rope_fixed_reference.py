#!/usr/bin/env python3
"""``rope_fixed_reference`` 的配置、配对、RNE、载荷和真实模型测试。"""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

try:
    from .linear_quant_reference import DEFAULT_ACTIVATION_SEED
    from .rope_fixed_reference import (
        DEFAULT_IMAGE,
        DEFAULT_MANIFEST,
        DEFAULT_POSITIONS,
        HALF_DIM,
        HEAD_DIM,
        INPUT_BYTES,
        K_VALUES,
        MAX_POSITION_EMBEDDINGS,
        Q28_SCALE,
        Q_HEADS,
        Q_VALUES,
        ROPE_THETA,
        ROTARY_DIM,
        TRIG_ROW_BYTES,
        apply_rope_fixed_q28,
        build_real_rope_cases,
        build_rope_case,
        build_upload_payload,
        generate_trig_row,
        load_rope_config,
        round_shift_rne,
        software_stress,
        validate_manifest,
        verify_payload_roundtrip,
    )
except ImportError:
    from linear_quant_reference import DEFAULT_ACTIVATION_SEED
    from rope_fixed_reference import (
        DEFAULT_IMAGE,
        DEFAULT_MANIFEST,
        DEFAULT_POSITIONS,
        HALF_DIM,
        HEAD_DIM,
        INPUT_BYTES,
        K_VALUES,
        MAX_POSITION_EMBEDDINGS,
        Q28_SCALE,
        Q_HEADS,
        Q_VALUES,
        ROPE_THETA,
        ROTARY_DIM,
        TRIG_ROW_BYTES,
        apply_rope_fixed_q28,
        build_real_rope_cases,
        build_rope_case,
        build_upload_payload,
        generate_trig_row,
        load_rope_config,
        round_shift_rne,
        software_stress,
        validate_manifest,
        verify_payload_roundtrip,
    )


class RoPEFixedReferenceTest(unittest.TestCase):
    def test_model_configuration(self) -> None:
        config = load_rope_config()
        self.assertEqual(config["head_dim"], HEAD_DIM)
        self.assertEqual(config["rotary_dim"], ROTARY_DIM)
        self.assertEqual(config["rope_theta"], ROPE_THETA)
        self.assertEqual(
            config["max_position_embeddings"], MAX_POSITION_EMBEDDINGS
        )
        self.assertEqual(HALF_DIM, 32)
        self.assertEqual(Q_VALUES, 896)
        self.assertEqual(K_VALUES, 128)
        self.assertEqual(INPUT_BYTES, 8192)
        self.assertEqual(TRIG_ROW_BYTES, 256)

    def test_round_shift_rne_positive_and_negative_ties(self) -> None:
        self.assertEqual(round_shift_rne(10, 2), 2)   # 2.5 -> 偶数 2
        self.assertEqual(round_shift_rne(14, 2), 4)   # 3.5 -> 偶数 4
        self.assertEqual(round_shift_rne(-10, 2), -2)
        self.assertEqual(round_shift_rne(-14, 2), -4)
        self.assertEqual(round_shift_rne(11, 2), 3)
        self.assertEqual(round_shift_rne(-11, 2), -3)

    def test_position_zero_is_exact_identity(self) -> None:
        rng = np.random.default_rng(20260730)
        q = rng.integers(
            -(1 << 40), 1 << 40, size=(Q_HEADS, HEAD_DIM), dtype=np.int64
        )
        k = rng.integers(
            -(1 << 40), 1 << 40, size=(2, HEAD_DIM), dtype=np.int64
        )
        case = build_rope_case(q, k, 0)
        np.testing.assert_array_equal(case.q_output_q28, q)
        np.testing.assert_array_equal(case.k_output_q28, k)
        self.assertEqual(case.max_abs_error, 0.0)

    def test_split_half_pairing_not_adjacent_pairing(self) -> None:
        values = np.zeros((1, HEAD_DIM), dtype=np.int64)
        values[0, 0] = Q28_SCALE
        values[0, HALF_DIM] = 2 * Q28_SCALE
        trig = generate_trig_row(1)
        output = apply_rope_fixed_q28(values, trig, heads=1)

        cos_q30 = int(trig.cos_q30[0])
        sin_q30 = int(trig.sin_q30[0])
        expected_first = round_shift_rne(
            Q28_SCALE * cos_q30 - 2 * Q28_SCALE * sin_q30, 30
        )
        expected_second = round_shift_rne(
            2 * Q28_SCALE * cos_q30 + Q28_SCALE * sin_q30, 30
        )
        self.assertEqual(int(output[0, 0]), expected_first)
        self.assertEqual(int(output[0, HALF_DIM]), expected_second)
        self.assertEqual(int(output[0, 1]), 0)

    def test_payload_roundtrip(self) -> None:
        q = np.arange(Q_VALUES, dtype=np.int64).reshape(Q_HEADS, HEAD_DIM)
        k = -np.arange(K_VALUES, dtype=np.int64).reshape(2, HEAD_DIM)
        positions = (0, 1, 2026)
        payload = build_upload_payload(q, k, positions)
        self.assertEqual(len(payload), INPUT_BYTES + len(positions) * TRIG_ROW_BYTES)
        self.assertEqual(len(verify_payload_roundtrip(q, k, positions)), 64)

    def test_small_random_error_stress(self) -> None:
        software_stress(rounds=20, seed=20260730)

    @unittest.skipUnless(Path(DEFAULT_IMAGE).is_file(), "本地没有真实 .p50 镜像")
    def test_real_layer0_qk_manifest(self) -> None:
        cases = build_real_rope_cases(
            DEFAULT_POSITIONS,
            activation_seed=DEFAULT_ACTIVATION_SEED,
        )
        self.assertEqual([case.position for case in cases], list(DEFAULT_POSITIONS))
        self.assertEqual(cases[0].q_output_q28.shape, (Q_HEADS, HEAD_DIM))
        self.assertEqual(cases[0].k_output_q28.shape, (2, HEAD_DIM))
        committed = validate_manifest(
            cases,
            DEFAULT_MANIFEST,
            activation_seed=DEFAULT_ACTIVATION_SEED,
        )
        self.assertEqual(committed["pairing_rule"]["name"], "qwen2_split_half_rotate_half")
        self.assertEqual(
            committed["cases"][2]["sha256"]["q_output_q28"],
            "6c266ff09ef200af907da2796b8fb1db4e5c050f0cad15ccb62e318a5953b0d6",
        )
        self.assertEqual(
            committed["cases"][2]["sha256"]["k_output_q28"],
            "0f8625c3063eb62726c7b3bfc933af4d70652014cd4b63a0ba772916a4c02622",
        )


if __name__ == "__main__":
    unittest.main()
