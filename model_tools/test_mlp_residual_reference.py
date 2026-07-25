#!/usr/bin/env python3
"""G1 MLP 第二处残差软件参考测试。"""

from __future__ import annotations

import unittest

import numpy as np

try:
    from .mlp_residual_reference import (
        DEFAULT_MANIFEST,
        K,
        Q_MAX,
        Q_MIN,
        RESCALE_SHIFT,
        build_fixed_real_cases,
        build_upload_payload,
        mlp_residual_q10,
        rescale_down_q28_to_q10,
        round_shift_rne_signed,
        software_stress,
        validate_manifest,
    )
except ImportError:
    from mlp_residual_reference import (
        DEFAULT_MANIFEST,
        K,
        Q_MAX,
        Q_MIN,
        RESCALE_SHIFT,
        build_fixed_real_cases,
        build_upload_payload,
        mlp_residual_q10,
        rescale_down_q28_to_q10,
        round_shift_rne_signed,
        software_stress,
        validate_manifest,
    )


class MLPResidualReferenceTests(unittest.TestCase):
    def test_signed_rne_ties(self) -> None:
        half = 1 << (RESCALE_SHIFT - 1)
        self.assertEqual(round_shift_rne_signed((2 << RESCALE_SHIFT) + half, RESCALE_SHIFT), 2)
        self.assertEqual(round_shift_rne_signed((3 << RESCALE_SHIFT) + half, RESCALE_SHIFT), 4)
        self.assertEqual(round_shift_rne_signed(-((2 << RESCALE_SHIFT) + half), RESCALE_SHIFT), -2)
        self.assertEqual(round_shift_rne_signed(-((3 << RESCALE_SHIFT) + half), RESCALE_SHIFT), -4)

    def test_q28_to_q10_saturation(self) -> None:
        values = np.zeros(K, dtype=np.int64)
        values[0] = np.iinfo(np.int64).max
        values[1] = np.iinfo(np.int64).min
        converted, saturated = rescale_down_q28_to_q10(values)
        self.assertEqual(int(converted[0]), Q_MAX)
        self.assertEqual(int(converted[1]), Q_MIN)
        self.assertEqual(saturated, 2)

    def test_residual_double_saturation(self) -> None:
        hidden = np.zeros(K, dtype=np.int16)
        hidden[0] = Q_MAX
        hidden[1] = Q_MIN
        down = np.zeros(K, dtype=np.int64)
        down[0] = 2 << RESCALE_SHIFT
        down[1] = -(2 << RESCALE_SHIFT)
        output, scaled, rescale_sat, add_sat = mlp_residual_q10(hidden, down)
        self.assertEqual(int(scaled[0]), 2)
        self.assertEqual(int(scaled[1]), -2)
        self.assertEqual(int(output[0]), Q_MAX)
        self.assertEqual(int(output[1]), Q_MIN)
        self.assertEqual(rescale_sat, 0)
        self.assertEqual(add_sat, 2)

    def test_random_software_stress(self) -> None:
        software_stress(rounds=50, seed=20260817)

    def test_real_coherent_query0_and_manifest(self) -> None:
        cases = build_fixed_real_cases()
        self.assertEqual(len(cases), 4)
        case = cases[0]
        self.assertEqual(case.query_position, 0)
        self.assertEqual(case.count, 1)
        self.assertEqual(case.residual_hidden_q10.shape, (K,))
        self.assertEqual(case.down_proj_q28.shape, (K,))
        self.assertEqual(case.output_q10.shape, (K,))
        self.assertEqual(len(build_upload_payload(case)), K * 10)

        manifest = validate_manifest(cases, DEFAULT_MANIFEST)
        self.assertEqual(manifest["definition"]["fixed_queries"], [0, 1, 5, 15])
        self.assertEqual(
            manifest["definition"]["forbidden_residual_source"],
            "post_attention_layernorm output",
        )


if __name__ == "__main__":
    unittest.main()
