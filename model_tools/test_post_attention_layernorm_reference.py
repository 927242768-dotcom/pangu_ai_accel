#!/usr/bin/env python3
"""G1 layer0 post_attention_layernorm 连贯输入、定点格式与压力测试。"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

try:
    from .p50_format import P50Image
    from .post_attention_layernorm_reference import (
        DEFAULT_GAMMA,
        DEFAULT_IMAGE,
        DEFAULT_MANIFEST,
        K,
        UPLOAD_BYTES,
        build_fixed_real_cases,
        build_upload_payload,
        case_from_input_q10,
        fixed_manifest,
        load_gamma,
        make_random_input_q10,
        software_stress,
        verify_payload_roundtrip,
    )
except ImportError:
    from p50_format import P50Image
    from post_attention_layernorm_reference import (
        DEFAULT_GAMMA,
        DEFAULT_IMAGE,
        DEFAULT_MANIFEST,
        K,
        UPLOAD_BYTES,
        build_fixed_real_cases,
        build_upload_payload,
        case_from_input_q10,
        fixed_manifest,
        load_gamma,
        make_random_input_q10,
        software_stress,
        verify_payload_roundtrip,
    )


@unittest.skipUnless(Path(DEFAULT_IMAGE).is_file(), "本地没有真实 .p50 镜像")
class PostAttentionLayerNormReferenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.image = P50Image(DEFAULT_IMAGE)
        cls.image.validate()
        cls.gamma = load_gamma(cls.image)

    def test_real_gamma_shape_and_storage(self) -> None:
        entry = self.image.tensor(DEFAULT_GAMMA)
        self.assertEqual(tuple(entry["shape"]), (K,))
        self.assertEqual(entry["storage"], "float16")
        self.assertEqual(self.gamma.shape, (K,))

    def test_q10_input_is_preserved_exactly(self) -> None:
        values = np.arange(K, dtype=np.int32)
        values = ((values * 977 + 12345) & 0xFFFF).astype(np.uint16).view(np.int16)
        case = case_from_input_q10(
            input_q10=values,
            gamma_values=self.gamma,
            label="exact q10 roundtrip",
        )
        np.testing.assert_array_equal(case.input_q10, values)
        self.assertEqual(case.input_clipped_count, 0)
        self.assertEqual(case.gamma_clipped_count, 0)

    def test_upload_layout_and_roundtrip(self) -> None:
        rng = np.random.default_rng(20260807)
        values = make_random_input_q10(rng, 1)
        case = case_from_input_q10(
            input_q10=values,
            gamma_values=self.gamma,
            label="payload",
        )
        self.assertEqual(len(build_upload_payload(case)), UPLOAD_BYTES)
        self.assertEqual(len(verify_payload_roundtrip(case)), 64)

    def test_four_coherent_attention_outputs_match_manifest(self) -> None:
        cases = build_fixed_real_cases(image_path=Path(DEFAULT_IMAGE))
        self.assertEqual([case.query_position for case in cases], [0, 1, 5, 15])
        self.assertEqual([case.count for case in cases], [1, 2, 6, 16])
        for case in cases:
            self.assertEqual(case.input_q10.shape, (K,))
            self.assertEqual(case.output_lut_q10.shape, (K,))
            self.assertEqual(case.input_clipped_count, 0)
            self.assertEqual(case.gamma_clipped_count, 0)
            self.assertLessEqual(
                int(
                    np.max(
                        np.abs(
                            case.output_lut_q10.astype(np.int32)
                            - case.output_exact_q10.astype(np.int32)
                        )
                    )
                ),
                3,
            )
        committed = json.loads(Path(DEFAULT_MANIFEST).read_text(encoding="utf-8"))
        self.assertEqual(fixed_manifest(cases), committed)

    def test_random_and_boundary_software_stress_1000(self) -> None:
        max_delta = software_stress(
            image_path=Path(DEFAULT_IMAGE),
            rounds=1000,
            seed=20260807,
        )
        # 全 int16 极值比 E1 的常规激活范围更宽，LUT256 最坏偏差为 8 LSB。
        self.assertLessEqual(max_delta, 8)


if __name__ == "__main__":
    unittest.main()
