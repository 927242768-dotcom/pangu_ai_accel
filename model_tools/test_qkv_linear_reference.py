#!/usr/bin/env python3
"""``qkv_linear_reference`` 的布局、载荷和真实 P50 集成测试。"""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

try:
    from .linear_quant_reference import DEFAULT_ACTIVATION_SEED
    from .p50_format import P50Image
    from .qkv_linear_reference import (
        DEFAULT_IMAGE,
        DEFAULT_MANIFEST,
        HEAD_DIM,
        K,
        KV_HEADS,
        PROJECTION_SPECS,
        Q_HEADS,
        build_qkv_cases,
        build_upload_payload,
        load_qkv_models,
        projection_sequence,
        projection_spec,
        qkv_manifest,
        reshape_heads,
        validate_gqa_layout,
        validate_manifest,
        verify_payload_roundtrip,
    )
except ImportError:
    from linear_quant_reference import DEFAULT_ACTIVATION_SEED
    from p50_format import P50Image
    from qkv_linear_reference import (
        DEFAULT_IMAGE,
        DEFAULT_MANIFEST,
        HEAD_DIM,
        K,
        KV_HEADS,
        PROJECTION_SPECS,
        Q_HEADS,
        build_qkv_cases,
        build_upload_payload,
        load_qkv_models,
        projection_sequence,
        projection_spec,
        qkv_manifest,
        reshape_heads,
        validate_gqa_layout,
        validate_manifest,
        verify_payload_roundtrip,
    )


class QKVLinearReferenceTest(unittest.TestCase):
    def test_projection_specs_match_gqa_configuration(self) -> None:
        q = projection_spec("q")
        k = projection_spec("K")
        v = projection_spec("v")
        self.assertEqual((q.rows, q.heads), (Q_HEADS * HEAD_DIM, Q_HEADS))
        self.assertEqual((k.rows, k.heads), (KV_HEADS * HEAD_DIM, KV_HEADS))
        self.assertEqual((v.rows, v.heads), (KV_HEADS * HEAD_DIM, KV_HEADS))
        self.assertEqual(q.upload_bytes, 488320)
        self.assertEqual(k.upload_bytes, 70528)
        self.assertEqual(v.upload_bytes, 70528)
        self.assertEqual([item.key for item in projection_sequence("all")], ["q", "k", "v"])

    def test_head_major_layout_roundtrip(self) -> None:
        for spec in PROJECTION_SPECS.values():
            flat = np.arange(spec.rows, dtype=np.int64)
            heads = reshape_heads(flat, spec)
            self.assertEqual(heads.shape, (spec.heads, HEAD_DIM))
            np.testing.assert_array_equal(heads[0], np.arange(HEAD_DIM))
            np.testing.assert_array_equal(heads.reshape(-1), flat)

    @unittest.skipUnless(Path(DEFAULT_IMAGE).is_file(), "本地没有真实 .p50 镜像")
    def test_real_qkv_fixed_reference_and_manifest(self) -> None:
        image = P50Image(DEFAULT_IMAGE)
        image.validate()
        models = load_qkv_models(image)
        cases = build_qkv_cases(models, activation_seed=DEFAULT_ACTIVATION_SEED)
        validate_gqa_layout(cases)
        self.assertEqual(cases["q"].weights.shape, (896, K))
        self.assertEqual(cases["k"].weights.shape, (128, K))
        self.assertEqual(cases["v"].weights.shape, (128, K))
        self.assertTrue(np.array_equal(cases["q"].activation, cases["k"].activation))
        self.assertTrue(np.array_equal(cases["q"].activation, cases["v"].activation))
        for key in ("q", "k", "v"):
            case = cases[key]
            self.assertEqual(len(build_upload_payload(case)), case.spec.upload_bytes)
            self.assertEqual(len(verify_payload_roundtrip(case)), 64)
        generated = qkv_manifest(cases, DEFAULT_ACTIVATION_SEED)
        committed = validate_manifest(cases, DEFAULT_MANIFEST, DEFAULT_ACTIVATION_SEED)
        self.assertEqual(generated, committed)
        self.assertEqual(
            generated["projections"]["q"]["sha256"]["output_fixed_q28"],
            "ea1f04bf4ff313dad07025ff35e66a088f13afd28d817422b89bb135f63525a0",
        )
        self.assertEqual(
            generated["projections"]["k"]["sha256"]["output_fixed_q28"],
            "20728d329c32c722b0194032897bc3cf9a3a31323317e389d8fd7b6f78745474",
        )
        self.assertEqual(
            generated["projections"]["v"]["sha256"]["output_fixed_q28"],
            "162622e05e0013ca342f28032cb280c264f428f93a197eb67dbfafd76e20a168",
        )


if __name__ == "__main__":
    unittest.main()
