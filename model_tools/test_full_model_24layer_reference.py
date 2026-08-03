#!/usr/bin/env python3
"""真实 24 层单 token 软件金标准的快速回归。"""

from __future__ import annotations

import unittest
from pathlib import Path

try:
    from .full_model_24layer_reference import (
        ACTIVE_LAYER_COUNT,
        FullModel24LayerReferenceError,
        build_24layer_sequence,
        build_initial_hidden_q10,
        build_single_token_layer_case,
        case_tensor_hashes,
        layer_projection_specs,
        load_layer_context,
        load_reference,
        sequence_manifest,
        tensor_hash_set_sha256,
        validate_layer_index,
    )
    from .full_model_layer_sequence import load_reference as load_sequence_reference
    from .p50_format import P50Image
    from .transformer_block_reference import build_fixed_real_cases, sha256_array
except ImportError:
    from full_model_24layer_reference import (
        ACTIVE_LAYER_COUNT,
        FullModel24LayerReferenceError,
        build_24layer_sequence,
        build_initial_hidden_q10,
        build_single_token_layer_case,
        case_tensor_hashes,
        layer_projection_specs,
        load_layer_context,
        load_reference,
        sequence_manifest,
        tensor_hash_set_sha256,
        validate_layer_index,
    )
    from full_model_layer_sequence import load_reference as load_sequence_reference
    from p50_format import P50Image
    from transformer_block_reference import build_fixed_real_cases, sha256_array


IMAGE = Path("model_output/yanbo_qwen25_0.5b_int4.p50")


class FullModel24LayerReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.image = P50Image(IMAGE)
        cls.image.validate()
        cls.layer0_context = load_layer_context(cls.image, 0)
        cls.initial = build_initial_hidden_q10(cls.layer0_context)
        cls.new_layer0 = build_single_token_layer_case(
            cls.layer0_context,
            cls.initial,
        )

    def test_real_layer_range_and_dynamic_tensor_names(self) -> None:
        self.assertEqual(ACTIVE_LAYER_COUNT, 24)
        self.assertEqual([validate_layer_index(i) for i in range(24)], list(range(24)))
        for invalid in (-1, 24, 27):
            with self.assertRaises(FullModel24LayerReferenceError):
                validate_layer_index(invalid)
        specs = layer_projection_specs(23)
        self.assertEqual(
            specs["q"].weight_name,
            "model.layers.23.self_attn.q_proj.weight",
        )
        self.assertEqual(
            specs["k"].bias_name,
            "model.layers.23.self_attn.k_proj.bias",
        )
        self.assertEqual(specs["v"].rows, 128)

    def test_layer0_is_exactly_the_verified_g2_query0_case(self) -> None:
        old = build_fixed_real_cases(queries=(0,))[0]
        self.assertEqual(
            sha256_array(self.initial, "<i2"),
            "26139d5cacc3a2c2cf018016f370effd02e043b0d2155f89573463683fba80f0",
        )
        self.assertEqual(case_tensor_hashes(self.new_layer0), case_tensor_hashes(old))
        self.assertEqual(
            sha256_array(self.new_layer0.output_q10, "<i2"),
            "630952eaf6fe179639773f2b60d1e9e990f380b0e698fa3d493dd7c279c96104",
        )

    def test_layer1_consumes_layer0_output_and_uses_layer1_parameters(self) -> None:
        context = load_layer_context(self.image, 1)
        case = build_single_token_layer_case(context, self.new_layer0.output_q10)
        self.assertEqual(context.layer_index, 1)
        self.assertEqual(
            context.block.attention.qkv_models["q"].spec.weight_name,
            "model.layers.1.self_attn.q_proj.weight",
        )
        self.assertEqual(
            context.block.attention.oproj_model.weight_name,
            "model.layers.1.self_attn.o_proj.weight",
        )
        self.assertEqual(
            context.block.down_model.weight_name,
            "model.layers.1.mlp.down_proj.weight",
        )
        self.assertEqual(
            sha256_array(case.block_input_q10, "<i2"),
            "630952eaf6fe179639773f2b60d1e9e990f380b0e698fa3d493dd7c279c96104",
        )
        self.assertEqual(
            sha256_array(case.output_q10, "<i2"),
            "884aa9a403d6ae98c5c86c22f9bfdaa37f75441c130745a83fb3c83090e1f8c9",
        )

    def test_two_layer_sequence_matches_frozen_prefix(self) -> None:
        sequence = build_24layer_sequence(end_layer=1)
        manifest = sequence_manifest(sequence)
        reference = load_reference()
        self.assertEqual(
            manifest["layer_output_sha256"],
            reference["layer_output_sha256"][:2],
        )
        self.assertEqual(
            manifest["layer_tensor_set_sha256"],
            reference["layer_tensor_set_sha256"][:2],
        )
        self.assertEqual(
            manifest["layer_parameter_upload_set_sha256"],
            reference["layer_parameter_upload_set_sha256"][:2],
        )
        with self.assertRaises(FullModel24LayerReferenceError):
            build_24layer_sequence(start_layer=1, end_layer=1)

    def test_reference_freezes_24_layer_hidden_chain(self) -> None:
        reference = load_reference()
        outputs = reference["layer_output_sha256"]
        self.assertEqual(len(outputs), 24)
        self.assertEqual(reference["definition"]["layer_range"], [0, 23])
        self.assertEqual(reference["layer_tensor_count"], 18)
        self.assertEqual(len(reference["layer_tensor_set_sha256"]), 24)
        self.assertEqual(len(set(reference["layer_tensor_set_sha256"])), 24)
        self.assertEqual(reference["final_hidden_sha256"], outputs[-1])
        self.assertEqual(
            outputs[-1],
            "e9708deff4856b400fb953575288fdceab6bfef6a895f15739ac18b488f5619a",
        )

    def test_parameter_hashes_match_h3_layer_upload_contract(self) -> None:
        reference = load_reference()
        sequence_reference = load_sequence_reference()
        self.assertEqual(
            reference["layer_parameter_upload_set_sha256"],
            sequence_reference["layer_upload_set_sha256"],
        )

    def test_only_real_nonzero_saturations_are_frozen(self) -> None:
        self.assertEqual(
            load_reference()["saturation_events"],
            [
                {"layer_index": 17, "stage": "second_residual", "count": 1},
                {"layer_index": 18, "stage": "first_residual", "count": 1},
                {"layer_index": 20, "stage": "first_residual", "count": 1},
                {"layer_index": 21, "stage": "first_residual", "count": 1},
                {"layer_index": 21, "stage": "second_residual", "count": 1},
                {"layer_index": 22, "stage": "second_residual", "count": 1},
                {"layer_index": 23, "stage": "first_residual", "count": 1},
            ],
        )

    def test_tensor_set_digest_covers_all_18_named_tensors(self) -> None:
        hashes = case_tensor_hashes(self.new_layer0)
        self.assertEqual(len(hashes), 18)
        self.assertEqual(
            tensor_hash_set_sha256(hashes),
            load_reference()["layer_tensor_set_sha256"][0],
        )

    def test_reference_explicitly_excludes_later_model_stages(self) -> None:
        self.assertEqual(
            load_reference()["definition"]["excluded"],
            ["embedding", "final_rmsnorm", "lm_head", "logits", "sampling"],
        )


if __name__ == "__main__":
    unittest.main()
