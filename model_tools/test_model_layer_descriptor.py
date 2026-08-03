#!/usr/bin/env python3
"""阶段 H1 真实模型层描述表与主机文件偏移测试。"""

from __future__ import annotations

import unittest

try:
    from .model_layer_descriptor import (
        LAYER_TENSOR_ROLES,
        build_model_layer_descriptor,
        expand_all_layers,
        expand_layer_tensors,
        load_reference,
    )
    from .p50_format import P50Image
except ImportError:
    from model_layer_descriptor import (
        LAYER_TENSOR_ROLES,
        build_model_layer_descriptor,
        expand_all_layers,
        expand_layer_tensors,
        load_reference,
    )
    from p50_format import P50Image


class ModelLayerDescriptorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.image = P50Image("model_output/yanbo_qwen25_0.5b_int4.p50")
        cls.descriptor = build_model_layer_descriptor()
        cls.reference = load_reference()

    def test_model_and_hardware_capacities_are_not_confused(self) -> None:
        model = self.descriptor["model"]
        hardware = self.descriptor["hardware_contract"]
        self.assertEqual(model["num_hidden_layers"], 24)
        self.assertEqual(hardware["active_layer_count"], 24)
        self.assertEqual(hardware["layer_capacity"], 28)
        self.assertEqual(hardware["unused_layer_slots"], 4)
        self.assertEqual(model["max_position_embeddings"], 32768)
        self.assertEqual(hardware["model_max_position_embeddings"], 32768)
        self.assertEqual(hardware["max_context"], 16384)
        self.assertTrue(hardware["context_limited_by_hardware"])

    def test_tensor_counts_match_p50_header(self) -> None:
        counts = self.descriptor["tensor_counts"]
        self.assertEqual(counts["global"], 2)
        self.assertEqual(counts["per_layer"], 12)
        self.assertEqual(counts["all_layers"], 288)
        self.assertEqual(counts["total"], 290)
        self.assertEqual(counts["total"], self.image.header.tensor_count)
        self.assertEqual(counts["global"] + counts["all_layers"], counts["total"])

    def test_uniform_layer_layout(self) -> None:
        layout = self.descriptor["layer_layout"]
        self.assertEqual(layout["layer_count"], 24)
        self.assertEqual(layout["tensor_count_per_layer"], 12)
        self.assertEqual(layout["first_layer_base_offset"], 72_851_456)
        self.assertEqual(layout["layer_stride_bytes"], 7_958_528)
        self.assertEqual(layout["layer_payload_span_bytes"], 7_955_456)
        self.assertEqual(layout["layer_alignment_gap_bytes"], 3_072)
        self.assertEqual(len(layout["layer_bases"]), 24)
        self.assertEqual(layout["layer_bases"][0], 72_851_456)
        self.assertEqual(layout["layer_bases"][-1], 255_897_600)
        self.assertEqual(
            {
                right - left
                for left, right in zip(
                    layout["layer_bases"], layout["layer_bases"][1:]
                )
            },
            {7_958_528},
        )

    def test_template_roles_shapes_and_quantization(self) -> None:
        templates = self.descriptor["tensor_templates"]
        self.assertEqual(
            [item["role"] for item in templates],
            [role for role, _ in LAYER_TENSOR_ROLES],
        )
        by_role = {item["role"]: item for item in templates}
        self.assertEqual(by_role["input_layernorm_weight"]["shape"], [896])
        self.assertEqual(by_role["q_proj_weight"]["shape"], [896, 896])
        self.assertEqual(by_role["k_proj_weight"]["shape"], [128, 896])
        self.assertEqual(by_role["v_proj_weight"]["shape"], [128, 896])
        self.assertEqual(by_role["o_proj_weight"]["shape"], [896, 896])
        self.assertEqual(by_role["gate_proj_weight"]["shape"], [4864, 896])
        self.assertEqual(by_role["up_proj_weight"]["shape"], [4864, 896])
        self.assertEqual(by_role["down_proj_weight"]["shape"], [896, 4864])
        self.assertEqual(by_role["down_proj_weight"]["groups_per_row"], 76)
        for role in (
            "q_proj_weight",
            "k_proj_weight",
            "v_proj_weight",
            "o_proj_weight",
            "gate_proj_weight",
            "up_proj_weight",
        ):
            self.assertEqual(by_role[role]["groups_per_row"], 14)
            self.assertEqual(by_role[role]["padded_columns"], 896)
        self.assertEqual(by_role["down_proj_weight"]["padded_columns"], 4864)
        for item in templates:
            if item["role"].endswith("weight") and item["role"] not in {
                "input_layernorm_weight",
                "post_attention_layernorm_weight",
            }:
                self.assertEqual(item["storage"], "int4_groupwise_symmetric")
            else:
                self.assertEqual(item["storage"], "float16")

    def test_all_288_layer_tensors_expand_to_exact_p50_entries(self) -> None:
        layers = expand_all_layers(self.descriptor)
        self.assertEqual(len(layers), 24)
        expanded_count = 0
        for layer in layers:
            layer_index = layer["layer_index"]
            tensors = layer["tensors"]
            self.assertEqual(len(tensors), 12)
            self.assertEqual(
                layer["base_offset"],
                self.descriptor["layer_layout"]["layer_bases"][layer_index],
            )
            for expanded in tensors:
                entry = self.image.tensor(expanded["name"])
                for key in (
                    "shape",
                    "source_dtype",
                    "storage",
                    "data_offset",
                    "data_nbytes",
                ):
                    self.assertEqual(expanded[key], entry[key], (expanded["name"], key))
                for key in (
                    "scale_offset",
                    "scale_nbytes",
                    "padded_columns",
                    "groups_per_row",
                ):
                    if key in entry:
                        self.assertEqual(
                            expanded[key], entry[key], (expanded["name"], key)
                        )
                    else:
                        self.assertNotIn(key, expanded)
                expanded_count += 1
        self.assertEqual(expanded_count, 288)

    def test_global_tensors_and_tied_lm_head(self) -> None:
        globals_by_role = {
            item["role"]: item for item in self.descriptor["global_tensors"]
        }
        embedding = globals_by_role["embedding_and_tied_lm_head_weight"]
        final_norm = globals_by_role["final_norm_weight"]
        self.assertEqual(embedding["name"], "model.embed_tokens.weight")
        self.assertEqual(embedding["shape"], [151936, 896])
        self.assertEqual(embedding["data_offset"], 528_384)
        self.assertEqual(embedding["scale_offset"], 68_595_712)
        self.assertEqual(final_norm["name"], "model.norm.weight")
        self.assertEqual(final_norm["shape"], [896])
        self.assertEqual(final_norm["data_offset"], 263_856_128)
        self.assertEqual(
            self.descriptor["lm_head"],
            {
                "separate_tensor_present": False,
                "tied_to": "model.embed_tokens.weight",
            },
        )

    def test_last_layer_ends_before_final_norm_without_overlap(self) -> None:
        last_layer = expand_layer_tensors(self.descriptor, 23)
        last_end = max(
            max(
                item["data_offset"] + item["data_nbytes"],
                item.get("scale_offset", 0) + item.get("scale_nbytes", 0),
            )
            for item in last_layer
        )
        final_norm_offset = self.image.tensor("model.norm.weight")["data_offset"]
        self.assertEqual(last_end, 263_853_056)
        self.assertLess(last_end, final_norm_offset)
        self.assertEqual(final_norm_offset - last_end, 3_072)

    def test_layer_index_bounds(self) -> None:
        with self.assertRaises(IndexError):
            expand_layer_tensors(self.descriptor, -1)
        with self.assertRaises(IndexError):
            expand_layer_tensors(self.descriptor, 24)

    def test_frozen_reference_matches_real_image(self) -> None:
        self.assertEqual(self.descriptor, self.reference)
        self.assertEqual(
            self.descriptor["image"]["metadata_sha256"],
            "f3746774d10cde045c21ba04bca47fabb409256801d91a5a2ffe6142c3c013bd",
        )


if __name__ == "__main__":
    unittest.main()
