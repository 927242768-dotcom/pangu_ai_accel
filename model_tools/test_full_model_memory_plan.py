#!/usr/bin/env python3
"""阶段 H2 完整模型内存与按层参数事务测试。"""

from __future__ import annotations

import unittest

try:
    from .full_model_memory_plan import (
        DDR_BYTES,
        SLOT_A_BASE,
        SLOT_B_BASE,
        build_full_model_memory_plan,
        expand_layer_transfer_plan,
        load_reference,
        reference_snapshot,
    )
    from .model_layer_descriptor import build_model_layer_descriptor
    from .p50_format import P50Image
except ImportError:
    from full_model_memory_plan import (
        DDR_BYTES,
        SLOT_A_BASE,
        SLOT_B_BASE,
        build_full_model_memory_plan,
        expand_layer_transfer_plan,
        load_reference,
        reference_snapshot,
    )
    from model_layer_descriptor import build_model_layer_descriptor
    from p50_format import P50Image


class FullModelMemoryPlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.plan = build_full_model_memory_plan()
        cls.descriptor = build_model_layer_descriptor()
        cls.image = P50Image("model_output/yanbo_qwen25_0.5b_int4.p50")

    def test_partitions_cover_exactly_one_gib_without_gaps(self) -> None:
        regions = self.plan["ddr"]["partitions"]
        self.assertEqual(regions[0]["byte_start"], 0)
        self.assertEqual(regions[-1]["byte_end"], DDR_BYTES)
        for left, right in zip(regions, regions[1:]):
            self.assertEqual(left["byte_end"], right["byte_start"])
        self.assertEqual(sum(item["size_bytes"] for item in regions), DDR_BYTES)

    def test_real_24_layer_kv_and_top_global_region(self) -> None:
        kv = self.plan["kv_cache"]
        self.assertEqual(kv["active_layers"], 24)
        self.assertEqual(kv["unused_capacity_layer_slots"], 4)
        self.assertEqual(kv["byte_start"], 0x0800_0000)
        self.assertEqual(kv["byte_end"], 0x3800_0000)
        self.assertEqual(kv["size_bytes"], 768 << 20)
        self.assertEqual(kv["layer_stride_bytes"], 32 << 20)
        self.assertEqual(kv["max_context"], 16_384)
        self.assertEqual(self.plan["global_layout"]["byte_start"], 0x3800_0000)

    def test_existing_g2_regions_remain_in_compatible_partitions(self) -> None:
        for region in self.plan["runtime"]["existing_regions"]:
            self.assertGreaterEqual(region["byte_start"], 0)
            self.assertLessEqual(region["byte_end"], 0x0100_0000)
        for region in self.plan["layer_parameter_slots"]["existing_slot_a_regions"]:
            self.assertGreaterEqual(region["byte_start"], SLOT_A_BASE)
            self.assertLessEqual(region["byte_end"], SLOT_A_BASE + 0x0100_0000)
        self.assertEqual(
            self.plan["layer_parameter_slots"]["slot_a_used_span_bytes"],
            9_065_472,
        )
        self.assertEqual(
            self.plan["layer_parameter_slots"]["slot_a_free_bytes"],
            7_711_744,
        )

    def test_global_resident_and_lm_head_buffers_fit(self) -> None:
        regions = {item["name"]: item for item in self.plan["global_layout"]["regions"]}
        self.assertEqual(regions["embedding_lm_head_int4"]["byte_start"], 0x3800_0000)
        self.assertEqual(regions["embedding_lm_head_int4"]["size_bytes"], 68_067_328)
        self.assertEqual(
            regions["embedding_lm_head_scale_fp16"]["size_bytes"], 4_254_208
        )
        self.assertEqual(regions["final_norm_gamma_q10"]["size_bytes"], 1_792)
        self.assertEqual(
            regions["lm_head_combined_scale_uq4_28"]["size_bytes"],
            151_936 * 14 * 4,
        )
        self.assertEqual(regions["lm_head_logits_q28"]["size_bytes"], 151_936 * 8)
        self.assertLessEqual(self.plan["global_layout"]["byte_end"], DDR_BYTES)
        self.assertEqual(self.plan["ddr"]["partitions"][-1]["size_bytes"], 52_162_560)

    def test_layer_transfer_template_has_19_exact_transactions(self) -> None:
        streaming = self.plan["layer_streaming"]
        template = streaming["transfer_template"]
        self.assertEqual(len(template), 19)
        self.assertEqual(streaming["source_bytes_per_layer"], 7_926_528)
        self.assertEqual(streaming["destination_bytes_per_layer"], 7_961_088)
        self.assertEqual(
            {item["transform"] for item in template},
            {
                "copy_int4",
                "copy_fp16_scale",
                "fp16_to_q6_10",
                "fp16_bias_to_q28_row32",
            },
        )
        self.assertEqual(
            [item["source_role"] for item in template].count("q_proj_weight"), 2
        )
        self.assertEqual(
            [item["source_role"] for item in template].count("down_proj_weight"),
            2,
        )

    def test_all_layer_source_offsets_match_real_p50(self) -> None:
        role_to_suffix = {
            item["role"]: item["name_suffix"]
            for item in self.descriptor["tensor_templates"]
        }
        total = 0
        for layer_index in range(24):
            transfers = expand_layer_transfer_plan(self.plan, layer_index, slot="A")
            self.assertEqual(len(transfers), 19)
            for item in transfers:
                tensor = self.image.tensor(
                    f"model.layers.{layer_index}.{role_to_suffix[item['source_role']]}"
                )
                expected = (
                    tensor["data_offset"]
                    if item["source_component"] == "data"
                    else tensor["scale_offset"]
                )
                self.assertEqual(item["source_byte_offset"], expected)
                self.assertLessEqual(
                    item["source_byte_offset"] + item["source_nbytes"],
                    self.image.file_size,
                )
                total += 1
        self.assertEqual(total, 24 * 19)

    def test_slot_b_only_shifts_parameter_destinations(self) -> None:
        slot_a = expand_layer_transfer_plan(self.plan, 23, slot="A")
        slot_b = expand_layer_transfer_plan(self.plan, 23, slot="B")
        self.assertEqual(len(slot_a), len(slot_b))
        for left, right in zip(slot_a, slot_b):
            self.assertEqual(left["source_byte_offset"], right["source_byte_offset"])
            if "destination_offset_in_slot" in left:
                self.assertEqual(
                    right["destination_byte_address"] - left["destination_byte_address"],
                    SLOT_B_BASE - SLOT_A_BASE,
                )
            else:
                self.assertEqual(
                    right["destination_byte_address"], left["destination_byte_address"]
                )

    def test_hidden_handoff_uses_two_existing_1792_byte_regions(self) -> None:
        handoff = self.plan["runtime"]["hidden_handoff"]
        self.assertEqual(handoff["ping_region"], "block_hidden_q10")
        self.assertEqual(handoff["pong_region"], "block_output_q10")
        self.assertEqual(handoff["ping_byte_address"], 0)
        self.assertEqual(handoff["pong_byte_address"], 0x0003_4000)
        self.assertEqual(handoff["size_bytes"], 1792)
        self.assertEqual(handoff["copy_bytes_per_layer"], 1792)
        self.assertIn("尚未实现", handoff["rtl_status"])

    def test_uart_is_explicitly_validation_only(self) -> None:
        transport = self.plan["transport"]
        self.assertEqual(transport["current_uart_baud"], 115200)
        self.assertAlmostEqual(transport["per_layer_uart_seconds"], 691.0666666667)
        self.assertGreater(transport["all_layers_reload_uart_hours_per_token"], 4.6)
        self.assertTrue(transport["validation_only"])
        self.assertTrue(transport["usable_inference_requires_faster_transport"])

    def test_reference_snapshot_matches(self) -> None:
        self.assertEqual(reference_snapshot(self.plan), load_reference())
        with self.assertRaises(IndexError):
            expand_layer_transfer_plan(self.plan, 24)
        with self.assertRaises(ValueError):
            expand_layer_transfer_plan(self.plan, 0, slot="C")


if __name__ == "__main__":
    unittest.main()
