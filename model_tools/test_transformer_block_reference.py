#!/usr/bin/env python3
"""G2 layer0 完整 Transformer Block 软件参考与集成契约测试。"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

import numpy as np

try:
    from .transformer_block_reference import (
        DEFAULT_MANIFEST,
        EXECUTION_HEADER,
        HIDDEN_SIZE,
        KV_CACHE_BASE_BYTES,
        LOW_DDR_LIMIT_BYTES,
        STAGES,
        build_execution_payload,
        build_fixed_real_cases,
        integration_contract,
        kv_slot_byte_addresses,
        linear_invocations,
        parameter_regions,
        scratch_regions,
        sha256_array,
        validate_manifest,
        validate_memory_layout,
        verify_execution_payload,
    )
except ImportError:
    from transformer_block_reference import (
        DEFAULT_MANIFEST,
        EXECUTION_HEADER,
        HIDDEN_SIZE,
        KV_CACHE_BASE_BYTES,
        LOW_DDR_LIMIT_BYTES,
        STAGES,
        build_execution_payload,
        build_fixed_real_cases,
        integration_contract,
        kv_slot_byte_addresses,
        linear_invocations,
        parameter_regions,
        scratch_regions,
        sha256_array,
        validate_manifest,
        validate_memory_layout,
        verify_execution_payload,
    )


EXPECTED_OUTPUT_SHA256 = [
    "630952eaf6fe179639773f2b60d1e9e990f380b0e698fa3d493dd7c279c96104",
    "1cd96d92e43203abda26cace446c2664a0cbaa1fad29a8658a0d94941fbfbea7",
    "b2365afdc2857c543628c9d15fd829005ede98aa99d35a8c4f973f61b35ed9dc",
    "c164aab5251afc3954b8689a826a1241d1c7d6757adef5c5a232e127b59a4032",
]


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RTL_ROOT = PROJECT_ROOT / "transformer_block_g2" / "rtl"


class TransformerBlockReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cases = build_fixed_real_cases()

    def test_complete_fixed_outputs_match_verified_g1(self) -> None:
        self.assertEqual([case.query_position for case in self.cases], [0, 1, 5, 15])
        self.assertEqual([case.count for case in self.cases], [1, 2, 6, 16])
        actual = [sha256_array(case.output_q10, "<i2") for case in self.cases]
        self.assertEqual(actual, EXPECTED_OUTPUT_SHA256)
        for case in self.cases:
            self.assertEqual(case.block_input_q10.shape, (HIDDEN_SIZE,))
            self.assertEqual(case.current_q_q28.shape, (14, 64))
            self.assertEqual(case.current_k_q28.shape, (2, 64))
            self.assertEqual(case.current_q_rope_q28.shape, (14, 64))
            self.assertEqual(case.current_k_rope_q28.shape, (2, 64))
            self.assertEqual(case.output_q10.shape, (HIDDEN_SIZE,))
            self.assertTrue(np.issubdtype(case.output_q10.dtype, np.signedinteger))

    def test_execution_payload_roundtrip_and_lengths(self) -> None:
        expected_lengths = [2112, 4160, 12352, 32832]
        for case, expected_length in zip(self.cases, expected_lengths, strict=True):
            payload = build_execution_payload(case)
            self.assertEqual(EXECUTION_HEADER.size, 64)
            self.assertEqual(len(payload), expected_length)
            self.assertEqual(len(verify_execution_payload(case)), 64)

    def test_memory_layout_and_f3_kv_boundary(self) -> None:
        validate_memory_layout()
        self.assertEqual(KV_CACHE_BASE_BYTES, LOW_DDR_LIMIT_BYTES)
        k_first, v_first = kv_slot_byte_addresses(0, 0)
        self.assertEqual(k_first, 0x08000000)
        self.assertEqual(v_first, 0x08000400)
        k_last, v_last = kv_slot_byte_addresses(27, 16383)
        self.assertEqual(k_last, 0x3FFFF800)
        self.assertEqual(v_last, 0x3FFFFC00)
        self.assertEqual(v_last + 1024, 0x40000000)

    def test_stage_and_region_contract(self) -> None:
        stage_ids = [stage[0] for stage in STAGES]
        stage_names = [stage[1] for stage in STAGES]
        self.assertEqual(len(stage_ids), len(set(stage_ids)))
        self.assertEqual(stage_names[0], "IDLE")
        self.assertEqual(stage_names[-1], "ERROR")
        self.assertIn("RESIDUAL2", stage_names)
        self.assertEqual(len(scratch_regions()), 28)
        self.assertEqual(len(parameter_regions()), 24)
        contract = integration_contract()
        self.assertEqual(contract["controller_address_unit_bytes"], 4)
        self.assertEqual(contract["kv_cache"]["positions"], 16384)

    def test_fixed_manifest(self) -> None:
        manifest = validate_manifest(self.cases, DEFAULT_MANIFEST)
        self.assertEqual(manifest["definition"]["fixed_queries"], [0, 1, 5, 15])
        self.assertEqual(
            manifest["definition"]["hardware_status"],
            "software reference and integration contract only; RTL/PDS/board pending",
        )

    def test_rtl_contract_addresses_and_stages_match_python(self) -> None:
        source = (RTL_ROOT / "transformer_block_contract.vh").read_text(encoding="utf-8")
        hex_macros = {
            name: int(value.replace("_", ""), 16)
            for name, value in re.findall(
                r"^`define\s+(\w+)\s+\d+'h([0-9a-fA-F_]+)", source, re.MULTILINE
            )
        }
        scratch_macro = {
            "block_hidden_q10": "G2_BLOCK_HIDDEN_CTRL_ADDR",
            "input_norm_q10": "G2_INPUT_NORM_CTRL_ADDR",
            "q_q28": "G2_Q_Q28_CTRL_ADDR",
            "k_q28": "G2_K_Q28_CTRL_ADDR",
            "v_q28": "G2_V_Q28_CTRL_ADDR",
            "q_rope_q28": "G2_Q_ROPE_CTRL_ADDR",
            "k_rope_q28": "G2_K_ROPE_CTRL_ADDR",
            "scores_q28": "G2_SCORES_CTRL_ADDR",
            "probabilities_q31": "G2_PROBABILITIES_CTRL_ADDR",
            "attention_concat_q28": "G2_ATTN_CONCAT_CTRL_ADDR",
            "oproj_q28": "G2_OPROJ_CTRL_ADDR",
            "attention_residual_q10": "G2_ATTN_RESIDUAL_CTRL_ADDR",
            "post_attention_norm_q10": "G2_POST_NORM_CTRL_ADDR",
            "gate_q28": "G2_GATE_CTRL_ADDR",
            "up_q28": "G2_UP_CTRL_ADDR",
            "silu_gate_q10": "G2_SILU_GATE_CTRL_ADDR",
            "silu_up_q28": "G2_SILU_UP_CTRL_ADDR",
            "down_proj_q28": "G2_DOWN_CTRL_ADDR",
            "block_output_q10": "G2_BLOCK_OUTPUT_CTRL_ADDR",
            "linear_activation_int8": "G2_LINEAR_ACT_INT8_CTRL_ADDR",
            "linear_quant_metadata": "G2_LINEAR_QUANT_META_CTRL_ADDR",
            "execution_payload": "G2_EXEC_PAYLOAD_CTRL_ADDR",
            "input_rms_gamma_q10": "G2_INPUT_RMS_GAMMA_CTRL_ADDR",
            "post_rms_gamma_q10": "G2_POST_RMS_GAMMA_CTRL_ADDR",
            "rms_lut_uq12_20": "G2_RMS_LUT_CTRL_ADDR",
            "softmax_exp_lut_q31": "G2_SOFTMAX_LUT_CTRL_ADDR",
            "silu_pwl_q10": "G2_SILU_PWL_CTRL_ADDR",
            "rope_trig_q30": "G2_ROPE_TRIG_CTRL_ADDR",
        }
        for region in scratch_regions():
            self.assertEqual(hex_macros[scratch_macro[region.name]], region.controller_address)

        parameter_macro = {
            "q_weight_int4": "G2_Q_WEIGHT_CTRL_ADDR",
            "q_scale_uq4_28": "G2_Q_SCALE_CTRL_ADDR",
            "q_bias_q28": "G2_Q_BIAS_CTRL_ADDR",
            "k_weight_int4": "G2_K_WEIGHT_CTRL_ADDR",
            "k_scale_uq4_28": "G2_K_SCALE_CTRL_ADDR",
            "k_bias_q28": "G2_K_BIAS_CTRL_ADDR",
            "v_weight_int4": "G2_V_WEIGHT_CTRL_ADDR",
            "v_scale_uq4_28": "G2_V_SCALE_CTRL_ADDR",
            "v_bias_q28": "G2_V_BIAS_CTRL_ADDR",
            "oproj_weight_int4": "G2_OPROJ_WEIGHT_CTRL_ADDR",
            "oproj_scale_uq4_28": "G2_OPROJ_SCALE_CTRL_ADDR",
            "gate_weight_int4": "G2_GATE_WEIGHT_CTRL_ADDR",
            "gate_scale_uq4_28": "G2_GATE_SCALE_CTRL_ADDR",
            "up_weight_int4": "G2_UP_WEIGHT_CTRL_ADDR",
            "up_scale_uq4_28": "G2_UP_SCALE_CTRL_ADDR",
            "down_weight_int4": "G2_DOWN_WEIGHT_CTRL_ADDR",
            "down_scale_uq4_28": "G2_DOWN_SCALE_CTRL_ADDR",
            "q_weight_scale_fp16": "G2_Q_RAW_SCALE_CTRL_ADDR",
            "k_weight_scale_fp16": "G2_K_RAW_SCALE_CTRL_ADDR",
            "v_weight_scale_fp16": "G2_V_RAW_SCALE_CTRL_ADDR",
            "oproj_weight_scale_fp16": "G2_OPROJ_RAW_SCALE_CTRL_ADDR",
            "gate_weight_scale_fp16": "G2_GATE_RAW_SCALE_CTRL_ADDR",
            "up_weight_scale_fp16": "G2_UP_RAW_SCALE_CTRL_ADDR",
            "down_weight_scale_fp16": "G2_DOWN_RAW_SCALE_CTRL_ADDR",
        }
        for region in parameter_regions():
            self.assertEqual(hex_macros[parameter_macro[region.name]], region.controller_address)

        for stage_id, stage_name, _, _ in STAGES:
            self.assertEqual(hex_macros[f"G2_STAGE_{stage_name}"], stage_id)

    def test_linear_invocation_matrix_contract(self) -> None:
        invocations = linear_invocations()
        self.assertEqual(
            [item.name for item in invocations],
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        self.assertEqual([item.rows for item in invocations], [896, 128, 128, 896, 4864, 4864, 896])
        self.assertEqual([item.columns for item in invocations], [896, 896, 896, 896, 896, 896, 4864])
        self.assertEqual([item.groups for item in invocations], [14, 14, 14, 14, 14, 14, 76])
        self.assertEqual([item.act_beats for item in invocations], [28, 28, 28, 28, 28, 28, 152])
        self.assertEqual(
            [item.weight_beats_per_row for item in invocations],
            [14, 14, 14, 14, 14, 14, 76],
        )
        self.assertEqual([item.scale_beats_per_row for item in invocations], [2, 2, 2, 2, 2, 2, 10])
        self.assertEqual(
            [item.has_bias for item in invocations],
            [True, True, True, False, False, False, False],
        )

    def test_shared_linear_runtime_shapes_are_explicit(self) -> None:
        source = (RTL_ROOT / "shared_linear_engine.v").read_text(encoding="utf-8")
        self.assertIn("cfg_k_blocks == 9'd56", source)
        self.assertIn("cfg_groups == 7'd14", source)
        self.assertIn("cfg_k_blocks == 9'd304", source)
        self.assertIn("cfg_groups == 7'd76", source)
        self.assertIn("MAX_SCALE_WORDS", source)
        self.assertIn("scale_mem [0:MAX_SCALE_WORDS-1]", source)
        self.assertIn("module shared_linear_engine", source)

        runtime_source = (RTL_ROOT / "runtime_linear_ctrl.v").read_text(encoding="utf-8")
        self.assertIn("cfg_m_rows == 13'd128", runtime_source)
        self.assertIn("cfg_m_rows == 13'd896", runtime_source)
        self.assertIn("cfg_m_rows == 13'd4864", runtime_source)
        self.assertIn("module runtime_linear_ctrl", runtime_source)

        scheduler_source = (RTL_ROOT / "transformer_block_scheduler.v").read_text(encoding="utf-8")
        self.assertIn("input  wire [21:0]  engine_done", scheduler_source)
        self.assertIn("output reg  [21:0]  engine_start", scheduler_source)
        self.assertIn("module transformer_block_scheduler", scheduler_source)

        divider_source = (RTL_ROOT / "unsigned_divider_rne.v").read_text(encoding="utf-8")
        self.assertIn("ST_ITER_PREP", divider_source)
        self.assertIn("ST_ITER_UPDATE", divider_source)
        self.assertIn("doubled_remainder", divider_source)
        self.assertIn("next_quotient[0]", divider_source)
        self.assertIn("module unsigned_divider_rne", divider_source)

        q10_source = (RTL_ROOT / "runtime_q10_activation_quantizer.v").read_text(encoding="utf-8")
        self.assertIn("ST_DIV_PREP", q10_source)
        self.assertIn("source_magnitude_reg * 7'd127", q10_source)
        self.assertIn(".WIDTH(32)", q10_source)

        q28_convert_source = (RTL_ROOT / "q28_to_binary32.v").read_text(encoding="utf-8")
        self.assertIn("binary64_mantissa[52:29]", q28_convert_source)
        self.assertIn("binary32_exponent_msb + 8'd99", q28_convert_source)
        self.assertIn("module q28_to_binary32", q28_convert_source)

        q28_seq_source = (RTL_ROOT / "q28_to_binary32_sequential.v").read_text(encoding="utf-8")
        self.assertIn("ST_CAPTURE", q28_seq_source)
        self.assertIn("input_reg", q28_seq_source)
        self.assertIn("module q28_to_binary32_sequential", q28_seq_source)

        q28_source = (RTL_ROOT / "runtime_q28_activation_quantizer.v").read_text(encoding="utf-8")
        self.assertIn("ST_LOAD_UPDATE", q28_source)
        self.assertIn("q28_to_binary32_sequential", q28_source)
        self.assertIn("exponent_difference_signed", q28_source)
        self.assertIn("ratio_numerator_base", q28_source)
        self.assertIn("ST_RATIO_SHIFT", q28_source)
        self.assertIn(".WIDTH(96)", q28_source)

        scale_source = (RTL_ROOT / "runtime_fp16_scale_builder.v").read_text(encoding="utf-8")
        self.assertIn("ST_SHIFT_INIT", scale_source)
        self.assertIn("ST_SHIFT_LOOP", scale_source)
        self.assertIn("? -11'sd24", scale_source)
        self.assertIn("effective_denominator_base = all_zero_reg ? 8'd1 : 8'd127", scale_source)
        self.assertIn("32'hffff_ffff", scale_source)


if __name__ == "__main__":
    unittest.main()
