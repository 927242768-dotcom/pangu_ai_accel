from __future__ import annotations

import re
import unittest
from pathlib import Path

try:
    from .transformer_block_reference import (
        KV_CACHE_BASE_BYTES,
        KV_LAYER_STRIDE_BYTES,
        KV_TOKEN_STRIDE_BYTES,
        KV_VALUE_OFFSET_BYTES,
        parameter_regions,
        scratch_regions,
    )
except ImportError:
    from transformer_block_reference import (
        KV_CACHE_BASE_BYTES,
        KV_LAYER_STRIDE_BYTES,
        KV_TOKEN_STRIDE_BYTES,
        KV_VALUE_OFFSET_BYTES,
        parameter_regions,
        scratch_regions,
    )


ROOT = Path(__file__).resolve().parents[1]
RTL = ROOT / "transformer_block_g2" / "rtl"
PNR = ROOT / "transformer_block_g2" / "pnr"
HOST_TOOL = ROOT / "tools" / "pangu_transformer_block_host.py"


class TransformerBlockG2IntegrationTests(unittest.TestCase):
    def test_all_g2_stage_modules_exist(self) -> None:
        required = {
            "g2_axi_stage_mux.v": "module g2_axi_stage_mux",
            "g2_rmsnorm_stage_ctrl.v": "module g2_rmsnorm_stage_ctrl",
            "g2_quant_sequence_unified_ctrl.v": "module g2_quant_sequence_unified_ctrl",
            "runtime_activation_quantizer_stream_ctrl.v": "module runtime_activation_quantizer_stream_ctrl",
            "runtime_quantizer_unified_ctrl.v": "module runtime_quantizer_unified_ctrl",
            "g2_linear_stage_ctrl.v": "module g2_linear_stage_ctrl",
            "g2_rope_stage_ctrl.v": "module g2_rope_stage_ctrl",
            "g2_kv_write_stage_ctrl.v": "module g2_kv_write_stage_ctrl",
            "g2_attention_score_stage_ctrl.v": "module g2_attention_score_stage_ctrl",
            "g2_softmax_stage_ctrl.v": "module g2_softmax_stage_ctrl",
            "g2_attention_output_stage_ctrl.v": "module g2_attention_output_stage_ctrl",
            "g2_stream_residual_ctrl.v": "module g2_stream_residual_ctrl",
            "g2_stream_silu_ctrl.v": "module g2_stream_silu_ctrl",
            "g2_stream_silu_up_mul_ctrl.v": "module g2_stream_silu_up_mul_ctrl",
            "transformer_block_ctrl.v": "module transformer_block_ctrl",
            "transformer_block_host_ctrl.v": "module transformer_block_host_ctrl",
            "transformer_block_top.v": "module transformer_block_top",
        }
        for filename, module_declaration in required.items():
            source = (RTL / filename).read_text(encoding="utf-8")
            self.assertIn(module_declaration, source, filename)

    def test_rtl_address_contract_matches_python_memory_map(self) -> None:
        contract = (RTL / "transformer_block_contract.vh").read_text(encoding="utf-8")
        macros = {
            name: int(value.replace("_", ""), 16)
            for name, value in re.findall(
                r"`define\s+G2_([A-Z0-9_]+)_CTRL_ADDR\s+32'h([0-9a-fA-F_]+)",
                contract,
            )
        }
        regions = {
            region.name: region
            for region in (*scratch_regions(), *parameter_regions())
        }
        expected = {
            "BLOCK_HIDDEN": "block_hidden_q10",
            "INPUT_NORM": "input_norm_q10",
            "Q_Q28": "q_q28",
            "K_Q28": "k_q28",
            "V_Q28": "v_q28",
            "Q_ROPE": "q_rope_q28",
            "K_ROPE": "k_rope_q28",
            "SCORES": "scores_q28",
            "PROBABILITIES": "probabilities_q31",
            "ATTN_CONCAT": "attention_concat_q28",
            "OPROJ": "oproj_q28",
            "ATTN_RESIDUAL": "attention_residual_q10",
            "POST_NORM": "post_attention_norm_q10",
            "GATE": "gate_q28",
            "UP": "up_q28",
            "SILU_GATE": "silu_gate_q10",
            "SILU_UP": "silu_up_q28",
            "DOWN": "down_proj_q28",
            "BLOCK_OUTPUT": "block_output_q10",
            "LINEAR_ACT_INT8": "linear_activation_int8",
            "LINEAR_QUANT_META": "linear_quant_metadata",
            "EXEC_PAYLOAD": "execution_payload",
            "INPUT_RMS_GAMMA": "input_rms_gamma_q10",
            "POST_RMS_GAMMA": "post_rms_gamma_q10",
            "RMS_LUT": "rms_lut_uq12_20",
            "SOFTMAX_LUT": "softmax_exp_lut_q31",
            "SILU_PWL": "silu_pwl_q10",
            "ROPE_TRIG": "rope_trig_q30",
            "Q_WEIGHT": "q_weight_int4",
            "Q_SCALE": "q_scale_uq4_28",
            "Q_BIAS": "q_bias_q28",
            "K_WEIGHT": "k_weight_int4",
            "K_SCALE": "k_scale_uq4_28",
            "K_BIAS": "k_bias_q28",
            "V_WEIGHT": "v_weight_int4",
            "V_SCALE": "v_scale_uq4_28",
            "V_BIAS": "v_bias_q28",
            "OPROJ_WEIGHT": "oproj_weight_int4",
            "OPROJ_SCALE": "oproj_scale_uq4_28",
            "GATE_WEIGHT": "gate_weight_int4",
            "GATE_SCALE": "gate_scale_uq4_28",
            "UP_WEIGHT": "up_weight_int4",
            "UP_SCALE": "up_scale_uq4_28",
            "DOWN_WEIGHT": "down_weight_int4",
            "DOWN_SCALE": "down_scale_uq4_28",
            "Q_RAW_SCALE": "q_weight_scale_fp16",
            "K_RAW_SCALE": "k_weight_scale_fp16",
            "V_RAW_SCALE": "v_weight_scale_fp16",
            "OPROJ_RAW_SCALE": "oproj_weight_scale_fp16",
            "GATE_RAW_SCALE": "gate_weight_scale_fp16",
            "UP_RAW_SCALE": "up_weight_scale_fp16",
            "DOWN_RAW_SCALE": "down_weight_scale_fp16",
        }
        self.assertEqual(set(macros) - {"KV_BASE"}, set(expected))
        for macro_name, region_name in expected.items():
            self.assertEqual(
                macros[macro_name],
                regions[region_name].controller_address,
                f"G2_{macro_name}_CTRL_ADDR != {region_name}",
            )

        kv_constants = {
            name: int(value.replace("_", ""), 16)
            for name, value in re.findall(
                r"`define\s+G2_(KV_[A-Z0-9_]+)_CTRL(?:_ADDR)?\s+32'h([0-9a-fA-F_]+)",
                contract,
            )
        }
        self.assertEqual(
            kv_constants,
            {
                "KV_BASE": KV_CACHE_BASE_BYTES // 4,
                "KV_LAYER_STRIDE": KV_LAYER_STRIDE_BYTES // 4,
                "KV_TOKEN_STRIDE": KV_TOKEN_STRIDE_BYTES // 4,
                "KV_V_OFFSET": KV_VALUE_OFFSET_BYTES // 4,
            },
        )

    def test_controller_covers_all_22_scheduler_stages(self) -> None:
        contract = (RTL / "transformer_block_contract.vh").read_text(encoding="utf-8")
        controller = (RTL / "transformer_block_ctrl.v").read_text(encoding="utf-8")
        scheduler = (RTL / "transformer_block_scheduler.v").read_text(encoding="utf-8")

        stage_matches = re.findall(
            r"`define\s+G2_STAGE_([A-Z0-9_]+)\s+5'h([0-9a-fA-F]+)", contract
        )
        compute_stages = [
            name for name, value in stage_matches if 1 <= int(value, 16) <= 22
        ]
        self.assertEqual(len(compute_stages), 22)
        self.assertIn("input  wire [21:0]  engine_done", scheduler)
        self.assertIn("output reg  [21:0]  engine_start", scheduler)

        for stage_name in compute_stages:
            self.assertIn(f"`G2_STAGE_{stage_name}", controller, stage_name)
        for engine_index in range(22):
            self.assertIn(f"assign engine_done[{engine_index}]", controller)
            self.assertIn(f"assign engine_error[{engine_index}]", controller)

    def test_controller_uses_one_unified_axi_mux(self) -> None:
        controller = (RTL / "transformer_block_ctrl.v").read_text(encoding="utf-8")
        self.assertIn("localparam integer NUM_MASTERS = 11", controller)
        self.assertEqual(controller.count("g2_axi_stage_mux #("), 1)
        self.assertIn(".NUM_MASTERS (NUM_MASTERS)", controller)
        self.assertIn("parameter integer ACTIVE_LAYER_COUNT = 1", controller)
        self.assertIn("(cfg_layer < ACTIVE_LAYER_COUNT)", controller)
        self.assertIn(".clk           (clk)", controller)
        self.assertIn(".rst_n         (rst_n)", controller)
        self.assertIn(".select_master (selected_master)", controller)
        mux = (RTL / "g2_axi_stage_mux.v").read_text(encoding="utf-8")
        self.assertIn("reg [255:0] return_rdata_pipe", mux)
        self.assertIn("return_rvalid_pipe <= axi_rvalid", mux)
        self.assertIn("return_master_pipe <= read_owner", mux)
        self.assertIn("reg                         write_active", mux)
        self.assertIn("reg                         read_ar_pending", mux)
        self.assertIn("write_awaddr_pipe", mux)
        self.assertIn("read_araddr_pipe", mux)
        self.assertIn("m_awready[selected_bit] = !write_active", mux)
        self.assertIn("m_arready[selected_bit] = !read_ar_pending", mux)
        self.assertIn("m_wready[write_complete_owner] = 1'b1", mux)
        self.assertIn("m_rdata   = {NUM_MASTERS{return_rdata_pipe}}", mux)
        for master_name in (
            "MASTER_RMS",
            "MASTER_QUANT",
            "MASTER_LINEAR",
            "MASTER_ROPE",
            "MASTER_KV",
            "MASTER_SCORE",
            "MASTER_SOFTMAX",
            "MASTER_ATTN_OUT",
            "MASTER_RESIDUAL",
            "MASTER_SILU",
            "MASTER_SILU_MUL",
        ):
            self.assertIn(master_name, controller)

    def test_attention_window_and_residual_branch_contracts(self) -> None:
        score = (RTL / "g2_attention_score_stage_ctrl.v").read_text(
            encoding="utf-8"
        )
        output = (RTL / "g2_attention_output_stage_ctrl.v").read_text(
            encoding="utf-8"
        )
        controller = (RTL / "transformer_block_ctrl.v").read_text(
            encoding="utf-8"
        )

        for source in (score, output):
            self.assertIn(
                "cfg_query_position == cfg_window_start + cfg_count - 1'b1",
                source,
            )
            self.assertIn("window_start_reg + token_index", source)
        self.assertIn("wire token_masked = token_index >= count_reg", score)
        self.assertIn(".token_count            (count_reg)", output)
        self.assertIn("`G2_KV_V_OFFSET_CTRL", output)

        self.assertIn(
            "`G2_ATTN_RESIDUAL_CTRL_ADDR : `G2_BLOCK_HIDDEN_CTRL_ADDR",
            controller,
        )
        self.assertIn(
            "`G2_DOWN_CTRL_ADDR : `G2_OPROJ_CTRL_ADDR",
            controller,
        )
        self.assertIn(
            "`G2_BLOCK_OUTPUT_CTRL_ADDR : `G2_ATTN_RESIDUAL_CTRL_ADDR",
            controller,
        )

    def test_runtime_quantization_is_inside_the_22_stage_chain(self) -> None:
        controller = (RTL / "transformer_block_ctrl.v").read_text(encoding="utf-8")
        quant = (RTL / "g2_quant_sequence_unified_ctrl.v").read_text(encoding="utf-8")
        stream = (RTL / "runtime_activation_quantizer_stream_ctrl.v").read_text(
            encoding="utf-8"
        )
        unified = (RTL / "runtime_quantizer_unified_ctrl.v").read_text(
            encoding="utf-8"
        )

        self.assertEqual(controller.count("g2_quant_sequence_unified_ctrl #("), 1)
        self.assertNotIn("u_g2_quant_sequence_q10", controller)
        self.assertNotIn("u_g2_quant_sequence_q28", controller)
        self.assertIn("engine_start[1] || engine_start[10]", controller)
        self.assertIn("engine_start[14] || engine_start[19]", controller)
        self.assertIn("MODE_QKV", quant)
        self.assertIn("MODE_OPROJ", quant)
        self.assertIn("MODE_GATE_UP", quant)
        self.assertIn("MODE_DOWN", quant)
        self.assertEqual(quant.count("runtime_quantizer_unified_ctrl #("), 1)
        self.assertEqual(unified.count("runtime_activation_quantizer_stream_ctrl #("), 1)
        self.assertNotIn("runtime_activation_quantizer_unified_ctrl #(", unified)
        self.assertEqual(stream.count("q28_to_binary32_sequential"), 1)
        self.assertEqual(stream.count("unsigned_divider_rne #("), 1)
        self.assertNotRegex(stream, r"source_mem\s*\[0:")
        self.assertIn("ST_P1_SETUP", stream)
        self.assertIn("ST_P2_SETUP", stream)
        self.assertIn("q10_extended <<< 18", stream)

    def test_shared_linear_bias_stride_matches_padded_payload_rows(self) -> None:
        linear = (RTL / "runtime_linear_ctrl.v").read_text(encoding="utf-8")
        payload = (
            ROOT / "model_tools" / "transformer_block_g2_payload.py"
        ).read_text(encoding="utf-8")
        self.assertIn("padded = np.zeros((rows, 4), dtype=\"<i8\")", payload)
        self.assertIn("padded[:, 0] = bias_q28.astype(\"<i8\")", payload)
        self.assertIn("bias_row_cache[63:0]", linear)
        self.assertIn("bias_row_addr <= bias_row_addr + 8", linear)

    def test_shared_linear_has_all_seven_fixed_modes(self) -> None:
        source = (RTL / "g2_linear_stage_ctrl.v").read_text(encoding="utf-8")
        for mode in (
            "MODE_Q",
            "MODE_K",
            "MODE_V",
            "MODE_O",
            "MODE_GATE",
            "MODE_UP",
            "MODE_DOWN",
        ):
            self.assertIn(mode, source)
        for rows in ("13'd128", "13'd896", "13'd4864"):
            self.assertIn(rows, source)
        self.assertIn("linear_k_blocks             = 9'd304", source)
        self.assertIn("linear_groups               = 7'd76", source)
        self.assertIn("linear_weight_beats_per_row = 7'd76", source)
        self.assertIn("linear_scale_beats_per_row  = 4'd10", source)
        self.assertEqual(source.count("runtime_linear_ctrl #("), 1)

    def test_high_drm_stages_are_streamed(self) -> None:
        residual = (RTL / "g2_stream_residual_ctrl.v").read_text(encoding="utf-8")
        silu = (RTL / "g2_stream_silu_ctrl.v").read_text(encoding="utf-8")
        multiply = (RTL / "g2_stream_silu_up_mul_ctrl.v").read_text(encoding="utf-8")

        self.assertNotRegex(residual, r"reg\s+\[255:0\]\s+\w+\s*\[0:")
        self.assertNotRegex(multiply, r"reg\s+\[255:0\]\s+\w+\s*\[0:")
        self.assertIn("pwl_mem [0:79]", silu)
        self.assertNotIn("[0:303]", silu)
        self.assertIn("rne_shift18_from_magnitude", residual)
        self.assertIn("rne_shift18_from_magnitude", silu)
        self.assertIn("rne_shift10_unsigned80", multiply)
        softmax_stage = (RTL / "g2_softmax_stage_ctrl.v").read_text(encoding="utf-8")
        softmax_core = (ROOT / "softmax_f5" / "rtl" / "softmax_core.v").read_text(
            encoding="utf-8"
        )
        self.assertIn(".PIPELINE_SCORE_DIFF (1)", softmax_stage)
        self.assertIn("parameter integer PIPELINE_SCORE_DIFF = 0", softmax_core)
        self.assertIn("ST_EXP_DECIDE", softmax_core)
        self.assertIn("ST_MUL_CAPTURE", multiply)
        self.assertIn("ST_MUL_ACCUM", multiply)
        self.assertIn("partial_product_reg", multiply)
        self.assertNotIn("ST_MUL3", multiply)

    def test_post_route_critical_paths_are_explicitly_pipelined(self) -> None:
        attention = (
            ROOT / "attention_score_f4" / "rtl" / "attention_score_core.v"
        ).read_text(encoding="utf-8")
        rmsnorm = (ROOT / "rmsnorm_k896" / "rtl" / "rmsnorm_k896_core.v").read_text(
            encoding="utf-8"
        )
        rms_stage = (RTL / "g2_rmsnorm_stage_ctrl.v").read_text(encoding="utf-8")
        host = (RTL / "transformer_block_host_ctrl.v").read_text(encoding="utf-8")

        self.assertIn("ST_FINALIZE", attention)
        self.assertIn("parameter integer PIPELINE_ACCUM = 0", attention)
        self.assertIn("aligned_partial_reg", attention)
        self.assertIn("magnitude_accumulator <= next_magnitude", attention)
        self.assertIn("$signed(~magnitude_accumulator + 1'b1)", attention)

        self.assertIn("parameter integer PIPELINE_NORMALIZE = 0", rmsnorm)
        self.assertIn("ST_NORMALIZE_SHIFT", rmsnorm)
        self.assertIn("variance_leading_bit_reg <= variance_leading_bit", rmsnorm)
        self.assertIn("parameter integer PIPELINE_X_RNE", rmsnorm)
        self.assertIn("ST_OUT_X_COMMIT", rmsnorm)
        self.assertIn("normalized_rounded_reg <= normalized_rounded_wire", rmsnorm)
        self.assertIn("parameter integer PIPELINE_GAMMA_INPUT", rmsnorm)
        self.assertIn("ST_OUT_G_CAPTURE", rmsnorm)
        self.assertIn("gamma_normalized_operand_reg <= normalized_q10", rmsnorm)
        self.assertIn("gamma_scale_operand_reg      <= selected_gamma", rmsnorm)
        self.assertIn(".PIPELINE_NORMALIZE   (1)", rms_stage)
        self.assertIn(".PIPELINE_X_RNE       (1)", rms_stage)
        self.assertIn(".PIPELINE_GAMMA_INPUT (1)", rms_stage)
        self.assertIn(".PIPELINE_RSQRT_SHIFT (1)", rms_stage)

        write_entry = host.split("8'h57, 8'h77: begin", 1)[1].split(
            "8'h52, 8'h72: begin", 1
        )[0]
        self.assertIn("transfer_beat     <= 256'd0", write_entry)
        check_write = host.split("ST_CHECK_WRITE: begin", 1)[1].split(
            "ST_RECV_WRITE_DATA: begin", 1
        )[0]
        self.assertNotIn("transfer_beat        <=", check_write)

    def test_pds_compile_script_contains_complete_controller_dependencies(self) -> None:
        script = (PNR / "build_stream_stage.tcl").read_text(encoding="utf-8")
        required_sources = (
            "transformer_block_ctrl.v",
            "transformer_block_scheduler.v",
            "g2_axi_stage_mux.v",
            "g2_quant_sequence_unified_ctrl.v",
            "g2_linear_stage_ctrl.v",
            "runtime_activation_quantizer_stream_ctrl.v",
            "runtime_quantizer_unified_ctrl.v",
            "runtime_linear_ctrl.v",
            "rmsnorm_k896_core.v",
            "rope_pair_q28_core.v",
            "attention_score_core.v",
            "softmax_core.v",
            "attention_output_core.v",
        )
        for filename in required_sources:
            self.assertIn(filename, script)
        self.assertIn("transformer_block_host_ctrl.v", script)
        self.assertIn("uart_rx.v", script)
        self.assertIn("uart_tx.v", script)
        self.assertIn("compile -top_module $top_name", script)
        self.assertIn("synthesize -ads -selected_syn_tool_opt 2", script)

    def test_host_protocol_keeps_stage_watchdog_enabled(self) -> None:
        host = (RTL / "transformer_block_host_ctrl.v").read_text(encoding="utf-8")
        top = (RTL / "transformer_block_top.v").read_text(encoding="utf-8")
        self.assertIn("32'd500000000", host)
        self.assertNotIn(".WATCHDOG_CYCLES (32'd0)", host)
        self.assertIn("PANGU50K G2 BLOCK V1", host)
        for command in (
            "8'h49", "8'h53", "8'h43", "8'h4c", "8'h57", "8'h52", "8'h4d", "8'h50", "8'h47"
        ):
            self.assertIn(command, host)
        self.assertIn("module transformer_block_top", top)
        self.assertIn("I_ipsxb_ddr_top", top)
        self.assertIn("u_transformer_block_host_ctrl", top)

    def test_board_host_verifies_linear_and_rope_boundaries(self) -> None:
        host = HOST_TOOL.read_text(encoding="utf-8")
        expected_order = (
            '"q_q28"',
            '"k_q28"',
            '"q_rope_q28"',
            '"k_rope_q28"',
        )
        offsets = [host.index(name) for name in expected_order]
        self.assertEqual(offsets, sorted(offsets))

    def test_full_pds_and_sram_only_scripts_exist(self) -> None:
        build = (PNR / "build_transformer_block.tcl").read_text(encoding="utf-8")
        program = (PNR / "program_transformer_block_sram.tcl").read_text(encoding="utf-8")
        for action in (
            "compile -top_module transformer_block_top",
            "synthesize -ads",
            "dev_map",
            "set pnr_args [list",
            "set gplace_seed 5",
            "set groute_seed 11",
            "eval pnr $pnr_args",
            "report_timing",
            "gen_bit_stream",
        ):
            self.assertIn(action, build)
        self.assertIn("G2_PLC_DECONGESTION", build)
        self.assertIn("G2_ROUTER_THREADS", build)
        self.assertIn("G2_SHARE_ROUTER_CONTROL", build)
        self.assertIn("cfg_program", program)
        for forbidden in ("cfg_erase", "flash_program", "program_flash", "erase_flash"):
            self.assertNotIn(forbidden, program.lower())


if __name__ == "__main__":
    unittest.main()
