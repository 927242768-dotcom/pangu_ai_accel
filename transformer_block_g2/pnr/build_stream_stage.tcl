# G2 流式阶段与 AXI mux 的独立 Compile/Synthesize 检查。
# 调用前通过环境变量 G2_TOP 指定顶层模块名。

if {![info exists ::env(G2_TOP)]} {
    error "G2_TOP environment variable is required"
}

set top_name $::env(G2_TOP)
add_design "../rtl/g2_axi_stage_mux.v"
add_design "../rtl/g2_stream_residual_ctrl.v"
add_design "../rtl/g2_stream_silu_ctrl.v"
add_design "../rtl/g2_stream_silu_up_mul_ctrl.v"
add_design "../rtl/g2_rmsnorm_stage_ctrl.v"
add_design "../rtl/g2_rope_stage_ctrl.v"
add_design "../rtl/g2_kv_write_stage_ctrl.v"
add_design "../rtl/g2_attention_score_stage_ctrl.v"
add_design "../rtl/g2_softmax_stage_ctrl.v"
add_design "../rtl/g2_attention_output_stage_ctrl.v"
add_design "../rtl/g2_quant_sequence_unified_ctrl.v"
add_design "../rtl/g2_linear_stage_ctrl.v"
add_design "../rtl/transformer_block_scheduler.v"
add_design "../rtl/transformer_block_ctrl.v"
add_design "../rtl/transformer_block_host_ctrl.v"
add_design "../../source/uart_rx.v"
add_design "../../source/uart_tx.v"
add_design "../rtl/int4_unpack16.v"
add_design "../rtl/int8_dot16_pipe.v"
add_design "../rtl/shared_linear_engine.v"
add_design "../rtl/runtime_linear_ctrl.v"
add_design "../rtl/unsigned_divider_rne.v"
add_design "../rtl/q28_to_binary32_sequential.v"
add_design "../rtl/runtime_fp16_scale_builder.v"
add_design "../rtl/runtime_activation_quantizer_stream_ctrl.v"
add_design "../rtl/runtime_scale_builder_ctrl.v"
add_design "../rtl/runtime_quantizer_unified_ctrl.v"
add_design "../../rmsnorm_k896/rtl/rmsnorm_k896_core.v"
add_design "../../rope_qk_layer0/rtl/rope_pair_q28_core.v"
add_design "../../attention_score_f4/rtl/attention_score_core.v"
add_design "../../softmax_f5/rtl/softmax_core.v"
add_design "../../attention_output_f6/rtl/attention_output_core.v"

set_arch -family Logos -device PGL50H -speedgrade -6 -package FBG484
compile -top_module $top_name
synthesize -ads -selected_syn_tool_opt 2
