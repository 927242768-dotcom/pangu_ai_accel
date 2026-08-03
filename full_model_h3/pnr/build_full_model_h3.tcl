# PGL50H H3 真实 24 层换层基线完整顶层构建。
#
# 默认执行完整流程；当前前端验收使用：
#   set H3_FRONTEND_ONLY=1
#   D:/Pango/PDS_2022.2-SP6.4/bin/pds_shell.exe \
#     -file build_full_model_h3.tcl -project_name full_model_h3_frontend

set ip_root "../../ipcore/pangu_ddr3_x32/pangu_ddr3_x32"
set g2_rtl "../../transformer_block_g2/rtl"

# H3 wrapper 与已验收 G2 完整 Block。
add_design "../rtl/full_model_h3_top.v"
add_design "$g2_rtl/g2_axi_stage_mux.v"
add_design "$g2_rtl/g2_stream_residual_ctrl.v"
add_design "$g2_rtl/g2_stream_silu_ctrl.v"
add_design "$g2_rtl/g2_stream_silu_up_mul_ctrl.v"
add_design "$g2_rtl/g2_rmsnorm_stage_ctrl.v"
add_design "$g2_rtl/g2_rope_stage_ctrl.v"
add_design "$g2_rtl/g2_kv_write_stage_ctrl.v"
add_design "$g2_rtl/g2_attention_score_stage_ctrl.v"
add_design "$g2_rtl/g2_softmax_stage_ctrl.v"
add_design "$g2_rtl/g2_attention_output_stage_ctrl.v"
add_design "$g2_rtl/g2_quant_sequence_unified_ctrl.v"
add_design "$g2_rtl/g2_linear_stage_ctrl.v"
add_design "$g2_rtl/transformer_block_scheduler.v"
add_design "$g2_rtl/transformer_block_ctrl.v"
add_design "$g2_rtl/transformer_block_host_ctrl.v"

# 共享 Linear 与运行时量化。
add_design "$g2_rtl/int4_unpack16.v"
add_design "$g2_rtl/int8_dot16_pipe.v"
add_design "$g2_rtl/shared_linear_engine.v"
add_design "$g2_rtl/runtime_linear_ctrl.v"
add_design "$g2_rtl/unsigned_divider_rne.v"
add_design "$g2_rtl/q28_to_binary32_sequential.v"
add_design "$g2_rtl/runtime_fp16_scale_builder.v"
add_design "$g2_rtl/runtime_activation_quantizer_stream_ctrl.v"
add_design "$g2_rtl/runtime_scale_builder_ctrl.v"
add_design "$g2_rtl/runtime_quantizer_unified_ctrl.v"

# 复用已真实上板验证的算术核心。
add_design "../../rmsnorm_k896/rtl/rmsnorm_k896_core.v"
add_design "../../rope_qk_layer0/rtl/rope_pair_q28_core.v"
add_design "../../attention_score_f4/rtl/attention_score_core.v"
add_design "../../softmax_f5/rtl/softmax_core.v"
add_design "../../attention_output_f6/rtl/attention_output_core.v"
add_design "../../source/uart_rx.v"
add_design "../../source/uart_tx.v"

# 已验证 DDR3 Controller + PHY。
foreach f [lsort [glob -nocomplain "$ip_root/rtl/ddrphy/*.vp"]] {
    add_design $f
}
add_design "$ip_root/rtl/ddrphy/ipsxb_ddrphy_slice_top_v1_5.v"
add_design "$ip_root/rtl/ipsxb_rst_sync_v1_1.v"
foreach f [lsort [glob -nocomplain "$ip_root/rtl/pll/*.v"]] {
    add_design $f
}
foreach f [lsort [glob -nocomplain "$ip_root/rtl/mcdq_ctrl/*.vp"]] {
    add_design $f
}
foreach f [lsort [glob -nocomplain "$ip_root/rtl/mcdq_ctrl/syn_mod/*.vp"]] {
    add_design $f
}
add_design "$ip_root/rtl/mcdq_ctrl/distributed_fifo/ipsxb_distributed_fifo_v1_0.v"
foreach f [lsort [glob -nocomplain "$ip_root/rtl/mcdq_ctrl/distributed_fifo/rtl/*.v"]] {
    add_design $f
}
add_design "$ip_root/pangu_ddr3_x32.v"
add_design "$ip_root/pangu_ddr3_x32_ddrphy_top.v"

add_constraint "$ip_root/pnr/ddr_test.fdc"

set_arch -family Logos -device PGL50H -speedgrade -6 -package FBG484
compile -top_module full_model_h3_top
synthesize -ads -selected_syn_tool_opt 2

if {![info exists ::env(H3_SYNTH_ONLY)] || $::env(H3_SYNTH_ONLY) ne "1"} {
    dev_map

    if {![info exists ::env(H3_FRONTEND_ONLY)] || $::env(H3_FRONTEND_ONLY) ne "1"} {
        set gplace_seed 5
        set groute_seed 11
        if {[info exists ::env(H3_GPLACE_SEED)] && $::env(H3_GPLACE_SEED) ne ""} {
            set gplace_seed $::env(H3_GPLACE_SEED)
        }
        if {[info exists ::env(H3_GROUTE_SEED)] && $::env(H3_GROUTE_SEED) ne ""} {
            set groute_seed $::env(H3_GROUTE_SEED)
        }
        set pnr_args [list \
            -gplace_seed $gplace_seed \
            -groute_seed $groute_seed \
            -fix_hold_violation_in_route TRUE \
            -max_hold_violated_paths_num 20000]
        if {[info exists ::env(H3_OPTIMIZE_MULTI_CORNER)] && $::env(H3_OPTIMIZE_MULTI_CORNER) eq "1"} {
            lappend pnr_args -optimize_multi_corner_timing
        }
        eval pnr $pnr_args
        report_timing
        gen_bit_stream
    }
}
