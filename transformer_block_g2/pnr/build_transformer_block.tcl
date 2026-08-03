# PGL50H G2 完整 layer0 Transformer Block 全流程构建。
# 在本目录运行：
# D:/Pango/PDS_2022.2-SP6.4/bin/pds_shell.exe \
#   -file build_transformer_block.tcl -project_name transformer_block_g2

set ip_root "../../ipcore/pangu_ddr3_x32/pangu_ddr3_x32"

# G2 scheduler、统一 AXI 仲裁、主机协议与板级顶层。
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
add_design "../rtl/transformer_block_top.v"

# 共享 Linear 与运行时量化。
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

# 复用已真实上板验证的算术核心，不复用其 UART controller/top。
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

# 复用已验证开发板 DDR3/UART/LED 引脚和时钟约束。
add_constraint "$ip_root/pnr/ddr_test.fdc"

set_arch -family Logos -device PGL50H -speedgrade -6 -package FBG484
compile -top_module transformer_block_top
synthesize -ads -selected_syn_tool_opt 2

# 设置 G2_SYNTH_ONLY=1 时，只执行 Compile/Synthesize，用于时序关键路径快速迭代。
if {![info exists ::env(G2_SYNTH_ONLY)] || $::env(G2_SYNTH_ONLY) ne "1"} {
    dev_map

    # 设置 G2_FRONTEND_ONLY=1 时，只完成完整顶层的 Compile/Synthesize/Dev-map，
    # 用于检查 RTL、DDR3 IP 集成与资源映射；默认仍执行完整 PnR/时序/位流。
    if {![info exists ::env(G2_FRONTEND_ONLY)] || $::env(G2_FRONTEND_ONLY) ne "1"} {
        # 默认保持已冻结 seed5/11；环境变量只改变物理实现策略，不改变 RTL。
        # G2_INPUT_PLACE_DB 仅接受 PDS PLC 阶段数据库。
        set gplace_seed 5
        if {[info exists ::env(G2_GPLACE_SEED)] && $::env(G2_GPLACE_SEED) ne ""} {
            set gplace_seed $::env(G2_GPLACE_SEED)
        }
        set groute_seed 11
        if {[info exists ::env(G2_GROUTE_SEED)] && $::env(G2_GROUTE_SEED) ne ""} {
            set groute_seed $::env(G2_GROUTE_SEED)
        }
        set pnr_args [list \
            -gplace_seed $gplace_seed \
            -groute_seed $groute_seed \
            -fix_hold_violation_in_route TRUE \
            -max_hold_violated_paths_num 20000]
        if {[info exists ::env(G2_INPUT_PLACE_DB)] && $::env(G2_INPUT_PLACE_DB) ne ""} {
            lappend pnr_args -input_place_db_file $::env(G2_INPUT_PLACE_DB)
        }
        if {[info exists ::env(G2_PLC_DECONGESTION)] && $::env(G2_PLC_DECONGESTION) ne ""} {
            lappend pnr_args -plc_decongestion $::env(G2_PLC_DECONGESTION)
        }
        if {[info exists ::env(G2_ROUTER_THREADS)] && $::env(G2_ROUTER_THREADS) ne ""} {
            lappend pnr_args -number_of_router_thread $::env(G2_ROUTER_THREADS)
        }
        if {[info exists ::env(G2_SHARE_ROUTER_CONTROL)] && $::env(G2_SHARE_ROUTER_CONTROL) ne ""} {
            lappend pnr_args -share_router_control_signal $::env(G2_SHARE_ROUTER_CONTROL)
        }
        if {[info exists ::env(G2_FAST_ROUTER)] && $::env(G2_FAST_ROUTER) ne ""} {
            lappend pnr_args -fast_router $::env(G2_FAST_ROUTER)
        }
        if {[info exists ::env(G2_SLACK_PRIOR)] && $::env(G2_SLACK_PRIOR) ne ""} {
            lappend pnr_args -slack_prior_in_global_router $::env(G2_SLACK_PRIOR)
        }
        if {[info exists ::env(G2_OPTIMIZE_MULTI_CORNER)] && $::env(G2_OPTIMIZE_MULTI_CORNER) eq "1"} {
            lappend pnr_args -optimize_multi_corner_timing
        }
        set place_only 0
        if {[info exists ::env(G2_PLACE_ONLY)] && $::env(G2_PLACE_ONLY) eq "1"} {
            set place_only 1
            lappend pnr_args -place_only
        }
        eval pnr $pnr_args
        if {!$place_only} {
            report_timing
            gen_bit_stream
        }
    }
}
