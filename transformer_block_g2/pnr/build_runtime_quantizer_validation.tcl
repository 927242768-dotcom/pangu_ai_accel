# PGL50H G2 Q6.10/Q28 运行时量化自动逐位闭环完整构建。
# 在本目录运行：
# D:/Pango/PDS_2022.2-SP6.4/bin/pds_shell.exe -file build_runtime_quantizer_validation.tcl -project_name runtime_quantizer_validation

set ip_root "../../ipcore/pangu_ddr3_x32/pangu_ddr3_x32"

add_design "../rtl/unsigned_divider_rne.v"
add_design "../rtl/runtime_q10_activation_quantizer.v"
add_design "../rtl/q28_to_binary32.v"
add_design "../rtl/q28_to_binary32_sequential.v"
add_design "../rtl/runtime_q28_activation_quantizer.v"
add_design "../rtl/runtime_fp16_scale_builder.v"
add_design "../rtl/runtime_activation_quantizer_ctrl.v"
add_design "../rtl/runtime_scale_builder_ctrl.v"
add_design "../rtl/runtime_quantizer_ctrl.v"
add_design "../rtl/runtime_quantizer_trace_checker.v"
add_design "../rtl/runtime_quantizer_validation_ctrl.v"
add_design "../rtl/runtime_quantizer_validation_top.v"
add_design "../../source/uart_rx.v"
add_design "../../source/uart_tx.v"

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
compile -top_module runtime_quantizer_validation_top
synthesize -ads -selected_syn_tool_opt 2
dev_map
pnr -gplace_seed 5 -groute_seed 11 \
    -fix_hold_violation_in_route TRUE \
    -max_hold_violated_paths_num 20000
report_timing
gen_bit_stream
