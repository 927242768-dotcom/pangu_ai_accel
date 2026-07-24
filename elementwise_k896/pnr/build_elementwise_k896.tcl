# PGL50H K=896 signed Q6.10 元素级算子构建脚本。
# 在本目录运行：
# D:/Pango/PDS_2022.2-SP6.4/bin/pds_shell.exe -file build_elementwise_k896.tcl -project_name elementwise_k896

set ip_root "../../ipcore/pangu_ddr3_x32/pangu_ddr3_x32"

add_design "../rtl/elementwise_k896_core.v"
add_design "../rtl/elementwise_k896_ctrl.v"
add_design "../rtl/elementwise_k896_top.v"
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

# 复用已真实上板通过的 DDR3、时钟、UART 和 LED 约束。
add_constraint "$ip_root/pnr/ddr_test.fdc"

set_arch -family Logos -device PGL50H -speedgrade -6 -package FBG484
compile -top_module elementwise_k896_top
synthesize -ads -selected_syn_tool_opt 2
dev_map
# 固定使用已通过多角时序的布局/布线种子，并强化保持修复。
pnr -gplace_seed 17 -groute_seed 29 \
    -fix_hold_violation_in_route TRUE \
    -max_hold_violated_paths_num 20000
report_timing
gen_bit_stream
