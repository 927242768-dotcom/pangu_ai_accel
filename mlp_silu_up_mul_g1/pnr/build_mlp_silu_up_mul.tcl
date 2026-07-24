# PGL50H layer0 MLP SiLU(gate) × up 构建脚本。
# 在本目录运行：
# D:/Pango/PDS_2022.2-SP6.4/bin/pds_shell.exe -file build_mlp_silu_up_mul.tcl -project_name mlp_silu_up_mul

set ip_root "../../ipcore/pangu_ddr3_x32/pangu_ddr3_x32"

add_design "../rtl/mlp_silu_up_mul_core.v"
add_design "../rtl/mlp_silu_up_mul_ctrl.v"
add_design "../rtl/mlp_silu_up_mul_top.v"
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

# 复用已经真实上板通过的 DDR3、时钟、UART 和 LED 约束。
add_constraint "$ip_root/pnr/ddr_test.fdc"

set_arch -family Logos -device PGL50H -speedgrade -6 -package FBG484
compile -top_module mlp_silu_up_mul_top
synthesize -ads -selected_syn_tool_opt 2
dev_map
pnr -max_hold_violated_paths_num 20000
report_timing
gen_bit_stream
