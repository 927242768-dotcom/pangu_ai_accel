# PGL50H packed INT4 GEMV（M=4、K=64）独立构建脚本。
# 在本目录运行：
# D:/Pango/PDS_2022.2-SP6.4/bin/pds_shell.exe -file build_gemv_m4k64.tcl -project_name gemv_m4k64_top

set ip_root "../../ipcore/pangu_ddr3_x32/pangu_ddr3_x32"

# 自研 RTL。
add_design "../rtl/gemv_m4k64_core.v"
add_design "../rtl/gemv_m4k64_ctrl.v"
add_design "../rtl/gemv_m4k64_top.v"
add_design "../../ddr_mac16_integration/rtl/int4_unpack16.v"
add_design "../../source/uart_rx.v"
add_design "../../source/uart_tx.v"
add_design "../../source/int8_dot16.v"

# DDR3 Controller + PHY 生成文件。
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
compile -top_module gemv_m4k64_top
synthesize -ads -selected_syn_tool_opt 2
dev_map
pnr
report_timing
gen_bit_stream
