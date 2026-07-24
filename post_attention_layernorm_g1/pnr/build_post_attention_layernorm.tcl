# PGL50H G1 layer0 post_attention_layernorm 独立构建脚本。
#
# 本阶段严格复用 E1 已真实上板通过的 RMSNorm RTL，不修改或覆盖 rmsnorm_k896
# 目录；仅在本独立 PDS 工作目录生成新的实现数据库与位流。
#
# 在本目录运行：
# D:/Pango/PDS_2022.2-SP6.4/bin/pds_shell.exe -file build_post_attention_layernorm.tcl -project_name post_attention_layernorm_g1

set ip_root "../../ipcore/pangu_ddr3_x32/pangu_ddr3_x32"

add_design "../../rmsnorm_k896/rtl/rmsnorm_k896_core.v"
add_design "../../rmsnorm_k896/rtl/rmsnorm_k896_ctrl.v"
add_design "../../rmsnorm_k896/rtl/rmsnorm_k896_top.v"
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
compile -top_module rmsnorm_k896_top
synthesize -ads -selected_syn_tool_opt 2
dev_map
pnr
report_timing
gen_bit_stream
