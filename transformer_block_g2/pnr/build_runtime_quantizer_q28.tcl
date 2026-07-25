# G2 单矩阵运行时量化 DDR3 controller（Q28 源）的独立综合检查。
# 只证明 RTL 可解析/综合，不代表数值、PnR、时序或板卡通过。

add_design "../rtl/unsigned_divider_rne.v"
add_design "../rtl/runtime_q10_activation_quantizer.v"
add_design "../rtl/q28_to_binary32.v"
add_design "../rtl/q28_to_binary32_sequential.v"
add_design "../rtl/runtime_q28_activation_quantizer.v"
add_design "../rtl/runtime_fp16_scale_builder.v"
add_design "../rtl/runtime_activation_quantizer_ctrl.v"
add_design "../rtl/runtime_scale_builder_ctrl.v"
add_design "../rtl/runtime_quantizer_ctrl.v"
add_design "../rtl/runtime_quantizer_q28_top.v"

set_arch -family Logos -device PGL50H -speedgrade -6 -package FBG484
compile -top_module runtime_quantizer_q28_top
synthesize -ads -selected_syn_tool_opt 2
