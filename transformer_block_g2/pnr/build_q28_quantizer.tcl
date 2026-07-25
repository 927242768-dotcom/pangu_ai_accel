# G2 Q28 -> binary32 -> INT8 运行时量化器的独立 Compile/Synthesize 检查。
# 仅验证 RTL 结构可综合，不代表数值仿真、PnR、时序或真实板卡通过。

add_design "../rtl/unsigned_divider_rne.v"
add_design "../rtl/q28_to_binary32.v"
add_design "../rtl/q28_to_binary32_sequential.v"
add_design "../rtl/runtime_q28_activation_quantizer.v"

set_arch -family Logos -device PGL50H -speedgrade -6 -package FBG484
compile -top_module runtime_q28_activation_quantizer
synthesize -ads -selected_syn_tool_opt 2
