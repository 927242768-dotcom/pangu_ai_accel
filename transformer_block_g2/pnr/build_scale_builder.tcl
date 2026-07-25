# G2 FP16 weight scale -> UQ4.28 combined scale builder 的独立综合检查。
# 仅验证 RTL 结构可综合，不代表数值仿真、PnR、时序或真实板卡通过。

add_design "../rtl/unsigned_divider_rne.v"
add_design "../rtl/runtime_fp16_scale_builder.v"

set_arch -family Logos -device PGL50H -speedgrade -6 -package FBG484
compile -top_module runtime_fp16_scale_builder
synthesize -ads -selected_syn_tool_opt 2
