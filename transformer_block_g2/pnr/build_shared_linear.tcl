# G2 共享 Linear engine 的独立 Compile/Synthesize 检查。
# 当前脚本只验证 RTL 可被 PDS 正式解析和综合，不代表完整 Block PnR/时序完成。

add_design "../rtl/int4_unpack16.v"
add_design "../rtl/int8_dot16_pipe.v"
add_design "../rtl/shared_linear_engine.v"

set_arch -family Logos -device PGL50H -speedgrade -6 -package FBG484
compile -top_module shared_linear_engine
synthesize -ads -selected_syn_tool_opt 2
