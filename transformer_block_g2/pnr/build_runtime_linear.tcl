# G2 运行时参数化 Linear DDR3 controller 的独立 Compile/Synthesize 检查。
# 当前脚本不包含 DDR3 PHY 顶层和板级 I/O，因此只证明 RTL 可解析、可综合。

add_design "../rtl/int4_unpack16.v"
add_design "../rtl/int8_dot16_pipe.v"
add_design "../rtl/shared_linear_engine.v"
add_design "../rtl/runtime_linear_ctrl.v"

set_arch -family Logos -device PGL50H -speedgrade -6 -package FBG484
compile -top_module runtime_linear_ctrl
synthesize -ads -selected_syn_tool_opt 2
