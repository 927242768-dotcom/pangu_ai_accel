# G2 Q10/Q28 运行时量化自动闭环 controller 独立综合检查。
# 该脚本不含 DDR3 IP，仅用于在接板级顶层前尽早捕获 RTL 解析/综合错误。

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
add_design "../../source/uart_rx.v"
add_design "../../source/uart_tx.v"

set_arch -family Logos -device PGL50H -speedgrade -6 -package FBG484
compile -top_module runtime_quantizer_validation_ctrl
synthesize -ads -selected_syn_tool_opt 2
