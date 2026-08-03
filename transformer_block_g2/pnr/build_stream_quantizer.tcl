# G2 两遍 DDR3 流式 activation quantizer 独立 Compile/Synthesize。
add_design "../rtl/unsigned_divider_rne.v"
add_design "../rtl/q28_to_binary32_sequential.v"
add_design "../rtl/runtime_activation_quantizer_stream_ctrl.v"
set_arch -family Logos -device PGL50H -speedgrade -6 -package FBG484
compile -top_module runtime_activation_quantizer_stream_ctrl
synthesize -ads -selected_syn_tool_opt 2
