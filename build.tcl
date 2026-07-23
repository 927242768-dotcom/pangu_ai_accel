add_design "source/uart_rx.v"
add_design "source/uart_tx.v"
add_design "source/int8_dot16.v"
add_design "source/top.v"
add_constraint "source/top.fdc"
set_arch -family Logos -device PGL50H -speedgrade -6 -package FBG484
compile -top_module top
synthesize -ads -selected_syn_tool_opt 2
dev_map
pnr
report_timing
gen_bit_stream
