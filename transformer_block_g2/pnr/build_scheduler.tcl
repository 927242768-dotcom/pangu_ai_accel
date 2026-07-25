# G2 18 阶段顺序 scheduler 的独立 Compile/Synthesize 检查。
# 当前脚本只验证 RTL 可被 PDS 正式解析和综合，不代表完整 Block PnR/时序完成。

add_design "../rtl/transformer_block_scheduler.v"

set_arch -family Logos -device PGL50H -speedgrade -6 -package FBG484
compile -top_module transformer_block_scheduler
synthesize -ads -selected_syn_tool_opt 2
