# 在已经完成 Compile/Synthesize/Device Map 的原 mlp_silu.pds 项目上下文内
# 重新执行默认种子 PnR、全角时序报告和位流生成。
#
# 用法：
# D:/Pango/PDS_2022.2-SP6.4/bin/pds_shell.exe \
#   -project mlp_silu.pds -file run_pnr_only.tcl
#
# 不要使用 -project_name 新建空项目，也不要把 XML 报告 dmr.db 当作阶段数据库。
pnr -max_hold_violated_paths_num 20000
report_timing
gen_bit_stream
