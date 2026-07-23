# 仅通过 JTAG 将 D1.3 GEMV 性能计数位流下载到 FPGA 易失性 SRAM。
# 不执行任何 Flash 擦写或编程命令。
cfg_connect -ip 127.0.0.1 -port 65420
cfg_scan_chain
cfg_assign_file -file "E:/50K/AI_LLM_FPGA/pangu_ai_accel/gemv_int4_perf/pnr/generate_bitstream/gemv_param_top.sbit" -device_index 0
cfg_program -device_index 0
