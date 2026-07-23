# 仅通过 JTAG 将 GEMV 验证位流下载到 FPGA 易失性 SRAM。
# 不执行任何 Flash 擦写或编程命令。
cfg_connect -ip 127.0.0.1 -port 65420
cfg_scan_chain
cfg_assign_file -file "E:/50K/AI_LLM_FPGA/pangu_ai_accel/gemv_int4_m4k64/pnr/generate_bitstream/gemv_m4k64_top.sbit" -device_index 0
cfg_program -device_index 0
