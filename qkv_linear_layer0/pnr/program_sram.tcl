# 仅通过 JTAG 将 Q/K/V Linear 位流下载到 FPGA 易失性 SRAM。
# 不执行任何 Flash 擦写或编程命令。
cfg_connect -ip 127.0.0.1 -port 65420
cfg_scan_chain
cfg_assign_file -file "E:/50K/AI_LLM_FPGA/pangu_ai_accel/qkv_linear_layer0/pnr_seed5/generate_bitstream/qkv_linear_top.sbit" -device_index 0
cfg_program -device_index 0
