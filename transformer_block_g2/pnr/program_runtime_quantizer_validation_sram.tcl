# G2 运行时量化自动逐位闭环位流：仅通过 JTAG 下载到 SRAM，绝不写 Flash。
# 使用 cdt_cfg_shell.exe 执行；本脚本不包含任何 Flash 擦除或编程命令。
cfg_connect -ip 127.0.0.1 -port 65420
cfg_scan_chain
cfg_assign_file -file "E:/50K/AI_LLM_FPGA/pangu_ai_accel/transformer_block_g2/pnr/generate_bitstream/runtime_quantizer_validation_top.sbit" -device_index 0
cfg_program -device_index 0
