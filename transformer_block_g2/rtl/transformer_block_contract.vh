`ifndef PANGU_TRANSFORMER_BLOCK_G2_CONTRACT_VH
`define PANGU_TRANSFORMER_BLOCK_G2_CONTRACT_VH

// G2 layer0 Transformer Block 集成契约。
// DDR3 Controller 地址单位为 32 bit；下列 *_CTRL_ADDR 均为控制器地址。
// Python 权威来源：model_tools/transformer_block_reference.py

`define G2_HIDDEN_SIZE                  16'd896
`define G2_INTERMEDIATE_SIZE            16'd4864
`define G2_Q_HEADS                      5'd14
`define G2_KV_HEADS                     3'd2
`define G2_HEAD_DIM                     7'd64
`define G2_MAX_WINDOW                   5'd16

// 动态 scratch / 中间张量。
`define G2_BLOCK_HIDDEN_CTRL_ADDR       32'h0000_0000
`define G2_INPUT_NORM_CTRL_ADDR         32'h0000_0400
`define G2_Q_Q28_CTRL_ADDR              32'h0000_0800
`define G2_K_Q28_CTRL_ADDR              32'h0000_1000
`define G2_V_Q28_CTRL_ADDR              32'h0000_1400
`define G2_Q_ROPE_CTRL_ADDR             32'h0000_1800
`define G2_K_ROPE_CTRL_ADDR             32'h0000_2000
`define G2_SCORES_CTRL_ADDR             32'h0000_2400
`define G2_PROBABILITIES_CTRL_ADDR      32'h0000_2800
`define G2_ATTN_CONCAT_CTRL_ADDR        32'h0000_2c00
`define G2_OPROJ_CTRL_ADDR              32'h0000_3400
`define G2_ATTN_RESIDUAL_CTRL_ADDR      32'h0000_3c00
`define G2_POST_NORM_CTRL_ADDR          32'h0000_4000
`define G2_GATE_CTRL_ADDR               32'h0000_4400
`define G2_UP_CTRL_ADDR                 32'h0000_6c00
`define G2_SILU_GATE_CTRL_ADDR          32'h0000_9400
`define G2_SILU_UP_CTRL_ADDR            32'h0000_a000
`define G2_DOWN_CTRL_ADDR               32'h0000_c800
`define G2_BLOCK_OUTPUT_CTRL_ADDR       32'h0000_d000
`define G2_LINEAR_ACT_INT8_CTRL_ADDR     32'h0000_d400
`define G2_LINEAR_QUANT_META_CTRL_ADDR   32'h0000_dc00
`define G2_EXEC_PAYLOAD_CTRL_ADDR       32'h0004_0000

// 常驻表与 RMSNorm 参数。
`define G2_INPUT_RMS_GAMMA_CTRL_ADDR    32'h0008_0000
`define G2_POST_RMS_GAMMA_CTRL_ADDR     32'h0008_0400
`define G2_RMS_LUT_CTRL_ADDR            32'h0008_0800
`define G2_SOFTMAX_LUT_CTRL_ADDR        32'h0008_0c00
`define G2_SILU_PWL_CTRL_ADDR           32'h0008_1000
`define G2_ROPE_TRIG_CTRL_ADDR          32'h0008_1400

// layer0 Linear 参数区。权重常驻；combined scale 按当前激活重建。
`define G2_Q_WEIGHT_CTRL_ADDR           32'h0040_0000
`define G2_Q_SCALE_CTRL_ADDR            32'h0041_8800
`define G2_Q_BIAS_CTRL_ADDR             32'h0041_c000
`define G2_K_WEIGHT_CTRL_ADDR           32'h0041_dc00
`define G2_K_SCALE_CTRL_ADDR            32'h0042_1400
`define G2_K_BIAS_CTRL_ADDR             32'h0042_1c00
`define G2_V_WEIGHT_CTRL_ADDR           32'h0042_2000
`define G2_V_SCALE_CTRL_ADDR            32'h0042_5800
`define G2_V_BIAS_CTRL_ADDR             32'h0042_6000
`define G2_OPROJ_WEIGHT_CTRL_ADDR       32'h0042_6400
`define G2_OPROJ_SCALE_CTRL_ADDR        32'h0043_ec00
`define G2_GATE_WEIGHT_CTRL_ADDR        32'h0044_2400
`define G2_GATE_SCALE_CTRL_ADDR         32'h004c_7400
`define G2_UP_WEIGHT_CTRL_ADDR          32'h004d_a400
`define G2_UP_SCALE_CTRL_ADDR           32'h0055_f400
`define G2_DOWN_WEIGHT_CTRL_ADDR        32'h0057_2400
`define G2_DOWN_SCALE_CTRL_ADDR         32'h005f_7400

// P50 原始 FP16 weight scale 常驻区；运行时量化器据此重建上方 UQ4.28。
`define G2_Q_RAW_SCALE_CTRL_ADDR        32'h0060_c000
`define G2_K_RAW_SCALE_CTRL_ADDR        32'h0060_dc00
`define G2_V_RAW_SCALE_CTRL_ADDR        32'h0060_e000
`define G2_OPROJ_RAW_SCALE_CTRL_ADDR    32'h0060_e400
`define G2_GATE_RAW_SCALE_CTRL_ADDR     32'h0061_0000
`define G2_UP_RAW_SCALE_CTRL_ADDR       32'h0061_8800
`define G2_DOWN_RAW_SCALE_CTRL_ADDR     32'h0062_1000

// F3 已验证 KV Cache 地址布局，禁止重新解释地址单位。
`define G2_KV_BASE_CTRL_ADDR            32'h0200_0000
`define G2_KV_LAYER_STRIDE_CTRL         32'h0080_0000
`define G2_KV_TOKEN_STRIDE_CTRL         32'h0000_0200
`define G2_KV_V_OFFSET_CTRL             32'h0000_0100
`define G2_KV_MAX_POSITIONS             15'd16384

// 顶层顺序调度阶段 ID。
`define G2_STAGE_IDLE                   5'h00
`define G2_STAGE_INPUT_RMS              5'h01
`define G2_STAGE_QKV_QUANT              5'h02
`define G2_STAGE_Q_LINEAR               5'h03
`define G2_STAGE_K_LINEAR               5'h04
`define G2_STAGE_V_LINEAR               5'h05
`define G2_STAGE_ROPE                   5'h06
`define G2_STAGE_KV_WRITE               5'h07
`define G2_STAGE_ATTENTION_SCORE        5'h08
`define G2_STAGE_SOFTMAX                5'h09
`define G2_STAGE_ATTENTION_OUTPUT       5'h0a
`define G2_STAGE_OPROJ_QUANT            5'h0b
`define G2_STAGE_OPROJ_LINEAR           5'h0c
`define G2_STAGE_RESIDUAL1              5'h0d
`define G2_STAGE_POST_RMS               5'h0e
`define G2_STAGE_GATE_UP_QUANT          5'h0f
`define G2_STAGE_GATE_LINEAR            5'h10
`define G2_STAGE_UP_LINEAR              5'h11
`define G2_STAGE_SILU                   5'h12
`define G2_STAGE_SILU_UP_MUL            5'h13
`define G2_STAGE_DOWN_QUANT             5'h14
`define G2_STAGE_DOWN_LINEAR            5'h15
`define G2_STAGE_RESIDUAL2              5'h16
`define G2_STAGE_DONE                   5'h17
`define G2_STAGE_ERROR                  5'h1f

// 第一版顶层握手：start 仅在 idle 接受；busy 覆盖完整执行；done 为单拍；
// error_code 为粘滞状态，必须复位后才能再次启动。

`endif
