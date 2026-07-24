`timescale 1ns/1ps

// 固定 M=4、K=896、group_size=64 的真实模型分组定点 GEMV 单行计算核心。
//
// 每个 64 元素 group 分成 4 个 MAC16 分块：
//   group_acc_int32 = sum(INT8 activation * INT4 weight)
//   product_q28     = group_acc_int32 * combined_scale_uq4_28
//   output_q28      = bias_q28 + sum(product_q28)
//
// 激活缓存覆盖完整 K=896；权重和 scale 缓存只保存当前输出行。
module gemv_group_q28_core #(
    parameter integer K             = 896,
    parameter integer GROUP_SIZE    = 64,
    parameter integer GROUPS        = K / GROUP_SIZE,
    parameter integer ACT_BEATS     = K / 32,
    parameter integer WEIGHT_BEATS  = K / 64
)(
    input  wire                    clk,
    input  wire                    rst_n,

    input  wire                    act_load_en,
    input  wire [4:0]              act_load_index,
    input  wire [255:0]            act_load_data,

    input  wire                    weight_load_en,
    input  wire [3:0]              weight_load_index,
    input  wire [255:0]            weight_load_data,

    input  wire                    scale_load_en,
    input  wire                    scale_load_beat_index,
    input  wire [255:0]            scale_load_data,

    input  wire                    start,
    input  wire signed [63:0]      bias_q28,
    output reg                     busy,
    output reg                     done,
    output reg signed [63:0]       y_q28
);

localparam integer K_BLOCKS = K / 16;

localparam [2:0] ST_IDLE           = 3'd0;
localparam [2:0] ST_READ_BLOCK     = 3'd1;
localparam [2:0] ST_PREPARE        = 3'd2;
localparam [2:0] ST_START_MAC      = 3'd3;
localparam [2:0] ST_WAIT_MAC       = 3'd4;
localparam [2:0] ST_MULTIPLY       = 3'd5;
localparam [2:0] ST_ACCUMULATE_Q28 = 3'd6;

reg [255:0] activation_mem [0:ACT_BEATS-1];
reg [255:0] weight_mem [0:WEIGHT_BEATS-1];
reg [31:0]  scale_mem [0:15];

reg [2:0] state;
reg [5:0] block_index;
reg [255:0] act_beat_reg;
reg [255:0] weight_beat_reg;
reg [127:0] mac_x_reg;
reg [127:0] mac_w_reg;
reg signed [31:0] group_accumulator;
reg signed [31:0] group_accumulator_final;
reg [31:0] scale_reg;
reg signed [63:0] product_reg;
reg signed [63:0] q28_accumulator;
reg signed [63:0] bias_reg;

wire [127:0] selected_x_block =
    block_index[0] ? act_beat_reg[255:128] : act_beat_reg[127:0];
reg [63:0] selected_w_block;
wire [127:0] unpacked_w_block;
wire dot_valid;
wire signed [31:0] dot_result;
wire signed [31:0] next_group_accumulator = group_accumulator + dot_result;

// scale 是无符号 32 位。前置 0 扩展为正的 signed 33 位，确保 bit31=1 时
// 仍按正数参与有符号乘法。当前数值范围保证结果可装入 signed int64。
wire signed [32:0] scale_positive = {1'b0, scale_reg};
wire signed [64:0] product_full =
    $signed(group_accumulator_final) * $signed(scale_positive);
wire signed [63:0] q28_after_product = q28_accumulator + product_reg;
wire signed [63:0] q28_with_bias = q28_after_product + bias_reg;

integer scale_lane;
always @(*) begin
    case (block_index[1:0])
        2'd0: selected_w_block = weight_beat_reg[63:0];
        2'd1: selected_w_block = weight_beat_reg[127:64];
        2'd2: selected_w_block = weight_beat_reg[191:128];
        default: selected_w_block = weight_beat_reg[255:192];
    endcase
end

int4_unpack16 u_int4_unpack16 (
    .packed_vec   (selected_w_block),
    .unpacked_vec (unpacked_w_block)
);

int8_dot16_pipe u_int8_dot16_pipe (
    .clk       (clk),
    .rst_n     (rst_n),
    .in_valid  (state == ST_START_MAC),
    .a_vec     (mac_x_reg),
    .b_vec     (mac_w_reg),
    .out_valid (dot_valid),
    .result    (dot_result)
);

// 缓存不复位；控制器只会读取本轮已经加载的有效位置。
always @(posedge clk) begin
    if (act_load_en)
        activation_mem[act_load_index] <= act_load_data;
    if (weight_load_en)
        weight_mem[weight_load_index] <= weight_load_data;
    if (scale_load_en) begin
        for (scale_lane = 0; scale_lane < 8; scale_lane = scale_lane + 1) begin
            scale_mem[{scale_load_beat_index, 3'b000} + scale_lane]
                <= scale_load_data[scale_lane*32 +: 32];
        end
    end
end

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        state                   <= ST_IDLE;
        block_index             <= 6'd0;
        act_beat_reg            <= 256'd0;
        weight_beat_reg         <= 256'd0;
        mac_x_reg               <= 128'd0;
        mac_w_reg               <= 128'd0;
        group_accumulator       <= 32'sd0;
        group_accumulator_final <= 32'sd0;
        scale_reg               <= 32'd0;
        product_reg             <= 64'sd0;
        q28_accumulator         <= 64'sd0;
        bias_reg                <= 64'sd0;
        y_q28                   <= 64'sd0;
        busy                    <= 1'b0;
        done                    <= 1'b0;
    end else begin
        done <= 1'b0;

        case (state)
            ST_IDLE: begin
                busy <= 1'b0;
                if (start) begin
                    block_index             <= 6'd0;
                    group_accumulator       <= 32'sd0;
                    group_accumulator_final <= 32'sd0;
                    product_reg             <= 64'sd0;
                    q28_accumulator         <= 64'sd0;
                    bias_reg                <= bias_q28;
                    y_q28                   <= 64'sd0;
                    busy                    <= 1'b1;
                    state                   <= ST_READ_BLOCK;
                end
            end

            ST_READ_BLOCK: begin
                // 同步读取当前 16 元素分块所在的激活拍和权重拍。
                act_beat_reg    <= activation_mem[block_index >> 1];
                weight_beat_reg <= weight_mem[block_index >> 2];
                state           <= ST_PREPARE;
            end

            ST_PREPARE: begin
                // 锁存 16 个激活和解包后的 INT4 权重。
                mac_x_reg <= selected_x_block;
                mac_w_reg <= unpacked_w_block;
                state     <= ST_START_MAC;
            end

            ST_START_MAC: begin
                // in_valid 在本状态拉高一个周期，启动显式平衡流水 MAC16。
                state <= ST_WAIT_MAC;
            end

            ST_WAIT_MAC: begin
                if (dot_valid) begin
                    if (block_index[1:0] == 2'd3) begin
                        // 每 4 个 MAC16 分块形成一个 64 元素 group 点积。
                        group_accumulator_final <= next_group_accumulator;
                        scale_reg               <= scale_mem[block_index[5:2]];
                        group_accumulator       <= 32'sd0;
                        state                   <= ST_MULTIPLY;
                    end else begin
                        group_accumulator <= next_group_accumulator;
                        block_index       <= block_index + 1'b1;
                        state             <= ST_READ_BLOCK;
                    end
                end
            end

            ST_MULTIPLY: begin
                // 单独流水一级完成 signed INT32 × unsigned UQ4.28。
                product_reg <= product_full[63:0];
                state       <= ST_ACCUMULATE_Q28;
            end

            ST_ACCUMULATE_Q28: begin
                if (block_index == K_BLOCKS - 1) begin
                    y_q28           <= q28_with_bias;
                    q28_accumulator <= 64'sd0;
                    busy            <= 1'b0;
                    done            <= 1'b1;
                    state           <= ST_IDLE;
                end else begin
                    q28_accumulator <= q28_after_product;
                    block_index     <= block_index + 1'b1;
                    state           <= ST_READ_BLOCK;
                end
            end

            default: begin
                state <= ST_IDLE;
                busy  <= 1'b0;
                done  <= 1'b0;
            end
        endcase
    end
end

endmodule
