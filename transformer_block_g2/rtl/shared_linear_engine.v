`timescale 1ns/1ps

// G2 共享单行 groupwise INT4 Linear engine。
//
// 统一覆盖：
//   Q/K/V/O_proj/gate/up: K=896,  k_blocks=56,  groups=14
//   down_proj:           K=4864, k_blocks=304, groups=76
//
// 一个输出行的硬件等价数学：
//   group_acc_int32 = sum_64(INT8 activation * signed INT4 weight)
//   product_q28     = group_acc_int32 * unsigned UQ4.28 combined scale
//   y_q28           = bias_q28 + sum_groups(product_q28)
//
// 外部 controller 负责：
//   1. 每个输入向量只加载一次 activation；
//   2. 每个输出行加载对应 packed weight / scale / bias；
//   3. 按目标矩阵的 M 行重复启动本 engine 并流式写回结果。
//
// 本模块只实现单行计算，不包含 DDR3 或矩阵行调度。
module shared_linear_engine #(
    parameter integer MAX_K             = 4864,
    parameter integer GROUP_SIZE        = 64,
    parameter integer MAX_GROUPS        = MAX_K / GROUP_SIZE,
    parameter integer MAX_SCALE_WORDS   = ((MAX_GROUPS + 7) / 8) * 8,
    parameter integer MAX_ACT_BEATS     = MAX_K / 32,
    parameter integer MAX_WEIGHT_BEATS  = MAX_K / 64
)(
    input  wire                    clk,
    input  wire                    rst_n,

    input  wire                    act_load_en,
    input  wire [7:0]              act_load_index,
    input  wire [255:0]            act_load_data,

    input  wire                    weight_load_en,
    input  wire [6:0]              weight_load_index,
    input  wire [255:0]            weight_load_data,

    input  wire                    scale_load_en,
    input  wire [3:0]              scale_load_beat_index,
    input  wire [255:0]            scale_load_data,

    // cfg_k_blocks 是 16 元素 block 数：56 或 304。
    // cfg_groups 必须等于 cfg_k_blocks / 4：14 或 76。
    input  wire [8:0]              cfg_k_blocks,
    input  wire [6:0]              cfg_groups,

    input  wire                    start,
    input  wire signed [63:0]      bias_q28,
    output reg                     busy,
    output reg                     done,
    output reg                     config_error,
    output reg signed [63:0]       y_q28
);

localparam [2:0] ST_IDLE           = 3'd0;
localparam [2:0] ST_READ_BLOCK     = 3'd1;
localparam [2:0] ST_PREPARE        = 3'd2;
localparam [2:0] ST_START_MAC      = 3'd3;
localparam [2:0] ST_WAIT_MAC       = 3'd4;
localparam [2:0] ST_MULTIPLY       = 3'd5;
localparam [2:0] ST_ACCUMULATE_Q28 = 3'd6;
localparam [2:0] ST_FETCH_BLOCK    = 3'd7;

reg [255:0] activation_mem [0:MAX_ACT_BEATS-1];
reg [255:0] weight_mem [0:MAX_WEIGHT_BEATS-1];
// DDR3 scale 行按 256 bit / 8 words 补齐；76 groups 必须容纳 80 words。
reg [31:0]  scale_mem [0:MAX_SCALE_WORDS-1];

reg [2:0] state;
reg [8:0] block_index;
reg [8:0] k_blocks_reg;
reg [6:0] groups_reg;
// 将宽 RAM 的地址和 4 路 lane 选择从高扇出的 block_index 独立寄存。
// ST_READ_BLOCK 只复制索引，ST_FETCH_BLOCK 下一拍执行同步 RAM 读取，
// 防止完整 G2 布局后 block_index 直接扇出到数百个 RAM/APM 端点。
reg [7:0] act_read_index_reg;
reg [6:0] weight_read_index_reg;
reg [1:0] block_lane_reg;
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

wire cfg_supported =
    ((cfg_k_blocks == 9'd56)  && (cfg_groups == 7'd14)) ||
    ((cfg_k_blocks == 9'd304) && (cfg_groups == 7'd76));
wire cfg_consistent =
    (cfg_k_blocks != 9'd0) &&
    (cfg_k_blocks[1:0] == 2'b00) &&
    ({2'b00, cfg_groups} == (cfg_k_blocks >> 2));
wire cfg_valid = cfg_supported && cfg_consistent;

wire [127:0] selected_x_block =
    block_lane_reg[0] ? act_beat_reg[255:128] : act_beat_reg[127:0];
reg [63:0] selected_w_block;
wire [127:0] unpacked_w_block;
wire dot_valid;
wire signed [31:0] dot_result;
wire signed [31:0] next_group_accumulator = group_accumulator + dot_result;

// combined scale 是 unsigned UQ4.28。显式零扩展为正的 signed 33 bit，
// 避免 bit31=1 时被有符号乘法错误解释为负数。
wire signed [32:0] scale_positive = {1'b0, scale_reg};
wire signed [64:0] product_full =
    $signed(group_accumulator_final) * $signed(scale_positive);
wire signed [63:0] q28_after_product = q28_accumulator + product_reg;
wire signed [63:0] q28_with_bias = q28_after_product + bias_reg;
wire [6:0] current_group_index = block_index[8:2];

integer scale_lane;
always @(*) begin
    case (block_lane_reg)
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

// 缓存不复位；外部 controller 只允许启动已经完整加载的配置。
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
        block_index             <= 9'd0;
        k_blocks_reg            <= 9'd0;
        groups_reg              <= 7'd0;
        act_read_index_reg      <= 8'd0;
        weight_read_index_reg   <= 7'd0;
        block_lane_reg          <= 2'd0;
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
        config_error            <= 1'b0;
    end else begin
        done <= 1'b0;

        case (state)
            ST_IDLE: begin
                busy <= 1'b0;
                if (start) begin
                    if (!cfg_valid) begin
                        config_error <= 1'b1;
                    end else begin
                        block_index             <= 9'd0;
                        k_blocks_reg            <= cfg_k_blocks;
                        groups_reg              <= cfg_groups;
                        group_accumulator       <= 32'sd0;
                        group_accumulator_final <= 32'sd0;
                        product_reg             <= 64'sd0;
                        q28_accumulator         <= 64'sd0;
                        bias_reg                <= bias_q28;
                        y_q28                   <= 64'sd0;
                        busy                    <= 1'b1;
                        config_error            <= 1'b0;
                        state                   <= ST_READ_BLOCK;
                    end
                end
            end

            ST_READ_BLOCK: begin
                act_read_index_reg    <= block_index[8:1];
                weight_read_index_reg <= block_index[8:2];
                block_lane_reg        <= block_index[1:0];
                state                 <= ST_FETCH_BLOCK;
            end

            ST_FETCH_BLOCK: begin
                act_beat_reg    <= activation_mem[act_read_index_reg];
                weight_beat_reg <= weight_mem[weight_read_index_reg];
                state           <= ST_PREPARE;
            end

            ST_PREPARE: begin
                mac_x_reg <= selected_x_block;
                mac_w_reg <= unpacked_w_block;
                state     <= ST_START_MAC;
            end

            ST_START_MAC: begin
                state <= ST_WAIT_MAC;
            end

            ST_WAIT_MAC: begin
                if (dot_valid) begin
                    if (block_index[1:0] == 2'd3) begin
                        group_accumulator_final <= next_group_accumulator;
                        scale_reg               <= scale_mem[current_group_index];
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
                product_reg <= product_full[63:0];
                state       <= ST_ACCUMULATE_Q28;
            end

            ST_ACCUMULATE_Q28: begin
                if (block_index == k_blocks_reg - 1'b1) begin
                    // cfg_groups 在启动时已验证与 k_blocks/4 一致；此断点同时
                    // 证明最后使用的 scale index 正好为 groups_reg-1。
                    if (current_group_index != groups_reg - 1'b1) begin
                        config_error <= 1'b1;
                        busy         <= 1'b0;
                        state        <= ST_IDLE;
                    end else begin
                        y_q28           <= q28_with_bias;
                        q28_accumulator <= 64'sd0;
                        busy            <= 1'b0;
                        done            <= 1'b1;
                        state           <= ST_IDLE;
                    end
                end else begin
                    q28_accumulator <= q28_after_product;
                    block_index     <= block_index + 1'b1;
                    state           <= ST_READ_BLOCK;
                end
            end

            default: begin
                state        <= ST_IDLE;
                busy         <= 1'b0;
                done         <= 1'b0;
                config_error <= 1'b1;
            end
        endcase
    end
end

endmodule
