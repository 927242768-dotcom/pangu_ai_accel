`timescale 1ns/1ps

// layer0 MLP SiLU(gate) × up 核心。
//
// 输入：
//   SiLU(gate) [4864] signed int16 Q6.10
//   up_proj    [4864] signed int64 Q28
// 输出：
//   product    [4864] signed int64 Q28
//
// 固定数值规则：
//   1. 完整 signed 16×64 乘法，保留 signed 80 bit Q38；
//   2. 对绝对值执行 round-to-nearest-even 右移 10 位；
//   3. 恢复符号并显式饱和到 signed int64；
//   4. 禁止任何隐含截断。
//
// 存储布局：一个 SiLU beat 含 16 个 int16；四个 up beat 共含对应 16 个 int64。
// 计算时复用一个 16×16 无符号乘法器，以四个 limb 精确重构 80 位乘积。
module mlp_silu_up_mul_core #(
    parameter integer M             = 4864,
    parameter integer INPUT_GROUPS  = M / 16,
    parameter integer SILU_BEATS    = M / 16,
    parameter integer UP_BEATS      = M / 4,
    parameter integer RESULT_BEATS  = M / 4
)(
    input  wire                    clk,
    input  wire                    rst_n,

    input  wire                    silu_load_en,
    input  wire [8:0]              silu_load_index,
    input  wire [255:0]            silu_load_data,

    input  wire                    up_load_en,
    input  wire [10:0]             up_load_index,
    input  wire [255:0]            up_load_data,

    input  wire                    start,
    output reg                     busy,
    output reg                     done,

    output reg  [255:0]            result_data,
    output reg                     result_valid,
    input  wire                    result_ready
);

localparam [3:0] ST_IDLE      = 4'd0;
localparam [3:0] ST_READ      = 4'd1;
localparam [3:0] ST_CAPTURE   = 4'd2;
localparam [3:0] ST_ABS       = 4'd3;
localparam [3:0] ST_MUL0      = 4'd4;
localparam [3:0] ST_MUL1      = 4'd5;
localparam [3:0] ST_MUL2      = 4'd6;
localparam [3:0] ST_MUL3      = 4'd7;
localparam [3:0] ST_ROUND     = 4'd8;
localparam [3:0] ST_SATURATE  = 4'd9;
localparam [3:0] ST_PACK      = 4'd10;
localparam [3:0] ST_WAIT      = 4'd11;

reg [255:0] silu_mem [0:SILU_BEATS-1];
reg [255:0] up_mem0  [0:INPUT_GROUPS-1];
reg [255:0] up_mem1  [0:INPUT_GROUPS-1];
reg [255:0] up_mem2  [0:INPUT_GROUPS-1];
reg [255:0] up_mem3  [0:INPUT_GROUPS-1];

reg [3:0] state;
reg [8:0] group_index;
reg [3:0] lane_index;
reg [255:0] silu_beat_reg;
reg [255:0] up_beat0_reg;
reg [255:0] up_beat1_reg;
reg [255:0] up_beat2_reg;
reg [255:0] up_beat3_reg;
reg [255:0] output_pack;

reg signed [15:0] silu_lane_reg;
reg signed [63:0] up_lane_reg;
reg               product_negative_reg;
reg [15:0]        silu_magnitude_reg;
reg [63:0]        up_magnitude_reg;
reg [79:0]        product_magnitude_reg;
reg [69:0]        rounded_magnitude_reg;
reg [63:0]        output_value_reg;

reg [15:0] selected_silu_bits;
reg [63:0] selected_up_bits;
always @(*) begin
    selected_silu_bits = silu_beat_reg[lane_index*16 +: 16];
    case (lane_index[3:2])
        2'd0: selected_up_bits = up_beat0_reg[lane_index[1:0]*64 +: 64];
        2'd1: selected_up_bits = up_beat1_reg[lane_index[1:0]*64 +: 64];
        2'd2: selected_up_bits = up_beat2_reg[lane_index[1:0]*64 +: 64];
        default: selected_up_bits = up_beat3_reg[lane_index[1:0]*64 +: 64];
    endcase
end

reg [15:0] selected_up_limb;
always @(*) begin
    case (state)
        ST_MUL0: selected_up_limb = up_magnitude_reg[15:0];
        ST_MUL1: selected_up_limb = up_magnitude_reg[31:16];
        ST_MUL2: selected_up_limb = up_magnitude_reg[47:32];
        default: selected_up_limb = up_magnitude_reg[63:48];
    endcase
end

wire [31:0] partial_product_wire = silu_magnitude_reg * selected_up_limb;
wire [79:0] partial_product_ext = {{48{1'b0}}, partial_product_wire};

function [69:0] rne_shift10_unsigned80;
    input [79:0] magnitude;
    reg [69:0] quotient;
    reg [9:0] remainder;
    begin
        quotient = magnitude >> 10;
        remainder = magnitude[9:0];
        if ((remainder > 10'h200) ||
            ((remainder == 10'h200) && quotient[0]))
            quotient = quotient + 1'b1;
        rne_shift10_unsigned80 = quotient;
    end
endfunction

localparam [69:0] POSITIVE_LIMIT = {6'd0, 64'h7fff_ffff_ffff_ffff};
localparam [69:0] NEGATIVE_LIMIT = {6'd0, 64'h8000_0000_0000_0000};

reg [255:0] output_pack_next;
always @(*) begin
    output_pack_next = output_pack;
    output_pack_next[lane_index[1:0]*64 +: 64] = output_value_reg;
end

always @(posedge clk) begin
    if (silu_load_en)
        silu_mem[silu_load_index] <= silu_load_data;

    if (up_load_en) begin
        case (up_load_index[1:0])
            2'd0: up_mem0[up_load_index[10:2]] <= up_load_data;
            2'd1: up_mem1[up_load_index[10:2]] <= up_load_data;
            2'd2: up_mem2[up_load_index[10:2]] <= up_load_data;
            2'd3: up_mem3[up_load_index[10:2]] <= up_load_data;
        endcase
    end
end

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        state                   <= ST_IDLE;
        group_index             <= 9'd0;
        lane_index              <= 4'd0;
        silu_beat_reg           <= 256'd0;
        up_beat0_reg            <= 256'd0;
        up_beat1_reg            <= 256'd0;
        up_beat2_reg            <= 256'd0;
        up_beat3_reg            <= 256'd0;
        output_pack             <= 256'd0;
        silu_lane_reg           <= 16'sd0;
        up_lane_reg             <= 64'sd0;
        product_negative_reg    <= 1'b0;
        silu_magnitude_reg      <= 16'd0;
        up_magnitude_reg        <= 64'd0;
        product_magnitude_reg   <= 80'd0;
        rounded_magnitude_reg   <= 70'd0;
        output_value_reg        <= 64'd0;
        result_data             <= 256'd0;
        result_valid            <= 1'b0;
        busy                    <= 1'b0;
        done                    <= 1'b0;
    end else begin
        done <= 1'b0;

        case (state)
            ST_IDLE: begin
                busy         <= 1'b0;
                result_valid <= 1'b0;
                if (start) begin
                    group_index <= 9'd0;
                    lane_index  <= 4'd0;
                    output_pack <= 256'd0;
                    busy        <= 1'b1;
                    state       <= ST_READ;
                end
            end

            ST_READ: begin
                silu_beat_reg <= silu_mem[group_index];
                up_beat0_reg  <= up_mem0[group_index];
                up_beat1_reg  <= up_mem1[group_index];
                up_beat2_reg  <= up_mem2[group_index];
                up_beat3_reg  <= up_mem3[group_index];
                lane_index    <= 4'd0;
                output_pack   <= 256'd0;
                state         <= ST_CAPTURE;
            end

            ST_CAPTURE: begin
                silu_lane_reg <= $signed(selected_silu_bits);
                up_lane_reg   <= $signed(selected_up_bits);
                state         <= ST_ABS;
            end

            ST_ABS: begin
                product_negative_reg <= silu_lane_reg[15] ^ up_lane_reg[63];
                silu_magnitude_reg <= silu_lane_reg[15] ?
                    (~silu_lane_reg + 1'b1) : silu_lane_reg;
                up_magnitude_reg <= up_lane_reg[63] ?
                    (~up_lane_reg + 1'b1) : up_lane_reg;
                product_magnitude_reg <= 80'd0;
                state <= ST_MUL0;
            end

            ST_MUL0: begin
                product_magnitude_reg <= partial_product_ext;
                state <= ST_MUL1;
            end

            ST_MUL1: begin
                product_magnitude_reg <=
                    product_magnitude_reg + (partial_product_ext << 16);
                state <= ST_MUL2;
            end

            ST_MUL2: begin
                product_magnitude_reg <=
                    product_magnitude_reg + (partial_product_ext << 32);
                state <= ST_MUL3;
            end

            ST_MUL3: begin
                product_magnitude_reg <=
                    product_magnitude_reg + (partial_product_ext << 48);
                state <= ST_ROUND;
            end

            ST_ROUND: begin
                rounded_magnitude_reg <= rne_shift10_unsigned80(product_magnitude_reg);
                state <= ST_SATURATE;
            end

            ST_SATURATE: begin
                if (product_negative_reg) begin
                    if (rounded_magnitude_reg >= NEGATIVE_LIMIT)
                        output_value_reg <= 64'h8000_0000_0000_0000;
                    else
                        output_value_reg <= (~rounded_magnitude_reg[63:0]) + 1'b1;
                end else begin
                    if (rounded_magnitude_reg > POSITIVE_LIMIT)
                        output_value_reg <= 64'h7fff_ffff_ffff_ffff;
                    else
                        output_value_reg <= rounded_magnitude_reg[63:0];
                end
                state <= ST_PACK;
            end

            ST_PACK: begin
                output_pack <= output_pack_next;
                if (lane_index[1:0] == 2'd3) begin
                    result_data  <= output_pack_next;
                    result_valid <= 1'b1;
                    state        <= ST_WAIT;
                end else begin
                    lane_index <= lane_index + 1'b1;
                    state      <= ST_CAPTURE;
                end
            end

            ST_WAIT: begin
                if (result_valid && result_ready) begin
                    result_valid <= 1'b0;
                    output_pack  <= 256'd0;
                    if (lane_index == 4'd15) begin
                        if (group_index == INPUT_GROUPS - 1) begin
                            busy  <= 1'b0;
                            done  <= 1'b1;
                            state <= ST_IDLE;
                        end else begin
                            group_index <= group_index + 1'b1;
                            state       <= ST_READ;
                        end
                    end else begin
                        lane_index <= lane_index + 1'b1;
                        state      <= ST_CAPTURE;
                    end
                end
            end

            default: begin
                state        <= ST_IDLE;
                busy         <= 1'b0;
                result_valid <= 1'b0;
            end
        endcase
    end
end

wire _unused_parameters = (UP_BEATS == RESULT_BEATS);

endmodule
