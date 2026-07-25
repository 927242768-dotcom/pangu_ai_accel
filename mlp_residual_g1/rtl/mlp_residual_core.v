`timescale 1ns/1ps

// layer0 MLP 第二处残差核心。
//
// 输入：
//   residual_hidden_q10[896] : 完整 Attention 第一处残差后的 signed int16 Q6.10
//   down_proj_q28[896]       : 已验证 down_proj signed int64 Q28
//
// 规则：
//   1. down_proj Q28 执行 signed RNE 右移 18 位；
//   2. 显式饱和到 signed int16 Q6.10；
//   3. 与 residual hidden 扩展相加；
//   4. 最终再次显式饱和到 signed int16 Q6.10。
module mlp_residual_core #(
    parameter integer K            = 896,
    parameter integer RESULT_BEATS = K / 16
)(
    input  wire                    clk,
    input  wire                    rst_n,

    input  wire                    hidden_load_en,
    input  wire [5:0]              hidden_load_index,
    input  wire [255:0]            hidden_load_data,

    input  wire                    down_load_en,
    input  wire [7:0]              down_load_index,
    input  wire [255:0]            down_load_data,

    input  wire                    start,
    output reg                     busy,
    output reg                     done,

    output reg  [255:0]            result_data,
    output reg                     result_valid,
    input  wire                    result_ready
);

localparam [3:0] ST_IDLE      = 4'd0;
localparam [3:0] ST_READ_BEAT = 4'd1;
localparam [3:0] ST_CAPTURE   = 4'd2;
localparam [3:0] ST_ABS       = 4'd3;
localparam [3:0] ST_ROUND     = 4'd4;
localparam [3:0] ST_SAT_DOWN  = 4'd5;
localparam [3:0] ST_ADD       = 4'd6;
localparam [3:0] ST_SAT_OUT   = 4'd7;
localparam [3:0] ST_PACK      = 4'd8;
localparam [3:0] ST_WAIT      = 4'd9;

reg [255:0] hidden_mem [0:RESULT_BEATS-1];
reg [255:0] down_mem0 [0:RESULT_BEATS-1];
reg [255:0] down_mem1 [0:RESULT_BEATS-1];
reg [255:0] down_mem2 [0:RESULT_BEATS-1];
reg [255:0] down_mem3 [0:RESULT_BEATS-1];

reg [3:0] state;
reg [5:0] beat_index;
reg [3:0] lane_index;
reg [255:0] hidden_beat_reg;
reg [255:0] down_beat0_reg;
reg [255:0] down_beat1_reg;
reg [255:0] down_beat2_reg;
reg [255:0] down_beat3_reg;
reg [255:0] output_pack;

reg signed [15:0] hidden_lane_reg;
reg signed [63:0] down_lane_reg;
reg               down_negative_reg;
reg [63:0]        down_magnitude_reg;
reg signed [63:0] down_scaled_wide_reg;
reg signed [15:0] down_q10_reg;
reg signed [17:0] residual_sum_reg;
reg [15:0]        output_saturated_reg;

reg [63:0] selected_down_bits;
always @(*) begin
    case (lane_index[3:2])
        2'd0: selected_down_bits = down_beat0_reg[lane_index[1:0]*64 +: 64];
        2'd1: selected_down_bits = down_beat1_reg[lane_index[1:0]*64 +: 64];
        2'd2: selected_down_bits = down_beat2_reg[lane_index[1:0]*64 +: 64];
        default: selected_down_bits = down_beat3_reg[lane_index[1:0]*64 +: 64];
    endcase
end

wire signed [15:0] selected_hidden =
    $signed(hidden_beat_reg[lane_index*16 +: 16]);

function signed [63:0] rne_shift18_from_magnitude;
    input [63:0] magnitude;
    input        negative;
    reg [63:0] quotient;
    reg [17:0] remainder;
    begin
        quotient = magnitude >> 18;
        remainder = magnitude[17:0];
        if ((remainder > 18'h20000) ||
            ((remainder == 18'h20000) && quotient[0]))
            quotient = quotient + 1'b1;
        rne_shift18_from_magnitude = negative ? -$signed(quotient) : $signed(quotient);
    end
endfunction

function signed [15:0] saturate_signed16_from64;
    input signed [63:0] value;
    begin
        if (value > 64'sd32767)
            saturate_signed16_from64 = 16'sh7fff;
        else if (value < -64'sd32768)
            saturate_signed16_from64 = 16'sh8000;
        else
            saturate_signed16_from64 = value[15:0];
    end
endfunction

function [15:0] saturate_signed16_from18;
    input signed [17:0] value;
    begin
        if (value > 18'sd32767)
            saturate_signed16_from18 = 16'h7fff;
        else if (value < -18'sd32768)
            saturate_signed16_from18 = 16'h8000;
        else
            saturate_signed16_from18 = value[15:0];
    end
endfunction

reg [255:0] output_pack_next;
always @(*) begin
    output_pack_next = output_pack;
    output_pack_next[lane_index*16 +: 16] = output_saturated_reg;
end

always @(posedge clk) begin
    if (hidden_load_en)
        hidden_mem[hidden_load_index] <= hidden_load_data;

    if (down_load_en) begin
        case (down_load_index[1:0])
            2'd0: down_mem0[down_load_index[7:2]] <= down_load_data;
            2'd1: down_mem1[down_load_index[7:2]] <= down_load_data;
            2'd2: down_mem2[down_load_index[7:2]] <= down_load_data;
            2'd3: down_mem3[down_load_index[7:2]] <= down_load_data;
        endcase
    end
end

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        state                <= ST_IDLE;
        beat_index           <= 6'd0;
        lane_index           <= 4'd0;
        hidden_beat_reg      <= 256'd0;
        down_beat0_reg       <= 256'd0;
        down_beat1_reg       <= 256'd0;
        down_beat2_reg       <= 256'd0;
        down_beat3_reg       <= 256'd0;
        output_pack          <= 256'd0;
        hidden_lane_reg      <= 16'sd0;
        down_lane_reg        <= 64'sd0;
        down_negative_reg    <= 1'b0;
        down_magnitude_reg   <= 64'd0;
        down_scaled_wide_reg <= 64'sd0;
        down_q10_reg         <= 16'sd0;
        residual_sum_reg     <= 18'sd0;
        output_saturated_reg <= 16'd0;
        result_data          <= 256'd0;
        result_valid         <= 1'b0;
        busy                 <= 1'b0;
        done                 <= 1'b0;
    end else begin
        done <= 1'b0;

        case (state)
            ST_IDLE: begin
                busy         <= 1'b0;
                result_valid <= 1'b0;
                if (start) begin
                    beat_index  <= 6'd0;
                    lane_index  <= 4'd0;
                    output_pack <= 256'd0;
                    busy        <= 1'b1;
                    state       <= ST_READ_BEAT;
                end
            end

            ST_READ_BEAT: begin
                hidden_beat_reg <= hidden_mem[beat_index];
                down_beat0_reg  <= down_mem0[beat_index];
                down_beat1_reg  <= down_mem1[beat_index];
                down_beat2_reg  <= down_mem2[beat_index];
                down_beat3_reg  <= down_mem3[beat_index];
                lane_index      <= 4'd0;
                output_pack     <= 256'd0;
                state           <= ST_CAPTURE;
            end

            ST_CAPTURE: begin
                hidden_lane_reg <= selected_hidden;
                down_lane_reg   <= $signed(selected_down_bits);
                state           <= ST_ABS;
            end

            ST_ABS: begin
                down_negative_reg  <= down_lane_reg[63];
                down_magnitude_reg <= down_lane_reg[63] ?
                    (~down_lane_reg + 1'b1) : down_lane_reg;
                state <= ST_ROUND;
            end

            ST_ROUND: begin
                down_scaled_wide_reg <= rne_shift18_from_magnitude(
                    down_magnitude_reg,
                    down_negative_reg
                );
                state <= ST_SAT_DOWN;
            end

            ST_SAT_DOWN: begin
                down_q10_reg <= saturate_signed16_from64(down_scaled_wide_reg);
                state <= ST_ADD;
            end

            ST_ADD: begin
                residual_sum_reg <=
                    {{2{hidden_lane_reg[15]}}, hidden_lane_reg} +
                    {{2{down_q10_reg[15]}}, down_q10_reg};
                state <= ST_SAT_OUT;
            end

            ST_SAT_OUT: begin
                output_saturated_reg <= saturate_signed16_from18(residual_sum_reg);
                state <= ST_PACK;
            end

            ST_PACK: begin
                output_pack <= output_pack_next;
                if (lane_index == 4'd15) begin
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
                    if (beat_index == RESULT_BEATS - 1) begin
                        busy  <= 1'b0;
                        done  <= 1'b1;
                        state <= ST_IDLE;
                    end else begin
                        beat_index <= beat_index + 1'b1;
                        state      <= ST_READ_BEAT;
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

endmodule
