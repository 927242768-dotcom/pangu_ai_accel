`timescale 1ns/1ps

// layer0 MLP SiLU(gate) 核心。
//
// 输入：gate_proj [4864] signed int64 Q28。
// 输出：SiLU(gate) [4864] signed int16 Q6.10。
//
// 固定数值规则：
//   1. Q28 对称 signed RNE 右移 18 位；
//   2. 显式饱和到 signed Q6.10 int16；
//   3. 复用 E2 已验证的 64 段端点 PWL SiLU；
//   4. x<-8 输出 0，x>=8 输出 x。
//
// 4864 个 Q28 输入按 4 个 bank 缓存；每个输出拍同步读取 16 个输入，
// 逐 lane 流水完成缩放、PWL 插值、饱和和 256 bit 打包。
module mlp_silu_core #(
    parameter integer M             = 4864,
    parameter integer RESULT_BEATS  = M / 16,
    parameter integer INPUT_BEATS   = M / 4,
    parameter integer PWL_ENTRIES   = 65
)(
    input  wire                    clk,
    input  wire                    rst_n,

    input  wire                    gate_load_en,
    input  wire [10:0]             gate_load_index,
    input  wire [255:0]            gate_load_data,

    input  wire                    pwl_load_en,
    input  wire [2:0]              pwl_load_index,
    input  wire [255:0]            pwl_load_data,

    input  wire                    start,
    output reg                     busy,
    output reg                     done,

    output reg  [255:0]            result_data,
    output reg                     result_valid,
    input  wire                    result_ready
);

localparam [3:0] ST_IDLE        = 4'd0;
localparam [3:0] ST_READ_BEAT   = 4'd1;
localparam [3:0] ST_CAPTURE     = 4'd2;
localparam [3:0] ST_ABS         = 4'd3;
localparam [3:0] ST_ROUND       = 4'd4;
localparam [3:0] ST_SAT_INPUT   = 4'd5;
localparam [3:0] ST_DISPATCH    = 4'd6;
localparam [3:0] ST_PWL_READ    = 4'd7;
localparam [3:0] ST_PWL_MULT    = 4'd8;
localparam [3:0] ST_PWL_INTERP  = 4'd9;
localparam [3:0] ST_PWL_ADD     = 4'd10;
localparam [3:0] ST_SAT_OUTPUT  = 4'd11;
localparam [3:0] ST_PACK        = 4'd12;
localparam [3:0] ST_WAIT        = 4'd13;

reg [255:0] gate_mem0 [0:RESULT_BEATS-1];
reg [255:0] gate_mem1 [0:RESULT_BEATS-1];
reg [255:0] gate_mem2 [0:RESULT_BEATS-1];
reg [255:0] gate_mem3 [0:RESULT_BEATS-1];
reg [15:0]  pwl_mem [0:PWL_ENTRIES-1];

reg [3:0] state;
reg [8:0] beat_index;
reg [3:0] lane_index;
reg [255:0] gate_beat0_reg;
reg [255:0] gate_beat1_reg;
reg [255:0] gate_beat2_reg;
reg [255:0] gate_beat3_reg;
reg [255:0] output_pack;

reg signed [63:0] gate_lane_reg;
reg               gate_negative_reg;
reg [63:0]        gate_magnitude_reg;
reg signed [63:0] gate_q10_wide_reg;
reg signed [15:0] gate_q10_reg;
reg signed [63:0] value_reg;
reg [5:0]         pwl_index_reg;
reg [7:0]         pwl_fraction_reg;
reg signed [16:0] pwl_endpoint0_reg;
reg signed [16:0] pwl_endpoint1_reg;
reg signed [26:0] pwl_product_reg;
reg signed [18:0] pwl_interp_reg;
reg [15:0]        output_saturated_reg;

integer load_lane;
integer load_global_index;

reg [63:0] selected_gate_bits;
always @(*) begin
    case (lane_index[3:2])
        2'd0: selected_gate_bits = gate_beat0_reg[lane_index[1:0]*64 +: 64];
        2'd1: selected_gate_bits = gate_beat1_reg[lane_index[1:0]*64 +: 64];
        2'd2: selected_gate_bits = gate_beat2_reg[lane_index[1:0]*64 +: 64];
        default: selected_gate_bits = gate_beat3_reg[lane_index[1:0]*64 +: 64];
    endcase
end

wire signed [16:0] gate_q10_ext = {gate_q10_reg[15], gate_q10_reg};
wire [13:0] silu_offset_wire = gate_q10_ext + 17'sd8192;
wire signed [17:0] pwl_delta_wire = pwl_endpoint1_reg - pwl_endpoint0_reg;
wire signed [8:0] pwl_fraction_signed = {1'b0, pwl_fraction_reg};
wire signed [26:0] pwl_product_wire = pwl_delta_wire * pwl_fraction_signed;
wire signed [19:0] pwl_add_wire =
    {{3{pwl_endpoint0_reg[16]}}, pwl_endpoint0_reg} +
    {pwl_interp_reg[18], pwl_interp_reg};

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

function signed [18:0] rne_shift8_signed27;
    input signed [26:0] value;
    reg [26:0] magnitude;
    reg [18:0] quotient;
    reg [7:0] remainder;
    begin
        magnitude = value[26] ? (~value + 1'b1) : value;
        quotient = magnitude >> 8;
        remainder = magnitude[7:0];
        if ((remainder > 8'h80) ||
            ((remainder == 8'h80) && quotient[0]))
            quotient = quotient + 1'b1;
        rne_shift8_signed27 = value[26] ? -$signed(quotient) : $signed(quotient);
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

function [15:0] saturate_signed16_from20;
    input signed [19:0] value;
    begin
        if (value > 20'sd32767)
            saturate_signed16_from20 = 16'h7fff;
        else if (value < -20'sd32768)
            saturate_signed16_from20 = 16'h8000;
        else
            saturate_signed16_from20 = value[15:0];
    end
endfunction

reg [255:0] output_pack_next;
always @(*) begin
    output_pack_next = output_pack;
    output_pack_next[lane_index*16 +: 16] = output_saturated_reg;
end

always @(posedge clk) begin
    if (gate_load_en) begin
        case (gate_load_index[1:0])
            2'd0: gate_mem0[gate_load_index[10:2]] <= gate_load_data;
            2'd1: gate_mem1[gate_load_index[10:2]] <= gate_load_data;
            2'd2: gate_mem2[gate_load_index[10:2]] <= gate_load_data;
            2'd3: gate_mem3[gate_load_index[10:2]] <= gate_load_data;
        endcase
    end

    if (pwl_load_en) begin
        for (load_lane = 0; load_lane < 16; load_lane = load_lane + 1) begin
            load_global_index = pwl_load_index * 16 + load_lane;
            if (load_global_index < PWL_ENTRIES)
                pwl_mem[load_global_index] <= pwl_load_data[load_lane*16 +: 16];
        end
    end
end

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        state                  <= ST_IDLE;
        beat_index             <= 9'd0;
        lane_index             <= 4'd0;
        gate_beat0_reg         <= 256'd0;
        gate_beat1_reg         <= 256'd0;
        gate_beat2_reg         <= 256'd0;
        gate_beat3_reg         <= 256'd0;
        output_pack            <= 256'd0;
        gate_lane_reg          <= 64'sd0;
        gate_negative_reg      <= 1'b0;
        gate_magnitude_reg     <= 64'd0;
        gate_q10_wide_reg      <= 64'sd0;
        gate_q10_reg           <= 16'sd0;
        value_reg              <= 64'sd0;
        pwl_index_reg          <= 6'd0;
        pwl_fraction_reg       <= 8'd0;
        pwl_endpoint0_reg      <= 17'sd0;
        pwl_endpoint1_reg      <= 17'sd0;
        pwl_product_reg        <= 27'sd0;
        pwl_interp_reg         <= 19'sd0;
        output_saturated_reg   <= 16'd0;
        result_data            <= 256'd0;
        result_valid           <= 1'b0;
        busy                   <= 1'b0;
        done                   <= 1'b0;
    end else begin
        done <= 1'b0;

        case (state)
            ST_IDLE: begin
                busy         <= 1'b0;
                result_valid <= 1'b0;
                if (start) begin
                    beat_index  <= 9'd0;
                    lane_index  <= 4'd0;
                    output_pack <= 256'd0;
                    busy        <= 1'b1;
                    state       <= ST_READ_BEAT;
                end
            end

            ST_READ_BEAT: begin
                gate_beat0_reg <= gate_mem0[beat_index];
                gate_beat1_reg <= gate_mem1[beat_index];
                gate_beat2_reg <= gate_mem2[beat_index];
                gate_beat3_reg <= gate_mem3[beat_index];
                lane_index     <= 4'd0;
                output_pack    <= 256'd0;
                state          <= ST_CAPTURE;
            end

            ST_CAPTURE: begin
                gate_lane_reg <= $signed(selected_gate_bits);
                state         <= ST_ABS;
            end

            ST_ABS: begin
                gate_negative_reg  <= gate_lane_reg[63];
                gate_magnitude_reg <= gate_lane_reg[63] ?
                    (~gate_lane_reg + 1'b1) : gate_lane_reg;
                state <= ST_ROUND;
            end

            ST_ROUND: begin
                gate_q10_wide_reg <= rne_shift18_from_magnitude(
                    gate_magnitude_reg,
                    gate_negative_reg
                );
                state <= ST_SAT_INPUT;
            end

            ST_SAT_INPUT: begin
                gate_q10_reg <= saturate_signed16_from64(gate_q10_wide_reg);
                state        <= ST_DISPATCH;
            end

            ST_DISPATCH: begin
                if (gate_q10_reg < -16'sd8192) begin
                    value_reg <= 64'sd0;
                    state     <= ST_SAT_OUTPUT;
                end else if (gate_q10_reg >= 16'sd8192) begin
                    value_reg <= {{48{gate_q10_reg[15]}}, gate_q10_reg};
                    state     <= ST_SAT_OUTPUT;
                end else begin
                    pwl_index_reg    <= silu_offset_wire[13:8];
                    pwl_fraction_reg <= silu_offset_wire[7:0];
                    state            <= ST_PWL_READ;
                end
            end

            ST_PWL_READ: begin
                pwl_endpoint0_reg <= $signed(pwl_mem[pwl_index_reg]);
                pwl_endpoint1_reg <= $signed(pwl_mem[pwl_index_reg + 7'd1]);
                state             <= ST_PWL_MULT;
            end

            ST_PWL_MULT: begin
                pwl_product_reg <= pwl_product_wire;
                state           <= ST_PWL_INTERP;
            end

            ST_PWL_INTERP: begin
                pwl_interp_reg <= rne_shift8_signed27(pwl_product_reg);
                state          <= ST_PWL_ADD;
            end

            ST_PWL_ADD: begin
                value_reg <= {{44{pwl_add_wire[19]}}, pwl_add_wire};
                state     <= ST_SAT_OUTPUT;
            end

            ST_SAT_OUTPUT: begin
                if ((gate_q10_reg < -16'sd8192) || (gate_q10_reg >= 16'sd8192))
                    output_saturated_reg <= saturate_signed16_from64(value_reg);
                else
                    output_saturated_reg <= saturate_signed16_from20(value_reg[19:0]);
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

wire _unused_input_beats = (INPUT_BEATS == 1216);

endmodule
