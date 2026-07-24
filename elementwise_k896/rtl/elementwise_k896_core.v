`timescale 1ns/1ps

// 固定 K=896 的 signed Q6.10 元素级算子核心。
//
// 操作模式：
//   0：残差加法 A+B，显式饱和；
//   1：定点缩放 A*scale，Q20 RNE 右移 10 位后饱和；
//   2：元素级乘法 A*B，Q20 RNE 右移 10 位后饱和；
//   3：SiLU 64 段端点分段线性，[-8,8) 外采用 0/x 尾部规则。
//
// 输入、scale、端点和输出均为 signed Q6.10 int16。每拍缓存 16 个元素，
// 核心逐 lane 计算并将 16 个结果重新打包为一个 256 bit 数据拍。
module elementwise_k896_core #(
    parameter integer K          = 896,
    parameter integer DATA_BEATS = K / 16,
    parameter integer PWL_ENTRIES= 65
)(
    input  wire                    clk,
    input  wire                    rst_n,

    input  wire                    input_a_load_en,
    input  wire [5:0]              input_a_load_index,
    input  wire [255:0]            input_a_load_data,

    input  wire                    input_b_load_en,
    input  wire [5:0]              input_b_load_index,
    input  wire [255:0]            input_b_load_data,

    input  wire                    pwl_load_en,
    input  wire [2:0]              pwl_load_index,
    input  wire [255:0]            pwl_load_data,

    input  wire [1:0]              op_mode,
    input  wire signed [15:0]      scale_q10,
    input  wire                    start,
    output reg                     busy,
    output reg                     done,

    output reg  [255:0]            result_data,
    output reg                     result_valid,
    input  wire                    result_ready
);

localparam [3:0] ST_IDLE          = 4'd0;
localparam [3:0] ST_READ_BEAT     = 4'd1;
localparam [3:0] ST_DISPATCH      = 4'd2;
localparam [3:0] ST_MUL_ROUND     = 4'd3;
localparam [3:0] ST_SILU_READ     = 4'd4;
localparam [3:0] ST_SILU_MULT     = 4'd5;
localparam [3:0] ST_SILU_INTERP   = 4'd6;
localparam [3:0] ST_SATURATE      = 4'd7;
localparam [3:0] ST_PACK          = 4'd8;
localparam [3:0] ST_WAIT          = 4'd9;
localparam [3:0] ST_SILU_ADD      = 4'd10;

reg [255:0] input_a_mem [0:DATA_BEATS-1];
reg [255:0] input_b_mem [0:DATA_BEATS-1];
reg [15:0]  pwl_mem [0:PWL_ENTRIES-1];

reg [3:0] state;
reg [5:0] beat_index;
reg [3:0] lane_index;
reg [255:0] input_a_beat_reg;
reg [255:0] input_b_beat_reg;
reg [255:0] output_pack;
reg [1:0] op_mode_reg;
reg signed [15:0] scale_q10_reg;

reg signed [63:0] product_reg;
reg signed [63:0] value_reg;
reg [5:0] pwl_index_reg;
reg [7:0] pwl_fraction_reg;
reg signed [16:0] pwl_endpoint0_reg;
reg signed [16:0] pwl_endpoint1_reg;
reg signed [26:0] pwl_product_reg;
reg signed [18:0] pwl_interp_reg;
reg [15:0] output_saturated_reg;

integer load_lane;
integer load_global_index;

wire signed [15:0] selected_a =
    $signed(input_a_beat_reg[lane_index*16 +: 16]);
wire signed [15:0] selected_b =
    $signed(input_b_beat_reg[lane_index*16 +: 16]);
wire signed [16:0] selected_a_ext = {selected_a[15], selected_a};
wire signed [16:0] selected_b_ext = {selected_b[15], selected_b};
wire signed [17:0] add_full = selected_a_ext + selected_b_ext;
wire signed [31:0] scale_product_full = selected_a * scale_q10_reg;
wire signed [31:0] element_product_full = selected_a * selected_b;
wire [13:0] silu_offset_wire = selected_a_ext + 17'sd8192;
wire signed [17:0] pwl_delta_wire = pwl_endpoint1_reg - pwl_endpoint0_reg;
wire signed [8:0] pwl_fraction_signed = {1'b0, pwl_fraction_reg};
wire signed [26:0] pwl_product_wire = pwl_delta_wire * pwl_fraction_signed;
wire signed [19:0] pwl_add_wire =
    {{3{pwl_endpoint0_reg[16]}}, pwl_endpoint0_reg} +
    {pwl_interp_reg[18], pwl_interp_reg};

function signed [63:0] rne_shift10_signed64;
    input signed [63:0] value;
    reg [63:0] magnitude;
    reg [63:0] quotient;
    reg [9:0] remainder;
    begin
        magnitude = value[63] ? (~value + 1'b1) : value;
        quotient = magnitude >> 10;
        remainder = magnitude[9:0];
        if ((remainder > 10'h200) ||
            ((remainder == 10'h200) && quotient[0]))
            quotient = quotient + 1'b1;
        rne_shift10_signed64 = value[63] ? -$signed(quotient) : $signed(quotient);
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

function [15:0] saturate_signed16;
    input signed [63:0] value;
    begin
        if (value > 64'sd32767)
            saturate_signed16 = 16'h7fff;
        else if (value < -64'sd32768)
            saturate_signed16 = 16'h8000;
        else
            saturate_signed16 = value[15:0];
    end
endfunction

reg [255:0] output_pack_next;
always @(*) begin
    output_pack_next = output_pack;
    output_pack_next[lane_index*16 +: 16] = output_saturated_reg;
end

always @(posedge clk) begin
    if (input_a_load_en)
        input_a_mem[input_a_load_index] <= input_a_load_data;
    if (input_b_load_en)
        input_b_mem[input_b_load_index] <= input_b_load_data;
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
        state                 <= ST_IDLE;
        beat_index            <= 6'd0;
        lane_index            <= 4'd0;
        input_a_beat_reg      <= 256'd0;
        input_b_beat_reg      <= 256'd0;
        output_pack           <= 256'd0;
        op_mode_reg           <= 2'd0;
        scale_q10_reg         <= 16'sd0;
        product_reg           <= 64'sd0;
        value_reg             <= 64'sd0;
        pwl_index_reg         <= 6'd0;
        pwl_fraction_reg      <= 8'd0;
        pwl_endpoint0_reg     <= 17'sd0;
        pwl_endpoint1_reg     <= 17'sd0;
        pwl_product_reg       <= 27'sd0;
        pwl_interp_reg        <= 19'sd0;
        output_saturated_reg  <= 16'd0;
        result_data           <= 256'd0;
        result_valid          <= 1'b0;
        busy                  <= 1'b0;
        done                  <= 1'b0;
    end else begin
        done <= 1'b0;

        case (state)
            ST_IDLE: begin
                busy         <= 1'b0;
                result_valid <= 1'b0;
                if (start) begin
                    beat_index       <= 6'd0;
                    lane_index       <= 4'd0;
                    output_pack      <= 256'd0;
                    op_mode_reg      <= op_mode;
                    scale_q10_reg    <= scale_q10;
                    busy             <= 1'b1;
                    state            <= ST_READ_BEAT;
                end
            end

            ST_READ_BEAT: begin
                input_a_beat_reg <= input_a_mem[beat_index];
                input_b_beat_reg <= input_b_mem[beat_index];
                lane_index       <= 4'd0;
                output_pack      <= 256'd0;
                state            <= ST_DISPATCH;
            end

            ST_DISPATCH: begin
                case (op_mode_reg)
                    2'd0: begin
                        value_reg <= {{46{add_full[17]}}, add_full};
                        state     <= ST_SATURATE;
                    end
                    2'd1: begin
                        product_reg <= {{32{scale_product_full[31]}}, scale_product_full};
                        state       <= ST_MUL_ROUND;
                    end
                    2'd2: begin
                        product_reg <= {{32{element_product_full[31]}}, element_product_full};
                        state       <= ST_MUL_ROUND;
                    end
                    2'd3: begin
                        if (selected_a < -16'sd8192) begin
                            value_reg <= 64'sd0;
                            state     <= ST_SATURATE;
                        end else if (selected_a >= 16'sd8192) begin
                            value_reg <= {{48{selected_a[15]}}, selected_a};
                            state     <= ST_SATURATE;
                        end else begin
                            pwl_index_reg    <= silu_offset_wire[13:8];
                            pwl_fraction_reg <= silu_offset_wire[7:0];
                            state            <= ST_SILU_READ;
                        end
                    end
                    default: begin
                        value_reg <= 64'sd0;
                        state     <= ST_SATURATE;
                    end
                endcase
            end

            ST_MUL_ROUND: begin
                value_reg <= rne_shift10_signed64(product_reg);
                state     <= ST_SATURATE;
            end

            ST_SILU_READ: begin
                pwl_endpoint0_reg <= $signed(pwl_mem[pwl_index_reg]);
                // 使用 7 位加法，避免最高段 index=63 时 6 位结果回绕到 0。
                pwl_endpoint1_reg <= $signed(pwl_mem[pwl_index_reg + 7'd1]);
                state             <= ST_SILU_MULT;
            end

            ST_SILU_MULT: begin
                pwl_product_reg <= pwl_product_wire;
                state           <= ST_SILU_INTERP;
            end

            ST_SILU_INTERP: begin
                pwl_interp_reg <= rne_shift8_signed27(pwl_product_reg);
                state          <= ST_SILU_ADD;
            end

            ST_SILU_ADD: begin
                value_reg <= {{44{pwl_add_wire[19]}}, pwl_add_wire};
                state     <= ST_SATURATE;
            end

            ST_SATURATE: begin
                output_saturated_reg <= saturate_signed16(value_reg);
                state                <= ST_PACK;
            end

            ST_PACK: begin
                output_pack <= output_pack_next;
                if (lane_index == 4'd15) begin
                    result_data  <= output_pack_next;
                    result_valid <= 1'b1;
                    state        <= ST_WAIT;
                end else begin
                    lane_index <= lane_index + 1'b1;
                    state      <= ST_DISPATCH;
                end
            end

            ST_WAIT: begin
                if (result_valid && result_ready) begin
                    result_valid <= 1'b0;
                    if (beat_index == DATA_BEATS - 1) begin
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
