`timescale 1ns/1ps

// 固定 K=896 的 Qwen2 RMSNorm 定点核心。
//
// 数值格式与 model_tools/rmsnorm_fixed_reference.py 完全一致：
//   input/gamma/output : signed Q6.10 int16
//   sum(x^2)           : 40 bit，保留 20 位小数
//   mean/epsilon       : Q12.20，epsilon_q20=1
//   rsqrt              : unsigned UQ12.20 uint32
//   rounding           : round-to-nearest-even（RNE）
//   saturation         : 输出显式饱和到 signed int16
//
// rsqrt 第一版采用归一化尾数 m∈[1,2) 的 256 项中点 LUT。LUT 由上位机随
// 固定载荷写入 DDR3，再读入片上缓存；指数奇偶通过 1/sqrt(2) 常数校正。
module rmsnorm_k896_core #(
    parameter integer K         = 896,
    parameter integer DATA_BEATS= K / 16,
    parameter integer LUT_BEATS = 32
)(
    input  wire                    clk,
    input  wire                    rst_n,

    input  wire                    input_load_en,
    input  wire [5:0]              input_load_index,
    input  wire [255:0]            input_load_data,

    input  wire                    gamma_load_en,
    input  wire [5:0]              gamma_load_index,
    input  wire [255:0]            gamma_load_data,

    input  wire                    lut_load_en,
    input  wire [4:0]              lut_load_index,
    input  wire [255:0]            lut_load_data,

    input  wire                    start,
    output reg                     busy,
    output reg                     done,

    output reg  [255:0]            result_data,
    output reg                     result_valid,
    input  wire                    result_ready,

    output reg  [39:0]             debug_sum_squares,
    output reg  [39:0]             debug_variance_q20,
    output reg  [31:0]             debug_rsqrt_q20
);

localparam [31:0] EPSILON_Q20      = 32'd1;
localparam [31:0] ONE_Q20          = 32'd1048576;
localparam [31:0] INV_SQRT2_Q20    = 32'd741455;
localparam [10:0] DIVISOR_K        = 11'd896;

localparam [4:0] ST_IDLE            = 5'd0;
localparam [4:0] ST_SUM_READ_BEAT   = 5'd1;
localparam [4:0] ST_SUM_ACCUM       = 5'd2;
localparam [4:0] ST_SUM_SQUARE      = 5'd18;
localparam [4:0] ST_DIV_INIT        = 5'd3;
localparam [4:0] ST_DIV_STEP        = 5'd4;
localparam [4:0] ST_NORMALIZE       = 5'd5;
localparam [4:0] ST_LUT_READ        = 5'd6;
localparam [4:0] ST_LUT_CAPTURE     = 5'd7;
localparam [4:0] ST_RSQRT_MULTIPLY  = 5'd8;
localparam [4:0] ST_RSQRT_ROUND     = 5'd9;
localparam [4:0] ST_RSQRT_SHIFT     = 5'd10;
localparam [4:0] ST_RSQRT_COMMIT    = 5'd11;
localparam [4:0] ST_OUT_READ_BEAT   = 5'd12;
localparam [4:0] ST_OUT_X_MULTIPLY  = 5'd13;
localparam [4:0] ST_OUT_X_ROUND     = 5'd14;
localparam [4:0] ST_OUT_G_MULTIPLY  = 5'd15;
localparam [4:0] ST_OUT_PACK        = 5'd16;
localparam [4:0] ST_OUT_WAIT        = 5'd17;
localparam [4:0] ST_OUT_ROUND       = 5'd19;
localparam [4:0] ST_OUT_SATURATE    = 5'd20;

reg [255:0] input_mem [0:DATA_BEATS-1];
reg [255:0] gamma_mem [0:DATA_BEATS-1];
reg [255:0] lut_mem   [0:LUT_BEATS-1];

reg [4:0] state;
reg [5:0] beat_index;
reg [3:0] lane_index;
reg [255:0] input_beat_reg;
reg [255:0] gamma_beat_reg;
reg [255:0] output_pack;

reg [31:0] square_reg;
reg [39:0] sum_squares;
reg [39:0] div_dividend;
reg [39:0] div_quotient;
reg [10:0] div_remainder;
reg [5:0]  div_bit_index;
reg [39:0] variance_q20;

reg [30:0] mantissa_q30;
reg signed [6:0] exponent_value;
reg signed [6:0] half_exponent;
reg exponent_odd;
reg [7:0] lut_index;
reg [2:0] lut_lane_index;
reg [255:0] lut_beat_reg;
reg [31:0] lut_seed_q20;
reg [63:0] rsqrt_product;
reg [63:0] rsqrt_after_constant_reg;
reg [63:0] rsqrt_scaled_reg;
reg [31:0] rsqrt_q20;

reg signed [63:0] x_rsqrt_product;
reg signed [31:0] normalized_q10;
reg signed [63:0] gamma_product;
reg signed [63:0] output_rounded_reg;
reg [15:0] output_saturated_reg;

wire signed [15:0] selected_input =
    $signed(input_beat_reg[lane_index*16 +: 16]);
wire signed [15:0] selected_gamma =
    $signed(gamma_beat_reg[lane_index*16 +: 16]);
wire signed [31:0] selected_input_square = selected_input * selected_input;

wire [10:0] div_remainder_shift =
    {div_remainder[9:0], div_dividend[div_bit_index]};
wire div_take = div_remainder_shift >= DIVISOR_K;
wire [10:0] div_remainder_next =
    div_take ? (div_remainder_shift - DIVISOR_K) : div_remainder_shift;
wire [39:0] div_quotient_bit = 40'd1 << div_bit_index;
wire [39:0] div_quotient_next =
    div_take ? (div_quotient | div_quotient_bit) : div_quotient;
wire div_rne_increment =
    ({div_remainder_next, 1'b0} > {1'b0, DIVISOR_K}) ||
    (({div_remainder_next, 1'b0} == {1'b0, DIVISOR_K}) &&
     div_quotient_next[0]);
wire [39:0] mean_square_rounded =
    div_quotient_next + (div_rne_increment ? 40'd1 : 40'd0);

function [5:0] leading_bit40;
    input [39:0] value;
    integer i;
    begin
        leading_bit40 = 6'd0;
        for (i = 0; i < 40; i = i + 1) begin
            if (value[i])
                leading_bit40 = i;
        end
    end
endfunction

wire [5:0] variance_leading_bit = leading_bit40(variance_q20);
wire [5:0] mantissa_left_shift = 6'd30 - variance_leading_bit;
wire [69:0] variance_shifted_wide =
    {30'd0, variance_q20} << mantissa_left_shift;
wire [30:0] normalized_mantissa_wire = variance_shifted_wide[30:0];
wire signed [6:0] exponent_wire =
    $signed({1'b0, variance_leading_bit}) - 7'sd20;
wire signed [6:0] half_exponent_wire = exponent_wire >>> 1;
wire exponent_odd_wire = exponent_wire[0];
wire [7:0] lut_index_wire = normalized_mantissa_wire[29:22];

wire [31:0] selected_lut_value =
    lut_beat_reg[lut_lane_index*32 +: 32];
wire [31:0] exponent_factor_q20 =
    exponent_odd ? INV_SQRT2_Q20 : ONE_Q20;
wire [63:0] rsqrt_product_rounded_base = rsqrt_product >> 20;
wire [19:0] rsqrt_product_remainder = rsqrt_product[19:0];
wire rsqrt_product_round_up =
    (rsqrt_product_remainder > 20'h80000) ||
    ((rsqrt_product_remainder == 20'h80000) &&
     rsqrt_product_rounded_base[0]);
wire [63:0] rsqrt_after_constant_wire =
    rsqrt_product_rounded_base + (rsqrt_product_round_up ? 64'd1 : 64'd0);

function [63:0] rne_shift_unsigned64;
    input [63:0] value;
    input [5:0] shift;
    reg [63:0] quotient;
    reg [63:0] remainder;
    reg [63:0] half;
    reg [63:0] mask;
    begin
        if (shift == 0) begin
            rne_shift_unsigned64 = value;
        end else begin
            quotient = value >> shift;
            mask = (64'd1 << shift) - 1'b1;
            remainder = value & mask;
            half = 64'd1 << (shift - 1'b1);
            if ((remainder > half) ||
                ((remainder == half) && quotient[0]))
                quotient = quotient + 1'b1;
            rne_shift_unsigned64 = quotient;
        end
    end
endfunction

wire [5:0] half_exponent_magnitude =
    half_exponent[6] ? -half_exponent : half_exponent;
wire [63:0] rsqrt_scaled_right =
    rne_shift_unsigned64(rsqrt_after_constant_reg, half_exponent_magnitude);
wire [63:0] rsqrt_scaled_left =
    rsqrt_after_constant_reg << half_exponent_magnitude;
wire [63:0] rsqrt_scaled_wire =
    half_exponent[6] ? rsqrt_scaled_left : rsqrt_scaled_right;

function signed [63:0] rne_shift20_signed64;
    input signed [63:0] value;
    reg [63:0] magnitude;
    reg [63:0] quotient;
    reg [19:0] remainder;
    begin
        magnitude = value[63] ? (~value + 1'b1) : value;
        quotient = magnitude >> 20;
        remainder = magnitude[19:0];
        if ((remainder > 20'h80000) ||
            ((remainder == 20'h80000) && quotient[0]))
            quotient = quotient + 1'b1;
        rne_shift20_signed64 = value[63] ? -$signed(quotient) : $signed(quotient);
    end
endfunction

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

wire signed [48:0] x_rsqrt_full =
    $signed(selected_input) * $signed({1'b0, rsqrt_q20});
wire signed [63:0] x_rsqrt_full_ext = {{15{x_rsqrt_full[48]}}, x_rsqrt_full};
wire signed [63:0] normalized_rounded_wire =
    rne_shift20_signed64(x_rsqrt_product);
wire signed [47:0] gamma_product_full =
    $signed(normalized_q10) * $signed(selected_gamma);
wire signed [63:0] gamma_product_full_ext =
    {{16{gamma_product_full[47]}}, gamma_product_full};
wire signed [63:0] output_rounded_wire = rne_shift10_signed64(gamma_product);
wire [15:0] output_saturated_wire = saturate_signed16(output_rounded_wire);

reg [255:0] output_pack_next;
always @(*) begin
    output_pack_next = output_pack;
    output_pack_next[lane_index*16 +: 16] = output_saturated_reg;
end

always @(posedge clk) begin
    if (input_load_en)
        input_mem[input_load_index] <= input_load_data;
    if (gamma_load_en)
        gamma_mem[gamma_load_index] <= gamma_load_data;
    if (lut_load_en)
        lut_mem[lut_load_index] <= lut_load_data;
end

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        state               <= ST_IDLE;
        beat_index          <= 6'd0;
        lane_index          <= 4'd0;
        input_beat_reg      <= 256'd0;
        gamma_beat_reg      <= 256'd0;
        output_pack         <= 256'd0;
        square_reg          <= 32'd0;
        sum_squares         <= 40'd0;
        div_dividend        <= 40'd0;
        div_quotient        <= 40'd0;
        div_remainder       <= 11'd0;
        div_bit_index       <= 6'd0;
        variance_q20        <= 40'd0;
        mantissa_q30        <= 31'd0;
        exponent_value      <= 7'sd0;
        half_exponent       <= 7'sd0;
        exponent_odd        <= 1'b0;
        lut_index           <= 8'd0;
        lut_lane_index      <= 3'd0;
        lut_beat_reg        <= 256'd0;
        lut_seed_q20              <= 32'd0;
        rsqrt_product             <= 64'd0;
        rsqrt_after_constant_reg  <= 64'd0;
        rsqrt_scaled_reg          <= 64'd0;
        rsqrt_q20                 <= 32'd0;
        x_rsqrt_product     <= 64'sd0;
        normalized_q10      <= 32'sd0;
        gamma_product       <= 64'sd0;
        output_rounded_reg  <= 64'sd0;
        output_saturated_reg<= 16'd0;
        result_data         <= 256'd0;
        result_valid        <= 1'b0;
        busy                <= 1'b0;
        done                <= 1'b0;
        debug_sum_squares   <= 40'd0;
        debug_variance_q20  <= 40'd0;
        debug_rsqrt_q20     <= 32'd0;
    end else begin
        done <= 1'b0;

        case (state)
            ST_IDLE: begin
                busy         <= 1'b0;
                result_valid <= 1'b0;
                if (start) begin
                    beat_index      <= 6'd0;
                    lane_index      <= 4'd0;
                    sum_squares     <= 40'd0;
                    output_pack     <= 256'd0;
                    busy            <= 1'b1;
                    state           <= ST_SUM_READ_BEAT;
                end
            end

            ST_SUM_READ_BEAT: begin
                input_beat_reg <= input_mem[beat_index];
                lane_index     <= 4'd0;
                state          <= ST_SUM_SQUARE;
            end

            ST_SUM_SQUARE: begin
                square_reg <= $unsigned(selected_input_square);
                state      <= ST_SUM_ACCUM;
            end

            ST_SUM_ACCUM: begin
                sum_squares <= sum_squares + square_reg;
                if (lane_index == 4'd15) begin
                    if (beat_index == DATA_BEATS - 1) begin
                        debug_sum_squares <= sum_squares + square_reg;
                        div_dividend      <= sum_squares + square_reg;
                        state             <= ST_DIV_INIT;
                    end else begin
                        beat_index <= beat_index + 1'b1;
                        state      <= ST_SUM_READ_BEAT;
                    end
                end else begin
                    lane_index <= lane_index + 1'b1;
                    state      <= ST_SUM_SQUARE;
                end
            end

            ST_DIV_INIT: begin
                div_quotient  <= 40'd0;
                div_remainder <= 11'd0;
                div_bit_index <= 6'd39;
                state         <= ST_DIV_STEP;
            end

            ST_DIV_STEP: begin
                div_quotient  <= div_quotient_next;
                div_remainder <= div_remainder_next;
                if (div_bit_index == 0) begin
                    variance_q20 <= mean_square_rounded + EPSILON_Q20;
                    debug_variance_q20 <= mean_square_rounded + EPSILON_Q20;
                    state <= ST_NORMALIZE;
                end else begin
                    div_bit_index <= div_bit_index - 1'b1;
                end
            end

            ST_NORMALIZE: begin
                mantissa_q30   <= normalized_mantissa_wire;
                exponent_value <= exponent_wire;
                half_exponent  <= half_exponent_wire;
                exponent_odd   <= exponent_odd_wire;
                lut_index      <= lut_index_wire;
                state          <= ST_LUT_READ;
            end

            ST_LUT_READ: begin
                lut_beat_reg   <= lut_mem[lut_index[7:3]];
                lut_lane_index <= lut_index[2:0];
                state          <= ST_LUT_CAPTURE;
            end

            ST_LUT_CAPTURE: begin
                lut_seed_q20 <= selected_lut_value;
                state        <= ST_RSQRT_MULTIPLY;
            end

            ST_RSQRT_MULTIPLY: begin
                rsqrt_product <= lut_seed_q20 * exponent_factor_q20;
                state         <= ST_RSQRT_ROUND;
            end

            // 将乘法后的 Q20 RNE、指数动态移位和 uint32 饱和拆成独立寄存级，
            // 避免它们与后级 x*rsqrt 乘法器输入串成一条长组合路径。
            ST_RSQRT_ROUND: begin
                rsqrt_after_constant_reg <= rsqrt_after_constant_wire;
                state                    <= ST_RSQRT_SHIFT;
            end

            ST_RSQRT_SHIFT: begin
                rsqrt_scaled_reg <= rsqrt_scaled_wire;
                state             <= ST_RSQRT_COMMIT;
            end

            ST_RSQRT_COMMIT: begin
                if (rsqrt_scaled_reg > 64'h0000_0000_ffff_ffff)
                    rsqrt_q20 <= 32'hffff_ffff;
                else
                    rsqrt_q20 <= rsqrt_scaled_reg[31:0];
                debug_rsqrt_q20 <=
                    (rsqrt_scaled_reg > 64'h0000_0000_ffff_ffff) ?
                    32'hffff_ffff : rsqrt_scaled_reg[31:0];
                beat_index  <= 6'd0;
                lane_index  <= 4'd0;
                output_pack <= 256'd0;
                state       <= ST_OUT_READ_BEAT;
            end

            ST_OUT_READ_BEAT: begin
                input_beat_reg <= input_mem[beat_index];
                gamma_beat_reg <= gamma_mem[beat_index];
                lane_index     <= 4'd0;
                output_pack    <= 256'd0;
                state          <= ST_OUT_X_MULTIPLY;
            end

            ST_OUT_X_MULTIPLY: begin
                x_rsqrt_product <= x_rsqrt_full_ext;
                state           <= ST_OUT_X_ROUND;
            end

            ST_OUT_X_ROUND: begin
                normalized_q10 <= normalized_rounded_wire[31:0];
                state          <= ST_OUT_G_MULTIPLY;
            end

            ST_OUT_G_MULTIPLY: begin
                gamma_product <= gamma_product_full_ext;
                state         <= ST_OUT_ROUND;
            end

            ST_OUT_ROUND: begin
                output_rounded_reg <= output_rounded_wire;
                state              <= ST_OUT_SATURATE;
            end

            ST_OUT_SATURATE: begin
                output_saturated_reg <= saturate_signed16(output_rounded_reg);
                state                <= ST_OUT_PACK;
            end

            ST_OUT_PACK: begin
                output_pack <= output_pack_next;
                if (lane_index == 4'd15) begin
                    result_data  <= output_pack_next;
                    result_valid <= 1'b1;
                    state        <= ST_OUT_WAIT;
                end else begin
                    lane_index <= lane_index + 1'b1;
                    state      <= ST_OUT_X_MULTIPLY;
                end
            end

            ST_OUT_WAIT: begin
                if (result_valid && result_ready) begin
                    result_valid <= 1'b0;
                    if (beat_index == DATA_BEATS - 1) begin
                        busy <= 1'b0;
                        done <= 1'b1;
                        state <= ST_IDLE;
                    end else begin
                        beat_index <= beat_index + 1'b1;
                        state      <= ST_OUT_READ_BEAT;
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
