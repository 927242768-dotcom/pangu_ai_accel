`timescale 1ns/1ps

// 逐位复现 G1/F6 软件路径：
//   signed int64 Q28 -> IEEE binary64 -> /2^28 -> IEEE binary32。
//
// 不能直接把 int64 一次舍入到 24 bit，因为旧软件在 >53 bit 整数上存在
// binary64 再到 binary32 的两次 RNE。Q28 全动态范围在 binary32 中均为 normal finite。
module q28_to_binary32 (
    input  wire signed [63:0] q28_value,
    output reg                sign,
    output reg                zero,
    output reg  [23:0]        mantissa,
    output reg  signed [9:0]  component_exponent,
    output reg  [31:0]        binary32_bits
);

function [6:0] msb_index64;
    input [63:0] value;
    integer bit_index;
    begin
        msb_index64 = 7'd0;
        for (bit_index = 0; bit_index < 64; bit_index = bit_index + 1) begin
            if (value[bit_index])
                msb_index64 = bit_index[6:0];
        end
    end
endfunction

reg [63:0] magnitude;
reg [6:0] msb_index;
reg [6:0] binary64_shift;
reg [52:0] binary64_retained;
reg [63:0] binary64_discarded;
reg [63:0] binary64_half;
reg binary64_round_up;
reg [53:0] binary64_rounded;
reg [52:0] binary64_mantissa;
reg [6:0] binary64_exponent_msb;

reg [23:0] binary32_retained;
reg [28:0] binary32_discarded;
reg binary32_round_up;
reg [24:0] binary32_rounded;
reg [23:0] binary32_mantissa;
reg [6:0] binary32_exponent_msb;
reg [7:0] binary32_exponent_bits;

always @(*) begin
    sign                     = q28_value[63];
    magnitude                = q28_value[63]
        ? (~q28_value[63:0] + 1'b1)
        : q28_value[63:0];
    zero                     = (magnitude == 64'd0);
    msb_index                = msb_index64(magnitude);
    binary64_shift           = 7'd0;
    binary64_retained        = 53'd0;
    binary64_discarded       = 64'd0;
    binary64_half            = 64'd0;
    binary64_round_up        = 1'b0;
    binary64_rounded         = 54'd0;
    binary64_mantissa        = 53'd0;
    binary64_exponent_msb    = 7'd0;
    binary32_retained        = 24'd0;
    binary32_discarded       = 29'd0;
    binary32_round_up        = 1'b0;
    binary32_rounded         = 25'd0;
    binary32_mantissa        = 24'd0;
    binary32_exponent_msb    = 7'd0;
    binary32_exponent_bits   = 8'd0;
    mantissa                 = 24'd0;
    component_exponent       = 10'sd0;
    binary32_bits            = 32'd0;

    if (!zero) begin
        // 第一舍入：signed int64 magnitude -> 53-bit binary64 significand。
        if (msb_index <= 7'd52) begin
            binary64_mantissa     = magnitude << (7'd52 - msb_index);
            binary64_exponent_msb = msb_index;
        end else begin
            binary64_shift     = msb_index - 7'd52;
            binary64_retained  = magnitude >> binary64_shift;
            binary64_discarded = magnitude -
                ((magnitude >> binary64_shift) << binary64_shift);
            binary64_half      = 64'd1 << (binary64_shift - 1'b1);
            binary64_round_up  =
                (binary64_discarded > binary64_half) ||
                ((binary64_discarded == binary64_half) && binary64_retained[0]);
            binary64_rounded = {1'b0, binary64_retained} + binary64_round_up;
            if (binary64_rounded[53]) begin
                binary64_mantissa     = binary64_rounded[53:1];
                binary64_exponent_msb = msb_index + 1'b1;
            end else begin
                binary64_mantissa     = binary64_rounded[52:0];
                binary64_exponent_msb = msb_index;
            end
        end

        // /2^28 仅改变指数；第二舍入：53-bit -> 24-bit binary32。
        binary32_retained  = binary64_mantissa[52:29];
        binary32_discarded = binary64_mantissa[28:0];
        binary32_round_up  =
            (binary32_discarded > 29'h1000_0000) ||
            ((binary32_discarded == 29'h1000_0000) && binary32_retained[0]);
        binary32_rounded = {1'b0, binary32_retained} + binary32_round_up;
        if (binary32_rounded[24]) begin
            binary32_mantissa     = binary32_rounded[24:1];
            binary32_exponent_msb = binary64_exponent_msb + 1'b1;
        end else begin
            binary32_mantissa     = binary32_rounded[23:0];
            binary32_exponent_msb = binary64_exponent_msb;
        end

        // value = mantissa24 * 2^(exponent_msb - 28 - 23)。
        mantissa               = binary32_mantissa;
        component_exponent     = $signed({3'd0, binary32_exponent_msb}) - 10'sd51;
        binary32_exponent_bits = binary32_exponent_msb + 8'd99;
        binary32_bits          = {
            sign,
            binary32_exponent_bits,
            binary32_mantissa[22:0]
        };
    end else begin
        sign = 1'b0;
    end
end

endmodule
