`timescale 1ns/1ps

// 无符号 restoring divider，并在最终商上执行 round-to-nearest-even。
//
// quotient = RNE(numerator / denominator)
// latency  = WIDTH cycles after accepted start。
// start 仅在 busy=0 时接受；done 为单周期。
module unsigned_divider_rne #(
    parameter integer WIDTH = 96,
    parameter integer COUNT_WIDTH = $clog2(WIDTH + 1)
)(
    input  wire                 clk,
    input  wire                 rst_n,
    input  wire                 start,
    input  wire [WIDTH-1:0]     numerator,
    input  wire [WIDTH-1:0]     denominator,
    output reg                  busy,
    output reg                  done,
    output reg                  divide_by_zero,
    output reg                  overflow,
    output reg  [WIDTH-1:0]     quotient,
    output reg  [WIDTH-1:0]     remainder
);

reg [WIDTH-1:0] dividend_reg;
reg [WIDTH-1:0] divisor_reg;
reg [WIDTH-1:0] quotient_work;
reg [WIDTH:0]   remainder_work;
reg [COUNT_WIDTH-1:0] bit_count;

wire [WIDTH:0] shifted_remainder =
    {remainder_work[WIDTH-1:0], dividend_reg[WIDTH-1]};
wire [WIDTH:0] divisor_extended = {1'b0, divisor_reg};
wire subtract_enable = shifted_remainder >= divisor_extended;
wire [WIDTH:0] next_remainder =
    subtract_enable ? shifted_remainder - divisor_extended : shifted_remainder;
wire [WIDTH-1:0] next_quotient =
    {quotient_work[WIDTH-2:0], subtract_enable};
wire [WIDTH:0] doubled_remainder = {next_remainder[WIDTH-1:0], 1'b0};
wire round_up =
    (doubled_remainder > divisor_extended) ||
    ((doubled_remainder == divisor_extended) && next_quotient[0]);
wire [WIDTH:0] rounded_quotient_extended = {1'b0, next_quotient} + round_up;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        dividend_reg   <= {WIDTH{1'b0}};
        divisor_reg    <= {WIDTH{1'b0}};
        quotient_work  <= {WIDTH{1'b0}};
        remainder_work <= {(WIDTH+1){1'b0}};
        bit_count      <= {COUNT_WIDTH{1'b0}};
        busy           <= 1'b0;
        done           <= 1'b0;
        divide_by_zero <= 1'b0;
        overflow       <= 1'b0;
        quotient       <= {WIDTH{1'b0}};
        remainder      <= {WIDTH{1'b0}};
    end else begin
        done <= 1'b0;

        if (start && !busy) begin
            divide_by_zero <= 1'b0;
            overflow       <= 1'b0;
            quotient       <= {WIDTH{1'b0}};
            remainder      <= {WIDTH{1'b0}};
            if (denominator == {WIDTH{1'b0}}) begin
                divide_by_zero <= 1'b1;
                done           <= 1'b1;
            end else begin
                dividend_reg   <= numerator;
                divisor_reg    <= denominator;
                quotient_work  <= {WIDTH{1'b0}};
                remainder_work <= {(WIDTH+1){1'b0}};
                bit_count      <= {COUNT_WIDTH{1'b0}};
                busy           <= 1'b1;
            end
        end else if (busy) begin
            dividend_reg   <= {dividend_reg[WIDTH-2:0], 1'b0};
            quotient_work  <= next_quotient;
            remainder_work <= next_remainder;

            if (bit_count == WIDTH - 1) begin
                quotient       <= rounded_quotient_extended[WIDTH-1:0];
                remainder      <= next_remainder[WIDTH-1:0];
                overflow       <= rounded_quotient_extended[WIDTH];
                busy           <= 1'b0;
                done           <= 1'b1;
            end else begin
                bit_count <= bit_count + 1'b1;
            end
        end
    end
end

endmodule
