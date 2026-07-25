`timescale 1ns/1ps

// Q28 -> IEEE binary32 的多周期逐位转换器。
//
// 数值契约与 q28_to_binary32 完全相同：
//   signed int64 -> binary64（53-bit RNE）-> /2^28 -> binary32（24-bit RNE）。
//
// 原组合版本包含 64-bit priority encoder、可变移位和两次舍入，无法在
// DDR3 100 MHz 用户时钟下收敛。本实现用逐拍右移查找 MSB、逐拍归一化，
// 每个时钟只保留短加法/比较路径；吞吐降低但量化器本来就是串行控制，
// 对独立验证和完整 Block 调度均可接受。
module q28_to_binary32_sequential (
    input  wire               clk,
    input  wire               rst_n,
    input  wire               start,
    input  wire signed [63:0] q28_value,
    output reg                busy,
    output reg                done,
    output reg                sign,
    output reg                zero,
    output reg  [23:0]        mantissa,
    output reg  signed [9:0] component_exponent,
    output reg  [31:0]        binary32_bits
);

localparam [3:0] ST_IDLE          = 4'd0;
localparam [3:0] ST_CAPTURE       = 4'd1;
localparam [3:0] ST_FIND_MSB      = 4'd2;
localparam [3:0] ST_PREP_NORM     = 4'd3;
localparam [3:0] ST_SHIFT_LEFT    = 4'd4;
localparam [3:0] ST_SHIFT_RIGHT   = 4'd5;
localparam [3:0] ST_FIRST_ROUND   = 4'd6;
localparam [3:0] ST_SECOND_ROUND  = 4'd7;
localparam [3:0] ST_FINISH        = 4'd8;

reg [3:0] state;
reg signed [63:0] input_reg;
reg [63:0] magnitude_reg;
reg [63:0] scan_reg;
reg [63:0] normalize_reg;
reg [6:0] msb_index_reg;
reg [6:0] shift_count_reg;
reg guard_reg;
reg sticky_reg;
reg [52:0] binary64_mantissa_reg;
reg [6:0] binary64_exponent_msb_reg;

wire [63:0] input_magnitude = input_reg[63]
    ? (~input_reg[63:0] + 1'b1)
    : input_reg[63:0];
wire first_round_up = guard_reg &&
    (sticky_reg || normalize_reg[0]);
wire [53:0] first_rounded =
    {1'b0, normalize_reg[52:0]} + first_round_up;
wire second_round_up = binary64_mantissa_reg[28] &&
    ((|binary64_mantissa_reg[27:0]) || binary64_mantissa_reg[29]);
wire [24:0] second_rounded =
    {1'b0, binary64_mantissa_reg[52:29]} + second_round_up;
wire [23:0] second_mantissa = second_rounded[24]
    ? second_rounded[24:1]
    : second_rounded[23:0];
wire [6:0] second_exponent_msb = second_rounded[24]
    ? (binary64_exponent_msb_reg + 1'b1)
    : binary64_exponent_msb_reg;
wire signed [9:0] second_component_exponent =
    $signed({3'd0, second_exponent_msb}) - 10'sd51;
wire [7:0] second_binary32_exponent = second_exponent_msb + 8'd99;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        state                       <= ST_IDLE;
        input_reg                   <= 64'sd0;
        magnitude_reg               <= 64'd0;
        scan_reg                    <= 64'd0;
        normalize_reg               <= 64'd0;
        msb_index_reg               <= 7'd0;
        shift_count_reg             <= 7'd0;
        guard_reg                   <= 1'b0;
        sticky_reg                  <= 1'b0;
        binary64_mantissa_reg       <= 53'd0;
        binary64_exponent_msb_reg   <= 7'd0;
        busy                        <= 1'b0;
        done                        <= 1'b0;
        sign                        <= 1'b0;
        zero                        <= 1'b1;
        mantissa                    <= 24'd0;
        component_exponent          <= 10'sd0;
        binary32_bits               <= 32'd0;
    end else begin
        done <= 1'b0;

        case (state)
            ST_IDLE: begin
                busy <= 1'b0;
                if (start) begin
                    // 先只锁存输入，避免 RAM/上游 mux 与 64 位取绝对值处于同一拍。
                    input_reg          <= q28_value;
                    msb_index_reg      <= 7'd0;
                    shift_count_reg    <= 7'd0;
                    guard_reg          <= 1'b0;
                    sticky_reg         <= 1'b0;
                    mantissa           <= 24'd0;
                    component_exponent <= 10'sd0;
                    binary32_bits      <= 32'd0;
                    busy               <= 1'b1;
                    state              <= ST_CAPTURE;
                end
            end

            ST_CAPTURE: begin
                magnitude_reg <= input_magnitude;
                scan_reg      <= input_magnitude;
                normalize_reg <= input_magnitude;
                zero          <= (input_magnitude == 64'd0);
                sign          <= (input_magnitude == 64'd0)
                    ? 1'b0 : input_reg[63];
                if (input_magnitude == 64'd0)
                    state <= ST_FINISH;
                else
                    state <= ST_FIND_MSB;
            end

            // scan_reg 每拍右移一位，msb_index_reg 最终等于 floor(log2(x))。
            ST_FIND_MSB: begin
                if (scan_reg[63:1] == 63'd0) begin
                    normalize_reg <= magnitude_reg;
                    state         <= ST_PREP_NORM;
                end else begin
                    scan_reg      <= scan_reg >> 1;
                    msb_index_reg <= msb_index_reg + 1'b1;
                end
            end

            ST_PREP_NORM: begin
                guard_reg  <= 1'b0;
                sticky_reg <= 1'b0;
                if (msb_index_reg < 7'd52) begin
                    shift_count_reg <= 7'd52 - msb_index_reg;
                    state           <= ST_SHIFT_LEFT;
                end else if (msb_index_reg > 7'd52) begin
                    shift_count_reg <= msb_index_reg - 7'd52;
                    state           <= ST_SHIFT_RIGHT;
                end else begin
                    binary64_mantissa_reg     <= normalize_reg[52:0];
                    binary64_exponent_msb_reg <= msb_index_reg;
                    state                     <= ST_SECOND_ROUND;
                end
            end

            ST_SHIFT_LEFT: begin
                normalize_reg <= normalize_reg << 1;
                if (shift_count_reg == 7'd1) begin
                    binary64_mantissa_reg     <= (normalize_reg << 1);
                    binary64_exponent_msb_reg <= msb_index_reg;
                    shift_count_reg           <= 7'd0;
                    state                     <= ST_SECOND_ROUND;
                end else begin
                    shift_count_reg <= shift_count_reg - 1'b1;
                end
            end

            // guard 保存最后移出的位，sticky 保存其余更低位的 OR。
            ST_SHIFT_RIGHT: begin
                normalize_reg <= normalize_reg >> 1;
                sticky_reg    <= sticky_reg || guard_reg;
                guard_reg     <= normalize_reg[0];
                if (shift_count_reg == 7'd1) begin
                    shift_count_reg <= 7'd0;
                    state           <= ST_FIRST_ROUND;
                end else begin
                    shift_count_reg <= shift_count_reg - 1'b1;
                end
            end

            ST_FIRST_ROUND: begin
                if (first_rounded[53]) begin
                    binary64_mantissa_reg     <= first_rounded[53:1];
                    binary64_exponent_msb_reg <= msb_index_reg + 1'b1;
                end else begin
                    binary64_mantissa_reg     <= first_rounded[52:0];
                    binary64_exponent_msb_reg <= msb_index_reg;
                end
                state <= ST_SECOND_ROUND;
            end

            ST_SECOND_ROUND: begin
                mantissa           <= second_mantissa;
                component_exponent <= second_component_exponent;
                binary32_bits      <= {
                    sign,
                    second_binary32_exponent,
                    second_mantissa[22:0]
                };
                state <= ST_FINISH;
            end

            ST_FINISH: begin
                busy  <= 1'b0;
                done  <= 1'b1;
                state <= ST_IDLE;
            end

            default: begin
                busy  <= 1'b0;
                done  <= 1'b0;
                state <= ST_IDLE;
            end
        endcase
    end
end

endmodule
