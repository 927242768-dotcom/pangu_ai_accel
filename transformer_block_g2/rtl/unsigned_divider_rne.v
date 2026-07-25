`timescale 1ns/1ps

// 无符号 restoring divider，并在最终商上执行 round-to-nearest-even。
//
// quotient = RNE(numerator / denominator)
// latency  = 2*WIDTH + 2 cycles after accepted start。
// start 仅在 busy=0 时接受；done 为单周期。
//
// 100 MHz 时序说明：
// 旧实现每拍串联 WIDTH-bit 比较、减法、商更新与末拍舍入。这里将每一位
// 除法拆成 PREP（移位+比较）与 UPDATE（减法+寄存）两拍，并将最终 RNE
// 拆成比较与加一两拍，使任一周期最多只有一条 WIDTH-bit carry chain。
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

localparam [2:0] ST_IDLE          = 3'd0;
localparam [2:0] ST_ITER_PREP     = 3'd1;
localparam [2:0] ST_ITER_UPDATE   = 3'd2;
localparam [2:0] ST_ROUND_COMPARE = 3'd3;
localparam [2:0] ST_ROUND_APPLY   = 3'd4;
localparam [2:0] ST_FINISH        = 3'd5;

reg [2:0] state;
reg [WIDTH-1:0] dividend_reg;
reg [WIDTH-1:0] divisor_reg;
reg [WIDTH-1:0] quotient_work;
reg [WIDTH:0]   remainder_work;
reg [WIDTH:0]   shifted_remainder_reg;
reg             subtract_enable_reg;
reg             round_up_reg;
reg [COUNT_WIDTH-1:0] bit_count;

wire [WIDTH:0] divisor_extended = {1'b0, divisor_reg};
wire [WIDTH:0] shifted_remainder_wire =
    {remainder_work[WIDTH-1:0], dividend_reg[WIDTH-1]};
wire [WIDTH:0] updated_remainder_wire = subtract_enable_reg
    ? shifted_remainder_reg - divisor_extended
    : shifted_remainder_reg;
wire [WIDTH-1:0] updated_quotient_wire =
    {quotient_work[WIDTH-2:0], subtract_enable_reg};
// 兼容原 RNE 结构契约：最终舍入拍中的 next_quotient 即已寄存完整商。
wire [WIDTH-1:0] next_quotient = quotient_work;
wire [WIDTH:0] doubled_remainder =
    {remainder_work[WIDTH-1:0], 1'b0};
wire final_round_up_wire =
    (doubled_remainder > divisor_extended) ||
    ((doubled_remainder == divisor_extended) && next_quotient[0]);
wire [WIDTH:0] rounded_quotient_wire =
    {1'b0, quotient_work} + round_up_reg;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        state                  <= ST_IDLE;
        dividend_reg           <= {WIDTH{1'b0}};
        divisor_reg            <= {WIDTH{1'b0}};
        quotient_work          <= {WIDTH{1'b0}};
        remainder_work         <= {(WIDTH+1){1'b0}};
        shifted_remainder_reg  <= {(WIDTH+1){1'b0}};
        subtract_enable_reg    <= 1'b0;
        round_up_reg           <= 1'b0;
        bit_count              <= {COUNT_WIDTH{1'b0}};
        busy                   <= 1'b0;
        done                   <= 1'b0;
        divide_by_zero         <= 1'b0;
        overflow               <= 1'b0;
        quotient               <= {WIDTH{1'b0}};
        remainder              <= {WIDTH{1'b0}};
    end else begin
        done <= 1'b0;

        case (state)
            ST_IDLE: begin
                busy <= 1'b0;
                if (start) begin
                    divide_by_zero <= 1'b0;
                    overflow       <= 1'b0;
                    quotient       <= {WIDTH{1'b0}};
                    remainder      <= {WIDTH{1'b0}};
                    if (denominator == {WIDTH{1'b0}}) begin
                        divide_by_zero <= 1'b1;
                        done           <= 1'b1;
                    end else begin
                        dividend_reg          <= numerator;
                        divisor_reg           <= denominator;
                        quotient_work         <= {WIDTH{1'b0}};
                        remainder_work        <= {(WIDTH+1){1'b0}};
                        shifted_remainder_reg <= {(WIDTH+1){1'b0}};
                        subtract_enable_reg   <= 1'b0;
                        round_up_reg          <= 1'b0;
                        bit_count             <= {COUNT_WIDTH{1'b0}};
                        busy                  <= 1'b1;
                        state                 <= ST_ITER_PREP;
                    end
                end
            end

            // 仅比较：寄存下一拍需要的移位余数和减法使能。
            ST_ITER_PREP: begin
                shifted_remainder_reg <= shifted_remainder_wire;
                subtract_enable_reg   <=
                    (shifted_remainder_wire >= divisor_extended);
                dividend_reg <= {dividend_reg[WIDTH-2:0], 1'b0};
                state <= ST_ITER_UPDATE;
            end

            // 仅减法/商移位：结果立即落寄存器，不在同拍继续比较。
            ST_ITER_UPDATE: begin
                remainder_work <= updated_remainder_wire;
                quotient_work  <= updated_quotient_wire;
                if (bit_count == WIDTH - 1) begin
                    remainder <= updated_remainder_wire[WIDTH-1:0];
                    state     <= ST_ROUND_COMPARE;
                end else begin
                    bit_count <= bit_count + 1'b1;
                    state     <= ST_ITER_PREP;
                end
            end

            // 只做最终 2*remainder 与 denominator 的比较和 tie-even 判定。
            ST_ROUND_COMPARE: begin
                round_up_reg <= final_round_up_wire;
                state        <= ST_ROUND_APPLY;
            end

            // 只做最终商加一。
            ST_ROUND_APPLY: begin
                quotient <= rounded_quotient_wire[WIDTH-1:0];
                overflow <= rounded_quotient_wire[WIDTH];
                state    <= ST_FINISH;
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
