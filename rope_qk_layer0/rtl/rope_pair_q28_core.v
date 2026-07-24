`timescale 1ns/1ps

// 64x32 有符号顺序乘法器。
//
// 为避免 PGL50H 上直接 64x32 组合乘法形成长 APM 级联，按 16 位 limb
// 拆成 8 个 16x16 无符号部分积。每个部分积单独寄存，再累加到 96 位幅值，
// 最终恢复符号。吞吐率不是本阶段目标，优先保证 100 MHz 可收敛和数学精确。
module signed_mul64x32_seq16 (
    input  wire                   clk,
    input  wire                   rst_n,
    input  wire                   start,
    input  wire signed [63:0]     a,
    input  wire signed [31:0]     b,
    output reg                    busy,
    output reg                    done,
    output reg signed [95:0]      product
);

localparam [1:0] ST_IDLE    = 2'd0;
localparam [1:0] ST_PREPARE = 2'd1;
localparam [1:0] ST_CAPTURE = 2'd2;
localparam [1:0] ST_ACCUM   = 2'd3;

reg [1:0] state;
reg [2:0] step;
reg result_negative;
reg [63:0] magnitude_a;
reg [31:0] magnitude_b;
reg [15:0] limb_a_reg;
reg [15:0] limb_b_reg;
reg [31:0] partial_product_reg;
reg [95:0] magnitude_accumulator;

reg [15:0] selected_limb_a;
reg [15:0] selected_limb_b;
reg [95:0] aligned_partial;
wire [31:0] partial_product = limb_a_reg * limb_b_reg;
wire [95:0] next_magnitude = magnitude_accumulator + aligned_partial;

always @(*) begin
    case (step)
        3'd0, 3'd4: selected_limb_a = magnitude_a[15:0];
        3'd1, 3'd5: selected_limb_a = magnitude_a[31:16];
        3'd2, 3'd6: selected_limb_a = magnitude_a[47:32];
        default:    selected_limb_a = magnitude_a[63:48];
    endcase

    if (step < 3'd4)
        selected_limb_b = magnitude_b[15:0];
    else
        selected_limb_b = magnitude_b[31:16];

    case (step)
        3'd0: aligned_partial = {64'd0, partial_product_reg};
        3'd1: aligned_partial = {48'd0, partial_product_reg, 16'd0};
        3'd2: aligned_partial = {32'd0, partial_product_reg, 32'd0};
        3'd3: aligned_partial = {16'd0, partial_product_reg, 48'd0};
        3'd4: aligned_partial = {48'd0, partial_product_reg, 16'd0};
        3'd5: aligned_partial = {32'd0, partial_product_reg, 32'd0};
        3'd6: aligned_partial = {16'd0, partial_product_reg, 48'd0};
        default: aligned_partial = {partial_product_reg, 64'd0};
    endcase
end

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        state                 <= ST_IDLE;
        step                  <= 3'd0;
        result_negative       <= 1'b0;
        magnitude_a           <= 64'd0;
        magnitude_b           <= 32'd0;
        limb_a_reg            <= 16'd0;
        limb_b_reg            <= 16'd0;
        partial_product_reg   <= 32'd0;
        magnitude_accumulator <= 96'd0;
        product               <= 96'sd0;
        busy                  <= 1'b0;
        done                  <= 1'b0;
    end else begin
        done <= 1'b0;

        case (state)
            ST_IDLE: begin
                busy <= 1'b0;
                if (start) begin
                    result_negative       <= a[63] ^ b[31];
                    magnitude_a           <= a[63] ? (~a + 1'b1) : a;
                    magnitude_b           <= b[31] ? (~b + 1'b1) : b;
                    magnitude_accumulator <= 96'd0;
                    step                  <= 3'd0;
                    busy                  <= 1'b1;
                    state                 <= ST_PREPARE;
                end
            end

            ST_PREPARE: begin
                limb_a_reg <= selected_limb_a;
                limb_b_reg <= selected_limb_b;
                state      <= ST_CAPTURE;
            end

            ST_CAPTURE: begin
                partial_product_reg <= partial_product;
                state               <= ST_ACCUM;
            end

            ST_ACCUM: begin
                if (step == 3'd7) begin
                    product <= result_negative ?
                        $signed(~next_magnitude + 1'b1) : $signed(next_magnitude);
                    busy  <= 1'b0;
                    done  <= 1'b1;
                    state <= ST_IDLE;
                end else begin
                    magnitude_accumulator <= next_magnitude;
                    step                  <= step + 1'b1;
                    state                 <= ST_PREPARE;
                end
            end

            default: begin
                state <= ST_IDLE;
                busy  <= 1'b0;
                done  <= 1'b0;
            end
        endcase
    end
end

endmodule


// Qwen2 split-half RoPE 单对定点旋转核心。
//
// 输入/输出：signed int64 Q28
// sin/cos： signed int32 Q1.30
// 配对规则：dim i 与 dim i+32
//
// y_first  = RNE((x_first*cos - x_second*sin) / 2^30)
// y_second = RNE((x_second*cos + x_first*sin) / 2^30)
//
// 四个乘积由上面的顺序乘法器复用完成；乘积相加/相减、求绝对值、RNE、
// 饱和分别放在独立寄存阶段，避免形成 97 位长组合路径。
module rope_pair_q28_core (
    input  wire                   clk,
    input  wire                   rst_n,
    input  wire                   start,
    input  wire signed [63:0]     x_first_q28,
    input  wire signed [63:0]     x_second_q28,
    input  wire signed [31:0]     cos_q30,
    input  wire signed [31:0]     sin_q30,
    output reg                    busy,
    output reg                    done,
    output reg signed [63:0]      y_first_q28,
    output reg signed [63:0]      y_second_q28
);

localparam [3:0] ST_IDLE      = 4'd0;
localparam [3:0] ST_START0    = 4'd1;
localparam [3:0] ST_WAIT0     = 4'd2;
localparam [3:0] ST_START1    = 4'd3;
localparam [3:0] ST_WAIT1     = 4'd4;
localparam [3:0] ST_START2    = 4'd5;
localparam [3:0] ST_WAIT2     = 4'd6;
localparam [3:0] ST_START3    = 4'd7;
localparam [3:0] ST_WAIT3     = 4'd8;
localparam [3:0] ST_COMBINE   = 4'd9;
localparam [3:0] ST_ABS       = 4'd10;
localparam [3:0] ST_ROUND     = 4'd11;
localparam [3:0] ST_SATURATE  = 4'd12;

reg [3:0] state;
reg signed [63:0] first_reg;
reg signed [63:0] second_reg;
reg signed [31:0] cos_reg;
reg signed [31:0] sin_reg;
reg signed [63:0] mul_a;
reg signed [31:0] mul_b;
reg mul_start;
wire mul_busy;
wire mul_done;
wire signed [95:0] mul_product;

reg signed [95:0] product0;
reg signed [95:0] product1;
reg signed [95:0] product2;
reg signed [95:0] product3;
reg signed [96:0] first_sum;
reg signed [96:0] second_sum;
reg first_negative;
reg second_negative;
reg [96:0] first_magnitude;
reg [96:0] second_magnitude;
reg [66:0] first_rounded;
reg [66:0] second_rounded;

wire [66:0] first_quotient = first_magnitude[96:30];
wire [66:0] second_quotient = second_magnitude[96:30];
wire [29:0] first_remainder = first_magnitude[29:0];
wire [29:0] second_remainder = second_magnitude[29:0];
wire first_round_up =
    (first_remainder > 30'h20000000) ||
    ((first_remainder == 30'h20000000) && first_quotient[0]);
wire second_round_up =
    (second_remainder > 30'h20000000) ||
    ((second_remainder == 30'h20000000) && second_quotient[0]);

function [63:0] signed_magnitude_sat64;
    input sign_value;
    input [66:0] magnitude;
    begin
        if (!sign_value) begin
            if (|magnitude[66:63])
                signed_magnitude_sat64 = 64'h7fff_ffff_ffff_ffff;
            else
                signed_magnitude_sat64 = magnitude[63:0];
        end else begin
            if (|magnitude[66:64] || (magnitude[63] && |magnitude[62:0]))
                signed_magnitude_sat64 = 64'h8000_0000_0000_0000;
            else if (magnitude[63])
                signed_magnitude_sat64 = 64'h8000_0000_0000_0000;
            else
                signed_magnitude_sat64 = (~magnitude[63:0]) + 1'b1;
        end
    end
endfunction

signed_mul64x32_seq16 u_signed_mul64x32_seq16 (
    .clk     (clk),
    .rst_n   (rst_n),
    .start   (mul_start),
    .a       (mul_a),
    .b       (mul_b),
    .busy    (mul_busy),
    .done    (mul_done),
    .product (mul_product)
);

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        state            <= ST_IDLE;
        first_reg        <= 64'sd0;
        second_reg       <= 64'sd0;
        cos_reg          <= 32'sd0;
        sin_reg          <= 32'sd0;
        mul_a            <= 64'sd0;
        mul_b            <= 32'sd0;
        mul_start        <= 1'b0;
        product0         <= 96'sd0;
        product1         <= 96'sd0;
        product2         <= 96'sd0;
        product3         <= 96'sd0;
        first_sum        <= 97'sd0;
        second_sum       <= 97'sd0;
        first_negative   <= 1'b0;
        second_negative  <= 1'b0;
        first_magnitude  <= 97'd0;
        second_magnitude <= 97'd0;
        first_rounded    <= 67'd0;
        second_rounded   <= 67'd0;
        y_first_q28      <= 64'sd0;
        y_second_q28     <= 64'sd0;
        busy             <= 1'b0;
        done             <= 1'b0;
    end else begin
        done      <= 1'b0;
        mul_start <= 1'b0;

        case (state)
            ST_IDLE: begin
                busy <= 1'b0;
                if (start) begin
                    first_reg  <= x_first_q28;
                    second_reg <= x_second_q28;
                    cos_reg    <= cos_q30;
                    sin_reg    <= sin_q30;
                    busy       <= 1'b1;
                    state      <= ST_START0;
                end
            end

            ST_START0: begin
                mul_a     <= first_reg;
                mul_b     <= cos_reg;
                mul_start <= 1'b1;
                state     <= ST_WAIT0;
            end

            ST_WAIT0: begin
                if (mul_done) begin
                    product0 <= mul_product;
                    state    <= ST_START1;
                end
            end

            ST_START1: begin
                mul_a     <= second_reg;
                mul_b     <= sin_reg;
                mul_start <= 1'b1;
                state     <= ST_WAIT1;
            end

            ST_WAIT1: begin
                if (mul_done) begin
                    product1 <= mul_product;
                    state    <= ST_START2;
                end
            end

            ST_START2: begin
                mul_a     <= second_reg;
                mul_b     <= cos_reg;
                mul_start <= 1'b1;
                state     <= ST_WAIT2;
            end

            ST_WAIT2: begin
                if (mul_done) begin
                    product2 <= mul_product;
                    state    <= ST_START3;
                end
            end

            ST_START3: begin
                mul_a     <= first_reg;
                mul_b     <= sin_reg;
                mul_start <= 1'b1;
                state     <= ST_WAIT3;
            end

            ST_WAIT3: begin
                if (mul_done) begin
                    product3 <= mul_product;
                    state    <= ST_COMBINE;
                end
            end

            ST_COMBINE: begin
                first_sum <=
                    $signed({product0[95], product0}) -
                    $signed({product1[95], product1});
                second_sum <=
                    $signed({product2[95], product2}) +
                    $signed({product3[95], product3});
                state <= ST_ABS;
            end

            ST_ABS: begin
                first_negative   <= first_sum[96];
                second_negative  <= second_sum[96];
                first_magnitude  <= first_sum[96] ? (~first_sum + 1'b1) : first_sum;
                second_magnitude <= second_sum[96] ? (~second_sum + 1'b1) : second_sum;
                state            <= ST_ROUND;
            end

            ST_ROUND: begin
                first_rounded  <= first_quotient + first_round_up;
                second_rounded <= second_quotient + second_round_up;
                state          <= ST_SATURATE;
            end

            ST_SATURATE: begin
                y_first_q28  <= signed_magnitude_sat64(first_negative, first_rounded);
                y_second_q28 <= signed_magnitude_sat64(second_negative, second_rounded);
                busy         <= 1'b0;
                done         <= 1'b1;
                state        <= ST_IDLE;
            end

            default: begin
                state <= ST_IDLE;
                busy  <= 1'b0;
                done  <= 1'b0;
            end
        endcase
    end
end

wire _unused_mul_busy = mul_busy;

endmodule
