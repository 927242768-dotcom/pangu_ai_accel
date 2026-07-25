`timescale 1ns/1ps

// G2 运行时 FP16 weight scale -> unsigned UQ4.28 combined scale。
//
// 非零激活：
//   combined = RNE(weight_fp16 * max_abs_binary32 / 127 * 2^28)
// 全零激活：
//   activation_scale 固定为 1.0，combined = RNE(weight_fp16 * 2^28)
//
// max_abs_binary32 使用精确分量：mantissa * 2^exponent。
// 输入 weight scale 必须是有限正 FP16；输出显式饱和到 uint32。
module runtime_fp16_scale_builder (
    input  wire                 clk,
    input  wire                 rst_n,

    input  wire                 all_zero,
    input  wire [23:0]          max_mantissa_binary32,
    input  wire signed [9:0]    max_exponent_binary32,

    input  wire                 scale_valid,
    output wire                 scale_ready,
    input  wire [15:0]          weight_scale_fp16,

    output reg                  combined_valid,
    input  wire                 combined_ready,
    output reg  [31:0]          combined_scale_uq4_28,
    output reg                  saturated,
    output reg                  error,
    output reg  [7:0]           error_code,
    output wire [2:0]           debug_state
);

localparam [2:0] ST_IDLE      = 3'd0;
localparam [2:0] ST_PREPARE   = 3'd1;
localparam [2:0] ST_DIV_START = 3'd2;
localparam [2:0] ST_DIV_WAIT  = 3'd3;
localparam [2:0] ST_OUTPUT    = 3'd4;
localparam [2:0] ST_ERROR     = 3'd7;

localparam [7:0] ERR_FP16       = 8'h01;
localparam [7:0] ERR_MAX_META   = 8'h02;
localparam [7:0] ERR_SHIFT      = 8'h03;
localparam [7:0] ERR_DIVIDER    = 8'h04;
localparam [7:0] ERR_INTERNAL   = 8'hff;

reg [2:0] state;
reg [15:0] scale_reg;
reg all_zero_reg;
reg [23:0] max_mantissa_reg;
reg signed [9:0] max_exponent_reg;
reg [95:0] numerator_reg;
reg [95:0] denominator_reg;
reg divider_start;

wire fp16_sign = scale_reg[15];
wire [4:0] fp16_exponent_bits = scale_reg[14:10];
wire [9:0] fp16_fraction = scale_reg[9:0];
wire fp16_zero = (fp16_exponent_bits == 5'd0) && (fp16_fraction == 10'd0);
wire fp16_special = fp16_exponent_bits == 5'h1f;
wire fp16_valid_positive = !fp16_sign && !fp16_zero && !fp16_special;
wire [10:0] fp16_mantissa =
    (fp16_exponent_bits == 5'd0) ? {1'b0, fp16_fraction} : {1'b1, fp16_fraction};
// binary16 normal: mantissa * 2^(exp_bits-15-10)；subnormal: fraction * 2^-24。
wire signed [10:0] fp16_component_exponent =
    (fp16_exponent_bits == 5'd0)
        ? -11'sd24
        : $signed({6'd0, fp16_exponent_bits}) - 11'sd25;

wire [34:0] nonzero_base_product = fp16_mantissa * max_mantissa_reg;
wire [34:0] effective_base_product =
    all_zero_reg ? {24'd0, fp16_mantissa} : nonzero_base_product;
wire [7:0] effective_denominator_base = all_zero_reg ? 8'd1 : 8'd127;
wire signed [10:0] effective_shift = all_zero_reg
    ? fp16_component_exponent + 11'sd28
    : fp16_component_exponent + $signed(max_exponent_reg) + 11'sd28;
wire shift_negative = effective_shift[10];
wire [10:0] shift_magnitude =
    shift_negative ? (~effective_shift + 1'b1) : effective_shift;
wire shift_too_large = shift_magnitude >= 11'd96;
wire [95:0] numerator_wire = shift_negative
    ? {{61{1'b0}}, effective_base_product}
    : ({{61{1'b0}}, effective_base_product} << shift_magnitude);
wire [95:0] denominator_wire = shift_negative
    ? ({{88{1'b0}}, effective_denominator_base} << shift_magnitude)
    : {{88{1'b0}}, effective_denominator_base};

wire divider_busy;
wire divider_done;
wire divider_divide_by_zero;
wire divider_overflow;
wire [95:0] divider_quotient;
wire [95:0] divider_remainder;
wire quotient_saturated = divider_overflow || (|divider_quotient[95:32]);

assign scale_ready = (state == ST_IDLE) && !error;
assign debug_state = state;

unsigned_divider_rne #(
    .WIDTH(96)
) u_unsigned_divider_rne (
    .clk            (clk),
    .rst_n          (rst_n),
    .start          (divider_start),
    .numerator      (numerator_reg),
    .denominator    (denominator_reg),
    .busy           (divider_busy),
    .done           (divider_done),
    .divide_by_zero (divider_divide_by_zero),
    .overflow       (divider_overflow),
    .quotient       (divider_quotient),
    .remainder      (divider_remainder)
);

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        state                   <= ST_IDLE;
        scale_reg               <= 16'd0;
        all_zero_reg            <= 1'b0;
        max_mantissa_reg        <= 24'd0;
        max_exponent_reg        <= 10'sd0;
        numerator_reg           <= 96'd0;
        denominator_reg         <= 96'd0;
        divider_start           <= 1'b0;
        combined_valid          <= 1'b0;
        combined_scale_uq4_28   <= 32'd0;
        saturated               <= 1'b0;
        error                   <= 1'b0;
        error_code              <= 8'd0;
    end else begin
        divider_start <= 1'b0;

        case (state)
            ST_IDLE: begin
                combined_valid <= 1'b0;
                saturated      <= 1'b0;
                if (scale_valid && !error) begin
                    scale_reg        <= weight_scale_fp16;
                    all_zero_reg     <= all_zero;
                    max_mantissa_reg <= max_mantissa_binary32;
                    max_exponent_reg <= max_exponent_binary32;
                    error_code       <= 8'd0;
                    state            <= ST_PREPARE;
                end
            end

            ST_PREPARE: begin
                if (!fp16_valid_positive) begin
                    error      <= 1'b1;
                    error_code <= ERR_FP16;
                    state      <= ST_ERROR;
                end else if (!all_zero_reg && max_mantissa_reg == 24'd0) begin
                    error      <= 1'b1;
                    error_code <= ERR_MAX_META;
                    state      <= ST_ERROR;
                end else if (shift_too_large) begin
                    error      <= 1'b1;
                    error_code <= ERR_SHIFT;
                    state      <= ST_ERROR;
                end else begin
                    numerator_reg   <= numerator_wire;
                    denominator_reg <= denominator_wire;
                    state           <= ST_DIV_START;
                end
            end

            ST_DIV_START: begin
                divider_start <= 1'b1;
                state         <= ST_DIV_WAIT;
            end

            ST_DIV_WAIT: begin
                if (divider_done) begin
                    if (divider_divide_by_zero) begin
                        error      <= 1'b1;
                        error_code <= ERR_DIVIDER;
                        state      <= ST_ERROR;
                    end else begin
                        combined_scale_uq4_28 <= quotient_saturated
                            ? 32'hffff_ffff
                            : divider_quotient[31:0];
                        saturated      <= quotient_saturated;
                        combined_valid <= 1'b1;
                        state          <= ST_OUTPUT;
                    end
                end
            end

            ST_OUTPUT: begin
                if (combined_valid && combined_ready) begin
                    combined_valid <= 1'b0;
                    saturated      <= 1'b0;
                    state          <= ST_IDLE;
                end
            end

            ST_ERROR: begin
                combined_valid <= 1'b0;
            end

            default: begin
                error      <= 1'b1;
                error_code <= ERR_INTERNAL;
                state      <= ST_ERROR;
            end
        endcase
    end
end

endmodule
