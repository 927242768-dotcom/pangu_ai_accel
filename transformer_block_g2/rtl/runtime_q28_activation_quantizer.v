`timescale 1ns/1ps

// G2 Q28 -> binary32 -> symmetric INT8 运行时量化器。
//
// 严格复现：
//   signed int64 Q28 -> binary64 -> /2^28 -> binary32
//   q = signed RNE(abs(binary32_x) * 127 / max_abs_binary32)
//
// 输入/输出是单元素 valid/ready 流；DDR3 adapter 负责 256-bit 解包和打包。
module runtime_q28_activation_quantizer #(
    parameter integer MAX_LENGTH = 4864,
    parameter integer INDEX_WIDTH = 13
)(
    input  wire                         clk,
    input  wire                         rst_n,

    input  wire                         load_start,
    input  wire [INDEX_WIDTH-1:0]       vector_length,
    input  wire                         source_valid,
    output wire                         source_ready,
    input  wire signed [63:0]           source_q28,
    input  wire                         source_last,

    input  wire                         quantize_start,
    output reg                          activation_valid,
    input  wire                         activation_ready,
    output reg  [INDEX_WIDTH-1:0]       activation_index,
    output reg  signed [7:0]            activation_int8,
    output reg                          activation_last,

    output reg  [23:0]                  max_mantissa_binary32,
    output reg  signed [9:0]            max_exponent_binary32,
    output reg  [31:0]                  max_abs_binary32_bits,
    output reg                          all_zero,
    output reg                          load_complete,
    output reg                          busy,
    output reg                          done,
    output reg                          error,
    output reg  [7:0]                   error_code,
    output wire [3:0]                   debug_state
);

localparam [3:0] ST_IDLE       = 4'd0;
localparam [3:0] ST_LOAD       = 4'd1;
localparam [3:0] ST_LOADED     = 4'd2;
localparam [3:0] ST_PREP_MAX   = 4'd3;
localparam [3:0] ST_READ       = 4'd4;
localparam [3:0] ST_RATIO_PREP = 4'd5;
localparam [3:0] ST_DIV_START  = 4'd6;
localparam [3:0] ST_DIV_WAIT   = 4'd7;
localparam [3:0] ST_OUTPUT     = 4'd8;
localparam [3:0] ST_FINISH     = 4'd9;
localparam [3:0] ST_ERROR      = 4'd15;

localparam [7:0] ERR_LENGTH       = 8'h01;
localparam [7:0] ERR_LOAD_COUNT   = 8'h02;
localparam [7:0] ERR_START_ORDER  = 8'h03;
localparam [7:0] ERR_EXPONENT     = 8'h04;
localparam [7:0] ERR_DIVIDER      = 8'h05;
localparam [7:0] ERR_INTERNAL     = 8'hff;

reg [3:0] state;
reg signed [63:0] source_mem [0:MAX_LENGTH-1];
reg [INDEX_WIDTH-1:0] length_reg;
reg [INDEX_WIDTH-1:0] load_count;
reg [INDEX_WIDTH-1:0] quant_index;
reg [63:0] max_abs_raw;
reg signed [63:0] source_reg;
reg [95:0] divider_numerator_reg;
reg [95:0] divider_denominator_reg;
reg divider_start;

wire [63:0] source_magnitude_wire = source_q28[63]
    ? (~source_q28[63:0] + 1'b1)
    : source_q28[63:0];
wire [63:0] loaded_max_candidate =
    (source_magnitude_wire > max_abs_raw) ? source_magnitude_wire : max_abs_raw;
wire vector_length_supported =
    (vector_length == 13'd896) || (vector_length == 13'd4864);
wire output_handshake = activation_valid && activation_ready;

wire max_sign_unused;
wire max_zero;
wire [23:0] max_mantissa_wire;
wire signed [9:0] max_exponent_wire;
wire [31:0] max_bits_wire;
wire source_sign_wire;
wire source_zero_wire;
wire [23:0] source_mantissa_wire;
wire signed [9:0] source_exponent_wire;
wire [31:0] source_bits_unused;

q28_to_binary32 u_max_q28_to_binary32 (
    .q28_value          ($signed(max_abs_raw)),
    .sign               (max_sign_unused),
    .zero               (max_zero),
    .mantissa           (max_mantissa_wire),
    .component_exponent (max_exponent_wire),
    .binary32_bits      (max_bits_wire)
);

q28_to_binary32 u_source_q28_to_binary32 (
    .q28_value          (source_reg),
    .sign               (source_sign_wire),
    .zero               (source_zero_wire),
    .mantissa           (source_mantissa_wire),
    .component_exponent (source_exponent_wire),
    .binary32_bits      (source_bits_unused)
);

wire signed [10:0] exponent_difference_signed =
    $signed(max_exponent_binary32) - $signed(source_exponent_wire);
wire exponent_difference_negative = exponent_difference_signed[10];
wire [10:0] exponent_difference = exponent_difference_signed[10:0];
wire exponent_difference_too_large = exponent_difference >= 11'd72;
wire [30:0] ratio_numerator_base = source_mantissa_wire * 7'd127;
wire [95:0] ratio_numerator_wire = {{65{1'b0}}, ratio_numerator_base};
wire [95:0] ratio_denominator_wire =
    {{72{1'b0}}, max_mantissa_binary32} << exponent_difference;

wire divider_busy;
wire divider_done;
wire divider_divide_by_zero;
wire divider_overflow;
wire [95:0] divider_quotient;
wire [95:0] divider_remainder;

assign source_ready = (state == ST_LOAD);
assign debug_state = state;

unsigned_divider_rne #(
    .WIDTH(96)
) u_unsigned_divider_rne (
    .clk            (clk),
    .rst_n          (rst_n),
    .start          (divider_start),
    .numerator      (divider_numerator_reg),
    .denominator    (divider_denominator_reg),
    .busy           (divider_busy),
    .done           (divider_done),
    .divide_by_zero (divider_divide_by_zero),
    .overflow       (divider_overflow),
    .quotient       (divider_quotient),
    .remainder      (divider_remainder)
);

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        state                    <= ST_IDLE;
        length_reg               <= {INDEX_WIDTH{1'b0}};
        load_count               <= {INDEX_WIDTH{1'b0}};
        quant_index              <= {INDEX_WIDTH{1'b0}};
        max_abs_raw              <= 64'd0;
        source_reg               <= 64'sd0;
        divider_numerator_reg    <= 96'd0;
        divider_denominator_reg  <= 96'd0;
        divider_start            <= 1'b0;
        activation_valid         <= 1'b0;
        activation_index         <= {INDEX_WIDTH{1'b0}};
        activation_int8          <= 8'sd0;
        activation_last          <= 1'b0;
        max_mantissa_binary32    <= 24'd0;
        max_exponent_binary32    <= 10'sd0;
        max_abs_binary32_bits    <= 32'd0;
        all_zero                 <= 1'b1;
        load_complete            <= 1'b0;
        busy                     <= 1'b0;
        done                     <= 1'b0;
        error                    <= 1'b0;
        error_code               <= 8'd0;
    end else begin
        divider_start <= 1'b0;
        done          <= 1'b0;

        case (state)
            ST_IDLE: begin
                busy             <= 1'b0;
                activation_valid <= 1'b0;
                activation_last  <= 1'b0;
                load_complete    <= 1'b0;
                if (load_start && !error) begin
                    if (!vector_length_supported) begin
                        error      <= 1'b1;
                        error_code <= ERR_LENGTH;
                        state      <= ST_ERROR;
                    end else begin
                        length_reg            <= vector_length;
                        load_count            <= {INDEX_WIDTH{1'b0}};
                        quant_index           <= {INDEX_WIDTH{1'b0}};
                        max_abs_raw            <= 64'd0;
                        max_mantissa_binary32 <= 24'd0;
                        max_exponent_binary32 <= 10'sd0;
                        max_abs_binary32_bits <= 32'd0;
                        all_zero              <= 1'b1;
                        error_code            <= 8'd0;
                        busy                  <= 1'b1;
                        state                 <= ST_LOAD;
                    end
                end
            end

            ST_LOAD: begin
                if (source_valid) begin
                    source_mem[load_count] <= source_q28;
                    max_abs_raw <= loaded_max_candidate;
                    if (source_magnitude_wire != 64'd0)
                        all_zero <= 1'b0;

                    if (source_last) begin
                        if (load_count + 1'b1 != length_reg) begin
                            error      <= 1'b1;
                            error_code <= ERR_LOAD_COUNT;
                            busy       <= 1'b0;
                            state      <= ST_ERROR;
                        end else begin
                            load_complete <= 1'b1;
                            state         <= ST_LOADED;
                        end
                    end else if (load_count + 1'b1 == length_reg) begin
                        error      <= 1'b1;
                        error_code <= ERR_LOAD_COUNT;
                        busy       <= 1'b0;
                        state      <= ST_ERROR;
                    end else begin
                        load_count <= load_count + 1'b1;
                    end
                end
            end

            ST_LOADED: begin
                if (load_start) begin
                    error      <= 1'b1;
                    error_code <= ERR_START_ORDER;
                    busy       <= 1'b0;
                    state      <= ST_ERROR;
                end else if (quantize_start) begin
                    quant_index <= {INDEX_WIDTH{1'b0}};
                    state       <= ST_PREP_MAX;
                end
            end

            ST_PREP_MAX: begin
                max_mantissa_binary32 <= max_mantissa_wire;
                max_exponent_binary32 <= max_zero ? 10'sd0 : max_exponent_wire;
                max_abs_binary32_bits <= max_zero ? 32'd0 : {1'b0, max_bits_wire[30:0]};
                state                 <= ST_READ;
            end

            ST_READ: begin
                source_reg <= source_mem[quant_index];
                state      <= ST_RATIO_PREP;
            end

            ST_RATIO_PREP: begin
                if (all_zero || source_zero_wire) begin
                    activation_index <= quant_index;
                    activation_int8  <= 8'sd0;
                    activation_last  <= (quant_index + 1'b1 == length_reg);
                    activation_valid <= 1'b1;
                    state            <= ST_OUTPUT;
                end else if (exponent_difference_negative || exponent_difference_too_large) begin
                    error      <= 1'b1;
                    error_code <= ERR_EXPONENT;
                    busy       <= 1'b0;
                    state      <= ST_ERROR;
                end else begin
                    divider_numerator_reg   <= ratio_numerator_wire;
                    divider_denominator_reg <= ratio_denominator_wire;
                    state                   <= ST_DIV_START;
                end
            end

            ST_DIV_START: begin
                divider_start <= 1'b1;
                state         <= ST_DIV_WAIT;
            end

            ST_DIV_WAIT: begin
                if (divider_done) begin
                    if (
                        divider_divide_by_zero || divider_overflow ||
                        (|divider_quotient[95:7]) || divider_quotient[6:0] > 7'd127
                    ) begin
                        error      <= 1'b1;
                        error_code <= ERR_DIVIDER;
                        busy       <= 1'b0;
                        state      <= ST_ERROR;
                    end else begin
                        activation_index <= quant_index;
                        activation_int8 <= source_sign_wire
                            ? -$signed({1'b0, divider_quotient[6:0]})
                            :  $signed({1'b0, divider_quotient[6:0]});
                        activation_last  <= (quant_index + 1'b1 == length_reg);
                        activation_valid <= 1'b1;
                        state            <= ST_OUTPUT;
                    end
                end
            end

            ST_OUTPUT: begin
                if (output_handshake) begin
                    activation_valid <= 1'b0;
                    activation_last  <= 1'b0;
                    if (quant_index + 1'b1 == length_reg) begin
                        state <= ST_FINISH;
                    end else begin
                        quant_index <= quant_index + 1'b1;
                        state       <= ST_READ;
                    end
                end
            end

            ST_FINISH: begin
                busy <= 1'b0;
                done <= 1'b1;
                state <= ST_IDLE;
            end

            ST_ERROR: begin
                busy             <= 1'b0;
                activation_valid <= 1'b0;
                activation_last  <= 1'b0;
            end

            default: begin
                error      <= 1'b1;
                error_code <= ERR_INTERNAL;
                busy       <= 1'b0;
                state      <= ST_ERROR;
            end
        endcase
    end
end

endmodule
