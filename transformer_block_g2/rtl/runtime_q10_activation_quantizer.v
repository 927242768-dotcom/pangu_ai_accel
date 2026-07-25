`timescale 1ns/1ps

// G2 Q6.10 -> symmetric INT8 运行时量化器。
//
// 精确规则：
//   max_abs = max(abs(x_q10))
//   q       = signed RNE(abs(x_q10) * 127 / max_abs)
//   all-zero 向量输出全 0，并向 scale builder 指示 activation_scale=1.0。
//
// 输入按单元素顺序加载；后续 DDR3 adapter 负责从 256-bit beat 解包。
// 输出同样为单元素 valid/ready 流，adapter 负责每 32 项打包写回。
module runtime_q10_activation_quantizer #(
    parameter integer MAX_LENGTH = 4864,
    parameter integer INDEX_WIDTH = 13
)(
    input  wire                         clk,
    input  wire                         rst_n,

    input  wire                         load_start,
    input  wire [INDEX_WIDTH-1:0]       vector_length,
    input  wire                         source_valid,
    output wire                         source_ready,
    input  wire signed [15:0]           source_q10,
    input  wire                         source_last,

    input  wire                         quantize_start,
    output reg                          activation_valid,
    input  wire                         activation_ready,
    output reg  [INDEX_WIDTH-1:0]       activation_index,
    output reg  signed [7:0]            activation_int8,
    output reg                          activation_last,

    output reg  [15:0]                  max_abs_q10,
    output reg  [23:0]                  max_mantissa_binary32,
    output reg  signed [9:0]            max_exponent_binary32,
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
localparam [3:0] ST_DIV_PREP   = 4'd5;
localparam [3:0] ST_DIV_START  = 4'd6;
localparam [3:0] ST_DIV_WAIT   = 4'd7;
localparam [3:0] ST_OUTPUT     = 4'd8;
localparam [3:0] ST_FINISH     = 4'd9;
localparam [3:0] ST_ERROR      = 4'd15;

localparam [7:0] ERR_LENGTH       = 8'h01;
localparam [7:0] ERR_LOAD_COUNT   = 8'h02;
localparam [7:0] ERR_START_ORDER  = 8'h03;
localparam [7:0] ERR_DIVIDER      = 8'h04;
localparam [7:0] ERR_INTERNAL     = 8'hff;

reg [3:0] state;
reg signed [15:0] source_mem [0:MAX_LENGTH-1];
reg [INDEX_WIDTH-1:0] length_reg;
reg [INDEX_WIDTH-1:0] load_count;
reg [INDEX_WIDTH-1:0] quant_index;
reg signed [15:0] source_reg;
reg [15:0] source_magnitude_reg;
reg divider_start;

wire [15:0] source_magnitude_wire =
    source_q10[15] ? (~source_q10[15:0] + 1'b1) : source_q10[15:0];
wire [15:0] loaded_max_candidate =
    (source_magnitude_wire > max_abs_q10) ? source_magnitude_wire : max_abs_q10;
wire vector_length_supported =
    (vector_length == 13'd896) || (vector_length == 13'd4864);
wire output_handshake = activation_valid && activation_ready;

wire divider_busy;
wire divider_done;
wire divider_divide_by_zero;
wire divider_overflow;
wire [31:0] divider_quotient;
wire [31:0] divider_remainder;
wire [31:0] divider_numerator = source_magnitude_reg * 7'd127;
wire [31:0] divider_denominator = {16'd0, max_abs_q10};

assign source_ready = (state == ST_LOAD);
assign debug_state = state;

function [4:0] msb_index16;
    input [15:0] value;
    integer bit_index;
    begin
        msb_index16 = 5'd0;
        for (bit_index = 0; bit_index < 16; bit_index = bit_index + 1) begin
            if (value[bit_index])
                msb_index16 = bit_index[4:0];
        end
    end
endfunction

wire [4:0] max_msb_index = msb_index16(max_abs_q10);
wire [23:0] max_mantissa_wire =
    (max_abs_q10 == 16'd0) ? 24'd0 :
    ({8'd0, max_abs_q10} << (5'd23 - max_msb_index));
wire signed [9:0] max_exponent_wire =
    $signed({1'b0, max_msb_index}) - 10'sd33;

unsigned_divider_rne #(
    .WIDTH(32)
) u_unsigned_divider_rne (
    .clk            (clk),
    .rst_n          (rst_n),
    .start          (divider_start),
    .numerator      (divider_numerator),
    .denominator    (divider_denominator),
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
        source_reg               <= 16'sd0;
        source_magnitude_reg     <= 16'd0;
        divider_start            <= 1'b0;
        activation_valid         <= 1'b0;
        activation_index         <= {INDEX_WIDTH{1'b0}};
        activation_int8          <= 8'sd0;
        activation_last          <= 1'b0;
        max_abs_q10              <= 16'd0;
        max_mantissa_binary32    <= 24'd0;
        max_exponent_binary32    <= 10'sd0;
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
                        max_abs_q10           <= 16'd0;
                        max_mantissa_binary32 <= 24'd0;
                        max_exponent_binary32 <= 10'sd0;
                        all_zero              <= 1'b1;
                        error_code            <= 8'd0;
                        busy                  <= 1'b1;
                        state                 <= ST_LOAD;
                    end
                end
            end

            ST_LOAD: begin
                if (source_valid) begin
                    source_mem[load_count] <= source_q10;
                    max_abs_q10 <= loaded_max_candidate;
                    if (source_magnitude_wire != 16'd0)
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
                max_exponent_binary32 <= all_zero ? 10'sd0 : max_exponent_wire;
                state                 <= ST_READ;
            end

            ST_READ: begin
                source_reg <= source_mem[quant_index];
                state      <= ST_DIV_PREP;
            end

            ST_DIV_PREP: begin
                source_magnitude_reg <=
                    source_reg[15] ? (~source_reg[15:0] + 1'b1) : source_reg[15:0];
                state <= ST_DIV_START;
            end

            ST_DIV_START: begin
                if (all_zero) begin
                    activation_index <= quant_index;
                    activation_int8  <= 8'sd0;
                    activation_last  <= (quant_index + 1'b1 == length_reg);
                    activation_valid <= 1'b1;
                    state            <= ST_OUTPUT;
                end else begin
                    divider_start <= 1'b1;
                    state         <= ST_DIV_WAIT;
                end
            end

            ST_DIV_WAIT: begin
                if (divider_done) begin
                    if (divider_divide_by_zero || divider_overflow || divider_quotient > 32'd127) begin
                        error      <= 1'b1;
                        error_code <= ERR_DIVIDER;
                        busy       <= 1'b0;
                        state      <= ST_ERROR;
                    end else begin
                        activation_index <= quant_index;
                        activation_int8 <= source_reg[15]
                            ? -$signed(divider_quotient[7:0])
                            :  $signed(divider_quotient[7:0]);
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
