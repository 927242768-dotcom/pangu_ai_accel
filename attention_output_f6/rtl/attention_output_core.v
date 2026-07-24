`timescale 1ns/1ps

// unsigned UQ1.31 32 bit × signed Q28 64 bit 顺序精确乘法器。
//
// 概率拆为 2 个 16 bit limb，V 的绝对值拆为 4 个 16 bit limb，
// 共 8 个 16x16 部分积。乘积为 signed 96 bit Q59。顺序复用窄乘法器，
// 避免 32x64 大组合乘法对 100 MHz 时序造成压力。
module uq31_mul_q28_seq8 (
    input  wire                   clk,
    input  wire                   rst_n,
    input  wire                   start,
    input  wire [31:0]            probability_q31,
    input  wire signed [63:0]     value_q28,
    output reg                    busy,
    output reg                    done,
    output reg signed [95:0]      product_q59
);

localparam [1:0] ST_IDLE    = 2'd0;
localparam [1:0] ST_PREPARE = 2'd1;
localparam [1:0] ST_CAPTURE = 2'd2;
localparam [1:0] ST_ACCUM   = 2'd3;

reg [1:0] state;
reg [2:0] step;
reg result_negative;
reg [31:0] probability_magnitude;
reg [63:0] value_magnitude;
reg [15:0] probability_limb_reg;
reg [15:0] value_limb_reg;
reg [31:0] partial_product_reg;
reg [95:0] magnitude_accumulator;

reg [15:0] selected_probability_limb;
reg [15:0] selected_value_limb;
reg [2:0] limb_sum;
reg [95:0] aligned_partial;
wire [31:0] partial_product = probability_limb_reg * value_limb_reg;
wire [95:0] next_magnitude = magnitude_accumulator + aligned_partial;

always @(*) begin
    selected_probability_limb = step[0] ?
        probability_magnitude[31:16] : probability_magnitude[15:0];

    case (step[2:1])
        2'd0: selected_value_limb = value_magnitude[15:0];
        2'd1: selected_value_limb = value_magnitude[31:16];
        2'd2: selected_value_limb = value_magnitude[47:32];
        default: selected_value_limb = value_magnitude[63:48];
    endcase

    limb_sum = {2'd0, step[0]} + {1'b0, step[2:1]};
    case (limb_sum)
        3'd0: aligned_partial = {64'd0, partial_product_reg};
        3'd1: aligned_partial = {48'd0, partial_product_reg, 16'd0};
        3'd2: aligned_partial = {32'd0, partial_product_reg, 32'd0};
        3'd3: aligned_partial = {16'd0, partial_product_reg, 48'd0};
        default: aligned_partial = {partial_product_reg, 64'd0};
    endcase
end

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        state                    <= ST_IDLE;
        step                     <= 3'd0;
        result_negative          <= 1'b0;
        probability_magnitude    <= 32'd0;
        value_magnitude          <= 64'd0;
        probability_limb_reg     <= 16'd0;
        value_limb_reg           <= 16'd0;
        partial_product_reg      <= 32'd0;
        magnitude_accumulator    <= 96'd0;
        product_q59              <= 96'sd0;
        busy                     <= 1'b0;
        done                     <= 1'b0;
    end else begin
        done <= 1'b0;
        case (state)
            ST_IDLE: begin
                busy <= 1'b0;
                if (start) begin
                    result_negative       <= value_q28[63] &&
                                             (probability_q31 != 32'd0);
                    probability_magnitude <= probability_q31;
                    value_magnitude       <= value_q28[63] ?
                                             (~value_q28 + 1'b1) : value_q28;
                    magnitude_accumulator <= 96'd0;
                    step                  <= 3'd0;
                    busy                  <= 1'b1;
                    state                 <= ST_PREPARE;
                end
            end

            ST_PREPARE: begin
                probability_limb_reg <= selected_probability_limb;
                value_limb_reg       <= selected_value_limb;
                state                <= ST_CAPTURE;
            end

            ST_CAPTURE: begin
                partial_product_reg <= partial_product;
                state               <= ST_ACCUM;
            end

            ST_ACCUM: begin
                if (step == 3'd7) begin
                    product_q59 <= result_negative ?
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


// F6 Attention 输出独立加权和核心。
//
// 概率缓存： [14,16] uint32 UQ1.31，共 28 个 256 bit beat。
// V 缓存：    [最多16,2,64] int64 Q28，token-major，共最多 512 beat。
// GQA：       Q head 0..6 -> KV0；Q head 7..13 -> KV1。
// 计算：      每项产生 signed Q59，最多 16 项在 signed 100 bit 中精确累加，
//             全部累加完成后仅执行一次 signed RNE >>31，恢复 signed Q28，
//             再显式饱和到 signed int64。
// 输出：      [14,64] head-major，逐元素 valid/ready 流式交给控制器。
module attention_output_core (
    input  wire                   clk,
    input  wire                   rst_n,

    input  wire                   probability_beat_we,
    input  wire [4:0]             probability_beat_index,
    input  wire [255:0]           probability_beat_data,
    input  wire                   v_beat_we,
    input  wire [8:0]             v_beat_index,
    input  wire [255:0]           v_beat_data,

    input  wire                   start,
    input  wire [4:0]             token_count,
    input  wire                   result_ready,
    output reg                    busy,
    output reg                    result_valid,
    output reg [3:0]              result_head,
    output reg [5:0]              result_dimension,
    output reg signed [63:0]      result_q28,
    output reg                    done
);

localparam [3:0] ST_IDLE       = 4'd0;
localparam [3:0] ST_READ       = 4'd1;
localparam [3:0] ST_SELECT     = 4'd2;
localparam [3:0] ST_MUL_START  = 4'd3;
localparam [3:0] ST_MUL_WAIT   = 4'd4;
localparam [3:0] ST_SUM_ABS    = 4'd5;
localparam [3:0] ST_SUM_ROUND  = 4'd6;
localparam [3:0] ST_RESULT_OUT = 4'd7;
localparam [3:0] ST_RESULT_WAIT= 4'd8;

reg [255:0] probability_mem [0:27];
reg [255:0] v_mem [0:511];
reg [255:0] probability_beat_reg;
reg [255:0] v_beat_reg;

reg [3:0] state;
reg [4:0] active_token_count;
reg [3:0] q_head;
reg [5:0] dimension;
reg [3:0] token_index;
reg signed [99:0] accumulator_q59;
reg signed [99:0] final_sum_q59;
reg final_negative;
reg [99:0] final_magnitude;
reg [69:0] rounded_magnitude;

reg [31:0] mul_probability_q31;
reg signed [63:0] mul_value_q28;
reg mul_start;
wire mul_busy;
wire mul_done;
wire signed [95:0] mul_product_q59;

wire [4:0] probability_read_beat_index =
    ({1'b0, q_head} << 1) + {4'd0, token_index[3]};
wire [8:0] v_read_beat_index =
    ({5'd0, token_index} << 5) +
    ((q_head < 4'd7) ? 9'd0 : 9'd16) +
    {5'd0, dimension[5:2]};

reg [31:0] selected_probability_q31;
reg signed [63:0] selected_v_q28;
wire signed [99:0] extended_product_q59 =
    {{4{mul_product_q59[95]}}, mul_product_q59};
wire signed [99:0] next_sum_q59 = accumulator_q59 + extended_product_q59;

wire [68:0] final_quotient = final_magnitude[99:31];
wire [30:0] final_remainder = final_magnitude[30:0];
wire final_round_up =
    (final_remainder > 31'h40000000) ||
    ((final_remainder == 31'h40000000) && final_quotient[0]);

always @(*) begin
    case (token_index[2:0])
        3'd0: selected_probability_q31 = probability_beat_reg[31:0];
        3'd1: selected_probability_q31 = probability_beat_reg[63:32];
        3'd2: selected_probability_q31 = probability_beat_reg[95:64];
        3'd3: selected_probability_q31 = probability_beat_reg[127:96];
        3'd4: selected_probability_q31 = probability_beat_reg[159:128];
        3'd5: selected_probability_q31 = probability_beat_reg[191:160];
        3'd6: selected_probability_q31 = probability_beat_reg[223:192];
        default: selected_probability_q31 = probability_beat_reg[255:224];
    endcase

    case (dimension[1:0])
        2'd0: selected_v_q28 = v_beat_reg[63:0];
        2'd1: selected_v_q28 = v_beat_reg[127:64];
        2'd2: selected_v_q28 = v_beat_reg[191:128];
        default: selected_v_q28 = v_beat_reg[255:192];
    endcase
end

function [63:0] signed_magnitude_sat64;
    input sign_value;
    input [69:0] magnitude;
    begin
        if (!sign_value) begin
            if (|magnitude[69:63])
                signed_magnitude_sat64 = 64'h7fff_ffff_ffff_ffff;
            else
                signed_magnitude_sat64 = magnitude[63:0];
        end else begin
            if (|magnitude[69:64])
                signed_magnitude_sat64 = 64'h8000_0000_0000_0000;
            else if (magnitude[63])
                signed_magnitude_sat64 = 64'h8000_0000_0000_0000;
            else
                signed_magnitude_sat64 = (~magnitude[63:0]) + 1'b1;
        end
    end
endfunction

uq31_mul_q28_seq8 u_uq31_mul_q28_seq8 (
    .clk             (clk),
    .rst_n           (rst_n),
    .start           (mul_start),
    .probability_q31 (mul_probability_q31),
    .value_q28       (mul_value_q28),
    .busy            (mul_busy),
    .done            (mul_done),
    .product_q59     (mul_product_q59)
);

// 缓存不复位；控制器只读取本次已经装载的有效 beat。该结构便于推断 DRM。
always @(posedge clk) begin
    if (probability_beat_we)
        probability_mem[probability_beat_index] <= probability_beat_data;
    if (v_beat_we)
        v_mem[v_beat_index] <= v_beat_data;
end

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        probability_beat_reg <= 256'd0;
        v_beat_reg           <= 256'd0;
        state                <= ST_IDLE;
        active_token_count   <= 5'd0;
        q_head               <= 4'd0;
        dimension            <= 6'd0;
        token_index          <= 4'd0;
        accumulator_q59      <= 100'sd0;
        final_sum_q59        <= 100'sd0;
        final_negative       <= 1'b0;
        final_magnitude      <= 100'd0;
        rounded_magnitude    <= 70'd0;
        mul_probability_q31  <= 32'd0;
        mul_value_q28        <= 64'sd0;
        mul_start            <= 1'b0;
        busy                 <= 1'b0;
        result_valid         <= 1'b0;
        result_head          <= 4'd0;
        result_dimension     <= 6'd0;
        result_q28           <= 64'sd0;
        done                 <= 1'b0;
    end else begin
        mul_start    <= 1'b0;
        result_valid <= 1'b0;
        done         <= 1'b0;

        case (state)
            ST_IDLE: begin
                busy <= 1'b0;
                if (start) begin
                    active_token_count <= token_count;
                    q_head             <= 4'd0;
                    dimension          <= 6'd0;
                    token_index        <= 4'd0;
                    accumulator_q59    <= 100'sd0;
                    busy               <= 1'b1;
                    state              <= ST_READ;
                end
            end

            ST_READ: begin
                probability_beat_reg <= probability_mem[probability_read_beat_index];
                v_beat_reg           <= v_mem[v_read_beat_index];
                state                <= ST_SELECT;
            end

            ST_SELECT: begin
                mul_probability_q31 <= selected_probability_q31;
                mul_value_q28       <= selected_v_q28;
                state               <= ST_MUL_START;
            end

            ST_MUL_START: begin
                mul_start <= 1'b1;
                state     <= ST_MUL_WAIT;
            end

            ST_MUL_WAIT: begin
                if (mul_done) begin
                    if (token_index == active_token_count - 1'b1) begin
                        final_sum_q59 <= next_sum_q59;
                        state         <= ST_SUM_ABS;
                    end else begin
                        accumulator_q59 <= next_sum_q59;
                        token_index     <= token_index + 1'b1;
                        state           <= ST_READ;
                    end
                end
            end

            ST_SUM_ABS: begin
                final_negative  <= final_sum_q59[99];
                final_magnitude <= final_sum_q59[99] ?
                                   (~final_sum_q59 + 1'b1) : final_sum_q59;
                state           <= ST_SUM_ROUND;
            end

            ST_SUM_ROUND: begin
                rounded_magnitude <= {1'b0, final_quotient} + final_round_up;
                state             <= ST_RESULT_OUT;
            end

            ST_RESULT_OUT: begin
                result_valid     <= 1'b1;
                result_head      <= q_head;
                result_dimension <= dimension;
                result_q28       <= signed_magnitude_sat64(
                    final_negative, rounded_magnitude
                );
                state <= ST_RESULT_WAIT;
            end

            ST_RESULT_WAIT: begin
                if (result_ready) begin
                    if ((q_head == 4'd13) && (dimension == 6'd63)) begin
                        done <= 1'b1;
                        busy <= 1'b0;
                        state <= ST_IDLE;
                    end else begin
                        token_index     <= 4'd0;
                        accumulator_q59 <= 100'sd0;
                        if (dimension == 6'd63) begin
                            dimension <= 6'd0;
                            q_head    <= q_head + 1'b1;
                        end else begin
                            dimension <= dimension + 1'b1;
                        end
                        state <= ST_READ;
                    end
                end
            end

            default: begin
                state <= ST_IDLE;
                busy  <= 1'b0;
            end
        endcase
    end
end

wire _unused_mul_busy = mul_busy;

endmodule
