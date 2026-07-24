`timescale 1ns/1ps

// 计算 reciprocal_q31 = RNE(2^62 / denominator_q31)。
// denominator_q31 为最多 16 个 UQ1.31 exp 的和，范围 [2^31, 16*2^31]。
// 使用 63 周期恢复除法，避免综合出长组合除法器。
module reciprocal_q31_divider (
    input  wire         clk,
    input  wire         rst_n,
    input  wire         start,
    input  wire [35:0]  denominator_q31,
    output reg          busy,
    output reg          done,
    output reg  [31:0]  reciprocal_q31
);

reg [35:0] denominator_reg;
reg [62:0] dividend_shift;
reg [62:0] quotient_work;
reg [36:0] remainder_work;
reg [5:0]  bit_count;

wire [36:0] shifted_remainder = {
    remainder_work[35:0], dividend_shift[62]
};
wire subtract_denominator =
    shifted_remainder >= {1'b0, denominator_reg};
wire [36:0] next_remainder = subtract_denominator ?
    (shifted_remainder - {1'b0, denominator_reg}) : shifted_remainder;
wire [62:0] next_quotient = {
    quotient_work[61:0], subtract_denominator
};
wire [62:0] next_dividend = {dividend_shift[61:0], 1'b0};
wire [37:0] doubled_remainder = {next_remainder, 1'b0};
wire [37:0] extended_denominator = {2'b0, denominator_reg};
wire quotient_round_up =
    (doubled_remainder > extended_denominator) ||
    ((doubled_remainder == extended_denominator) && next_quotient[0]);
wire [63:0] rounded_quotient = {1'b0, next_quotient} + quotient_round_up;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        denominator_reg <= 36'd0;
        dividend_shift  <= 63'd0;
        quotient_work   <= 63'd0;
        remainder_work  <= 37'd0;
        bit_count       <= 6'd0;
        busy            <= 1'b0;
        done            <= 1'b0;
        reciprocal_q31  <= 32'd0;
    end else begin
        done <= 1'b0;
        if (!busy) begin
            if (start && (denominator_q31 != 36'd0)) begin
                denominator_reg <= denominator_q31;
                dividend_shift  <= {1'b1, 62'd0}; // 2^62
                quotient_work   <= 63'd0;
                remainder_work  <= 37'd0;
                bit_count       <= 6'd0;
                busy            <= 1'b1;
            end
        end else begin
            dividend_shift <= next_dividend;
            quotient_work  <= next_quotient;
            remainder_work <= next_remainder;
            if (bit_count == 6'd62) begin
                if (|rounded_quotient[63:32])
                    reciprocal_q31 <= 32'hffff_ffff;
                else
                    reciprocal_q31 <= rounded_quotient[31:0];
                busy <= 1'b0;
                done <= 1'b1;
            end else begin
                bit_count <= bit_count + 1'b1;
            end
        end
    end
end

endmodule


// F5 固定 14x16 mask 感知 Softmax 核心。
//
// 输入 score： [14,16] signed int64 Q28，INT64_MIN 表示 mask。
// exp LUT：    513 个 unsigned UQ1.31 uint32 端点，[-16,0]，步长 1/32。
// 输出概率：   [14,16] unsigned UQ1.31 uint32。
module softmax_core (
    input  wire                 clk,
    input  wire                 rst_n,

    input  wire                 score_beat_we,
    input  wire [5:0]           score_beat_index,
    input  wire [255:0]         score_beat_data,
    input  wire                 lut_beat_we,
    input  wire [6:0]           lut_beat_index,
    input  wire [255:0]         lut_beat_data,

    input  wire                 start,
    input  wire                 probability_ready,
    output reg                  busy,
    output reg                  probability_valid,
    output reg [3:0]            probability_head,
    output reg [3:0]            probability_token,
    output reg [31:0]           probability_q31,
    output reg                  done
);

localparam [4:0] ST_IDLE              = 5'd0;
localparam [4:0] ST_SCORE_READ_REQ    = 5'd1;
localparam [4:0] ST_SCORE_READ_STORE  = 5'd2;
localparam [4:0] ST_MAX_INIT          = 5'd3;
localparam [4:0] ST_MAX_SCAN          = 5'd4;
localparam [4:0] ST_EXP_INIT          = 5'd5;
localparam [4:0] ST_EXP_PREP          = 5'd6;
localparam [4:0] ST_LUT_USE           = 5'd7;
localparam [4:0] ST_LUT_NEXT_USE      = 5'd8;
localparam [4:0] ST_INTERP_DELTA      = 5'd9;
localparam [4:0] ST_INTERP_PARTIAL    = 5'd10;
localparam [4:0] ST_INTERP_PAIR_SUM   = 5'd11;
localparam [4:0] ST_INTERP_FINAL_SUM  = 5'd12;
localparam [4:0] ST_INTERP_ROUND      = 5'd13;
localparam [4:0] ST_EXP_COMMIT        = 5'd14;
localparam [4:0] ST_DIV_START         = 5'd15;
localparam [4:0] ST_DIV_WAIT          = 5'd16;
localparam [4:0] ST_OUT_INIT          = 5'd17;
localparam [4:0] ST_OUT_SELECT        = 5'd18;
localparam [4:0] ST_OUT_MUL           = 5'd19;
localparam [4:0] ST_OUT_ROUND         = 5'd20;
localparam [4:0] ST_OUT_VALID         = 5'd21;
localparam [4:0] ST_OUT_WAIT          = 5'd22;

localparam signed [64:0] EXP_MIN_Q28 = -65'sd4294967296; // -16 * 2^28
localparam [31:0] PROB_ONE = 32'h8000_0000;
localparam [63:0] MASK_VALUE = 64'h8000_0000_0000_0000;

reg [255:0] score_mem [0:55];
reg [255:0] lut_mem [0:64];
reg signed [63:0] head_scores [0:15];
reg [31:0] exp_values [0:15];

reg [4:0] state;
reg [3:0] active_head;
reg [1:0] load_beat;
reg [3:0] token_index;
reg [255:0] score_beat_reg;
reg [255:0] lut_beat_reg;
reg signed [63:0] max_score_q28;
reg max_valid;
reg active_all_masked;
reg [35:0] sum_exp_q31;
reg [35:0] divider_denominator_q31;
reg [31:0] reciprocal_q31;
reg [31:0] pending_exp_q31;
reg [31:0] selected_exp_q31;
reg selected_output_masked;
reg [31:0] pending_probability_q31;

reg [6:0] lut_beat_address;
reg [2:0] lut_lane;
reg [22:0] lut_remainder;
reg lut_exact_endpoint;
reg [31:0] lut_left;
reg [31:0] lut_right;
reg [25:0] lut_delta;
reg [24:0] interp_partial00;
reg [23:0] interp_partial01;
reg [24:0] interp_partial10;
reg [23:0] interp_partial11;
reg [36:0] interp_pair_sum_low;
reg [35:0] interp_pair_sum_high;
reg [48:0] interp_product;
reg [63:0] norm_product;

reg divider_start;
wire divider_busy;
wire divider_done;
wire [31:0] divider_result_q31;

integer reset_index;

wire [5:0] score_read_address =
    ({2'd0, active_head} << 2) + {4'd0, load_beat};
wire signed [63:0] current_score_q28 = head_scores[token_index];
wire current_score_valid = (current_score_q28 != $signed(MASK_VALUE));
wire take_current_max = current_score_valid &&
    (!max_valid || ($signed(current_score_q28) > $signed(max_score_q28)));
wire signed [64:0] score_difference_q28 =
    {current_score_q28[63], current_score_q28} -
    {max_score_q28[63], max_score_q28};
wire [64:0] difference_magnitude = score_difference_q28[64] ?
    (~score_difference_q28 + 1'b1) : score_difference_q28;
wire [35:0] sum_with_pending =
    sum_exp_q31 + {4'd0, pending_exp_q31};

wire [32:0] lut_delta_full = {1'b0, lut_left} - {1'b0, lut_right};
wire [25:0] interp_quotient = interp_product[48:23];
wire [22:0] interp_remainder = interp_product[22:0];
wire interp_round_up =
    (interp_remainder > 23'h400000) ||
    ((interp_remainder == 23'h400000) && interp_quotient[0]);
wire [26:0] interp_rounded = {1'b0, interp_quotient} + interp_round_up;

wire [32:0] norm_quotient = norm_product[63:31];
wire [30:0] norm_remainder = norm_product[30:0];
wire norm_round_up =
    (norm_remainder > 31'h40000000) ||
    ((norm_remainder == 31'h40000000) && norm_quotient[0]);
wire [33:0] norm_rounded = {1'b0, norm_quotient} + norm_round_up;
wire [31:0] norm_probability_sat =
    (norm_rounded >= 34'd2147483648) ? PROB_ONE : norm_rounded[31:0];

function [31:0] select_lut_value;
    input [255:0] beat;
    input [2:0] lane;
    begin
        case (lane)
            3'd0: select_lut_value = beat[31:0];
            3'd1: select_lut_value = beat[63:32];
            3'd2: select_lut_value = beat[95:64];
            3'd3: select_lut_value = beat[127:96];
            3'd4: select_lut_value = beat[159:128];
            3'd5: select_lut_value = beat[191:160];
            3'd6: select_lut_value = beat[223:192];
            default: select_lut_value = beat[255:224];
        endcase
    end
endfunction

reciprocal_q31_divider u_reciprocal_q31_divider (
    .clk             (clk),
    .rst_n           (rst_n),
    .start           (divider_start),
    .denominator_q31 (divider_denominator_q31),
    .busy            (divider_busy),
    .done            (divider_done),
    .reciprocal_q31  (divider_result_q31)
);

// 两个缓存均不复位；控制器只会在完整加载后启动计算。
always @(posedge clk) begin
    if (score_beat_we)
        score_mem[score_beat_index] <= score_beat_data;
    if (lut_beat_we)
        lut_mem[lut_beat_index] <= lut_beat_data;
end

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        state                    <= ST_IDLE;
        active_head              <= 4'd0;
        load_beat                <= 2'd0;
        token_index              <= 4'd0;
        score_beat_reg           <= 256'd0;
        lut_beat_reg             <= 256'd0;
        max_score_q28            <= 64'sd0;
        max_valid                <= 1'b0;
        active_all_masked        <= 1'b0;
        sum_exp_q31              <= 36'd0;
        divider_denominator_q31  <= 36'd0;
        reciprocal_q31           <= 32'd0;
        pending_exp_q31          <= 32'd0;
        selected_exp_q31         <= 32'd0;
        selected_output_masked   <= 1'b0;
        pending_probability_q31  <= 32'd0;
        lut_beat_address         <= 7'd0;
        lut_lane                 <= 3'd0;
        lut_remainder            <= 23'd0;
        lut_exact_endpoint       <= 1'b0;
        lut_left                 <= 32'd0;
        lut_right                <= 32'd0;
        lut_delta                <= 26'd0;
        interp_partial00         <= 25'd0;
        interp_partial01         <= 24'd0;
        interp_partial10         <= 25'd0;
        interp_partial11         <= 24'd0;
        interp_pair_sum_low      <= 37'd0;
        interp_pair_sum_high     <= 36'd0;
        interp_product           <= 49'd0;
        norm_product             <= 64'd0;
        divider_start            <= 1'b0;
        busy                     <= 1'b0;
        probability_valid        <= 1'b0;
        probability_head         <= 4'd0;
        probability_token        <= 4'd0;
        probability_q31          <= 32'd0;
        done                     <= 1'b0;
        for (reset_index = 0; reset_index < 16; reset_index = reset_index + 1) begin
            head_scores[reset_index] <= 64'sd0;
            exp_values[reset_index]  <= 32'd0;
        end
    end else begin
        divider_start     <= 1'b0;
        probability_valid <= 1'b0;
        done              <= 1'b0;

        case (state)
            ST_IDLE: begin
                busy <= 1'b0;
                if (start) begin
                    active_head <= 4'd0;
                    load_beat   <= 2'd0;
                    busy        <= 1'b1;
                    state       <= ST_SCORE_READ_REQ;
                end
            end

            ST_SCORE_READ_REQ: begin
                score_beat_reg <= score_mem[score_read_address];
                state          <= ST_SCORE_READ_STORE;
            end

            ST_SCORE_READ_STORE: begin
                case (load_beat)
                    2'd0: begin
                        head_scores[0] <= score_beat_reg[63:0];
                        head_scores[1] <= score_beat_reg[127:64];
                        head_scores[2] <= score_beat_reg[191:128];
                        head_scores[3] <= score_beat_reg[255:192];
                    end
                    2'd1: begin
                        head_scores[4] <= score_beat_reg[63:0];
                        head_scores[5] <= score_beat_reg[127:64];
                        head_scores[6] <= score_beat_reg[191:128];
                        head_scores[7] <= score_beat_reg[255:192];
                    end
                    2'd2: begin
                        head_scores[8]  <= score_beat_reg[63:0];
                        head_scores[9]  <= score_beat_reg[127:64];
                        head_scores[10] <= score_beat_reg[191:128];
                        head_scores[11] <= score_beat_reg[255:192];
                    end
                    default: begin
                        head_scores[12] <= score_beat_reg[63:0];
                        head_scores[13] <= score_beat_reg[127:64];
                        head_scores[14] <= score_beat_reg[191:128];
                        head_scores[15] <= score_beat_reg[255:192];
                    end
                endcase
                if (load_beat == 2'd3) begin
                    state <= ST_MAX_INIT;
                end else begin
                    load_beat <= load_beat + 1'b1;
                    state     <= ST_SCORE_READ_REQ;
                end
            end

            ST_MAX_INIT: begin
                token_index       <= 4'd0;
                max_score_q28     <= 64'sd0;
                max_valid         <= 1'b0;
                active_all_masked <= 1'b0;
                state             <= ST_MAX_SCAN;
            end

            ST_MAX_SCAN: begin
                if (take_current_max)
                    max_score_q28 <= current_score_q28;
                if (current_score_valid)
                    max_valid <= 1'b1;

                if (token_index == 4'd15) begin
                    token_index <= 4'd0;
                    if (max_valid || current_score_valid) begin
                        active_all_masked <= 1'b0;
                        state             <= ST_EXP_INIT;
                    end else begin
                        active_all_masked <= 1'b1;
                        state             <= ST_OUT_INIT;
                    end
                end else begin
                    token_index <= token_index + 1'b1;
                end
            end

            ST_EXP_INIT: begin
                token_index <= 4'd0;
                sum_exp_q31 <= 36'd0;
                state       <= ST_EXP_PREP;
            end

            ST_EXP_PREP: begin
                if (!current_score_valid) begin
                    pending_exp_q31 <= 32'd0;
                    state           <= ST_EXP_COMMIT;
                end else if (!score_difference_q28[64]) begin
                    pending_exp_q31 <= PROB_ONE;
                    state           <= ST_EXP_COMMIT;
                end else if (score_difference_q28 < EXP_MIN_Q28) begin
                    pending_exp_q31 <= 32'd0;
                    state           <= ST_EXP_COMMIT;
                end else begin
                    lut_beat_address   <= difference_magnitude[32:26];
                    lut_lane           <= difference_magnitude[25:23];
                    lut_remainder      <= difference_magnitude[22:0];
                    lut_exact_endpoint <= difference_magnitude[32];
                    lut_beat_reg       <= lut_mem[difference_magnitude[32:26]];
                    state              <= ST_LUT_USE;
                end
            end

            ST_LUT_USE: begin
                if (lut_exact_endpoint) begin
                    pending_exp_q31 <= select_lut_value(lut_beat_reg, lut_lane);
                    state           <= ST_EXP_COMMIT;
                end else if (lut_lane == 3'd7) begin
                    lut_left     <= select_lut_value(lut_beat_reg, lut_lane);
                    lut_beat_reg <= lut_mem[lut_beat_address + 1'b1];
                    state        <= ST_LUT_NEXT_USE;
                end else begin
                    lut_left  <= select_lut_value(lut_beat_reg, lut_lane);
                    lut_right <= select_lut_value(lut_beat_reg, lut_lane + 1'b1);
                    state     <= ST_INTERP_DELTA;
                end
            end

            ST_LUT_NEXT_USE: begin
                lut_right <= select_lut_value(lut_beat_reg, 3'd0);
                state     <= ST_INTERP_DELTA;
            end

            ST_INTERP_DELTA: begin
                // 正式 513 点 exp LUT 的最大相邻差为 66,071,126，恰需 26 位。
                // 对异常上传表执行 26 位饱和，避免重新引入宽乘法器。
                lut_delta <= |lut_delta_full[32:26] ?
                             26'h3ff_ffff : lut_delta_full[25:0];
                state <= ST_INTERP_PARTIAL;
            end

            ST_INTERP_PARTIAL: begin
                // 26x23 拆成四个独立 13x12/11 部分积，避免 APM 级联长路径。
                interp_partial00 <= lut_delta[12:0]  * lut_remainder[11:0];
                interp_partial01 <= lut_delta[12:0]  * lut_remainder[22:12];
                interp_partial10 <= lut_delta[25:13] * lut_remainder[11:0];
                interp_partial11 <= lut_delta[25:13] * lut_remainder[22:12];
                state <= ST_INTERP_PAIR_SUM;
            end

            ST_INTERP_PAIR_SUM: begin
                interp_pair_sum_low <=
                    {12'd0, interp_partial00} +
                    {1'b0, interp_partial01, 12'd0};
                interp_pair_sum_high <=
                    {11'd0, interp_partial10} +
                    {interp_partial11, 12'd0};
                state <= ST_INTERP_FINAL_SUM;
            end

            ST_INTERP_FINAL_SUM: begin
                interp_product <=
                    {12'd0, interp_pair_sum_low} +
                    {interp_pair_sum_high, 13'd0};
                state <= ST_INTERP_ROUND;
            end

            ST_INTERP_ROUND: begin
                pending_exp_q31 <= lut_left - interp_rounded;
                state <= ST_EXP_COMMIT;
            end

            ST_EXP_COMMIT: begin
                exp_values[token_index] <= pending_exp_q31;
                sum_exp_q31             <= sum_with_pending;
                if (token_index == 4'd15) begin
                    divider_denominator_q31 <= sum_with_pending;
                    state                   <= ST_DIV_START;
                end else begin
                    token_index <= token_index + 1'b1;
                    state       <= ST_EXP_PREP;
                end
            end

            ST_DIV_START: begin
                divider_start <= 1'b1;
                state         <= ST_DIV_WAIT;
            end

            ST_DIV_WAIT: begin
                if (divider_done) begin
                    reciprocal_q31 <= divider_result_q31;
                    state          <= ST_OUT_INIT;
                end
            end

            ST_OUT_INIT: begin
                token_index <= 4'd0;
                state       <= ST_OUT_SELECT;
            end

            ST_OUT_SELECT: begin
                selected_exp_q31       <= exp_values[token_index];
                selected_output_masked <= active_all_masked || !current_score_valid;
                state                  <= ST_OUT_MUL;
            end

            ST_OUT_MUL: begin
                if (selected_output_masked) begin
                    pending_probability_q31 <= 32'd0;
                    state                   <= ST_OUT_VALID;
                end else begin
                    norm_product <= selected_exp_q31 * reciprocal_q31;
                    state        <= ST_OUT_ROUND;
                end
            end

            ST_OUT_ROUND: begin
                pending_probability_q31 <= norm_probability_sat;
                state                   <= ST_OUT_VALID;
            end

            ST_OUT_VALID: begin
                probability_valid <= 1'b1;
                probability_head  <= active_head;
                probability_token <= token_index;
                probability_q31   <= pending_probability_q31;
                state             <= ST_OUT_WAIT;
            end

            ST_OUT_WAIT: begin
                if (probability_ready) begin
                    if (token_index == 4'd15) begin
                        if (active_head == 4'd13) begin
                            busy <= 1'b0;
                            done <= 1'b1;
                            state <= ST_IDLE;
                        end else begin
                            active_head <= active_head + 1'b1;
                            load_beat   <= 2'd0;
                            state       <= ST_SCORE_READ_REQ;
                        end
                    end else begin
                        token_index <= token_index + 1'b1;
                        state       <= ST_OUT_SELECT;
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

wire _unused_divider_busy = divider_busy;
wire _unused_interp_high = interp_rounded[26];

endmodule
