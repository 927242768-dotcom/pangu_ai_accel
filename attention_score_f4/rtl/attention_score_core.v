`timescale 1ns/1ps

// 64x64 有符号顺序乘法器。
//
// 两个操作数取绝对值后拆成 4x4 个 16 位 limb。16 个 16x16 部分积
// 逐项寄存并累加，最后恢复符号。优先保证数学精确和 100 MHz 时序。
module signed_mul64x64_seq16 #(
    // 完整 G2 Block 可在 APM 部分积寄存与 128 位累加器之间再插入
    // 一拍对齐寄存，切断 APM tco + 对齐选择 + 宽加法的后布线长路径。
    // 默认关闭，保持 F4 已验证工程的周期不变。
    parameter integer PIPELINE_ACCUM = 0,
    // 完整 G2 可把最终 128 位补码恢复拆成低/高 64 位两拍，
    // 避免 magnitude_accumulator 到 product 高位的整条进位链。
    parameter integer PIPELINE_SIGN_RESTORE = 0
)(
    input  wire                   clk,
    input  wire                   rst_n,
    input  wire                   start,
    input  wire signed [63:0]     a,
    input  wire signed [63:0]     b,
    output reg                    busy,
    output reg                    done,
    output reg signed [127:0]     product
);

localparam [2:0] ST_IDLE     = 3'd0;
localparam [2:0] ST_PREPARE  = 3'd1;
localparam [2:0] ST_CAPTURE  = 3'd2;
localparam [2:0] ST_ACCUM    = 3'd3;
localparam [2:0] ST_FINALIZE      = 3'd4;
localparam [2:0] ST_ALIGN         = 3'd5;
localparam [2:0] ST_FINALIZE_HIGH = 3'd6;

reg [2:0] state;
reg [3:0] step;
reg result_negative;
reg [63:0] magnitude_a;
reg [63:0] magnitude_b;
reg [15:0] limb_a_reg;
reg [15:0] limb_b_reg;
reg [31:0] partial_product_reg;
reg [127:0] aligned_partial_reg;
reg [127:0] magnitude_accumulator;
reg finalize_carry;

reg [15:0] selected_limb_a;
reg [15:0] selected_limb_b;
reg [2:0] limb_sum;
reg [127:0] aligned_partial;
wire [31:0] partial_product = limb_a_reg * limb_b_reg;
wire [127:0] accum_partial =
    (PIPELINE_ACCUM != 0) ? aligned_partial_reg : aligned_partial;
wire [127:0] next_magnitude = magnitude_accumulator + accum_partial;

always @(*) begin
    case (step[1:0])
        2'd0: selected_limb_a = magnitude_a[15:0];
        2'd1: selected_limb_a = magnitude_a[31:16];
        2'd2: selected_limb_a = magnitude_a[47:32];
        default: selected_limb_a = magnitude_a[63:48];
    endcase

    case (step[3:2])
        2'd0: selected_limb_b = magnitude_b[15:0];
        2'd1: selected_limb_b = magnitude_b[31:16];
        2'd2: selected_limb_b = magnitude_b[47:32];
        default: selected_limb_b = magnitude_b[63:48];
    endcase

    limb_sum = {1'b0, step[1:0]} + {1'b0, step[3:2]};
    case (limb_sum)
        3'd0: aligned_partial = {96'd0, partial_product_reg};
        3'd1: aligned_partial = {80'd0, partial_product_reg, 16'd0};
        3'd2: aligned_partial = {64'd0, partial_product_reg, 32'd0};
        3'd3: aligned_partial = {48'd0, partial_product_reg, 48'd0};
        3'd4: aligned_partial = {32'd0, partial_product_reg, 64'd0};
        3'd5: aligned_partial = {16'd0, partial_product_reg, 80'd0};
        default: aligned_partial = {partial_product_reg, 96'd0};
    endcase
end

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        state                 <= ST_IDLE;
        step                  <= 4'd0;
        result_negative       <= 1'b0;
        magnitude_a           <= 64'd0;
        magnitude_b           <= 64'd0;
        limb_a_reg            <= 16'd0;
        limb_b_reg            <= 16'd0;
        partial_product_reg   <= 32'd0;
        aligned_partial_reg   <= 128'd0;
        magnitude_accumulator <= 128'd0;
        finalize_carry        <= 1'b0;
        product               <= 128'sd0;
        busy                  <= 1'b0;
        done                  <= 1'b0;
    end else begin
        done <= 1'b0;
        case (state)
            ST_IDLE: begin
                busy <= 1'b0;
                if (start) begin
                    result_negative       <= a[63] ^ b[63];
                    magnitude_a           <= a[63] ? (~a + 1'b1) : a;
                    magnitude_b           <= b[63] ? (~b + 1'b1) : b;
                    magnitude_accumulator <= 128'd0;
                    step                  <= 4'd0;
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
                state <= (PIPELINE_ACCUM != 0) ? ST_ALIGN : ST_ACCUM;
            end
            ST_ALIGN: begin
                aligned_partial_reg <= aligned_partial;
                state               <= ST_ACCUM;
            end
            ST_ACCUM: begin
                magnitude_accumulator <= next_magnitude;
                if (step == 4'd15) begin
                    // 最后一个部分积先独立写入累加器，下一拍再恢复符号。
                    // 由此切断“APM 输出 -> 128 位累加 -> 128 位取负 -> product”长路径。
                    state <= ST_FINALIZE;
                end else begin
                    step  <= step + 1'b1;
                    state <= ST_PREPARE;
                end
            end
            ST_FINALIZE: begin
                if (PIPELINE_SIGN_RESTORE != 0) begin
                    if (result_negative) begin
                        {finalize_carry, product[63:0]} <=
                            {1'b0, ~magnitude_accumulator[63:0]} + 65'd1;
                    end else begin
                        product[63:0] <= magnitude_accumulator[63:0];
                        finalize_carry <= 1'b0;
                    end
                    state <= ST_FINALIZE_HIGH;
                end else begin
                    product <= result_negative ?
                        $signed(~magnitude_accumulator + 1'b1) :
                        $signed(magnitude_accumulator);
                    busy  <= 1'b0;
                    done  <= 1'b1;
                    state <= ST_IDLE;
                end
            end
            ST_FINALIZE_HIGH: begin
                product[127:64] <= result_negative ?
                    ((~magnitude_accumulator[127:64]) + finalize_carry) :
                    magnitude_accumulator[127:64];
                busy  <= 1'b0;
                done  <= 1'b1;
                state <= ST_IDLE;
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


// F4 单 token Attention Score 核心。
//
// Q 缓存和 K 缓存均按 256 bit beat 保存，并采用同步读结构，便于 PDS 推断
// DRM18K，而不是把 65536 bit 数据展开为触发器。
//
// Q : [14,64] signed int64 Q28，224 beats
// K : [2,64]  signed int64 Q28，32 beats
// GQA: q_head 0..6 -> kv_head 0；q_head 7..13 -> kv_head 1
// score = RNE((sum(Q*K)) / 2^31)，输出 signed int64 Q28
module attention_score_core #(
    parameter integer PIPELINE_MUL_ACCUM = 0,
    // 完整 G2 可把乘法器最终 128 位符号恢复拆成低/高 64 位两拍。
    // 默认关闭，保持独立 F4 已验证工程的周期不变。
    parameter integer PIPELINE_MUL_SIGN = 0,
    // 完整 G2 可把每维的 128 位 dot_accumulator + mul_product
    // 拆成低 64 位与高 64 位两拍，切断后布线宽进位链。
    // 默认关闭，保持独立 F4 已验证工程的周期不变。
    parameter integer PIPELINE_DOT_ACCUM = 0
)(
    input  wire                   clk,
    input  wire                   rst_n,

    input  wire                   q_beat_we,
    input  wire [7:0]             q_beat_index,
    input  wire [255:0]           q_beat_data,
    input  wire                   k_beat_we,
    input  wire [4:0]             k_beat_index,
    input  wire [255:0]           k_beat_data,

    input  wire                   start_token,
    input  wire                   token_masked,
    input  wire                   score_ready,
    output reg                    busy,
    output reg                    score_valid,
    output reg [3:0]              score_head,
    output reg signed [63:0]      score_q28,
    output reg                    token_done
);

localparam [3:0] ST_IDLE       = 4'd0;
localparam [3:0] ST_MASK_OUT   = 4'd1;
localparam [3:0] ST_MASK_WAIT  = 4'd2;
localparam [3:0] ST_READ_VALUE = 4'd3;
localparam [3:0] ST_MUL_START  = 4'd4;
localparam [3:0] ST_MUL_WAIT   = 4'd5;
localparam [3:0] ST_DOT_ABS    = 4'd6;
localparam [3:0] ST_DOT_ROUND  = 4'd7;
localparam [3:0] ST_DOT_OUT    = 4'd8;
localparam [3:0] ST_DOT_WAIT   = 4'd9;
localparam [3:0] ST_DOT_ACCUM_HI = 4'd10;

reg [255:0] q_mem [0:223];
reg [255:0] k_mem [0:31];
reg [255:0] q_beat_reg;
reg [255:0] k_beat_reg;
reg [3:0] state;
reg active_masked;
reg [3:0] q_head;
reg [5:0] dimension;
reg signed [127:0] dot_accumulator;
reg signed [127:0] dot_sum;
// G2 专用分段累加寄存：低 64 位先计算并保留进位，下一拍完成高 64 位。
reg [64:0] dot_low_sum_reg;
reg [63:0] dot_acc_high_reg;
reg [63:0] mul_high_reg;
reg dot_accum_last_reg;
reg dot_negative;
reg [127:0] dot_magnitude;
reg [97:0] rounded_magnitude;

reg signed [63:0] mul_a;
reg signed [63:0] mul_b;
reg mul_start;
wire mul_busy;
wire mul_done;
wire signed [127:0] mul_product;

wire [7:0] q_read_beat_index = ({4'd0, q_head} << 4) + {6'd0, dimension[5:2]};
wire [4:0] k_read_beat_index = (q_head < 4'd7) ?
                               {1'b0, dimension[5:2]} :
                               (5'd16 + {1'b0, dimension[5:2]});
reg signed [63:0] selected_q_value;
reg signed [63:0] selected_k_value;

wire [96:0] dot_quotient = dot_magnitude[127:31];
wire [30:0] dot_remainder = dot_magnitude[30:0];
wire dot_round_up =
    (dot_remainder > 31'h40000000) ||
    ((dot_remainder == 31'h40000000) && dot_quotient[0]);
wire [64:0] dot_high_sum_ext =
    {1'b0, dot_acc_high_reg} + {1'b0, mul_high_reg} + dot_low_sum_reg[64];
wire signed [127:0] dot_split_sum =
    $signed({dot_high_sum_ext[63:0], dot_low_sum_reg[63:0]});

always @(*) begin
    case (dimension[1:0])
        2'd0: begin
            selected_q_value = q_beat_reg[63:0];
            selected_k_value = k_beat_reg[63:0];
        end
        2'd1: begin
            selected_q_value = q_beat_reg[127:64];
            selected_k_value = k_beat_reg[127:64];
        end
        2'd2: begin
            selected_q_value = q_beat_reg[191:128];
            selected_k_value = k_beat_reg[191:128];
        end
        default: begin
            selected_q_value = q_beat_reg[255:192];
            selected_k_value = k_beat_reg[255:192];
        end
    endcase
end

function [63:0] signed_magnitude_sat64;
    input sign_value;
    input [97:0] magnitude;
    begin
        if (!sign_value) begin
            if (|magnitude[97:63])
                signed_magnitude_sat64 = 64'h7fff_ffff_ffff_ffff;
            else
                signed_magnitude_sat64 = magnitude[63:0];
        end else begin
            if (|magnitude[97:64])
                signed_magnitude_sat64 = 64'h8000_0000_0000_0000;
            else if (magnitude[63])
                signed_magnitude_sat64 = 64'h8000_0000_0000_0000;
            else
                signed_magnitude_sat64 = (~magnitude[63:0]) + 1'b1;
        end
    end
endfunction

signed_mul64x64_seq16 #(
    .PIPELINE_ACCUM        (PIPELINE_MUL_ACCUM),
    .PIPELINE_SIGN_RESTORE (PIPELINE_MUL_SIGN)
) u_signed_mul64x64_seq16 (
    .clk     (clk),
    .rst_n   (rst_n),
    .start   (mul_start),
    .a       (mul_a),
    .b       (mul_b),
    .busy    (mul_busy),
    .done    (mul_done),
    .product (mul_product)
);

// 缓存不复位；控制器只会读取本次已加载的有效 beat。
// 独立无复位写端口是 PDS 推断 DRM/LUTRAM 的关键结构。
always @(posedge clk) begin
    if (q_beat_we)
        q_mem[q_beat_index] <= q_beat_data;
    if (k_beat_we)
        k_mem[k_beat_index] <= k_beat_data;
end

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        q_beat_reg       <= 256'd0;
        k_beat_reg       <= 256'd0;
        state            <= ST_IDLE;
        active_masked    <= 1'b0;
        q_head           <= 4'd0;
        dimension        <= 6'd0;
        dot_accumulator  <= 128'sd0;
        dot_sum          <= 128'sd0;
        dot_low_sum_reg  <= 65'd0;
        dot_acc_high_reg <= 64'd0;
        mul_high_reg     <= 64'd0;
        dot_accum_last_reg <= 1'b0;
        dot_negative     <= 1'b0;
        dot_magnitude    <= 128'd0;
        rounded_magnitude <= 98'd0;
        mul_a            <= 64'sd0;
        mul_b            <= 64'sd0;
        mul_start        <= 1'b0;
        busy             <= 1'b0;
        score_valid      <= 1'b0;
        score_head       <= 4'd0;
        score_q28        <= 64'sd0;
        token_done       <= 1'b0;
    end else begin
        mul_start   <= 1'b0;
        score_valid <= 1'b0;
        token_done  <= 1'b0;

        case (state)
            ST_IDLE: begin
                busy <= 1'b0;
                if (start_token) begin
                    active_masked   <= token_masked;
                    q_head          <= 4'd0;
                    dimension       <= 6'd0;
                    dot_accumulator <= 128'sd0;
                    busy            <= 1'b1;
                    state           <= token_masked ? ST_MASK_OUT : ST_READ_VALUE;
                end
            end

            ST_MASK_OUT: begin
                score_valid <= 1'b1;
                score_head  <= q_head;
                score_q28   <= 64'h8000_0000_0000_0000;
                state       <= ST_MASK_WAIT;
            end

            ST_MASK_WAIT: begin
                if (score_ready) begin
                    if (q_head == 4'd13) begin
                        token_done <= 1'b1;
                        busy       <= 1'b0;
                        state      <= ST_IDLE;
                    end else begin
                        q_head <= q_head + 1'b1;
                        state  <= ST_MASK_OUT;
                    end
                end
            end

            ST_READ_VALUE: begin
                q_beat_reg <= q_mem[q_read_beat_index];
                k_beat_reg <= k_mem[k_read_beat_index];
                state      <= ST_MUL_START;
            end

            ST_MUL_START: begin
                mul_a     <= selected_q_value;
                mul_b     <= selected_k_value;
                mul_start <= 1'b1;
                state     <= ST_MUL_WAIT;
            end

            ST_MUL_WAIT: begin
                if (mul_done) begin
                    if (PIPELINE_DOT_ACCUM != 0) begin
                        // 二补码 128 位加法按低/高 64 位拆分，逐位结果与原表达式一致。
                        dot_low_sum_reg    <= {1'b0, dot_accumulator[63:0]} +
                                              {1'b0, mul_product[63:0]};
                        dot_acc_high_reg   <= dot_accumulator[127:64];
                        mul_high_reg       <= mul_product[127:64];
                        dot_accum_last_reg <= (dimension == 6'd63);
                        state              <= ST_DOT_ACCUM_HI;
                    end else if (dimension == 6'd63) begin
                        dot_sum <= dot_accumulator + mul_product;
                        state   <= ST_DOT_ABS;
                    end else begin
                        dot_accumulator <= dot_accumulator + mul_product;
                        dimension       <= dimension + 1'b1;
                        state           <= ST_READ_VALUE;
                    end
                end
            end

            ST_DOT_ACCUM_HI: begin
                if (dot_accum_last_reg) begin
                    dot_sum <= dot_split_sum;
                    state   <= ST_DOT_ABS;
                end else begin
                    dot_accumulator <= dot_split_sum;
                    dimension       <= dimension + 1'b1;
                    state           <= ST_READ_VALUE;
                end
            end

            ST_DOT_ABS: begin
                dot_negative  <= dot_sum[127];
                dot_magnitude <= dot_sum[127] ? (~dot_sum + 1'b1) : dot_sum;
                state         <= ST_DOT_ROUND;
            end

            ST_DOT_ROUND: begin
                rounded_magnitude <= {1'b0, dot_quotient} + dot_round_up;
                state             <= ST_DOT_OUT;
            end

            ST_DOT_OUT: begin
                score_valid <= 1'b1;
                score_head  <= q_head;
                score_q28   <= signed_magnitude_sat64(
                    dot_negative, rounded_magnitude
                );
                state <= ST_DOT_WAIT;
            end

            ST_DOT_WAIT: begin
                if (score_ready) begin
                    if (q_head == 4'd13) begin
                        token_done <= 1'b1;
                        busy       <= 1'b0;
                        state      <= ST_IDLE;
                    end else begin
                        q_head          <= q_head + 1'b1;
                        dimension       <= 6'd0;
                        dot_accumulator <= 128'sd0;
                        state           <= active_masked ? ST_MASK_OUT : ST_READ_VALUE;
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
