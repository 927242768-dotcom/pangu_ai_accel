`timescale 1ns/1ps

// 固定 K=896 的 tied Embedding 定点转换核心。
//
// 每个 Token 行槽由 16 个 256 bit 数据拍组成：
//   beat 0..13：每拍一个 64 元素 group 的 packed signed INT4 权重；
//   beat 14：group 0..7 的 UQ4.28 scale；
//   beat 15：group 8..13 的 UQ4.28 scale，末尾 8 B 为 0 padding。
//
// 每个元素执行：
//   signed INT4 * unsigned UQ4.28
//   -> RNE 右移 18 位
//   -> signed Q6.10 int16 显式饱和。
//
// 核心逐元素计算，每 16 个输出打包为一个 256 bit 数据拍，共输出 56 拍。
module embedding_k896_core #(
    parameter integer K           = 896,
    parameter integer GROUP_SIZE  = 64,
    parameter integer GROUPS      = K / GROUP_SIZE,
    parameter integer RESULT_BEATS= K / 16
)(
    input  wire                    clk,
    input  wire                    rst_n,

    input  wire                    row_load_en,
    input  wire [3:0]              row_load_index,
    input  wire [255:0]            row_load_data,

    input  wire                    start,
    output reg                     busy,
    output reg                     done,

    output reg  [255:0]            result_data,
    output reg                     result_valid,
    input  wire                    result_ready
);

localparam [2:0] ST_IDLE       = 3'd0;
localparam [2:0] ST_READ_GROUP = 3'd1;
localparam [2:0] ST_MULTIPLY   = 3'd2;
localparam [2:0] ST_ROUND      = 3'd3;
localparam [2:0] ST_SATURATE   = 3'd4;
localparam [2:0] ST_PACK       = 3'd5;
localparam [2:0] ST_WAIT       = 3'd6;

reg [255:0] weight_mem [0:GROUPS-1];
reg [31:0]  scale_mem  [0:GROUPS-1];

reg [2:0] state;
reg [3:0] group_index;
reg [5:0] lane_index;
reg [255:0] weight_beat_reg;
reg [255:0] output_pack;
reg signed [37:0] product_reg;
reg signed [37:0] rounded_reg;
reg [15:0] output_saturated_reg;

integer scale_lane;

wire [7:0] selected_packed_byte =
    weight_beat_reg[(lane_index[5:1] * 8) +: 8];
wire [3:0] selected_nibble =
    lane_index[0] ? selected_packed_byte[7:4] : selected_packed_byte[3:0];
wire signed [4:0] selected_weight =
    selected_nibble[3] ? {1'b1, selected_nibble} : {1'b0, selected_nibble};

// scale 必须按无符号正数参与乘法。前置 0 扩展为 signed 33 位，
// 可正确覆盖 bit31=1 和 0xffffffff 边界。
wire signed [32:0] selected_scale_positive = {1'b0, scale_mem[group_index]};
wire signed [37:0] product_full =
    $signed(selected_weight) * $signed(selected_scale_positive);

function signed [37:0] rne_shift18_signed38;
    input signed [37:0] value;
    reg [37:0] magnitude;
    reg [37:0] quotient;
    reg [17:0] remainder;
    begin
        magnitude = value[37] ? (~value + 1'b1) : value;
        quotient = magnitude >> 18;
        remainder = magnitude[17:0];
        if ((remainder > 18'h20000) ||
            ((remainder == 18'h20000) && quotient[0]))
            quotient = quotient + 1'b1;
        rne_shift18_signed38 = value[37] ? -$signed(quotient) : $signed(quotient);
    end
endfunction

function [15:0] saturate_signed16;
    input signed [37:0] value;
    begin
        if (value > 38'sd32767)
            saturate_signed16 = 16'h7fff;
        else if (value < -38'sd32768)
            saturate_signed16 = 16'h8000;
        else
            saturate_signed16 = value[15:0];
    end
endfunction

reg [255:0] output_pack_next;
always @(*) begin
    output_pack_next = output_pack;
    output_pack_next[lane_index[3:0]*16 +: 16] = output_saturated_reg;
end

// 缓存不复位；控制器只会在完整读取当前 Token 行后启动计算。
always @(posedge clk) begin
    if (row_load_en) begin
        if (row_load_index < GROUPS) begin
            weight_mem[row_load_index] <= row_load_data;
        end else if (row_load_index == 4'd14) begin
            for (scale_lane = 0; scale_lane < 8; scale_lane = scale_lane + 1)
                scale_mem[scale_lane] <= row_load_data[scale_lane*32 +: 32];
        end else if (row_load_index == 4'd15) begin
            for (scale_lane = 0; scale_lane < 6; scale_lane = scale_lane + 1)
                scale_mem[scale_lane + 8] <= row_load_data[scale_lane*32 +: 32];
        end
    end
end

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        state                <= ST_IDLE;
        group_index          <= 4'd0;
        lane_index           <= 6'd0;
        weight_beat_reg      <= 256'd0;
        output_pack          <= 256'd0;
        product_reg          <= 38'sd0;
        rounded_reg          <= 38'sd0;
        output_saturated_reg <= 16'd0;
        result_data          <= 256'd0;
        result_valid         <= 1'b0;
        busy                 <= 1'b0;
        done                 <= 1'b0;
    end else begin
        done <= 1'b0;

        case (state)
            ST_IDLE: begin
                busy         <= 1'b0;
                result_valid <= 1'b0;
                if (start) begin
                    group_index <= 4'd0;
                    lane_index  <= 6'd0;
                    output_pack <= 256'd0;
                    busy        <= 1'b1;
                    state       <= ST_READ_GROUP;
                end
            end

            ST_READ_GROUP: begin
                weight_beat_reg <= weight_mem[group_index];
                lane_index      <= 6'd0;
                output_pack     <= 256'd0;
                state           <= ST_MULTIPLY;
            end

            ST_MULTIPLY: begin
                product_reg <= product_full;
                state       <= ST_ROUND;
            end

            ST_ROUND: begin
                rounded_reg <= rne_shift18_signed38(product_reg);
                state       <= ST_SATURATE;
            end

            ST_SATURATE: begin
                output_saturated_reg <= saturate_signed16(rounded_reg);
                state                <= ST_PACK;
            end

            ST_PACK: begin
                output_pack <= output_pack_next;
                if (lane_index[3:0] == 4'd15) begin
                    result_data  <= output_pack_next;
                    result_valid <= 1'b1;
                    state        <= ST_WAIT;
                end else begin
                    lane_index <= lane_index + 1'b1;
                    state      <= ST_MULTIPLY;
                end
            end

            ST_WAIT: begin
                if (result_valid && result_ready) begin
                    result_valid <= 1'b0;
                    if (lane_index == GROUP_SIZE - 1) begin
                        if (group_index == GROUPS - 1) begin
                            busy  <= 1'b0;
                            done  <= 1'b1;
                            state <= ST_IDLE;
                        end else begin
                            group_index <= group_index + 1'b1;
                            lane_index  <= 6'd0;
                            output_pack <= 256'd0;
                            state       <= ST_READ_GROUP;
                        end
                    end else begin
                        lane_index  <= lane_index + 1'b1;
                        output_pack <= 256'd0;
                        state       <= ST_MULTIPLY;
                    end
                end
            end

            default: begin
                state        <= ST_IDLE;
                busy         <= 1'b0;
                result_valid <= 1'b0;
            end
        endcase
    end
end

endmodule
