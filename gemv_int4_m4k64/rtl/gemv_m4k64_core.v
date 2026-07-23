`timescale 1ns/1ps

// 固定尺寸 packed INT4 GEMV 计算核心：
//   W: 4 x 64，有符号 INT4，按行存储，每两个权重打包为 1 字节
//   x: 64 维有符号 INT8
//   y: 4 维有符号 INT32
//
// 每行拆成 4 个 MAC16 分块，跨分块进行 INT32 累加。
module gemv_m4k64_core (
    input  wire          clk,
    input  wire          rst_n,
    input  wire          start,
    input  wire [511:0]  x_vec,
    input  wire [1023:0] w_packed,
    output reg           busy,
    output reg           done,
    output reg  [127:0]  y_vec
);

localparam [1:0] ST_IDLE       = 2'd0;
localparam [1:0] ST_PREPARE    = 2'd1;
localparam [1:0] ST_CAPTURE_MAC = 2'd2;
localparam [1:0] ST_ACCUMULATE = 2'd3;

reg [1:0] state;
reg [1:0] row_index;
reg [1:0] block_index;
reg [127:0] mac_x_reg;
reg [127:0] mac_w_reg;
reg signed [31:0] dot_result_reg;
reg signed [31:0] row_accumulator;

wire [127:0] selected_x_block;
wire [63:0]  selected_w_block;
wire [127:0] unpacked_w_block;
wire signed [31:0] dot_result;
wire signed [31:0] accumulated_result;

assign selected_x_block = x_vec[block_index*128 +: 128];
assign selected_w_block = w_packed[row_index*256 + block_index*64 +: 64];
assign accumulated_result = row_accumulator + dot_result_reg;

int4_unpack16 u_int4_unpack16 (
    .packed_vec   (selected_w_block),
    .unpacked_vec (unpacked_w_block)
);

int8_dot16 u_int8_dot16 (
    .a_vec  (mac_x_reg),
    .b_vec  (mac_w_reg),
    .result (dot_result)
);

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        state           <= ST_IDLE;
        row_index       <= 2'd0;
        block_index     <= 2'd0;
        mac_x_reg       <= 128'd0;
        mac_w_reg       <= 128'd0;
        dot_result_reg  <= 32'sd0;
        row_accumulator <= 32'sd0;
        busy            <= 1'b0;
        done            <= 1'b0;
        y_vec           <= 128'd0;
    end else begin
        done <= 1'b0;

        case (state)
            ST_IDLE: begin
                if (start) begin
                    row_index       <= 2'd0;
                    block_index     <= 2'd0;
                    row_accumulator <= 32'sd0;
                    y_vec           <= 128'd0;
                    busy            <= 1'b1;
                    state           <= ST_PREPARE;
                end
            end

            ST_PREPARE: begin
                // 流水级 1：选择当前行/分块并完成 INT4 解包。
                mac_x_reg <= selected_x_block;
                mac_w_reg <= unpacked_w_block;
                state     <= ST_CAPTURE_MAC;
            end

            ST_CAPTURE_MAC: begin
                // 流水级 2：只寄存 MAC16 结果，切断乘加树与跨分块累加器的长路径。
                dot_result_reg <= dot_result;
                state          <= ST_ACCUMULATE;
            end

            ST_ACCUMULATE: begin
                // 流水级 3：使用已寄存的 MAC16 结果完成跨分块 INT32 累加。
                if (block_index == 2'd3) begin
                    y_vec[row_index*32 +: 32] <= accumulated_result;
                    row_accumulator           <= 32'sd0;
                    block_index               <= 2'd0;

                    if (row_index == 2'd3) begin
                        busy  <= 1'b0;
                        done  <= 1'b1;
                        state <= ST_IDLE;
                    end else begin
                        row_index <= row_index + 1'b1;
                        state     <= ST_PREPARE;
                    end
                end else begin
                    row_accumulator <= accumulated_result;
                    block_index     <= block_index + 1'b1;
                    state           <= ST_PREPARE;
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
