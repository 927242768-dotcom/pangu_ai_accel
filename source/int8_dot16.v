`timescale 1ns/1ps

// 16 路有符号 INT8 点积：sum(a[i] * b[i])。
// 这是 Transformer/MLP 中矩阵乘法的基础计算单元。
module int8_dot16 (
    input  wire [127:0] a_vec,
    input  wire [127:0] b_vec,
    output reg  signed [31:0] result
);

integer i;
reg signed [7:0] a_item;
reg signed [7:0] b_item;
reg signed [15:0] product;
reg signed [31:0] accumulator;

always @(*) begin
    accumulator = 32'sd0;
    a_item = 8'sd0;
    b_item = 8'sd0;
    product = 16'sd0;

    for (i = 0; i < 16; i = i + 1) begin
        a_item = $signed(a_vec[i*8 +: 8]);
        b_item = $signed(b_vec[i*8 +: 8]);
        product = a_item * b_item;
        accumulator = accumulator + product;
    end

    result = accumulator;
end

endmodule
