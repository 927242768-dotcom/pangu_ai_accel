`timescale 1ns/1ps

// 将 8 个字节中的 16 个二补码 INT4 权重解包为 16 个有符号 INT8。
// 字节内顺序：低 4 bit 为偶数下标权重，高 4 bit 为奇数下标权重。
module int4_unpack16 (
    input  wire [63:0]  packed_vec,
    output wire [127:0] unpacked_vec
);

genvar i;
generate
    for (i = 0; i < 16; i = i + 1) begin : gen_unpack
        wire [3:0] nibble = packed_vec[i*4 +: 4];
        assign unpacked_vec[i*8 +: 8] = {{4{nibble[3]}}, nibble};
    end
endgenerate

endmodule
