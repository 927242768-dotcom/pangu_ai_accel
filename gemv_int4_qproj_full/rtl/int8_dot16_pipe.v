`timescale 1ns/1ps

// 16 路 signed INT8 点积的显式平衡流水实现。
// 五级寄存将 16 个乘法、两两求和和最终归约分开，避免 MAC16 输出捕获
// 成为 100 MHz 慢角关键路径。
module int8_dot16_pipe (
    input  wire                 clk,
    input  wire                 rst_n,
    input  wire                 in_valid,
    input  wire [127:0]         a_vec,
    input  wire [127:0]         b_vec,
    output reg                  out_valid,
    output reg signed [31:0]    result
);

reg signed [15:0] product_reg [0:15];
reg signed [16:0] sum1_reg [0:7];
reg signed [17:0] sum2_reg [0:3];
reg signed [18:0] sum3_reg [0:1];
reg valid_product;
reg valid_sum1;
reg valid_sum2;
reg valid_sum3;

integer lane;
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        valid_product <= 1'b0;
        valid_sum1    <= 1'b0;
        valid_sum2    <= 1'b0;
        valid_sum3    <= 1'b0;
        out_valid     <= 1'b0;
        result        <= 32'sd0;
        for (lane = 0; lane < 16; lane = lane + 1)
            product_reg[lane] <= 16'sd0;
        for (lane = 0; lane < 8; lane = lane + 1)
            sum1_reg[lane] <= 17'sd0;
        for (lane = 0; lane < 4; lane = lane + 1)
            sum2_reg[lane] <= 18'sd0;
        for (lane = 0; lane < 2; lane = lane + 1)
            sum3_reg[lane] <= 19'sd0;
    end else begin
        valid_product <= in_valid;
        valid_sum1    <= valid_product;
        valid_sum2    <= valid_sum1;
        valid_sum3    <= valid_sum2;
        out_valid     <= valid_sum3;

        if (in_valid) begin
            for (lane = 0; lane < 16; lane = lane + 1) begin
                product_reg[lane] <=
                    $signed(a_vec[lane*8 +: 8]) *
                    $signed(b_vec[lane*8 +: 8]);
            end
        end

        if (valid_product) begin
            for (lane = 0; lane < 8; lane = lane + 1)
                sum1_reg[lane] <=
                    product_reg[lane*2] + product_reg[lane*2+1];
        end

        if (valid_sum1) begin
            for (lane = 0; lane < 4; lane = lane + 1)
                sum2_reg[lane] <= sum1_reg[lane*2] + sum1_reg[lane*2+1];
        end

        if (valid_sum2) begin
            sum3_reg[0] <= sum2_reg[0] + sum2_reg[1];
            sum3_reg[1] <= sum2_reg[2] + sum2_reg[3];
        end

        if (valid_sum3)
            result <= $signed(sum3_reg[0]) + $signed(sum3_reg[1]);
    end
end

endmodule
