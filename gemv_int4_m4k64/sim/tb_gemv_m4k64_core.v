`timescale 1ns/1ps

module tb_gemv_m4k64_core;

reg clk;
reg rst_n;
reg start;
reg [511:0] x_vec;
reg [1023:0] w_packed;
wire busy;
wire done;
wire [127:0] y_vec;

integer row;
integer index;
integer signed weight_value;

gemv_m4k64_core dut (
    .clk      (clk),
    .rst_n    (rst_n),
    .start    (start),
    .x_vec    (x_vec),
    .w_packed (w_packed),
    .busy     (busy),
    .done     (done),
    .y_vec    (y_vec)
);

always #5 clk = ~clk;

initial begin
    clk      = 1'b0;
    rst_n    = 1'b0;
    start    = 1'b0;
    x_vec    = 512'd0;
    w_packed = 1024'd0;

    for (index = 0; index < 64; index = index + 1)
        x_vec[index*8 +: 8] = index - 32;

    for (row = 0; row < 4; row = row + 1) begin
        for (index = 0; index < 64; index = index + 1) begin
            case (row)
                0: weight_value = (index % 16) - 8;
                1: weight_value = 7 - (index % 16);
                2: weight_value = ((index * 3) % 16) - 8;
                3: weight_value = (index % 2 == 0) ? -8 : 7;
                default: weight_value = 0;
            endcase
            w_packed[row*256 + index*4 +: 4] = weight_value[3:0];
        end
    end

    #30;
    rst_n = 1'b1;
    #20;
    start = 1'b1;
    #10;
    start = 1'b0;

    wait (done == 1'b1);
    #1;

    if ($signed(y_vec[31:0]) !== 32'sd1376) begin
        $display("FAIL row0: %0d", $signed(y_vec[31:0]));
        $finish;
    end
    if ($signed(y_vec[63:32]) !== -32'sd1344) begin
        $display("FAIL row1: %0d", $signed(y_vec[63:32]));
        $finish;
    end
    if ($signed(y_vec[95:64]) !== 32'sd416) begin
        $display("FAIL row2: %0d", $signed(y_vec[95:64]));
        $finish;
    end
    if ($signed(y_vec[127:96]) !== 32'sd256) begin
        $display("FAIL row3: %0d", $signed(y_vec[127:96]));
        $finish;
    end

    $display("PASS gemv_m4k64_core: [%0d, %0d, %0d, %0d]",
             $signed(y_vec[31:0]), $signed(y_vec[63:32]),
             $signed(y_vec[95:64]), $signed(y_vec[127:96]));
    $finish;
end

initial begin
    #5000;
    $display("FAIL timeout");
    $finish;
end

endmodule
