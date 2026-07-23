`timescale 1ns/1ps

module uart_tx #(
    parameter integer CLKS_PER_BIT = 434
)(
    input  wire       clk,
    input  wire       rst_n,
    input  wire [7:0] data,
    input  wire       start,
    output reg        tx,
    output reg        busy
);

reg [9:0] frame;
reg [3:0] bit_index;
reg [15:0] clk_count;

always @(posedge clk) begin
    if (!rst_n) begin
        tx        <= 1'b1;
        busy      <= 1'b0;
        frame     <= 10'h3ff;
        bit_index <= 4'd0;
        clk_count <= 16'd0;
    end else begin
        if (!busy) begin
            tx        <= 1'b1;
            bit_index <= 4'd0;
            clk_count <= 16'd0;

            if (start) begin
                frame <= {1'b1, data, 1'b0};
                tx    <= 1'b0;
                busy  <= 1'b1;
            end
        end else begin
            if (clk_count == CLKS_PER_BIT - 1) begin
                clk_count <= 16'd0;
                if (bit_index == 4'd9) begin
                    tx   <= 1'b1;
                    busy <= 1'b0;
                end else begin
                    bit_index <= bit_index + 1'b1;
                    tx <= frame[bit_index + 1'b1];
                end
            end else begin
                clk_count <= clk_count + 1'b1;
            end
        end
    end
end

endmodule
