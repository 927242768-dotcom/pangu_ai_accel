`timescale 1ns/1ps

module top (
    input  wire       clk_50m,
    input  wire       uart_rx_i,
    output wire       uart_tx_o,
    output wire [7:0] led
);

localparam integer CLKS_PER_BIT = 434; // 50 MHz / 115200 baud

localparam [2:0] ST_IDLE        = 3'd0;
localparam [2:0] ST_RECV_DOT    = 3'd1;
localparam [2:0] ST_RUN_DOT     = 3'd2;
localparam [2:0] ST_RUN_TEST    = 3'd3;
localparam [2:0] ST_SEND_INFO   = 3'd4;
localparam [2:0] ST_SEND_TEST   = 3'd5;
localparam [2:0] ST_SEND_RESULT = 3'd6;
localparam [2:0] ST_SEND_ERROR  = 3'd7;

// 字节 0 位于向量最低 8 位。
localparam [127:0] SELF_A = {
    8'h10, 8'h0f, 8'h0e, 8'h0d,
    8'h0c, 8'h0b, 8'h0a, 8'h09,
    8'h08, 8'h07, 8'h06, 8'h05,
    8'h04, 8'h03, 8'h02, 8'h01
};

localparam [127:0] SELF_B = {
    8'h07, 8'h06, 8'h05, 8'h04,
    8'h03, 8'h02, 8'h01, 8'h00,
    8'hff, 8'hfe, 8'hfd, 8'hfc,
    8'hfb, 8'hfa, 8'hf9, 8'hf8
};

reg [15:0] power_on_count = 16'd0;
wire rst_n = &power_on_count;

always @(posedge clk_50m) begin
    if (!rst_n)
        power_on_count <= power_on_count + 1'b1;
end

wire [7:0] rx_data;
wire       rx_valid;
reg  [7:0] tx_data;
reg        tx_start;
wire       tx_busy;

uart_rx #(
    .CLKS_PER_BIT(CLKS_PER_BIT)
) u_uart_rx (
    .clk   (clk_50m),
    .rst_n (rst_n),
    .rx    (uart_rx_i),
    .data  (rx_data),
    .valid (rx_valid)
);

uart_tx #(
    .CLKS_PER_BIT(CLKS_PER_BIT)
) u_uart_tx (
    .clk   (clk_50m),
    .rst_n (rst_n),
    .data  (tx_data),
    .start (tx_start),
    .tx    (uart_tx_o),
    .busy  (tx_busy)
);

reg [127:0] a_vec;
reg [127:0] b_vec;
wire signed [31:0] dot_result;
reg  signed [31:0] result_reg;

int8_dot16 u_int8_dot16 (
    .a_vec  (a_vec),
    .b_vec  (b_vec),
    .result (dot_result)
);

reg [2:0] state;
reg [5:0] rx_count;
reg [5:0] tx_index;
reg       selftest_pass;
reg       protocol_error;
reg [25:0] heartbeat_count;
reg        heartbeat;
reg [21:0] rx_activity_count;

function [7:0] info_char;
    input [5:0] index;
    begin
        case (index)
            6'd0:  info_char = "P";
            6'd1:  info_char = "A";
            6'd2:  info_char = "N";
            6'd3:  info_char = "G";
            6'd4:  info_char = "U";
            6'd5:  info_char = "5";
            6'd6:  info_char = "0";
            6'd7:  info_char = "K";
            6'd8:  info_char = " ";
            6'd9:  info_char = "A";
            6'd10: info_char = "I";
            6'd11: info_char = " ";
            6'd12: info_char = "I";
            6'd13: info_char = "N";
            6'd14: info_char = "T";
            6'd15: info_char = "8";
            6'd16: info_char = " ";
            6'd17: info_char = "M";
            6'd18: info_char = "A";
            6'd19: info_char = "C";
            6'd20: info_char = "1";
            6'd21: info_char = "6";
            6'd22: info_char = " ";
            6'd23: info_char = "V";
            6'd24: info_char = "1";
            6'd25: info_char = 8'h0d;
            6'd26: info_char = 8'h0a;
            default: info_char = 8'h00;
        endcase
    end
endfunction

function [7:0] test_char;
    input [5:0] index;
    input       pass;
    begin
        case (index)
            6'd0: test_char = pass ? "P" : "F";
            6'd1: test_char = pass ? "A" : "A";
            6'd2: test_char = pass ? "S" : "I";
            6'd3: test_char = pass ? "S" : "L";
            6'd4: test_char = 8'h0d;
            6'd5: test_char = 8'h0a;
            default: test_char = 8'h00;
        endcase
    end
endfunction

always @(posedge clk_50m) begin
    if (!rst_n) begin
        state             <= ST_IDLE;
        rx_count          <= 6'd0;
        tx_index          <= 6'd0;
        tx_data           <= 8'h00;
        tx_start          <= 1'b0;
        a_vec             <= 128'd0;
        b_vec             <= 128'd0;
        result_reg        <= 32'sd0;
        selftest_pass     <= 1'b0;
        protocol_error    <= 1'b0;
        heartbeat_count   <= 26'd0;
        heartbeat         <= 1'b0;
        rx_activity_count <= 22'd0;
    end else begin
        tx_start <= 1'b0;

        if (heartbeat_count == 26'd24_999_999) begin
            heartbeat_count <= 26'd0;
            heartbeat <= ~heartbeat;
        end else begin
            heartbeat_count <= heartbeat_count + 1'b1;
        end

        if (rx_valid)
            rx_activity_count <= 22'h3f_ffff;
        else if (rx_activity_count != 0)
            rx_activity_count <= rx_activity_count - 1'b1;

        case (state)
            ST_IDLE: begin
                rx_count <= 6'd0;
                tx_index <= 6'd0;

                if (rx_valid) begin
                    case (rx_data)
                        8'h49, 8'h69: begin // I / i: board information
                            state <= ST_SEND_INFO;
                            tx_index <= 6'd0;
                        end

                        8'h54, 8'h74: begin // T / t: fixed-vector self-test
                            a_vec <= SELF_A;
                            b_vec <= SELF_B;
                            state <= ST_RUN_TEST;
                        end

                        8'h44, 8'h64: begin // D / d: 16 A bytes + 16 B bytes
                            a_vec <= 128'd0;
                            b_vec <= 128'd0;
                            rx_count <= 6'd0;
                            state <= ST_RECV_DOT;
                        end

                        default: begin
                            protocol_error <= 1'b1;
                            tx_index <= 6'd0;
                            state <= ST_SEND_ERROR;
                        end
                    endcase
                end
            end

            ST_RECV_DOT: begin
                if (rx_valid) begin
                    if (rx_count < 6'd16)
                        a_vec <= {rx_data, a_vec[127:8]};
                    else
                        b_vec <= {rx_data, b_vec[127:8]};

                    if (rx_count == 6'd31) begin
                        rx_count <= 6'd0;
                        state <= ST_RUN_DOT;
                    end else begin
                        rx_count <= rx_count + 1'b1;
                    end
                end
            end

            ST_RUN_DOT: begin
                result_reg <= dot_result;
                tx_index <= 6'd0;
                state <= ST_SEND_RESULT;
            end

            ST_RUN_TEST: begin
                selftest_pass <= (dot_result == 32'sd272);
                if (dot_result != 32'sd272)
                    protocol_error <= 1'b1;
                tx_index <= 6'd0;
                state <= ST_SEND_TEST;
            end

            ST_SEND_INFO: begin
                if (!tx_busy && !tx_start) begin
                    if (tx_index < 6'd27) begin
                        tx_data <= info_char(tx_index);
                        tx_start <= 1'b1;
                        tx_index <= tx_index + 1'b1;
                    end else begin
                        state <= ST_IDLE;
                    end
                end
            end

            ST_SEND_TEST: begin
                if (!tx_busy && !tx_start) begin
                    if (tx_index < 6'd6) begin
                        tx_data <= test_char(tx_index, selftest_pass);
                        tx_start <= 1'b1;
                        tx_index <= tx_index + 1'b1;
                    end else begin
                        state <= ST_IDLE;
                    end
                end
            end

            ST_SEND_RESULT: begin
                if (!tx_busy && !tx_start) begin
                    if (tx_index < 6'd5) begin
                        case (tx_index)
                            6'd0: tx_data <= 8'h52; // 'R'
                            6'd1: tx_data <= result_reg[7:0];
                            6'd2: tx_data <= result_reg[15:8];
                            6'd3: tx_data <= result_reg[23:16];
                            6'd4: tx_data <= result_reg[31:24];
                            default: tx_data <= 8'h00;
                        endcase
                        tx_start <= 1'b1;
                        tx_index <= tx_index + 1'b1;
                    end else begin
                        state <= ST_IDLE;
                    end
                end
            end

            ST_SEND_ERROR: begin
                if (!tx_busy && !tx_start) begin
                    if (tx_index < 6'd3) begin
                        case (tx_index)
                            6'd0: tx_data <= "?";
                            6'd1: tx_data <= 8'h0d;
                            6'd2: tx_data <= 8'h0a;
                            default: tx_data <= 8'h00;
                        endcase
                        tx_start <= 1'b1;
                        tx_index <= tx_index + 1'b1;
                    end else begin
                        state <= ST_IDLE;
                    end
                end
            end

            default: state <= ST_IDLE;
        endcase
    end
end

assign led[0] = heartbeat;
assign led[1] = (rx_activity_count != 0);
assign led[2] = (state != ST_IDLE);
assign led[3] = selftest_pass;
assign led[4] = protocol_error;
assign led[7:5] = state;

endmodule
