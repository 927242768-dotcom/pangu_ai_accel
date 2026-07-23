`timescale 1ns/1ps

// DDR3 + packed INT4 GEMV（M=4、K=64）验证控制器。
//
// UART 协议（115200 8N1）：
//   I -> "PANGU50K GEMV M4K64 V1\r\n"
//   S -> 'S' + 状态字节 + "\r\n"
//   M + 64B INT8激活 + 128B packed INT4权重 -> 写入 DDR3，回复 "K\r\n"
//   G -> 读取激活和权重，执行 GEMV，写回结果，回复 'R' + 4个little-endian int32
//
// DDR3 控制器地址单位为 32 bit；每个 256 bit 数据拍占 8 个地址单位：
//   0x00, 0x08 : 64 个 INT8 激活
//   0x10..0x28 : 4 行 packed INT4 权重，每行 32 字节
//   0x30       : 4 个 INT32 输出，位于低 128 bit
module gemv_m4k64_ctrl #(
    parameter integer CTRL_ADDR_WIDTH = 28,
    parameter integer CLKS_PER_BIT    = 868
)(
    input  wire                         core_clk,
    input  wire                         core_rst_n,
    input  wire                         ddr_init_done,

    input  wire                         uart_rx_i,
    output wire                         uart_tx_o,

    output reg  [CTRL_ADDR_WIDTH-1:0]   axi_awaddr,
    output wire                         axi_awuser_ap,
    output wire [3:0]                   axi_awuser_id,
    output wire [3:0]                   axi_awlen,
    input  wire                         axi_awready,
    output reg                          axi_awvalid,

    output reg  [255:0]                 axi_wdata,
    output reg  [31:0]                  axi_wstrb,
    input  wire                         axi_wready,

    output reg  [CTRL_ADDR_WIDTH-1:0]   axi_araddr,
    output wire                         axi_aruser_ap,
    output wire [3:0]                   axi_aruser_id,
    output reg  [3:0]                   axi_arlen,
    input  wire                         axi_arready,
    output reg                          axi_arvalid,

    input  wire [255:0]                 axi_rdata,
    input  wire                         axi_rvalid,

    output wire [5:0]                   debug_state,
    output reg                          protocol_error,
    output reg                          loaded,
    output reg                          result_valid
);

localparam [CTRL_ADDR_WIDTH-1:0] ADDR_ACT    = {CTRL_ADDR_WIDTH{1'b0}};
localparam [CTRL_ADDR_WIDTH-1:0] ADDR_WEIGHT = {{(CTRL_ADDR_WIDTH-5){1'b0}}, 5'h10};
localparam [CTRL_ADDR_WIDTH-1:0] ADDR_RESULT = {{(CTRL_ADDR_WIDTH-6){1'b0}}, 6'h30};

localparam [5:0] ST_IDLE               = 6'd0;
localparam [5:0] ST_RECV_LOAD          = 6'd1;
localparam [5:0] ST_SETUP_LOAD_WRITE   = 6'd2;
localparam [5:0] ST_WRITE_LOAD         = 6'd3;
localparam [5:0] ST_READ_ACT           = 6'd4;
localparam [5:0] ST_SETUP_WEIGHT_READ  = 6'd5;
localparam [5:0] ST_READ_WEIGHT        = 6'd6;
localparam [5:0] ST_START_CORE         = 6'd7;
localparam [5:0] ST_WAIT_CORE          = 6'd8;
localparam [5:0] ST_SETUP_RESULT_WRITE = 6'd9;
localparam [5:0] ST_WRITE_RESULT       = 6'd10;
localparam [5:0] ST_SEND_INFO          = 6'd11;
localparam [5:0] ST_SEND_STATUS        = 6'd12;
localparam [5:0] ST_SEND_ACK           = 6'd13;
localparam [5:0] ST_SEND_RESULT        = 6'd14;
localparam [5:0] ST_SEND_ERROR         = 6'd15;

reg [5:0] state;
reg [5:0] tx_index;
reg [7:0] tx_data;
reg       tx_start;
wire      tx_busy;
wire [7:0] rx_data;
wire       rx_valid;

reg [255:0] upload_beat;
reg [4:0]   rx_byte_index;
reg [2:0]   load_block_index;
reg [1:0]   read_beat_index;

reg [511:0]  activation_cache;
reg [1023:0] weight_cache;
reg [127:0]  result_cache;

reg core_start;
wire core_busy;
wire core_done;
wire [127:0] core_y_vec;

reg aw_seen;
reg w_seen;
reg ar_seen;
reg [7:0] status_snapshot;
reg [7:0] error_code;

wire aw_handshake = axi_awvalid && axi_awready;
wire ar_handshake = axi_arvalid && axi_arready;
wire write_data_handshake = axi_wready && (aw_seen || aw_handshake);
wire read_data_handshake  = axi_rvalid && (ar_seen || ar_handshake);

assign axi_awuser_ap = 1'b0;
assign axi_awuser_id = 4'h0;
assign axi_awlen     = 4'h0;
assign axi_aruser_ap = 1'b0;
assign axi_aruser_id = 4'h0;
assign debug_state   = state;

uart_rx #(
    .CLKS_PER_BIT(CLKS_PER_BIT)
) u_uart_rx (
    .clk   (core_clk),
    .rst_n (core_rst_n),
    .rx    (uart_rx_i),
    .data  (rx_data),
    .valid (rx_valid)
);

uart_tx #(
    .CLKS_PER_BIT(CLKS_PER_BIT)
) u_uart_tx (
    .clk   (core_clk),
    .rst_n (core_rst_n),
    .data  (tx_data),
    .start (tx_start),
    .tx    (uart_tx_o),
    .busy  (tx_busy)
);

gemv_m4k64_core u_gemv_core (
    .clk      (core_clk),
    .rst_n    (core_rst_n),
    .start    (core_start),
    .x_vec    (activation_cache),
    .w_packed (weight_cache),
    .busy     (core_busy),
    .done     (core_done),
    .y_vec    (core_y_vec)
);

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
            6'd9:  info_char = "G";
            6'd10: info_char = "E";
            6'd11: info_char = "M";
            6'd12: info_char = "V";
            6'd13: info_char = " ";
            6'd14: info_char = "M";
            6'd15: info_char = "4";
            6'd16: info_char = "K";
            6'd17: info_char = "6";
            6'd18: info_char = "4";
            6'd19: info_char = " ";
            6'd20: info_char = "V";
            6'd21: info_char = "1";
            6'd22: info_char = 8'h0d;
            6'd23: info_char = 8'h0a;
            default: info_char = 8'h00;
        endcase
    end
endfunction

always @(posedge core_clk or negedge core_rst_n) begin
    if (!core_rst_n) begin
        state             <= ST_IDLE;
        tx_index          <= 6'd0;
        tx_data           <= 8'h00;
        tx_start          <= 1'b0;
        upload_beat       <= 256'd0;
        rx_byte_index     <= 5'd0;
        load_block_index  <= 3'd0;
        read_beat_index   <= 2'd0;
        activation_cache  <= 512'd0;
        weight_cache      <= 1024'd0;
        result_cache      <= 128'd0;
        core_start        <= 1'b0;
        axi_awaddr        <= {CTRL_ADDR_WIDTH{1'b0}};
        axi_awvalid       <= 1'b0;
        axi_wdata         <= 256'd0;
        axi_wstrb         <= 32'd0;
        axi_araddr        <= {CTRL_ADDR_WIDTH{1'b0}};
        axi_arlen         <= 4'd0;
        axi_arvalid       <= 1'b0;
        aw_seen           <= 1'b0;
        w_seen            <= 1'b0;
        ar_seen           <= 1'b0;
        status_snapshot   <= 8'd0;
        error_code        <= 8'd0;
        protocol_error    <= 1'b0;
        loaded            <= 1'b0;
        result_valid      <= 1'b0;
    end else begin
        tx_start   <= 1'b0;
        core_start <= 1'b0;

        case (state)
            ST_IDLE: begin
                axi_awvalid <= 1'b0;
                axi_arvalid <= 1'b0;
                aw_seen     <= 1'b0;
                w_seen      <= 1'b0;
                ar_seen     <= 1'b0;
                tx_index    <= 6'd0;

                if (rx_valid) begin
                    case (rx_data)
                        8'h49, 8'h69: begin // I / i
                            tx_index <= 6'd0;
                            state    <= ST_SEND_INFO;
                        end

                        8'h53, 8'h73: begin // S / s
                            status_snapshot <= {4'd0, core_busy, result_valid, loaded, ddr_init_done};
                            tx_index        <= 6'd0;
                            state           <= ST_SEND_STATUS;
                        end

                        8'h4d, 8'h6d: begin // M / m：加载固定 M4K64 GEMV 数据
                            if (ddr_init_done) begin
                                upload_beat      <= 256'd0;
                                rx_byte_index    <= 5'd0;
                                load_block_index <= 3'd0;
                                loaded           <= 1'b0;
                                result_valid     <= 1'b0;
                                state            <= ST_RECV_LOAD;
                            end else begin
                                protocol_error <= 1'b1;
                                error_code     <= 8'h02;
                                tx_index       <= 6'd0;
                                state          <= ST_SEND_ERROR;
                            end
                        end

                        8'h47, 8'h67: begin // G / g：执行 GEMV
                            if (!ddr_init_done) begin
                                protocol_error <= 1'b1;
                                error_code     <= 8'h02;
                                tx_index       <= 6'd0;
                                state          <= ST_SEND_ERROR;
                            end else if (!loaded) begin
                                protocol_error <= 1'b1;
                                error_code     <= 8'h03;
                                tx_index       <= 6'd0;
                                state          <= ST_SEND_ERROR;
                            end else begin
                                result_valid   <= 1'b0;
                                activation_cache <= 512'd0;
                                weight_cache     <= 1024'd0;
                                axi_araddr      <= ADDR_ACT;
                                axi_arlen       <= 4'd1;
                                axi_arvalid     <= 1'b1;
                                ar_seen         <= 1'b0;
                                read_beat_index <= 2'd0;
                                state           <= ST_READ_ACT;
                            end
                        end

                        default: begin
                            protocol_error <= 1'b1;
                            error_code     <= 8'h01;
                            tx_index       <= 6'd0;
                            state          <= ST_SEND_ERROR;
                        end
                    endcase
                end
            end

            ST_RECV_LOAD: begin
                if (rx_valid) begin
                    upload_beat[rx_byte_index*8 +: 8] <= rx_data;
                    if (rx_byte_index == 5'd31) begin
                        state <= ST_SETUP_LOAD_WRITE;
                    end else begin
                        rx_byte_index <= rx_byte_index + 1'b1;
                    end
                end
            end

            ST_SETUP_LOAD_WRITE: begin
                axi_awaddr  <= {{(CTRL_ADDR_WIDTH-3){1'b0}}, load_block_index} << 3;
                axi_awvalid <= 1'b1;
                axi_wdata   <= upload_beat;
                axi_wstrb   <= 32'hffff_ffff;
                aw_seen     <= 1'b0;
                w_seen      <= 1'b0;
                state       <= ST_WRITE_LOAD;
            end

            ST_WRITE_LOAD: begin
                if (aw_handshake) begin
                    axi_awvalid <= 1'b0;
                    aw_seen     <= 1'b1;
                end
                if (write_data_handshake)
                    w_seen <= 1'b1;

                if ((aw_seen || aw_handshake) && (w_seen || write_data_handshake)) begin
                    axi_awvalid <= 1'b0;
                    aw_seen     <= 1'b0;
                    w_seen      <= 1'b0;

                    if (load_block_index == 3'd5) begin
                        loaded  <= 1'b1;
                        tx_index <= 6'd0;
                        state   <= ST_SEND_ACK;
                    end else begin
                        load_block_index <= load_block_index + 1'b1;
                        rx_byte_index    <= 5'd0;
                        upload_beat      <= 256'd0;
                        state            <= ST_RECV_LOAD;
                    end
                end
            end

            ST_READ_ACT: begin
                if (ar_handshake) begin
                    axi_arvalid <= 1'b0;
                    ar_seen     <= 1'b1;
                end

                if (read_data_handshake) begin
                    activation_cache[read_beat_index*256 +: 256] <= axi_rdata;
                    if (read_beat_index == 2'd1) begin
                        ar_seen           <= 1'b0;
                        read_beat_index   <= 2'd0;
                        state             <= ST_SETUP_WEIGHT_READ;
                    end else begin
                        read_beat_index <= read_beat_index + 1'b1;
                    end
                end
            end

            ST_SETUP_WEIGHT_READ: begin
                axi_araddr      <= ADDR_WEIGHT;
                axi_arlen       <= 4'd3;
                axi_arvalid     <= 1'b1;
                ar_seen         <= 1'b0;
                read_beat_index <= 2'd0;
                state           <= ST_READ_WEIGHT;
            end

            ST_READ_WEIGHT: begin
                if (ar_handshake) begin
                    axi_arvalid <= 1'b0;
                    ar_seen     <= 1'b1;
                end

                if (read_data_handshake) begin
                    weight_cache[read_beat_index*256 +: 256] <= axi_rdata;
                    if (read_beat_index == 2'd3) begin
                        ar_seen         <= 1'b0;
                        read_beat_index <= 2'd0;
                        state           <= ST_START_CORE;
                    end else begin
                        read_beat_index <= read_beat_index + 1'b1;
                    end
                end
            end

            ST_START_CORE: begin
                core_start <= 1'b1;
                state      <= ST_WAIT_CORE;
            end

            ST_WAIT_CORE: begin
                if (core_done) begin
                    result_cache <= core_y_vec;
                    state        <= ST_SETUP_RESULT_WRITE;
                end
            end

            ST_SETUP_RESULT_WRITE: begin
                axi_awaddr  <= ADDR_RESULT;
                axi_awvalid <= 1'b1;
                axi_wdata   <= {128'd0, result_cache};
                axi_wstrb   <= 32'h0000_ffff;
                aw_seen     <= 1'b0;
                w_seen      <= 1'b0;
                state       <= ST_WRITE_RESULT;
            end

            ST_WRITE_RESULT: begin
                if (aw_handshake) begin
                    axi_awvalid <= 1'b0;
                    aw_seen     <= 1'b1;
                end
                if (write_data_handshake)
                    w_seen <= 1'b1;

                if ((aw_seen || aw_handshake) && (w_seen || write_data_handshake)) begin
                    axi_awvalid  <= 1'b0;
                    aw_seen      <= 1'b0;
                    w_seen       <= 1'b0;
                    result_valid <= 1'b1;
                    tx_index     <= 6'd0;
                    state        <= ST_SEND_RESULT;
                end
            end

            ST_SEND_INFO: begin
                if (!tx_busy && !tx_start) begin
                    if (tx_index < 6'd24) begin
                        tx_data  <= info_char(tx_index);
                        tx_start <= 1'b1;
                        tx_index <= tx_index + 1'b1;
                    end else begin
                        state <= ST_IDLE;
                    end
                end
            end

            ST_SEND_STATUS: begin
                if (!tx_busy && !tx_start) begin
                    if (tx_index < 6'd4) begin
                        case (tx_index)
                            6'd0: tx_data <= "S";
                            6'd1: tx_data <= status_snapshot;
                            6'd2: tx_data <= 8'h0d;
                            6'd3: tx_data <= 8'h0a;
                            default: tx_data <= 8'h00;
                        endcase
                        tx_start <= 1'b1;
                        tx_index <= tx_index + 1'b1;
                    end else begin
                        state <= ST_IDLE;
                    end
                end
            end

            ST_SEND_ACK: begin
                if (!tx_busy && !tx_start) begin
                    if (tx_index < 6'd3) begin
                        case (tx_index)
                            6'd0: tx_data <= "K";
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

            ST_SEND_RESULT: begin
                if (!tx_busy && !tx_start) begin
                    if (tx_index < 6'd17) begin
                        if (tx_index == 6'd0)
                            tx_data <= "R";
                        else
                            tx_data <= result_cache[(tx_index-1'b1)*8 +: 8];
                        tx_start <= 1'b1;
                        tx_index <= tx_index + 1'b1;
                    end else begin
                        state <= ST_IDLE;
                    end
                end
            end

            ST_SEND_ERROR: begin
                if (!tx_busy && !tx_start) begin
                    if (tx_index < 6'd4) begin
                        case (tx_index)
                            6'd0: tx_data <= "E";
                            6'd1: tx_data <= error_code;
                            6'd2: tx_data <= 8'h0d;
                            6'd3: tx_data <= 8'h0a;
                            default: tx_data <= 8'h00;
                        endcase
                        tx_start <= 1'b1;
                        tx_index <= tx_index + 1'b1;
                    end else begin
                        state <= ST_IDLE;
                    end
                end
            end

            default: begin
                protocol_error <= 1'b1;
                error_code     <= 8'hff;
                tx_index       <= 6'd0;
                axi_awvalid    <= 1'b0;
                axi_arvalid    <= 1'b0;
                state          <= ST_SEND_ERROR;
            end
        endcase
    end
end

endmodule
