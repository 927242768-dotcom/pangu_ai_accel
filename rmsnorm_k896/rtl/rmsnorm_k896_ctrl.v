`timescale 1ns/1ps

// 固定 K=896 RMSNorm 的 UART、DDR3 和计算调度控制器。
//
// UART 协议（115200 8N1）：
//   I -> "PANGU50K RMSNORM K896 V1\r\n"
//   S -> 'S' + flags + "\r\n"
//   L + 4608 B 固定载荷 -> 写入 DDR3，回复 "K\r\n"
//   G -> 计算并回复 'R' + 896 个 little-endian signed int16 Q6.10
//
// 上传载荷：
//   input_q6_10[896]        1792 B / 56 拍
//   gamma_q6_10[896]        1792 B / 56 拍
//   rsqrt_lut256_uq12_20    1024 B / 32 拍
module rmsnorm_k896_ctrl #(
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

    output wire [4:0]                   debug_state,
    output reg                          protocol_error,
    output reg                          loaded,
    output reg                          result_valid
);

localparam integer DATA_BEATS       = 56;
localparam integer LUT_BEATS        = 32;
localparam integer RESULT_BEATS     = 56;
localparam integer LOAD_TOTAL_BEATS = DATA_BEATS + DATA_BEATS + LUT_BEATS;

localparam [CTRL_ADDR_WIDTH-1:0] ADDR_INPUT  = 28'h0000000;
localparam [CTRL_ADDR_WIDTH-1:0] ADDR_GAMMA  = 28'h0001000;
localparam [CTRL_ADDR_WIDTH-1:0] ADDR_LUT    = 28'h0002000;
localparam [CTRL_ADDR_WIDTH-1:0] ADDR_RESULT = 28'h0003000;

localparam [4:0] ST_IDLE                = 5'd0;
localparam [4:0] ST_RECV_LOAD           = 5'd1;
localparam [4:0] ST_SETUP_LOAD_WRITE    = 5'd2;
localparam [4:0] ST_WRITE_LOAD          = 5'd3;
localparam [4:0] ST_SETUP_INPUT_READ    = 5'd4;
localparam [4:0] ST_READ_INPUT          = 5'd5;
localparam [4:0] ST_SETUP_GAMMA_READ    = 5'd6;
localparam [4:0] ST_READ_GAMMA          = 5'd7;
localparam [4:0] ST_SETUP_LUT_READ      = 5'd8;
localparam [4:0] ST_READ_LUT            = 5'd9;
localparam [4:0] ST_START_CORE          = 5'd10;
localparam [4:0] ST_WAIT_CORE_RESULT    = 5'd11;
localparam [4:0] ST_SETUP_RESULT_WRITE  = 5'd12;
localparam [4:0] ST_WRITE_RESULT        = 5'd13;
localparam [4:0] ST_SETUP_RESULT_READ   = 5'd14;
localparam [4:0] ST_READ_RESULT         = 5'd15;
localparam [4:0] ST_SEND_RESULT_PREFIX  = 5'd16;
localparam [4:0] ST_SEND_RESULT_BYTES   = 5'd17;
localparam [4:0] ST_SEND_INFO           = 5'd18;
localparam [4:0] ST_SEND_STATUS         = 5'd19;
localparam [4:0] ST_SEND_ACK            = 5'd20;
localparam [4:0] ST_SEND_ERROR          = 5'd21;

reg [4:0] state;
reg [5:0] tx_index;
reg [7:0] tx_data;
reg tx_start;
wire tx_busy;
wire [7:0] rx_data;
wire rx_valid;

reg [5:0] rx_byte_index;
reg [255:0] upload_beat;
reg [7:0] load_beat_index;

reg [5:0] read_base_beat;
reg [4:0] active_read_burst_beats;
reg [4:0] read_beat_index;

reg [5:0] result_write_beat_index;
reg [255:0] result_write_cache;
reg [5:0] result_read_beat_index;
reg [5:0] result_tx_byte_index;
reg [255:0] result_tx_cache;
reg result_prefix_sent;

reg core_start;
wire core_busy;
wire core_done;
wire [255:0] core_result_data;
wire core_result_valid;
wire core_result_ready = (state == ST_WAIT_CORE_RESULT);
wire [39:0] core_sum_squares;
wire [39:0] core_variance_q20;
wire [31:0] core_rsqrt_q20;

reg aw_seen;
reg w_seen;
reg ar_seen;
reg [7:0] status_snapshot;
reg [7:0] error_code;

wire aw_handshake = axi_awvalid && axi_awready;
wire ar_handshake = axi_arvalid && axi_arready;
wire write_data_handshake = axi_wready && (aw_seen || aw_handshake);
wire read_data_handshake = axi_rvalid && (ar_seen || ar_handshake);

wire [6:0] data_beats_remaining = DATA_BEATS - read_base_beat;
wire [4:0] next_data_burst_beats =
    (data_beats_remaining > 7'd16) ? 5'd16 : data_beats_remaining[4:0];
wire [6:0] lut_beats_remaining = LUT_BEATS - read_base_beat;
wire [4:0] next_lut_burst_beats =
    (lut_beats_remaining > 7'd16) ? 5'd16 : lut_beats_remaining[4:0];

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

rmsnorm_k896_core u_rmsnorm_k896_core (
    .clk               (core_clk),
    .rst_n             (core_rst_n),
    .input_load_en     ((state == ST_READ_INPUT) && read_data_handshake),
    .input_load_index  (read_base_beat + read_beat_index),
    .input_load_data   (axi_rdata),
    .gamma_load_en     ((state == ST_READ_GAMMA) && read_data_handshake),
    .gamma_load_index  (read_base_beat + read_beat_index),
    .gamma_load_data   (axi_rdata),
    .lut_load_en       ((state == ST_READ_LUT) && read_data_handshake),
    .lut_load_index    (read_base_beat[4:0] + read_beat_index),
    .lut_load_data     (axi_rdata),
    .start             (core_start),
    .busy              (core_busy),
    .done              (core_done),
    .result_data       (core_result_data),
    .result_valid      (core_result_valid),
    .result_ready      (core_result_ready),
    .debug_sum_squares (core_sum_squares),
    .debug_variance_q20(core_variance_q20),
    .debug_rsqrt_q20   (core_rsqrt_q20)
);

function [7:0] info_char;
    input [4:0] index;
    begin
        case (index)
            5'd0:  info_char = "P";
            5'd1:  info_char = "A";
            5'd2:  info_char = "N";
            5'd3:  info_char = "G";
            5'd4:  info_char = "U";
            5'd5:  info_char = "5";
            5'd6:  info_char = "0";
            5'd7:  info_char = "K";
            5'd8:  info_char = " ";
            5'd9:  info_char = "R";
            5'd10: info_char = "M";
            5'd11: info_char = "S";
            5'd12: info_char = "N";
            5'd13: info_char = "O";
            5'd14: info_char = "R";
            5'd15: info_char = "M";
            5'd16: info_char = " ";
            5'd17: info_char = "K";
            5'd18: info_char = "8";
            5'd19: info_char = "9";
            5'd20: info_char = "6";
            5'd21: info_char = " ";
            5'd22: info_char = "V";
            5'd23: info_char = "1";
            5'd24: info_char = 8'h0d;
            5'd25: info_char = 8'h0a;
            default: info_char = 8'h00;
        endcase
    end
endfunction

always @(posedge core_clk or negedge core_rst_n) begin
    if (!core_rst_n) begin
        state                    <= ST_IDLE;
        tx_index                 <= 6'd0;
        tx_data                  <= 8'h00;
        tx_start                 <= 1'b0;
        rx_byte_index            <= 6'd0;
        upload_beat              <= 256'd0;
        load_beat_index          <= 8'd0;
        read_base_beat           <= 6'd0;
        active_read_burst_beats  <= 5'd0;
        read_beat_index          <= 5'd0;
        result_write_beat_index  <= 6'd0;
        result_write_cache       <= 256'd0;
        result_read_beat_index   <= 6'd0;
        result_tx_byte_index     <= 6'd0;
        result_tx_cache          <= 256'd0;
        result_prefix_sent       <= 1'b0;
        core_start               <= 1'b0;
        axi_awaddr               <= {CTRL_ADDR_WIDTH{1'b0}};
        axi_awvalid              <= 1'b0;
        axi_wdata                <= 256'd0;
        axi_wstrb                <= 32'd0;
        axi_araddr               <= {CTRL_ADDR_WIDTH{1'b0}};
        axi_arlen                <= 4'd0;
        axi_arvalid              <= 1'b0;
        aw_seen                  <= 1'b0;
        w_seen                   <= 1'b0;
        ar_seen                  <= 1'b0;
        status_snapshot          <= 8'd0;
        error_code               <= 8'd0;
        protocol_error           <= 1'b0;
        loaded                   <= 1'b0;
        result_valid             <= 1'b0;
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
                        8'h49, 8'h69: state <= ST_SEND_INFO; // I / i

                        8'h53, 8'h73: begin // S / s
                            status_snapshot <= {
                                4'd0, core_busy, result_valid, loaded, ddr_init_done
                            };
                            state <= ST_SEND_STATUS;
                        end

                        8'h4c, 8'h6c: begin // L / l
                            if (!ddr_init_done) begin
                                protocol_error <= 1'b1;
                                error_code     <= 8'h02;
                                state          <= ST_SEND_ERROR;
                            end else begin
                                upload_beat     <= 256'd0;
                                rx_byte_index   <= 6'd0;
                                load_beat_index <= 8'd0;
                                loaded          <= 1'b0;
                                result_valid    <= 1'b0;
                                state           <= ST_RECV_LOAD;
                            end
                        end

                        8'h47, 8'h67: begin // G / g
                            if (!ddr_init_done) begin
                                protocol_error <= 1'b1;
                                error_code     <= 8'h02;
                                state          <= ST_SEND_ERROR;
                            end else if (!loaded) begin
                                protocol_error <= 1'b1;
                                error_code     <= 8'h04;
                                state          <= ST_SEND_ERROR;
                            end else begin
                                result_valid            <= 1'b0;
                                read_base_beat           <= 6'd0;
                                read_beat_index          <= 5'd0;
                                result_write_beat_index  <= 6'd0;
                                result_read_beat_index   <= 6'd0;
                                result_tx_byte_index     <= 6'd0;
                                result_prefix_sent       <= 1'b0;
                                state                    <= ST_SETUP_INPUT_READ;
                            end
                        end

                        default: begin
                            protocol_error <= 1'b1;
                            error_code     <= 8'h01;
                            state          <= ST_SEND_ERROR;
                        end
                    endcase
                end
            end

            ST_RECV_LOAD: begin
                if (rx_valid) begin
                    upload_beat[rx_byte_index*8 +: 8] <= rx_data;
                    if (rx_byte_index == 6'd31)
                        state <= ST_SETUP_LOAD_WRITE;
                    else
                        rx_byte_index <= rx_byte_index + 1'b1;
                end
            end

            ST_SETUP_LOAD_WRITE: begin
                if (load_beat_index < DATA_BEATS)
                    axi_awaddr <= ADDR_INPUT + (load_beat_index << 3);
                else if (load_beat_index < 2*DATA_BEATS)
                    axi_awaddr <= ADDR_GAMMA +
                        ((load_beat_index - DATA_BEATS) << 3);
                else
                    axi_awaddr <= ADDR_LUT +
                        ((load_beat_index - 2*DATA_BEATS) << 3);
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
                    if (load_beat_index + 1'b1 == LOAD_TOTAL_BEATS) begin
                        loaded <= 1'b1;
                        state  <= ST_SEND_ACK;
                    end else begin
                        load_beat_index <= load_beat_index + 1'b1;
                        rx_byte_index   <= 6'd0;
                        upload_beat     <= 256'd0;
                        state           <= ST_RECV_LOAD;
                    end
                end
            end

            ST_SETUP_INPUT_READ: begin
                axi_araddr              <= ADDR_INPUT + (read_base_beat << 3);
                axi_arlen               <= next_data_burst_beats - 1'b1;
                axi_arvalid             <= 1'b1;
                ar_seen                 <= 1'b0;
                read_beat_index         <= 5'd0;
                active_read_burst_beats <= next_data_burst_beats;
                state                   <= ST_READ_INPUT;
            end

            ST_READ_INPUT: begin
                if (ar_handshake) begin
                    axi_arvalid <= 1'b0;
                    ar_seen     <= 1'b1;
                end
                if (read_data_handshake) begin
                    if (read_beat_index + 1'b1 == active_read_burst_beats) begin
                        ar_seen <= 1'b0;
                        if (read_base_beat + active_read_burst_beats == DATA_BEATS) begin
                            read_base_beat <= 6'd0;
                            state <= ST_SETUP_GAMMA_READ;
                        end else begin
                            read_base_beat <= read_base_beat + active_read_burst_beats;
                            state <= ST_SETUP_INPUT_READ;
                        end
                    end else begin
                        read_beat_index <= read_beat_index + 1'b1;
                    end
                end
            end

            ST_SETUP_GAMMA_READ: begin
                axi_araddr              <= ADDR_GAMMA + (read_base_beat << 3);
                axi_arlen               <= next_data_burst_beats - 1'b1;
                axi_arvalid             <= 1'b1;
                ar_seen                 <= 1'b0;
                read_beat_index         <= 5'd0;
                active_read_burst_beats <= next_data_burst_beats;
                state                   <= ST_READ_GAMMA;
            end

            ST_READ_GAMMA: begin
                if (ar_handshake) begin
                    axi_arvalid <= 1'b0;
                    ar_seen     <= 1'b1;
                end
                if (read_data_handshake) begin
                    if (read_beat_index + 1'b1 == active_read_burst_beats) begin
                        ar_seen <= 1'b0;
                        if (read_base_beat + active_read_burst_beats == DATA_BEATS) begin
                            read_base_beat <= 6'd0;
                            state <= ST_SETUP_LUT_READ;
                        end else begin
                            read_base_beat <= read_base_beat + active_read_burst_beats;
                            state <= ST_SETUP_GAMMA_READ;
                        end
                    end else begin
                        read_beat_index <= read_beat_index + 1'b1;
                    end
                end
            end

            ST_SETUP_LUT_READ: begin
                axi_araddr              <= ADDR_LUT + (read_base_beat << 3);
                axi_arlen               <= next_lut_burst_beats - 1'b1;
                axi_arvalid             <= 1'b1;
                ar_seen                 <= 1'b0;
                read_beat_index         <= 5'd0;
                active_read_burst_beats <= next_lut_burst_beats;
                state                   <= ST_READ_LUT;
            end

            ST_READ_LUT: begin
                if (ar_handshake) begin
                    axi_arvalid <= 1'b0;
                    ar_seen     <= 1'b1;
                end
                if (read_data_handshake) begin
                    if (read_beat_index + 1'b1 == active_read_burst_beats) begin
                        ar_seen <= 1'b0;
                        if (read_base_beat + active_read_burst_beats == LUT_BEATS)
                            state <= ST_START_CORE;
                        else begin
                            read_base_beat <= read_base_beat + active_read_burst_beats;
                            state <= ST_SETUP_LUT_READ;
                        end
                    end else begin
                        read_beat_index <= read_beat_index + 1'b1;
                    end
                end
            end

            ST_START_CORE: begin
                core_start <= 1'b1;
                state      <= ST_WAIT_CORE_RESULT;
            end

            ST_WAIT_CORE_RESULT: begin
                if (core_result_valid) begin
                    result_write_cache <= core_result_data;
                    state <= ST_SETUP_RESULT_WRITE;
                end
            end

            ST_SETUP_RESULT_WRITE: begin
                axi_awaddr  <= ADDR_RESULT + (result_write_beat_index << 3);
                axi_awvalid <= 1'b1;
                axi_wdata   <= result_write_cache;
                axi_wstrb   <= 32'hffff_ffff;
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
                    axi_awvalid <= 1'b0;
                    aw_seen     <= 1'b0;
                    w_seen      <= 1'b0;
                    if (result_write_beat_index + 1'b1 == RESULT_BEATS) begin
                        result_valid            <= 1'b1;
                        result_read_beat_index  <= 6'd0;
                        result_tx_byte_index    <= 6'd0;
                        result_prefix_sent      <= 1'b0;
                        state                   <= ST_SETUP_RESULT_READ;
                    end else begin
                        result_write_beat_index <= result_write_beat_index + 1'b1;
                        state <= ST_WAIT_CORE_RESULT;
                    end
                end
            end

            ST_SETUP_RESULT_READ: begin
                axi_araddr      <= ADDR_RESULT + (result_read_beat_index << 3);
                axi_arlen       <= 4'd0;
                axi_arvalid     <= 1'b1;
                ar_seen         <= 1'b0;
                state           <= ST_READ_RESULT;
            end

            ST_READ_RESULT: begin
                if (ar_handshake) begin
                    axi_arvalid <= 1'b0;
                    ar_seen     <= 1'b1;
                end
                if (read_data_handshake) begin
                    result_tx_cache      <= axi_rdata;
                    result_tx_byte_index <= 6'd0;
                    ar_seen              <= 1'b0;
                    if (result_prefix_sent)
                        state <= ST_SEND_RESULT_BYTES;
                    else
                        state <= ST_SEND_RESULT_PREFIX;
                end
            end

            ST_SEND_RESULT_PREFIX: begin
                if (!tx_busy && !tx_start) begin
                    tx_data            <= "R";
                    tx_start           <= 1'b1;
                    result_prefix_sent <= 1'b1;
                    state              <= ST_SEND_RESULT_BYTES;
                end
            end

            ST_SEND_RESULT_BYTES: begin
                if (!tx_busy && !tx_start) begin
                    tx_data  <= result_tx_cache[result_tx_byte_index*8 +: 8];
                    tx_start <= 1'b1;
                    if (result_tx_byte_index == 6'd31) begin
                        if (result_read_beat_index + 1'b1 == RESULT_BEATS)
                            state <= ST_IDLE;
                        else begin
                            result_read_beat_index <= result_read_beat_index + 1'b1;
                            state <= ST_SETUP_RESULT_READ;
                        end
                    end else begin
                        result_tx_byte_index <= result_tx_byte_index + 1'b1;
                    end
                end
            end

            ST_SEND_INFO: begin
                if (!tx_busy && !tx_start) begin
                    if (tx_index < 6'd26) begin
                        tx_data  <= info_char(tx_index[4:0]);
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
                axi_awvalid    <= 1'b0;
                axi_arvalid    <= 1'b0;
                tx_index       <= 6'd0;
                state          <= ST_SEND_ERROR;
            end
        endcase
    end
end

wire _unused_core_debug = &{1'b0, core_done, core_sum_squares,
                            core_variance_q20, core_rsqrt_q20};

endmodule
