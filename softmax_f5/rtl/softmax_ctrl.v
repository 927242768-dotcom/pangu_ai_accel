`timescale 1ns/1ps

// F5 Softmax 独立验证控制器。
//
// DDR3 低端临时区：
//   scores        : ctrl 0x0000800，1792 B / 56 beats，[14,16] int64 Q28
//   probabilities : ctrl 0x0000A00， 896 B / 28 beats，[14,16] uint32 UQ1.31
//   exp LUT       : ctrl 0x0000B00，2080 B / 65 beats，513 个 uint32 + 28 B padding
//
// UART 协议（115200 8N1）：
//   I -> "PANGU50K SOFTMAX F5 V1\r\n"
//   S -> 'S' + flags + CRLF
//   L + 1792 B scores -> "K\r\n"
//   T + 2080 B exp LUT -> "K\r\n"
//   G -> 读取 scores/LUT，执行 14 heads Softmax，写回概率，回复 "K\r\n"
//   R -> 'D' + 896 B probabilities
//
// flags：bit0 DDR ready，bit1 scores loaded，bit2 LUT loaded，bit3 result valid，
// bit4 core busy，bit5 protocol error。
module softmax_ctrl #(
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

    output wire [6:0]                   debug_state,
    output reg                          protocol_error,
    output reg                          scores_loaded,
    output reg                          lut_loaded,
    output reg                          result_valid
);

localparam integer SCORE_BEATS = 56;
localparam integer LUT_BEATS   = 65;
localparam integer PROB_BEATS  = 28;

localparam [CTRL_ADDR_WIDTH-1:0] SCORE_BASE_CTRL = 28'h0000800;
localparam [CTRL_ADDR_WIDTH-1:0] PROB_BASE_CTRL  = 28'h0000A00;
localparam [CTRL_ADDR_WIDTH-1:0] LUT_BASE_CTRL   = 28'h0000B00;

localparam [6:0] ST_IDLE                 = 7'd0;
localparam [6:0] ST_RECV_SCORE           = 7'd1;
localparam [6:0] ST_SETUP_SCORE_WRITE    = 7'd2;
localparam [6:0] ST_SCORE_WRITE          = 7'd3;
localparam [6:0] ST_RECV_LUT             = 7'd4;
localparam [6:0] ST_SETUP_LUT_WRITE      = 7'd5;
localparam [6:0] ST_LUT_WRITE            = 7'd6;
localparam [6:0] ST_COMPUTE_INIT         = 7'd7;
localparam [6:0] ST_SETUP_SCORE_READ     = 7'd8;
localparam [6:0] ST_SCORE_READ           = 7'd9;
localparam [6:0] ST_SETUP_LUT_READ       = 7'd10;
localparam [6:0] ST_LUT_READ             = 7'd11;
localparam [6:0] ST_CORE_START           = 7'd12;
localparam [6:0] ST_CORE_WAIT            = 7'd13;
localparam [6:0] ST_SETUP_PROB_WRITE     = 7'd14;
localparam [6:0] ST_PROB_WRITE           = 7'd15;
localparam [6:0] ST_SETUP_RESULT_READ    = 7'd16;
localparam [6:0] ST_RESULT_READ          = 7'd17;
localparam [6:0] ST_SEND_RESULT_HEADER   = 7'd18;
localparam [6:0] ST_SEND_RESULT_BYTES    = 7'd19;
localparam [6:0] ST_SEND_INFO            = 7'd20;
localparam [6:0] ST_SEND_STATUS          = 7'd21;
localparam [6:0] ST_SEND_ACK             = 7'd22;
localparam [6:0] ST_SEND_ERROR           = 7'd23;

reg [6:0] state;
wire [7:0] uart_rx_data;
wire uart_rx_valid;
reg [7:0] tx_data;
reg tx_start;
wire tx_busy;
reg [6:0] tx_index;

reg [5:0] rx_byte_index;
reg [255:0] upload_beat;
reg [5:0] score_write_beat;
reg [6:0] lut_write_beat;

reg [6:0] read_global_beat;
reg [4:0] read_burst_count;
reg [4:0] read_capture_index;
reg [4:0] result_send_beat_index;
reg [5:0] result_send_byte_index;
reg [255:0] read_buffer [0:15];

reg core_score_beat_we;
reg [5:0] core_score_beat_index;
reg [255:0] core_score_beat_data;
reg core_lut_beat_we;
reg [6:0] core_lut_beat_index;
reg [255:0] core_lut_beat_data;
reg core_start;
reg core_probability_ready;
wire core_busy;
wire core_probability_valid;
wire [3:0] core_probability_head;
wire [3:0] core_probability_token;
wire [31:0] core_probability_q31;
wire core_done;

reg [3:0] pending_probability_head;
reg [3:0] pending_probability_token;
reg [31:0] pending_probability_q31;

reg aw_seen;
reg w_seen;
reg ar_seen;
reg [7:0] status_snapshot;
reg [7:0] error_code;
integer clear_index;

wire aw_handshake = axi_awvalid && axi_awready;
wire ar_handshake = axi_arvalid && axi_arready;
wire write_data_handshake = axi_wready && (aw_seen || aw_handshake);
wire read_data_handshake = axi_rvalid && (ar_seen || ar_handshake);

wire [6:0] score_read_remaining = SCORE_BEATS - read_global_beat;
wire [7:0] lut_read_remaining = LUT_BEATS - read_global_beat;
wire [6:0] prob_read_remaining = PROB_BEATS - read_global_beat;
wire [4:0] next_score_burst =
    (score_read_remaining > 7'd16) ? 5'd16 : score_read_remaining[4:0];
wire [4:0] next_lut_burst =
    (lut_read_remaining > 8'd16) ? 5'd16 : lut_read_remaining[4:0];
wire [4:0] next_prob_burst =
    (prob_read_remaining > 7'd16) ? 5'd16 : prob_read_remaining[4:0];
wire [255:0] selected_result_beat = read_buffer[result_send_beat_index];
wire [4:0] pending_probability_beat =
    ({1'b0, pending_probability_head} << 1) +
    {4'd0, pending_probability_token[3]};

assign axi_awuser_ap = 1'b0;
assign axi_awuser_id = 4'h0;
assign axi_awlen = 4'h0;
assign axi_aruser_ap = 1'b0;
assign axi_aruser_id = 4'h0;
assign debug_state = state;

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
            5'd9:  info_char = "S";
            5'd10: info_char = "O";
            5'd11: info_char = "F";
            5'd12: info_char = "T";
            5'd13: info_char = "M";
            5'd14: info_char = "A";
            5'd15: info_char = "X";
            5'd16: info_char = " ";
            5'd17: info_char = "F";
            5'd18: info_char = "5";
            5'd19: info_char = " ";
            5'd20: info_char = "V";
            5'd21: info_char = "1";
            5'd22: info_char = 8'h0d;
            5'd23: info_char = 8'h0a;
            default: info_char = 8'h00;
        endcase
    end
endfunction

uart_rx #(
    .CLKS_PER_BIT(CLKS_PER_BIT)
) u_uart_rx (
    .clk   (core_clk),
    .rst_n (core_rst_n),
    .rx    (uart_rx_i),
    .data  (uart_rx_data),
    .valid (uart_rx_valid)
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

softmax_core u_softmax_core (
    .clk                 (core_clk),
    .rst_n               (core_rst_n),
    .score_beat_we       (core_score_beat_we),
    .score_beat_index    (core_score_beat_index),
    .score_beat_data     (core_score_beat_data),
    .lut_beat_we         (core_lut_beat_we),
    .lut_beat_index      (core_lut_beat_index),
    .lut_beat_data       (core_lut_beat_data),
    .start               (core_start),
    .probability_ready   (core_probability_ready),
    .busy                (core_busy),
    .probability_valid   (core_probability_valid),
    .probability_head    (core_probability_head),
    .probability_token   (core_probability_token),
    .probability_q31     (core_probability_q31),
    .done                (core_done)
);

always @(posedge core_clk or negedge core_rst_n) begin
    if (!core_rst_n) begin
        state                     <= ST_IDLE;
        tx_data                   <= 8'd0;
        tx_start                  <= 1'b0;
        tx_index                  <= 7'd0;
        rx_byte_index             <= 6'd0;
        upload_beat               <= 256'd0;
        score_write_beat          <= 6'd0;
        lut_write_beat            <= 7'd0;
        read_global_beat          <= 7'd0;
        read_burst_count          <= 5'd0;
        read_capture_index        <= 5'd0;
        result_send_beat_index    <= 5'd0;
        result_send_byte_index    <= 6'd0;
        core_score_beat_we        <= 1'b0;
        core_score_beat_index     <= 6'd0;
        core_score_beat_data      <= 256'd0;
        core_lut_beat_we          <= 1'b0;
        core_lut_beat_index       <= 7'd0;
        core_lut_beat_data        <= 256'd0;
        core_start                <= 1'b0;
        core_probability_ready    <= 1'b0;
        pending_probability_head  <= 4'd0;
        pending_probability_token <= 4'd0;
        pending_probability_q31   <= 32'd0;
        axi_awaddr                <= {CTRL_ADDR_WIDTH{1'b0}};
        axi_awvalid               <= 1'b0;
        axi_wdata                 <= 256'd0;
        axi_wstrb                 <= 32'd0;
        axi_araddr                <= {CTRL_ADDR_WIDTH{1'b0}};
        axi_arlen                 <= 4'd0;
        axi_arvalid               <= 1'b0;
        aw_seen                   <= 1'b0;
        w_seen                    <= 1'b0;
        ar_seen                   <= 1'b0;
        status_snapshot           <= 8'd0;
        error_code                <= 8'd0;
        protocol_error            <= 1'b0;
        scores_loaded             <= 1'b0;
        lut_loaded                <= 1'b0;
        result_valid              <= 1'b0;
        for (clear_index = 0; clear_index < 16; clear_index = clear_index + 1)
            read_buffer[clear_index] <= 256'd0;
    end else begin
        tx_start               <= 1'b0;
        core_score_beat_we     <= 1'b0;
        core_lut_beat_we       <= 1'b0;
        core_start             <= 1'b0;
        core_probability_ready <= 1'b0;

        case (state)
            ST_IDLE: begin
                axi_awvalid <= 1'b0;
                axi_arvalid <= 1'b0;
                aw_seen     <= 1'b0;
                w_seen      <= 1'b0;
                ar_seen     <= 1'b0;
                tx_index    <= 7'd0;

                if (uart_rx_valid && !tx_busy) begin
                    case (uart_rx_data)
                        8'h49, 8'h69: state <= ST_SEND_INFO; // I/i

                        8'h53, 8'h73: begin // S/s
                            status_snapshot <= {
                                2'b00,
                                protocol_error,
                                core_busy,
                                result_valid,
                                lut_loaded,
                                scores_loaded,
                                ddr_init_done
                            };
                            state <= ST_SEND_STATUS;
                        end

                        8'h4c, 8'h6c: begin // L/l：load scores
                            if (!ddr_init_done) begin
                                protocol_error <= 1'b1;
                                error_code     <= 8'h02;
                                state          <= ST_SEND_ERROR;
                            end else begin
                                upload_beat      <= 256'd0;
                                rx_byte_index    <= 6'd0;
                                score_write_beat <= 6'd0;
                                scores_loaded    <= 1'b0;
                                result_valid     <= 1'b0;
                                state            <= ST_RECV_SCORE;
                            end
                        end

                        8'h54, 8'h74: begin // T/t：load exp table
                            if (!ddr_init_done) begin
                                protocol_error <= 1'b1;
                                error_code     <= 8'h02;
                                state          <= ST_SEND_ERROR;
                            end else begin
                                upload_beat   <= 256'd0;
                                rx_byte_index <= 6'd0;
                                lut_write_beat <= 7'd0;
                                lut_loaded    <= 1'b0;
                                result_valid  <= 1'b0;
                                state         <= ST_RECV_LUT;
                            end
                        end

                        8'h47, 8'h67: begin // G/g
                            if (!ddr_init_done) begin
                                protocol_error <= 1'b1;
                                error_code     <= 8'h02;
                                state          <= ST_SEND_ERROR;
                            end else if (!scores_loaded) begin
                                protocol_error <= 1'b1;
                                error_code     <= 8'h03;
                                state          <= ST_SEND_ERROR;
                            end else if (!lut_loaded) begin
                                protocol_error <= 1'b1;
                                error_code     <= 8'h04;
                                state          <= ST_SEND_ERROR;
                            end else begin
                                result_valid <= 1'b0;
                                state        <= ST_COMPUTE_INIT;
                            end
                        end

                        8'h52, 8'h72: begin // R/r
                            if (!result_valid) begin
                                protocol_error <= 1'b1;
                                error_code     <= 8'h05;
                                state          <= ST_SEND_ERROR;
                            end else begin
                                state <= ST_SEND_RESULT_HEADER;
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

            ST_RECV_SCORE: begin
                if (uart_rx_valid) begin
                    upload_beat[rx_byte_index*8 +: 8] <= uart_rx_data;
                    if (rx_byte_index == 6'd31)
                        state <= ST_SETUP_SCORE_WRITE;
                    else
                        rx_byte_index <= rx_byte_index + 1'b1;
                end
            end

            ST_SETUP_SCORE_WRITE: begin
                axi_awaddr  <= SCORE_BASE_CTRL + ({22'd0, score_write_beat} << 3);
                axi_awvalid <= 1'b1;
                axi_wdata   <= upload_beat;
                axi_wstrb   <= 32'hffff_ffff;
                aw_seen     <= 1'b0;
                w_seen      <= 1'b0;
                state       <= ST_SCORE_WRITE;
            end

            ST_SCORE_WRITE: begin
                if (aw_handshake) begin
                    axi_awvalid <= 1'b0;
                    aw_seen     <= 1'b1;
                end
                if (write_data_handshake)
                    w_seen <= 1'b1;
                if ((aw_seen || aw_handshake) &&
                    (w_seen || write_data_handshake)) begin
                    axi_awvalid <= 1'b0;
                    aw_seen     <= 1'b0;
                    w_seen      <= 1'b0;
                    if (score_write_beat == SCORE_BEATS - 1) begin
                        scores_loaded <= 1'b1;
                        state         <= ST_SEND_ACK;
                    end else begin
                        score_write_beat <= score_write_beat + 1'b1;
                        rx_byte_index    <= 6'd0;
                        upload_beat      <= 256'd0;
                        state            <= ST_RECV_SCORE;
                    end
                end
            end

            ST_RECV_LUT: begin
                if (uart_rx_valid) begin
                    upload_beat[rx_byte_index*8 +: 8] <= uart_rx_data;
                    if (rx_byte_index == 6'd31)
                        state <= ST_SETUP_LUT_WRITE;
                    else
                        rx_byte_index <= rx_byte_index + 1'b1;
                end
            end

            ST_SETUP_LUT_WRITE: begin
                axi_awaddr  <= LUT_BASE_CTRL + ({21'd0, lut_write_beat} << 3);
                axi_awvalid <= 1'b1;
                axi_wdata   <= upload_beat;
                axi_wstrb   <= 32'hffff_ffff;
                aw_seen     <= 1'b0;
                w_seen      <= 1'b0;
                state       <= ST_LUT_WRITE;
            end

            ST_LUT_WRITE: begin
                if (aw_handshake) begin
                    axi_awvalid <= 1'b0;
                    aw_seen     <= 1'b1;
                end
                if (write_data_handshake)
                    w_seen <= 1'b1;
                if ((aw_seen || aw_handshake) &&
                    (w_seen || write_data_handshake)) begin
                    axi_awvalid <= 1'b0;
                    aw_seen     <= 1'b0;
                    w_seen      <= 1'b0;
                    if (lut_write_beat == LUT_BEATS - 1) begin
                        lut_loaded <= 1'b1;
                        state      <= ST_SEND_ACK;
                    end else begin
                        lut_write_beat <= lut_write_beat + 1'b1;
                        rx_byte_index  <= 6'd0;
                        upload_beat    <= 256'd0;
                        state          <= ST_RECV_LUT;
                    end
                end
            end

            ST_COMPUTE_INIT: begin
                read_global_beat   <= 7'd0;
                read_capture_index <= 5'd0;
                state              <= ST_SETUP_SCORE_READ;
            end

            ST_SETUP_SCORE_READ: begin
                read_burst_count   <= next_score_burst;
                read_capture_index <= 5'd0;
                axi_araddr         <= SCORE_BASE_CTRL + ({21'd0, read_global_beat} << 3);
                axi_arlen          <= next_score_burst - 1'b1;
                axi_arvalid        <= 1'b1;
                ar_seen            <= 1'b0;
                state              <= ST_SCORE_READ;
            end

            ST_SCORE_READ: begin
                if (ar_handshake) begin
                    axi_arvalid <= 1'b0;
                    ar_seen     <= 1'b1;
                end
                if (read_data_handshake) begin
                    core_score_beat_we    <= 1'b1;
                    core_score_beat_index <= read_global_beat[5:0] + read_capture_index;
                    core_score_beat_data  <= axi_rdata;
                    if (read_capture_index == read_burst_count - 1'b1) begin
                        ar_seen <= 1'b0;
                        if (read_global_beat + read_burst_count == SCORE_BEATS) begin
                            read_global_beat   <= 7'd0;
                            read_capture_index <= 5'd0;
                            state              <= ST_SETUP_LUT_READ;
                        end else begin
                            read_global_beat <= read_global_beat + read_burst_count;
                            state            <= ST_SETUP_SCORE_READ;
                        end
                    end else begin
                        read_capture_index <= read_capture_index + 1'b1;
                    end
                end
            end

            ST_SETUP_LUT_READ: begin
                read_burst_count   <= next_lut_burst;
                read_capture_index <= 5'd0;
                axi_araddr         <= LUT_BASE_CTRL + ({21'd0, read_global_beat} << 3);
                axi_arlen          <= next_lut_burst - 1'b1;
                axi_arvalid        <= 1'b1;
                ar_seen            <= 1'b0;
                state              <= ST_LUT_READ;
            end

            ST_LUT_READ: begin
                if (ar_handshake) begin
                    axi_arvalid <= 1'b0;
                    ar_seen     <= 1'b1;
                end
                if (read_data_handshake) begin
                    core_lut_beat_we    <= 1'b1;
                    core_lut_beat_index <= read_global_beat + read_capture_index;
                    core_lut_beat_data  <= axi_rdata;
                    if (read_capture_index == read_burst_count - 1'b1) begin
                        ar_seen <= 1'b0;
                        if (read_global_beat + read_burst_count == LUT_BEATS) begin
                            state <= ST_CORE_START;
                        end else begin
                            read_global_beat <= read_global_beat + read_burst_count;
                            state            <= ST_SETUP_LUT_READ;
                        end
                    end else begin
                        read_capture_index <= read_capture_index + 1'b1;
                    end
                end
            end

            ST_CORE_START: begin
                core_start <= 1'b1;
                state      <= ST_CORE_WAIT;
            end

            ST_CORE_WAIT: begin
                if (core_probability_valid) begin
                    pending_probability_head  <= core_probability_head;
                    pending_probability_token <= core_probability_token;
                    pending_probability_q31   <= core_probability_q31;
                    state                     <= ST_SETUP_PROB_WRITE;
                end else if (core_done) begin
                    result_valid <= 1'b1;
                    state        <= ST_SEND_ACK;
                end
            end

            ST_SETUP_PROB_WRITE: begin
                axi_awaddr  <= PROB_BASE_CTRL + ({23'd0, pending_probability_beat} << 3);
                axi_awvalid <= 1'b1;
                case (pending_probability_token[2:0])
                    3'd0: begin
                        axi_wdata <= {224'd0, pending_probability_q31};
                        axi_wstrb <= 32'h0000_000f;
                    end
                    3'd1: begin
                        axi_wdata <= {192'd0, pending_probability_q31, 32'd0};
                        axi_wstrb <= 32'h0000_00f0;
                    end
                    3'd2: begin
                        axi_wdata <= {160'd0, pending_probability_q31, 64'd0};
                        axi_wstrb <= 32'h0000_0f00;
                    end
                    3'd3: begin
                        axi_wdata <= {128'd0, pending_probability_q31, 96'd0};
                        axi_wstrb <= 32'h0000_f000;
                    end
                    3'd4: begin
                        axi_wdata <= {96'd0, pending_probability_q31, 128'd0};
                        axi_wstrb <= 32'h000f_0000;
                    end
                    3'd5: begin
                        axi_wdata <= {64'd0, pending_probability_q31, 160'd0};
                        axi_wstrb <= 32'h00f0_0000;
                    end
                    3'd6: begin
                        axi_wdata <= {32'd0, pending_probability_q31, 192'd0};
                        axi_wstrb <= 32'h0f00_0000;
                    end
                    default: begin
                        axi_wdata <= {pending_probability_q31, 224'd0};
                        axi_wstrb <= 32'hf000_0000;
                    end
                endcase
                aw_seen <= 1'b0;
                w_seen  <= 1'b0;
                state   <= ST_PROB_WRITE;
            end

            ST_PROB_WRITE: begin
                if (aw_handshake) begin
                    axi_awvalid <= 1'b0;
                    aw_seen     <= 1'b1;
                end
                if (write_data_handshake)
                    w_seen <= 1'b1;
                if ((aw_seen || aw_handshake) &&
                    (w_seen || write_data_handshake)) begin
                    axi_awvalid           <= 1'b0;
                    aw_seen               <= 1'b0;
                    w_seen                <= 1'b0;
                    core_probability_ready <= 1'b1;
                    state                 <= ST_CORE_WAIT;
                end
            end

            ST_SEND_RESULT_HEADER: begin
                if (!tx_busy && !tx_start) begin
                    tx_data  <= "D";
                    tx_start <= 1'b1;
                    read_global_beat       <= 7'd0;
                    read_capture_index     <= 5'd0;
                    result_send_beat_index <= 5'd0;
                    result_send_byte_index <= 6'd0;
                    state                  <= ST_SETUP_RESULT_READ;
                end
            end

            ST_SETUP_RESULT_READ: begin
                read_burst_count   <= next_prob_burst;
                read_capture_index <= 5'd0;
                axi_araddr         <= PROB_BASE_CTRL + ({21'd0, read_global_beat} << 3);
                axi_arlen          <= next_prob_burst - 1'b1;
                axi_arvalid        <= 1'b1;
                ar_seen            <= 1'b0;
                state              <= ST_RESULT_READ;
            end

            ST_RESULT_READ: begin
                if (ar_handshake) begin
                    axi_arvalid <= 1'b0;
                    ar_seen     <= 1'b1;
                end
                if (read_data_handshake) begin
                    read_buffer[read_capture_index] <= axi_rdata;
                    if (read_capture_index == read_burst_count - 1'b1) begin
                        ar_seen                    <= 1'b0;
                        result_send_beat_index     <= 5'd0;
                        result_send_byte_index     <= 6'd0;
                        state                      <= ST_SEND_RESULT_BYTES;
                    end else begin
                        read_capture_index <= read_capture_index + 1'b1;
                    end
                end
            end

            ST_SEND_RESULT_BYTES: begin
                if (!tx_busy && !tx_start) begin
                    tx_data  <= selected_result_beat[result_send_byte_index*8 +: 8];
                    tx_start <= 1'b1;
                    if (result_send_byte_index == 6'd31) begin
                        result_send_byte_index <= 6'd0;
                        if (result_send_beat_index == read_burst_count - 1'b1) begin
                            result_send_beat_index <= 5'd0;
                            if (read_global_beat + read_burst_count == PROB_BEATS) begin
                                state <= ST_IDLE;
                            end else begin
                                read_global_beat <= read_global_beat + read_burst_count;
                                state            <= ST_SETUP_RESULT_READ;
                            end
                        end else begin
                            result_send_beat_index <= result_send_beat_index + 1'b1;
                        end
                    end else begin
                        result_send_byte_index <= result_send_byte_index + 1'b1;
                    end
                end
            end

            ST_SEND_INFO: begin
                if (!tx_busy && !tx_start) begin
                    tx_data  <= info_char(tx_index[4:0]);
                    tx_start <= 1'b1;
                    if (tx_index == 7'd23) begin
                        tx_index <= 7'd0;
                        state    <= ST_IDLE;
                    end else begin
                        tx_index <= tx_index + 1'b1;
                    end
                end
            end

            ST_SEND_STATUS: begin
                if (!tx_busy && !tx_start) begin
                    case (tx_index)
                        7'd0: tx_data <= "S";
                        7'd1: tx_data <= status_snapshot;
                        7'd2: tx_data <= 8'h0d;
                        default: tx_data <= 8'h0a;
                    endcase
                    tx_start <= 1'b1;
                    if (tx_index == 7'd3) begin
                        tx_index <= 7'd0;
                        state    <= ST_IDLE;
                    end else begin
                        tx_index <= tx_index + 1'b1;
                    end
                end
            end

            ST_SEND_ACK: begin
                if (!tx_busy && !tx_start) begin
                    case (tx_index)
                        7'd0: tx_data <= "K";
                        7'd1: tx_data <= 8'h0d;
                        default: tx_data <= 8'h0a;
                    endcase
                    tx_start <= 1'b1;
                    if (tx_index == 7'd2) begin
                        tx_index <= 7'd0;
                        state    <= ST_IDLE;
                    end else begin
                        tx_index <= tx_index + 1'b1;
                    end
                end
            end

            ST_SEND_ERROR: begin
                if (!tx_busy && !tx_start) begin
                    case (tx_index)
                        7'd0: tx_data <= "E";
                        7'd1: tx_data <= error_code;
                        7'd2: tx_data <= 8'h0d;
                        default: tx_data <= 8'h0a;
                    endcase
                    tx_start <= 1'b1;
                    if (tx_index == 7'd3) begin
                        tx_index <= 7'd0;
                        state    <= ST_IDLE;
                    end else begin
                        tx_index <= tx_index + 1'b1;
                    end
                end
            end

            default: begin
                state          <= ST_IDLE;
                axi_awvalid    <= 1'b0;
                axi_arvalid    <= 1'b0;
                protocol_error <= 1'b1;
                error_code     <= 8'hff;
            end
        endcase
    end
end

endmodule
