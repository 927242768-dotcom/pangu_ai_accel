`timescale 1ns/1ps

// F6 Attention 输出独立验证控制器。
//
// DDR3 低端临时区：
//   output        : ctrl 0x0000000，7168 B / 224 beats，[14,64] int64 Q28
//   probabilities : ctrl 0x0000A00， 896 B /  28 beats，[14,16] uint32 UQ1.31
//
// V 完全复用 F3 KV Cache 高端地址：
//   V = 0x02000000 + layer * 0x00800000 + position * 0x00000200 + 0x00000100
//   每个 V 为 [2,64] int64 Q28，共 1024 B / 32 beats。
//
// UART 协议（115200 8N1）：
//   I -> "PANGU50K ATTN OUTPUT V1\r\n"
//   S -> 'S' + flags + layer + start_u16 + count + v_loaded + CRLF
//   C + layer_u8 + start_u16 + count_u8 -> 配置，回复 "K\r\n"
//   P + 896 B probabilities -> 写入低端 DDR3，回复 "K\r\n"
//   V + position_u16 + 1024 B -> 写入 F3 V Cache，回复
//        'K' + position_u16 + CRLF
//   G -> 读取 probabilities/V，执行 14x64 加权和，写回 output，回复 "K\r\n"
//   R -> 回复 'D' + layer + start_u16 + count_u8 + 7168 B output
//
// flags：bit0 DDR ready，bit1 configured，bit2 probabilities loaded，
// bit3 V loaded，bit4 result valid，bit5 core busy，bit6 protocol error。
module attention_output_ctrl #(
    parameter integer CTRL_ADDR_WIDTH = 28,
    parameter integer CLKS_PER_BIT    = 868,
    parameter integer NUM_LAYERS      = 28,
    parameter integer MAX_CONTEXT     = 16384,
    parameter integer MAX_TOKENS      = 16
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
    output wire [4:0]                   debug_layer,
    output wire [15:0]                  debug_window_start,
    output reg                          protocol_error,
    output reg                          configured,
    output reg                          probabilities_loaded,
    output reg                          result_valid
);

localparam integer PROB_BEATS   = 28;
localparam integer V_BEATS      = 32;
localparam integer OUTPUT_BEATS = 224;

localparam [CTRL_ADDR_WIDTH-1:0] OUTPUT_BASE_CTRL = 28'h0000000;
localparam [CTRL_ADDR_WIDTH-1:0] PROB_BASE_CTRL   = 28'h0000A00;
localparam [CTRL_ADDR_WIDTH-1:0] KV_BASE_CTRL     = 28'h2000000;
localparam [CTRL_ADDR_WIDTH-1:0] V_OFFSET_CTRL    = 28'h0000100;

localparam [6:0] ST_IDLE                = 7'd0;
localparam [6:0] ST_RECV_CONFIG         = 7'd1;
localparam [6:0] ST_APPLY_CONFIG        = 7'd2;
localparam [6:0] ST_RECV_PROB           = 7'd3;
localparam [6:0] ST_SETUP_PROB_WRITE    = 7'd4;
localparam [6:0] ST_PROB_WRITE          = 7'd5;
localparam [6:0] ST_RECV_V_POSITION     = 7'd6;
localparam [6:0] ST_APPLY_V_POSITION    = 7'd7;
localparam [6:0] ST_RECV_V              = 7'd8;
localparam [6:0] ST_SETUP_V_WRITE       = 7'd9;
localparam [6:0] ST_V_WRITE             = 7'd10;
localparam [6:0] ST_COMPUTE_INIT        = 7'd11;
localparam [6:0] ST_SETUP_PROB_READ     = 7'd12;
localparam [6:0] ST_PROB_READ           = 7'd13;
localparam [6:0] ST_TOKEN_DISPATCH      = 7'd14;
localparam [6:0] ST_SETUP_V_READ        = 7'd15;
localparam [6:0] ST_V_READ              = 7'd16;
localparam [6:0] ST_CORE_START          = 7'd17;
localparam [6:0] ST_CORE_WAIT           = 7'd18;
localparam [6:0] ST_SETUP_OUTPUT_WRITE  = 7'd19;
localparam [6:0] ST_OUTPUT_WRITE        = 7'd20;
localparam [6:0] ST_SEND_RESULT_HEADER  = 7'd21;
localparam [6:0] ST_SETUP_RESULT_READ   = 7'd22;
localparam [6:0] ST_RESULT_READ         = 7'd23;
localparam [6:0] ST_SEND_RESULT_BYTES   = 7'd24;
localparam [6:0] ST_SEND_INFO           = 7'd25;
localparam [6:0] ST_SEND_STATUS         = 7'd26;
localparam [6:0] ST_SEND_ACK            = 7'd27;
localparam [6:0] ST_SEND_V_ACK          = 7'd28;
localparam [6:0] ST_SEND_ERROR          = 7'd29;

reg [6:0] state;
wire [7:0] uart_rx_data;
wire uart_rx_valid;
reg [7:0] tx_data;
reg tx_start;
wire tx_busy;
reg [6:0] tx_index;

reg [31:0] config_word;
reg [1:0] config_byte_index;
reg [4:0] configured_layer;
reg [15:0] window_start;
reg [4:0] window_count;
reg [7:0] v_loaded_count;

reg [5:0] rx_byte_index;
reg [255:0] upload_beat;
reg [4:0] probability_write_beat;
reg [4:0] v_write_beat;
reg [15:0] v_position_word;
reg v_position_byte_index;
reg [15:0] v_upload_position;

reg [8:0] read_global_beat;
reg [4:0] read_burst_count;
reg [4:0] read_capture_index;
reg [4:0] result_send_beat_index;
reg [5:0] result_send_byte_index;
reg [255:0] read_buffer [0:15];

reg [3:0] token_index;
reg [15:0] token_position;
reg [3:0] pending_output_head;
reg [5:0] pending_output_dimension;
reg signed [63:0] pending_output_q28;

reg core_probability_beat_we;
reg [4:0] core_probability_beat_index;
reg [255:0] core_probability_beat_data;
reg core_v_beat_we;
reg [8:0] core_v_beat_index;
reg [255:0] core_v_beat_data;
reg core_start;
reg core_result_ready;
wire core_busy;
wire core_result_valid;
wire [3:0] core_result_head;
wire [5:0] core_result_dimension;
wire signed [63:0] core_result_q28;
wire core_done;

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

wire [16:0] config_start_ext = {1'b0, config_word[23:8]};
wire [17:0] config_window_end = {1'b0, config_word[23:8]} +
                                {10'd0, config_word[31:24]};
wire [16:0] v_position_ext = {1'b0, v_position_word};
wire [5:0] probability_read_remaining = PROB_BEATS - read_global_beat[5:0];
wire [5:0] v_read_remaining = V_BEATS - read_global_beat[5:0];
wire [8:0] result_read_remaining = OUTPUT_BEATS - read_global_beat;
wire [4:0] next_probability_burst =
    (probability_read_remaining > 6'd16) ? 5'd16 :
    probability_read_remaining[4:0];
wire [4:0] next_v_burst =
    (v_read_remaining > 6'd16) ? 5'd16 : v_read_remaining[4:0];
wire [4:0] next_result_burst =
    (result_read_remaining > 9'd16) ? 5'd16 : result_read_remaining[4:0];
wire [255:0] selected_result_beat = read_buffer[result_send_beat_index];

wire [CTRL_ADDR_WIDTH-1:0] configured_layer_offset =
    ({23'd0, configured_layer} << 23);
wire [CTRL_ADDR_WIDTH-1:0] v_upload_position_offset =
    ({12'd0, v_upload_position} << 9);
wire [CTRL_ADDR_WIDTH-1:0] current_token_offset =
    ({12'd0, token_position} << 9);
wire [8:0] current_v_cache_beat =
    ({5'd0, token_index} << 5) + read_global_beat + read_capture_index;
wire [7:0] pending_output_beat =
    ({4'd0, pending_output_head} << 4) +
    {4'd0, pending_output_dimension[5:2]};

assign axi_awuser_ap = 1'b0;
assign axi_awuser_id = 4'h0;
assign axi_awlen = 4'h0;
assign axi_aruser_ap = 1'b0;
assign axi_aruser_id = 4'h0;
assign debug_state = state;
assign debug_layer = configured_layer;
assign debug_window_start = window_start;

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
            5'd9:  info_char = "A";
            5'd10: info_char = "T";
            5'd11: info_char = "T";
            5'd12: info_char = "N";
            5'd13: info_char = " ";
            5'd14: info_char = "O";
            5'd15: info_char = "U";
            5'd16: info_char = "T";
            5'd17: info_char = "P";
            5'd18: info_char = "U";
            5'd19: info_char = "T";
            5'd20: info_char = " ";
            5'd21: info_char = "V";
            5'd22: info_char = "1";
            5'd23: info_char = 8'h0d;
            5'd24: info_char = 8'h0a;
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

attention_output_core u_attention_output_core (
    .clk                    (core_clk),
    .rst_n                  (core_rst_n),
    .probability_beat_we    (core_probability_beat_we),
    .probability_beat_index (core_probability_beat_index),
    .probability_beat_data  (core_probability_beat_data),
    .v_beat_we              (core_v_beat_we),
    .v_beat_index           (core_v_beat_index),
    .v_beat_data            (core_v_beat_data),
    .start                  (core_start),
    .token_count            (window_count),
    .result_ready           (core_result_ready),
    .busy                   (core_busy),
    .result_valid           (core_result_valid),
    .result_head            (core_result_head),
    .result_dimension       (core_result_dimension),
    .result_q28             (core_result_q28),
    .done                   (core_done)
);

always @(posedge core_clk or negedge core_rst_n) begin
    if (!core_rst_n) begin
        state                       <= ST_IDLE;
        tx_data                     <= 8'd0;
        tx_start                    <= 1'b0;
        tx_index                    <= 7'd0;
        config_word                 <= 32'd0;
        config_byte_index           <= 2'd0;
        configured_layer            <= 5'd0;
        window_start                <= 16'd0;
        window_count                <= 5'd0;
        v_loaded_count              <= 8'd0;
        rx_byte_index               <= 6'd0;
        upload_beat                 <= 256'd0;
        probability_write_beat      <= 5'd0;
        v_write_beat                <= 5'd0;
        v_position_word             <= 16'd0;
        v_position_byte_index       <= 1'b0;
        v_upload_position           <= 16'd0;
        read_global_beat            <= 9'd0;
        read_burst_count            <= 5'd0;
        read_capture_index          <= 5'd0;
        result_send_beat_index      <= 5'd0;
        result_send_byte_index      <= 6'd0;
        token_index                 <= 4'd0;
        token_position              <= 16'd0;
        pending_output_head         <= 4'd0;
        pending_output_dimension    <= 6'd0;
        pending_output_q28          <= 64'sd0;
        core_probability_beat_we    <= 1'b0;
        core_probability_beat_index <= 5'd0;
        core_probability_beat_data  <= 256'd0;
        core_v_beat_we              <= 1'b0;
        core_v_beat_index           <= 9'd0;
        core_v_beat_data            <= 256'd0;
        core_start                  <= 1'b0;
        core_result_ready           <= 1'b0;
        axi_awaddr                  <= {CTRL_ADDR_WIDTH{1'b0}};
        axi_awvalid                 <= 1'b0;
        axi_wdata                   <= 256'd0;
        axi_wstrb                   <= 32'd0;
        axi_araddr                  <= {CTRL_ADDR_WIDTH{1'b0}};
        axi_arlen                   <= 4'd0;
        axi_arvalid                 <= 1'b0;
        aw_seen                     <= 1'b0;
        w_seen                      <= 1'b0;
        ar_seen                     <= 1'b0;
        status_snapshot             <= 8'd0;
        error_code                  <= 8'd0;
        protocol_error              <= 1'b0;
        configured                 <= 1'b0;
        probabilities_loaded       <= 1'b0;
        result_valid               <= 1'b0;
        for (clear_index = 0; clear_index < 16; clear_index = clear_index + 1)
            read_buffer[clear_index] <= 256'd0;
    end else begin
        tx_start                 <= 1'b0;
        core_probability_beat_we <= 1'b0;
        core_v_beat_we           <= 1'b0;
        core_start               <= 1'b0;
        core_result_ready        <= 1'b0;

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
                                1'b0,
                                protocol_error,
                                core_busy,
                                result_valid,
                                (v_loaded_count != 8'd0),
                                probabilities_loaded,
                                configured,
                                ddr_init_done
                            };
                            state <= ST_SEND_STATUS;
                        end

                        8'h43, 8'h63: begin // C/c
                            config_word       <= 32'd0;
                            config_byte_index <= 2'd0;
                            state             <= ST_RECV_CONFIG;
                        end

                        8'h50, 8'h70: begin // P/p
                            if (!ddr_init_done) begin
                                protocol_error <= 1'b1;
                                error_code     <= 8'h02;
                                state          <= ST_SEND_ERROR;
                            end else if (!configured) begin
                                protocol_error <= 1'b1;
                                error_code     <= 8'h03;
                                state          <= ST_SEND_ERROR;
                            end else begin
                                upload_beat            <= 256'd0;
                                rx_byte_index          <= 6'd0;
                                probability_write_beat <= 5'd0;
                                probabilities_loaded   <= 1'b0;
                                result_valid           <= 1'b0;
                                state                  <= ST_RECV_PROB;
                            end
                        end

                        8'h56, 8'h76: begin // V/v
                            if (!ddr_init_done) begin
                                protocol_error <= 1'b1;
                                error_code     <= 8'h02;
                                state          <= ST_SEND_ERROR;
                            end else if (!configured) begin
                                protocol_error <= 1'b1;
                                error_code     <= 8'h03;
                                state          <= ST_SEND_ERROR;
                            end else begin
                                v_position_word       <= 16'd0;
                                v_position_byte_index <= 1'b0;
                                state                 <= ST_RECV_V_POSITION;
                            end
                        end

                        8'h47, 8'h67: begin // G/g
                            if (!ddr_init_done) begin
                                protocol_error <= 1'b1;
                                error_code     <= 8'h02;
                                state          <= ST_SEND_ERROR;
                            end else if (!configured) begin
                                protocol_error <= 1'b1;
                                error_code     <= 8'h03;
                                state          <= ST_SEND_ERROR;
                            end else if (!probabilities_loaded) begin
                                protocol_error <= 1'b1;
                                error_code     <= 8'h05;
                                state          <= ST_SEND_ERROR;
                            end else if (v_loaded_count < window_count) begin
                                protocol_error <= 1'b1;
                                error_code     <= 8'h07;
                                state          <= ST_SEND_ERROR;
                            end else begin
                                result_valid <= 1'b0;
                                state        <= ST_COMPUTE_INIT;
                            end
                        end

                        8'h52, 8'h72: begin // R/r
                            if (!result_valid) begin
                                protocol_error <= 1'b1;
                                error_code     <= 8'h08;
                                state          <= ST_SEND_ERROR;
                            end else begin
                                tx_index <= 7'd0;
                                state    <= ST_SEND_RESULT_HEADER;
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

            ST_RECV_CONFIG: begin
                if (uart_rx_valid) begin
                    config_word[config_byte_index*8 +: 8] <= uart_rx_data;
                    if (config_byte_index == 2'd3)
                        state <= ST_APPLY_CONFIG;
                    else
                        config_byte_index <= config_byte_index + 1'b1;
                end
            end

            ST_APPLY_CONFIG: begin
                if ((config_word[7:0] >= NUM_LAYERS) ||
                    (config_start_ext >= MAX_CONTEXT) ||
                    (config_word[31:24] == 8'd0) ||
                    (config_word[31:24] > MAX_TOKENS) ||
                    (config_window_end > MAX_CONTEXT)) begin
                    protocol_error <= 1'b1;
                    error_code     <= 8'h04;
                    state          <= ST_SEND_ERROR;
                end else begin
                    configured_layer      <= config_word[4:0];
                    window_start          <= config_word[23:8];
                    window_count          <= config_word[28:24];
                    v_loaded_count        <= 8'd0;
                    configured           <= 1'b1;
                    probabilities_loaded <= 1'b0;
                    result_valid         <= 1'b0;
                    state                <= ST_SEND_ACK;
                end
            end

            ST_RECV_PROB: begin
                if (uart_rx_valid) begin
                    upload_beat[rx_byte_index*8 +: 8] <= uart_rx_data;
                    if (rx_byte_index == 6'd31)
                        state <= ST_SETUP_PROB_WRITE;
                    else
                        rx_byte_index <= rx_byte_index + 1'b1;
                end
            end

            ST_SETUP_PROB_WRITE: begin
                axi_awaddr  <= PROB_BASE_CTRL +
                               ({23'd0, probability_write_beat} << 3);
                axi_awvalid <= 1'b1;
                axi_wdata   <= upload_beat;
                axi_wstrb   <= 32'hffff_ffff;
                aw_seen     <= 1'b0;
                w_seen      <= 1'b0;
                state       <= ST_PROB_WRITE;
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
                    axi_awvalid <= 1'b0;
                    aw_seen     <= 1'b0;
                    w_seen      <= 1'b0;
                    if (probability_write_beat == PROB_BEATS - 1) begin
                        probabilities_loaded <= 1'b1;
                        state                <= ST_SEND_ACK;
                    end else begin
                        probability_write_beat <= probability_write_beat + 1'b1;
                        rx_byte_index          <= 6'd0;
                        upload_beat            <= 256'd0;
                        state                  <= ST_RECV_PROB;
                    end
                end
            end

            ST_RECV_V_POSITION: begin
                if (uart_rx_valid) begin
                    v_position_word[v_position_byte_index*8 +: 8] <= uart_rx_data;
                    if (v_position_byte_index)
                        state <= ST_APPLY_V_POSITION;
                    else
                        v_position_byte_index <= 1'b1;
                end
            end

            ST_APPLY_V_POSITION: begin
                if (v_position_ext >= MAX_CONTEXT) begin
                    protocol_error <= 1'b1;
                    error_code     <= 8'h06;
                    state          <= ST_SEND_ERROR;
                end else begin
                    v_upload_position <= v_position_word;
                    upload_beat       <= 256'd0;
                    rx_byte_index     <= 6'd0;
                    v_write_beat      <= 5'd0;
                    result_valid      <= 1'b0;
                    state             <= ST_RECV_V;
                end
            end

            ST_RECV_V: begin
                if (uart_rx_valid) begin
                    upload_beat[rx_byte_index*8 +: 8] <= uart_rx_data;
                    if (rx_byte_index == 6'd31)
                        state <= ST_SETUP_V_WRITE;
                    else
                        rx_byte_index <= rx_byte_index + 1'b1;
                end
            end

            ST_SETUP_V_WRITE: begin
                axi_awaddr <= KV_BASE_CTRL + configured_layer_offset +
                              v_upload_position_offset + V_OFFSET_CTRL +
                              ({23'd0, v_write_beat} << 3);
                axi_awvalid <= 1'b1;
                axi_wdata   <= upload_beat;
                axi_wstrb   <= 32'hffff_ffff;
                aw_seen     <= 1'b0;
                w_seen      <= 1'b0;
                state       <= ST_V_WRITE;
            end

            ST_V_WRITE: begin
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
                    if (v_write_beat == V_BEATS - 1) begin
                        if (v_loaded_count != 8'hff)
                            v_loaded_count <= v_loaded_count + 1'b1;
                        state <= ST_SEND_V_ACK;
                    end else begin
                        v_write_beat  <= v_write_beat + 1'b1;
                        rx_byte_index <= 6'd0;
                        upload_beat   <= 256'd0;
                        state         <= ST_RECV_V;
                    end
                end
            end

            ST_COMPUTE_INIT: begin
                read_global_beat   <= 9'd0;
                read_capture_index <= 5'd0;
                token_index        <= 4'd0;
                token_position     <= window_start;
                state              <= ST_SETUP_PROB_READ;
            end

            ST_SETUP_PROB_READ: begin
                read_burst_count   <= next_probability_burst;
                read_capture_index <= 5'd0;
                axi_araddr         <= PROB_BASE_CTRL +
                                      ({19'd0, read_global_beat} << 3);
                axi_arlen          <= next_probability_burst - 1'b1;
                axi_arvalid        <= 1'b1;
                ar_seen            <= 1'b0;
                state              <= ST_PROB_READ;
            end

            ST_PROB_READ: begin
                if (ar_handshake) begin
                    axi_arvalid <= 1'b0;
                    ar_seen     <= 1'b1;
                end
                if (read_data_handshake) begin
                    core_probability_beat_we    <= 1'b1;
                    core_probability_beat_index <=
                        read_global_beat[4:0] + read_capture_index;
                    core_probability_beat_data <= axi_rdata;
                    if (read_capture_index == read_burst_count - 1'b1) begin
                        ar_seen <= 1'b0;
                        if (read_global_beat + read_burst_count == PROB_BEATS) begin
                            token_index    <= 4'd0;
                            token_position <= window_start;
                            state          <= ST_TOKEN_DISPATCH;
                        end else begin
                            read_global_beat <= read_global_beat + read_burst_count;
                            state            <= ST_SETUP_PROB_READ;
                        end
                    end else begin
                        read_capture_index <= read_capture_index + 1'b1;
                    end
                end
            end

            ST_TOKEN_DISPATCH: begin
                token_position     <= window_start + {12'd0, token_index};
                read_global_beat   <= 9'd0;
                read_capture_index <= 5'd0;
                state              <= ST_SETUP_V_READ;
            end

            ST_SETUP_V_READ: begin
                read_burst_count   <= next_v_burst;
                read_capture_index <= 5'd0;
                axi_araddr <= KV_BASE_CTRL + configured_layer_offset +
                              current_token_offset + V_OFFSET_CTRL +
                              ({19'd0, read_global_beat} << 3);
                axi_arlen   <= next_v_burst - 1'b1;
                axi_arvalid <= 1'b1;
                ar_seen     <= 1'b0;
                state       <= ST_V_READ;
            end

            ST_V_READ: begin
                if (ar_handshake) begin
                    axi_arvalid <= 1'b0;
                    ar_seen     <= 1'b1;
                end
                if (read_data_handshake) begin
                    core_v_beat_we    <= 1'b1;
                    core_v_beat_index <= current_v_cache_beat;
                    core_v_beat_data  <= axi_rdata;
                    if (read_capture_index == read_burst_count - 1'b1) begin
                        ar_seen <= 1'b0;
                        if (read_global_beat + read_burst_count == V_BEATS) begin
                            if (token_index == window_count - 1'b1) begin
                                state <= ST_CORE_START;
                            end else begin
                                token_index    <= token_index + 1'b1;
                                token_position <= window_start +
                                                  {12'd0, token_index} + 1'b1;
                                state          <= ST_TOKEN_DISPATCH;
                            end
                        end else begin
                            read_global_beat <= read_global_beat + read_burst_count;
                            state            <= ST_SETUP_V_READ;
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
                if (core_result_valid) begin
                    pending_output_head      <= core_result_head;
                    pending_output_dimension <= core_result_dimension;
                    pending_output_q28       <= core_result_q28;
                    state                    <= ST_SETUP_OUTPUT_WRITE;
                end else if (core_done) begin
                    result_valid <= 1'b1;
                    state        <= ST_SEND_ACK;
                end
            end

            ST_SETUP_OUTPUT_WRITE: begin
                axi_awaddr  <= OUTPUT_BASE_CTRL +
                               ({20'd0, pending_output_beat} << 3);
                axi_awvalid <= 1'b1;
                case (pending_output_dimension[1:0])
                    2'd0: begin
                        axi_wdata <= {192'd0, pending_output_q28};
                        axi_wstrb <= 32'h0000_00ff;
                    end
                    2'd1: begin
                        axi_wdata <= {128'd0, pending_output_q28, 64'd0};
                        axi_wstrb <= 32'h0000_ff00;
                    end
                    2'd2: begin
                        axi_wdata <= {64'd0, pending_output_q28, 128'd0};
                        axi_wstrb <= 32'h00ff_0000;
                    end
                    default: begin
                        axi_wdata <= {pending_output_q28, 192'd0};
                        axi_wstrb <= 32'hff00_0000;
                    end
                endcase
                aw_seen <= 1'b0;
                w_seen  <= 1'b0;
                state   <= ST_OUTPUT_WRITE;
            end

            ST_OUTPUT_WRITE: begin
                if (aw_handshake) begin
                    axi_awvalid <= 1'b0;
                    aw_seen     <= 1'b1;
                end
                if (write_data_handshake)
                    w_seen <= 1'b1;
                if ((aw_seen || aw_handshake) &&
                    (w_seen || write_data_handshake)) begin
                    axi_awvalid       <= 1'b0;
                    aw_seen           <= 1'b0;
                    w_seen            <= 1'b0;
                    core_result_ready <= 1'b1;
                    state             <= ST_CORE_WAIT;
                end
            end

            ST_SEND_RESULT_HEADER: begin
                if (!tx_busy && !tx_start) begin
                    case (tx_index)
                        7'd0: tx_data <= "D";
                        7'd1: tx_data <= {3'd0, configured_layer};
                        7'd2: tx_data <= window_start[7:0];
                        7'd3: tx_data <= window_start[15:8];
                        default: tx_data <= {3'd0, window_count};
                    endcase
                    tx_start <= 1'b1;
                    if (tx_index == 7'd4) begin
                        read_global_beat       <= 9'd0;
                        read_capture_index     <= 5'd0;
                        result_send_beat_index <= 5'd0;
                        result_send_byte_index <= 6'd0;
                        tx_index               <= 7'd0;
                        state                  <= ST_SETUP_RESULT_READ;
                    end else begin
                        tx_index <= tx_index + 1'b1;
                    end
                end
            end

            ST_SETUP_RESULT_READ: begin
                read_burst_count   <= next_result_burst;
                read_capture_index <= 5'd0;
                axi_araddr         <= OUTPUT_BASE_CTRL +
                                      ({19'd0, read_global_beat} << 3);
                axi_arlen          <= next_result_burst - 1'b1;
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
                        ar_seen                <= 1'b0;
                        result_send_beat_index <= 5'd0;
                        result_send_byte_index <= 6'd0;
                        state                  <= ST_SEND_RESULT_BYTES;
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
                            if (read_global_beat + read_burst_count == OUTPUT_BEATS) begin
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
                    if (tx_index == 7'd24) begin
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
                        7'd2: tx_data <= {3'd0, configured_layer};
                        7'd3: tx_data <= window_start[7:0];
                        7'd4: tx_data <= window_start[15:8];
                        7'd5: tx_data <= {3'd0, window_count};
                        7'd6: tx_data <= v_loaded_count;
                        7'd7: tx_data <= 8'h0d;
                        default: tx_data <= 8'h0a;
                    endcase
                    tx_start <= 1'b1;
                    if (tx_index == 7'd8) begin
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

            ST_SEND_V_ACK: begin
                if (!tx_busy && !tx_start) begin
                    case (tx_index)
                        7'd0: tx_data <= "K";
                        7'd1: tx_data <= v_upload_position[7:0];
                        7'd2: tx_data <= v_upload_position[15:8];
                        7'd3: tx_data <= 8'h0d;
                        default: tx_data <= 8'h0a;
                    endcase
                    tx_start <= 1'b1;
                    if (tx_index == 7'd4) begin
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
