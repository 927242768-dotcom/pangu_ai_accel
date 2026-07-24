`timescale 1ns/1ps

// F4 Attention Score 独立验证控制器。
//
// 低端 DDR3：
//   Q      : ctrl 0x0000000，7168 B / 224 beats，布局 [14,64] int64 Q28
//   scores : ctrl 0x0000800，1792 B / 56 beats，布局 [14,16] int64 Q28
//
// K 完全复用 F3 KV Cache 高端地址：
//   K = 0x02000000 + layer * 0x00800000 + position * 0x00000200
//   每个 K 为 [2,64] int64 Q28，共 1024 B / 32 beats。
//
// UART 协议（115200 8N1）：
//   I -> "PANGU50K ATTN SCORE V1\r\n"
//   S -> 'S' + flags + layer + query_u16 + start_u16 + count + k_loaded + CRLF
//   C + layer_u8 + query_u16 + start_u16 + count_u8 -> 配置，回复 "K\r\n"
//   Q + 7168 B -> 上传当前 query 的 RoPE 后 Q，回复 "K\r\n"
//   K + position_u16 + 1024 B -> 写入 F3 K Cache 地址，回复
//        'K' + position_u16 + CRLF
//   G -> 读取 Q/K、计算 14x16 固定 score、写回 DDR3，回复 "K\r\n"
//   R -> 回复 'D' + layer + query_u16 + start_u16 + count_u8 + 1792 B scores
//
// score 定义：Q/K signed Q28；64 维点积为 Q56；RNE 右移 31 位，等价于
// 先乘 1/sqrt(64)=1/8 再转回 signed Q28。未来位置和未使用槽输出 INT64_MIN。
module attention_score_ctrl #(
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
    output wire [15:0]                  debug_query_position,
    output reg                          protocol_error,
    output reg                          configured,
    output reg                          q_loaded,
    output reg                          result_valid
);

localparam integer Q_BEATS     = 224;
localparam integer K_BEATS     = 32;
localparam integer SCORE_BEATS = 56;
localparam [CTRL_ADDR_WIDTH-1:0] Q_BASE_CTRL     = 28'h0000000;
localparam [CTRL_ADDR_WIDTH-1:0] SCORE_BASE_CTRL = 28'h0000800;
localparam [CTRL_ADDR_WIDTH-1:0] KV_BASE_CTRL    = 28'h2000000;

localparam [6:0] ST_IDLE                = 7'd0;
localparam [6:0] ST_RECV_CONFIG         = 7'd1;
localparam [6:0] ST_APPLY_CONFIG        = 7'd2;
localparam [6:0] ST_RECV_Q              = 7'd3;
localparam [6:0] ST_SETUP_Q_WRITE       = 7'd4;
localparam [6:0] ST_Q_WRITE             = 7'd5;
localparam [6:0] ST_RECV_K_POSITION     = 7'd6;
localparam [6:0] ST_APPLY_K_POSITION    = 7'd7;
localparam [6:0] ST_RECV_K              = 7'd8;
localparam [6:0] ST_SETUP_K_WRITE       = 7'd9;
localparam [6:0] ST_K_WRITE             = 7'd10;
localparam [6:0] ST_COMPUTE_INIT        = 7'd11;
localparam [6:0] ST_SETUP_Q_READ        = 7'd12;
localparam [6:0] ST_Q_READ              = 7'd13;
localparam [6:0] ST_TOKEN_DISPATCH      = 7'd14;
localparam [6:0] ST_SETUP_K_READ        = 7'd15;
localparam [6:0] ST_K_READ              = 7'd16;
localparam [6:0] ST_CORE_START          = 7'd17;
localparam [6:0] ST_CORE_WAIT           = 7'd18;
localparam [6:0] ST_SETUP_MASK_WRITE    = 7'd19;
localparam [6:0] ST_MASK_WRITE          = 7'd20;
localparam [6:0] ST_SEND_RESULT_HEADER  = 7'd21;
localparam [6:0] ST_SETUP_RESULT_READ   = 7'd22;
localparam [6:0] ST_RESULT_READ         = 7'd23;
localparam [6:0] ST_SEND_RESULT_BYTES   = 7'd24;
localparam [6:0] ST_SEND_INFO           = 7'd25;
localparam [6:0] ST_SEND_STATUS         = 7'd26;
localparam [6:0] ST_SEND_ACK            = 7'd27;
localparam [6:0] ST_SEND_K_ACK          = 7'd28;
localparam [6:0] ST_SEND_ERROR          = 7'd29;
localparam [6:0] ST_SETUP_SCORE_WRITE   = 7'd30;
localparam [6:0] ST_SCORE_WRITE         = 7'd31;

reg [6:0] state;
wire [7:0] uart_rx_data;
wire uart_rx_valid;
reg [7:0] tx_data;
reg tx_start;
wire tx_busy;
reg [6:0] tx_index;

reg [47:0] config_word;
reg [2:0] config_byte_index;
reg [4:0] configured_layer;
reg [15:0] query_position;
reg [15:0] window_start;
reg [4:0] window_count;
reg [7:0] k_loaded_count;

reg [5:0] rx_byte_index;
reg [255:0] upload_beat;
reg [7:0] q_write_beat;
reg [4:0] k_write_beat;
reg [15:0] k_position_word;
reg k_position_byte_index;
reg [15:0] k_upload_position;

reg [8:0] read_global_beat;
reg [4:0] read_burst_count;
reg [4:0] read_capture_index;
reg [5:0] result_send_beat_index;
reg [5:0] result_send_byte_index;
reg [255:0] read_buffer [0:15];

reg [3:0] token_index;
reg [15:0] token_position;
reg [5:0] mask_write_beat;
reg [3:0] pending_score_head;
reg signed [63:0] pending_score_q28;

reg core_q_beat_we;
reg [7:0] core_q_beat_index;
reg [255:0] core_q_beat_data;
reg core_k_beat_we;
reg [4:0] core_k_beat_index;
reg [255:0] core_k_beat_data;
reg core_start_token;
reg core_token_masked;
reg core_score_ready;
wire core_busy;
wire core_score_valid;
wire [3:0] core_score_head;
wire signed [63:0] core_score_q28;
wire core_token_done;

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

wire [16:0] config_query_ext = {1'b0, config_word[23:8]};
wire [16:0] config_start_ext = {1'b0, config_word[39:24]};
wire [17:0] config_window_end = {1'b0, config_word[39:24]} +
                                {10'd0, config_word[47:40]};
wire [16:0] k_position_ext = {1'b0, k_position_word};
wire [8:0] q_read_remaining = Q_BEATS - read_global_beat;
wire [8:0] k_read_remaining = K_BEATS - read_global_beat;
wire [6:0] result_read_remaining = SCORE_BEATS - read_global_beat[6:0];
wire [4:0] next_q_burst =
    (q_read_remaining > 9'd16) ? 5'd16 : q_read_remaining[4:0];
wire [4:0] next_k_burst =
    (k_read_remaining > 9'd16) ? 5'd16 : k_read_remaining[4:0];
wire [4:0] next_result_burst =
    (result_read_remaining > 7'd16) ? 5'd16 : result_read_remaining[4:0];
wire [255:0] selected_result_beat = read_buffer[result_send_beat_index];

wire [CTRL_ADDR_WIDTH-1:0] configured_layer_offset =
    ({23'd0, configured_layer} << 23);
wire [CTRL_ADDR_WIDTH-1:0] k_upload_position_offset =
    ({12'd0, k_upload_position} << 9);
wire [CTRL_ADDR_WIDTH-1:0] current_token_offset =
    ({12'd0, token_position} << 9);
wire current_token_masked = (token_position > query_position);
wire [5:0] pending_score_beat =
    ({2'd0, pending_score_head} << 2) + {4'd0, token_index[3:2]};

assign axi_awuser_ap = 1'b0;
assign axi_awuser_id = 4'h0;
assign axi_awlen = 4'h0;
assign axi_aruser_ap = 1'b0;
assign axi_aruser_id = 4'h0;
assign debug_state = state;
assign debug_layer = configured_layer;
assign debug_query_position = query_position;

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
            5'd14: info_char = "S";
            5'd15: info_char = "C";
            5'd16: info_char = "O";
            5'd17: info_char = "R";
            5'd18: info_char = "E";
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

attention_score_core u_attention_score_core (
    .clk             (core_clk),
    .rst_n           (core_rst_n),
    .q_beat_we       (core_q_beat_we),
    .q_beat_index    (core_q_beat_index),
    .q_beat_data     (core_q_beat_data),
    .k_beat_we       (core_k_beat_we),
    .k_beat_index    (core_k_beat_index),
    .k_beat_data     (core_k_beat_data),
    .start_token     (core_start_token),
    .token_masked    (core_token_masked),
    .score_ready     (core_score_ready),
    .busy            (core_busy),
    .score_valid     (core_score_valid),
    .score_head      (core_score_head),
    .score_q28       (core_score_q28),
    .token_done      (core_token_done)
);

always @(posedge core_clk or negedge core_rst_n) begin
    if (!core_rst_n) begin
        state                    <= ST_IDLE;
        tx_data                  <= 8'd0;
        tx_start                 <= 1'b0;
        tx_index                 <= 7'd0;
        config_word              <= 48'd0;
        config_byte_index        <= 3'd0;
        configured_layer         <= 5'd0;
        query_position           <= 16'd0;
        window_start             <= 16'd0;
        window_count             <= 5'd0;
        k_loaded_count           <= 8'd0;
        rx_byte_index            <= 6'd0;
        upload_beat              <= 256'd0;
        q_write_beat             <= 8'd0;
        k_write_beat             <= 5'd0;
        k_position_word          <= 16'd0;
        k_position_byte_index    <= 1'b0;
        k_upload_position        <= 16'd0;
        read_global_beat         <= 9'd0;
        read_burst_count         <= 5'd0;
        read_capture_index       <= 5'd0;
        result_send_beat_index   <= 6'd0;
        result_send_byte_index   <= 6'd0;
        token_index              <= 4'd0;
        token_position           <= 16'd0;
        mask_write_beat          <= 6'd0;
        pending_score_head       <= 4'd0;
        pending_score_q28        <= 64'sd0;
        core_q_beat_we           <= 1'b0;
        core_q_beat_index        <= 8'd0;
        core_q_beat_data         <= 256'd0;
        core_k_beat_we           <= 1'b0;
        core_k_beat_index        <= 5'd0;
        core_k_beat_data         <= 256'd0;
        core_start_token         <= 1'b0;
        core_token_masked        <= 1'b0;
        core_score_ready         <= 1'b0;
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
        configured              <= 1'b0;
        q_loaded                <= 1'b0;
        result_valid            <= 1'b0;
        for (clear_index = 0; clear_index < 16; clear_index = clear_index + 1)
            read_buffer[clear_index] <= 256'd0;
    end else begin
        tx_start          <= 1'b0;
        core_q_beat_we    <= 1'b0;
        core_k_beat_we    <= 1'b0;
        core_start_token  <= 1'b0;
        core_score_ready  <= 1'b0;

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
                                (k_loaded_count != 8'd0),
                                q_loaded,
                                configured,
                                ddr_init_done
                            };
                            state <= ST_SEND_STATUS;
                        end

                        8'h43, 8'h63: begin // C/c
                            config_word       <= 48'd0;
                            config_byte_index <= 3'd0;
                            state             <= ST_RECV_CONFIG;
                        end

                        8'h51, 8'h71: begin // Q/q
                            if (!ddr_init_done) begin
                                protocol_error <= 1'b1;
                                error_code     <= 8'h02;
                                state          <= ST_SEND_ERROR;
                            end else if (!configured) begin
                                protocol_error <= 1'b1;
                                error_code     <= 8'h03;
                                state          <= ST_SEND_ERROR;
                            end else begin
                                upload_beat   <= 256'd0;
                                rx_byte_index <= 6'd0;
                                q_write_beat  <= 8'd0;
                                q_loaded      <= 1'b0;
                                result_valid  <= 1'b0;
                                state         <= ST_RECV_Q;
                            end
                        end

                        8'h4b, 8'h6b: begin // K/k
                            if (!ddr_init_done) begin
                                protocol_error <= 1'b1;
                                error_code     <= 8'h02;
                                state          <= ST_SEND_ERROR;
                            end else if (!configured) begin
                                protocol_error <= 1'b1;
                                error_code     <= 8'h03;
                                state          <= ST_SEND_ERROR;
                            end else begin
                                k_position_word       <= 16'd0;
                                k_position_byte_index <= 1'b0;
                                state                 <= ST_RECV_K_POSITION;
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
                            end else if (!q_loaded) begin
                                protocol_error <= 1'b1;
                                error_code     <= 8'h05;
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
                    if (config_byte_index == 3'd5)
                        state <= ST_APPLY_CONFIG;
                    else
                        config_byte_index <= config_byte_index + 1'b1;
                end
            end

            ST_APPLY_CONFIG: begin
                if ((config_word[7:0] >= NUM_LAYERS) ||
                    (config_query_ext >= MAX_CONTEXT) ||
                    (config_start_ext >= MAX_CONTEXT) ||
                    (config_word[47:40] == 8'd0) ||
                    (config_word[47:40] > MAX_TOKENS) ||
                    (config_window_end > MAX_CONTEXT)) begin
                    protocol_error <= 1'b1;
                    error_code     <= 8'h04;
                    state          <= ST_SEND_ERROR;
                end else begin
                    configured_layer <= config_word[4:0];
                    query_position   <= config_word[23:8];
                    window_start     <= config_word[39:24];
                    window_count     <= config_word[44:40];
                    k_loaded_count   <= 8'd0;
                    configured      <= 1'b1;
                    q_loaded        <= 1'b0;
                    result_valid    <= 1'b0;
                    state           <= ST_SEND_ACK;
                end
            end

            ST_RECV_Q: begin
                if (uart_rx_valid) begin
                    upload_beat[rx_byte_index*8 +: 8] <= uart_rx_data;
                    if (rx_byte_index == 6'd31)
                        state <= ST_SETUP_Q_WRITE;
                    else
                        rx_byte_index <= rx_byte_index + 1'b1;
                end
            end

            ST_SETUP_Q_WRITE: begin
                axi_awaddr  <= Q_BASE_CTRL + ({20'd0, q_write_beat} << 3);
                axi_awvalid <= 1'b1;
                axi_wdata   <= upload_beat;
                axi_wstrb   <= 32'hffff_ffff;
                aw_seen     <= 1'b0;
                w_seen      <= 1'b0;
                state       <= ST_Q_WRITE;
            end

            ST_Q_WRITE: begin
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
                    if (q_write_beat == Q_BEATS - 1) begin
                        q_loaded <= 1'b1;
                        state    <= ST_SEND_ACK;
                    end else begin
                        q_write_beat <= q_write_beat + 1'b1;
                        rx_byte_index <= 6'd0;
                        upload_beat  <= 256'd0;
                        state        <= ST_RECV_Q;
                    end
                end
            end

            ST_RECV_K_POSITION: begin
                if (uart_rx_valid) begin
                    k_position_word[k_position_byte_index*8 +: 8] <= uart_rx_data;
                    if (k_position_byte_index)
                        state <= ST_APPLY_K_POSITION;
                    else
                        k_position_byte_index <= 1'b1;
                end
            end

            ST_APPLY_K_POSITION: begin
                if (k_position_ext >= MAX_CONTEXT) begin
                    protocol_error <= 1'b1;
                    error_code     <= 8'h06;
                    state          <= ST_SEND_ERROR;
                end else begin
                    k_upload_position <= k_position_word;
                    upload_beat       <= 256'd0;
                    rx_byte_index     <= 6'd0;
                    k_write_beat      <= 5'd0;
                    result_valid      <= 1'b0;
                    state             <= ST_RECV_K;
                end
            end

            ST_RECV_K: begin
                if (uart_rx_valid) begin
                    upload_beat[rx_byte_index*8 +: 8] <= uart_rx_data;
                    if (rx_byte_index == 6'd31)
                        state <= ST_SETUP_K_WRITE;
                    else
                        rx_byte_index <= rx_byte_index + 1'b1;
                end
            end

            ST_SETUP_K_WRITE: begin
                axi_awaddr <= KV_BASE_CTRL + configured_layer_offset +
                              k_upload_position_offset +
                              ({23'd0, k_write_beat} << 3);
                axi_awvalid <= 1'b1;
                axi_wdata   <= upload_beat;
                axi_wstrb   <= 32'hffff_ffff;
                aw_seen     <= 1'b0;
                w_seen      <= 1'b0;
                state       <= ST_K_WRITE;
            end

            ST_K_WRITE: begin
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
                    if (k_write_beat == K_BEATS - 1) begin
                        if (k_loaded_count != 8'hff)
                            k_loaded_count <= k_loaded_count + 1'b1;
                        state <= ST_SEND_K_ACK;
                    end else begin
                        k_write_beat <= k_write_beat + 1'b1;
                        rx_byte_index <= 6'd0;
                        upload_beat  <= 256'd0;
                        state        <= ST_RECV_K;
                    end
                end
            end

            ST_COMPUTE_INIT: begin
                mask_write_beat    <= 6'd0;
                read_global_beat   <= 9'd0;
                read_capture_index <= 5'd0;
                token_index        <= 4'd0;
                token_position     <= window_start;
                state              <= ST_SETUP_MASK_WRITE;
            end

            ST_SETUP_MASK_WRITE: begin
                axi_awaddr  <= SCORE_BASE_CTRL + ({22'd0, mask_write_beat} << 3);
                axi_awvalid <= 1'b1;
                axi_wdata   <= {4{64'h8000_0000_0000_0000}};
                axi_wstrb   <= 32'hffff_ffff;
                aw_seen     <= 1'b0;
                w_seen      <= 1'b0;
                state       <= ST_MASK_WRITE;
            end

            ST_MASK_WRITE: begin
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
                    if (mask_write_beat == SCORE_BEATS - 1) begin
                        read_global_beat   <= 9'd0;
                        read_capture_index <= 5'd0;
                        state              <= ST_SETUP_Q_READ;
                    end else begin
                        mask_write_beat <= mask_write_beat + 1'b1;
                        state           <= ST_SETUP_MASK_WRITE;
                    end
                end
            end

            ST_SETUP_Q_READ: begin
                read_burst_count   <= next_q_burst;
                read_capture_index <= 5'd0;
                axi_araddr         <= Q_BASE_CTRL + ({19'd0, read_global_beat} << 3);
                axi_arlen          <= next_q_burst - 1'b1;
                axi_arvalid        <= 1'b1;
                ar_seen            <= 1'b0;
                state              <= ST_Q_READ;
            end

            ST_Q_READ: begin
                if (ar_handshake) begin
                    axi_arvalid <= 1'b0;
                    ar_seen     <= 1'b1;
                end
                if (read_data_handshake) begin
                    core_q_beat_we    <= 1'b1;
                    core_q_beat_index <= read_global_beat[7:0] + read_capture_index;
                    core_q_beat_data  <= axi_rdata;
                    if (read_capture_index == read_burst_count - 1'b1) begin
                        ar_seen <= 1'b0;
                        if (read_global_beat + read_burst_count == Q_BEATS) begin
                            token_index    <= 4'd0;
                            token_position <= window_start;
                            state          <= ST_TOKEN_DISPATCH;
                        end else begin
                            read_global_beat <= read_global_beat + read_burst_count;
                            state            <= ST_SETUP_Q_READ;
                        end
                    end else begin
                        read_capture_index <= read_capture_index + 1'b1;
                    end
                end
            end

            ST_TOKEN_DISPATCH: begin
                token_position <= window_start + {12'd0, token_index};
                if ((window_start + {12'd0, token_index}) > query_position) begin
                    core_token_masked <= 1'b1;
                    core_start_token  <= 1'b1;
                    state             <= ST_CORE_WAIT;
                end else begin
                    read_global_beat   <= 9'd0;
                    read_capture_index <= 5'd0;
                    state              <= ST_SETUP_K_READ;
                end
            end

            ST_SETUP_K_READ: begin
                read_burst_count   <= next_k_burst;
                read_capture_index <= 5'd0;
                axi_araddr <= KV_BASE_CTRL + configured_layer_offset +
                              current_token_offset +
                              ({19'd0, read_global_beat} << 3);
                axi_arlen   <= next_k_burst - 1'b1;
                axi_arvalid <= 1'b1;
                ar_seen     <= 1'b0;
                state       <= ST_K_READ;
            end

            ST_K_READ: begin
                if (ar_handshake) begin
                    axi_arvalid <= 1'b0;
                    ar_seen     <= 1'b1;
                end
                if (read_data_handshake) begin
                    core_k_beat_we    <= 1'b1;
                    core_k_beat_index <= read_global_beat[4:0] + read_capture_index;
                    core_k_beat_data  <= axi_rdata;
                    if (read_capture_index == read_burst_count - 1'b1) begin
                        ar_seen <= 1'b0;
                        if (read_global_beat + read_burst_count == K_BEATS) begin
                            state <= ST_CORE_START;
                        end else begin
                            read_global_beat <= read_global_beat + read_burst_count;
                            state            <= ST_SETUP_K_READ;
                        end
                    end else begin
                        read_capture_index <= read_capture_index + 1'b1;
                    end
                end
            end

            ST_CORE_START: begin
                core_token_masked <= 1'b0;
                core_start_token  <= 1'b1;
                state             <= ST_CORE_WAIT;
            end

            ST_CORE_WAIT: begin
                if (core_score_valid) begin
                    pending_score_head <= core_score_head;
                    pending_score_q28  <= core_score_q28;
                    state              <= ST_SETUP_SCORE_WRITE;
                end else if (core_token_done) begin
                    if (token_index == window_count - 1'b1) begin
                        result_valid <= 1'b1;
                        state        <= ST_SEND_ACK;
                    end else begin
                        token_index    <= token_index + 1'b1;
                        token_position <= window_start + {12'd0, token_index} + 1'b1;
                        state          <= ST_TOKEN_DISPATCH;
                    end
                end
            end

            ST_SETUP_SCORE_WRITE: begin
                axi_awaddr  <= SCORE_BASE_CTRL + ({22'd0, pending_score_beat} << 3);
                axi_awvalid <= 1'b1;
                case (token_index[1:0])
                    2'd0: begin
                        axi_wdata <= {192'd0, pending_score_q28};
                        axi_wstrb <= 32'h0000_00ff;
                    end
                    2'd1: begin
                        axi_wdata <= {128'd0, pending_score_q28, 64'd0};
                        axi_wstrb <= 32'h0000_ff00;
                    end
                    2'd2: begin
                        axi_wdata <= {64'd0, pending_score_q28, 128'd0};
                        axi_wstrb <= 32'h00ff_0000;
                    end
                    default: begin
                        axi_wdata <= {pending_score_q28, 192'd0};
                        axi_wstrb <= 32'hff00_0000;
                    end
                endcase
                aw_seen <= 1'b0;
                w_seen  <= 1'b0;
                state   <= ST_SCORE_WRITE;
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
                    axi_awvalid     <= 1'b0;
                    aw_seen         <= 1'b0;
                    w_seen          <= 1'b0;
                    core_score_ready <= 1'b1;
                    state           <= ST_CORE_WAIT;
                end
            end

            ST_SEND_RESULT_HEADER: begin
                if (!tx_busy && !tx_start) begin
                    case (tx_index)
                        7'd0: tx_data <= "D";
                        7'd1: tx_data <= {3'd0, configured_layer};
                        7'd2: tx_data <= query_position[7:0];
                        7'd3: tx_data <= query_position[15:8];
                        7'd4: tx_data <= window_start[7:0];
                        7'd5: tx_data <= window_start[15:8];
                        default: tx_data <= {3'd0, window_count};
                    endcase
                    tx_start <= 1'b1;
                    if (tx_index == 7'd6) begin
                        read_global_beat       <= 9'd0;
                        read_capture_index     <= 5'd0;
                        result_send_beat_index <= 6'd0;
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
                axi_araddr         <= SCORE_BASE_CTRL + ({19'd0, read_global_beat} << 3);
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
                        ar_seen                    <= 1'b0;
                        result_send_beat_index     <= 6'd0;
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
                            result_send_beat_index <= 6'd0;
                            if (read_global_beat + read_burst_count == SCORE_BEATS) begin
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
                        7'd2: tx_data <= {3'd0, configured_layer};
                        7'd3: tx_data <= query_position[7:0];
                        7'd4: tx_data <= query_position[15:8];
                        7'd5: tx_data <= window_start[7:0];
                        7'd6: tx_data <= window_start[15:8];
                        7'd7: tx_data <= {3'd0, window_count};
                        7'd8: tx_data <= k_loaded_count;
                        7'd9: tx_data <= 8'h0d;
                        default: tx_data <= 8'h0a;
                    endcase
                    tx_start <= 1'b1;
                    if (tx_index == 7'd10) begin
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

            ST_SEND_K_ACK: begin
                if (!tx_busy && !tx_start) begin
                    case (tx_index)
                        7'd0: tx_data <= "K";
                        7'd1: tx_data <= k_upload_position[7:0];
                        7'd2: tx_data <= k_upload_position[15:8];
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
