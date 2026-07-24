`timescale 1ns/1ps

// F3 KV Cache 独立验证控制器。
//
// DDR3 地址布局（控制器地址单位为 32 bit）：
//   K = 0x02000000 + layer * 0x00800000 + position * 0x00000200
//   V = K + 0x00000100
//
// 每个 token 槽为 2048 B：K=[2,64] signed int64 Q28（1024 B）后接
// V=[2,64] signed int64 Q28（1024 B）。每层 16384 token、32 MiB，
// 28 层共 896 MiB；低端 128 MiB 保留，最后一个槽恰好结束于 1 GiB。
//
// UART 协议（115200 8N1）：
//   I -> "PANGU50K KV CACHE V1\r\n"
//   S -> 'S' + flags + layer + start_u16 + current_u16 + written_u16 + CRLF
//   C + layer_u8 + start_u16 -> 配置层和起始位置，回复 "K\r\n"
//   W + 2048 B K/V -> 写入 current_position，回复
//        'K' + layer + written_position_u16 + CRLF，并将 current_position 自动加 1
//   R + start_u16 + count_u8 -> 历史顺序读取，回复
//        'D' + layer + start_u16 + count_u8 + count*2048 B
//   Z -> current_position 复位到配置起点、written_count 清零，回复 "K\r\n"
module kv_cache_ctrl #(
    parameter integer CTRL_ADDR_WIDTH = 28,
    parameter integer CLKS_PER_BIT    = 868,
    parameter integer NUM_LAYERS      = 28,
    parameter integer MAX_CONTEXT     = 16384,
    parameter integer MAX_READ_TOKENS = 16
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
    output wire [4:0]                   debug_layer,
    output wire [15:0]                  debug_position,
    output reg                          protocol_error,
    output reg                          configured,
    output reg                          write_valid,
    output reg                          read_valid
);

localparam integer TOKEN_BEATS = 64;
localparam [CTRL_ADDR_WIDTH-1:0] KV_BASE_CTRL = 28'h2000000;

localparam [5:0] ST_IDLE               = 6'd0;
localparam [5:0] ST_RECV_CONFIG        = 6'd1;
localparam [5:0] ST_APPLY_CONFIG       = 6'd2;
localparam [5:0] ST_RECV_WRITE         = 6'd3;
localparam [5:0] ST_SETUP_WRITE        = 6'd4;
localparam [5:0] ST_WRITE              = 6'd5;
localparam [5:0] ST_RECV_READ_REQ      = 6'd6;
localparam [5:0] ST_APPLY_READ_REQ     = 6'd7;
localparam [5:0] ST_SEND_READ_HEADER   = 6'd8;
localparam [5:0] ST_SETUP_READ_BURST   = 6'd9;
localparam [5:0] ST_READ_BURST         = 6'd10;
localparam [5:0] ST_SEND_READ_BYTES    = 6'd11;
localparam [5:0] ST_SEND_INFO          = 6'd12;
localparam [5:0] ST_SEND_STATUS        = 6'd13;
localparam [5:0] ST_SEND_ACK           = 6'd14;
localparam [5:0] ST_SEND_WRITE_ACK     = 6'd15;
localparam [5:0] ST_SEND_ERROR         = 6'd16;

reg [5:0] state;
wire [7:0] uart_rx_data;
wire uart_rx_valid;
reg [7:0] tx_data;
reg tx_start;
wire tx_busy;
reg [5:0] tx_index;

reg [23:0] config_word;
reg [1:0] config_byte_index;
reg [4:0] configured_layer;
reg [15:0] start_position;
reg [15:0] current_position;
reg [15:0] written_count;
reg [15:0] written_position;

reg [5:0] rx_byte_index;
reg [255:0] upload_beat;
reg [5:0] write_beat_index;

reg [23:0] read_req_word;
reg [1:0] read_req_byte_index;
reg [15:0] read_start_position;
reg [4:0] read_count;
reg [CTRL_ADDR_WIDTH-1:0] read_base_ctrl;
reg [10:0] read_total_beats;
reg [10:0] read_global_beat;
reg [4:0] read_burst_count;
reg [4:0] read_capture_index;
reg [4:0] read_send_beat_index;
reg [5:0] read_send_byte_index;
reg [255:0] read_buffer [0:15];

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

wire context_full = (current_position >= MAX_CONTEXT);
wire any_written = (written_count != 16'd0);
wire [16:0] config_start_ext = {1'b0, config_word[23:8]};
wire [16:0] read_request_end = {1'b0, read_req_word[15:0]} +
                               {9'd0, read_req_word[23:16]};
wire [10:0] read_remaining_beats = read_total_beats - read_global_beat;
wire [4:0] next_burst_count =
    (read_remaining_beats > 11'd16) ? 5'd16 : read_remaining_beats[4:0];
wire [255:0] selected_read_beat = read_buffer[read_send_beat_index];

wire [CTRL_ADDR_WIDTH-1:0] configured_layer_offset =
    ({23'd0, configured_layer} << 23);
wire [CTRL_ADDR_WIDTH-1:0] current_position_offset =
    ({12'd0, current_position} << 9);
wire [CTRL_ADDR_WIDTH-1:0] current_slot_base_ctrl =
    KV_BASE_CTRL + configured_layer_offset + current_position_offset;

assign axi_awuser_ap = 1'b0;
assign axi_awuser_id = 4'h0;
assign axi_awlen = 4'h0;
assign axi_aruser_ap = 1'b0;
assign axi_aruser_id = 4'h0;
assign debug_state = state;
assign debug_layer = configured_layer;
assign debug_position = current_position;

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
            5'd9:  info_char = "K";
            5'd10: info_char = "V";
            5'd11: info_char = " ";
            5'd12: info_char = "C";
            5'd13: info_char = "A";
            5'd14: info_char = "C";
            5'd15: info_char = "H";
            5'd16: info_char = "E";
            5'd17: info_char = " ";
            5'd18: info_char = "V";
            5'd19: info_char = "1";
            5'd20: info_char = 8'h0d;
            5'd21: info_char = 8'h0a;
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

always @(posedge core_clk or negedge core_rst_n) begin
    if (!core_rst_n) begin
        state                  <= ST_IDLE;
        tx_data                <= 8'd0;
        tx_start               <= 1'b0;
        tx_index               <= 6'd0;
        config_word            <= 24'd0;
        config_byte_index      <= 2'd0;
        configured_layer       <= 5'd0;
        start_position         <= 16'd0;
        current_position       <= 16'd0;
        written_count          <= 16'd0;
        written_position       <= 16'd0;
        rx_byte_index          <= 6'd0;
        upload_beat            <= 256'd0;
        write_beat_index       <= 6'd0;
        read_req_word          <= 24'd0;
        read_req_byte_index    <= 2'd0;
        read_start_position    <= 16'd0;
        read_count             <= 5'd0;
        read_base_ctrl         <= {CTRL_ADDR_WIDTH{1'b0}};
        read_total_beats       <= 11'd0;
        read_global_beat       <= 11'd0;
        read_burst_count       <= 5'd0;
        read_capture_index     <= 5'd0;
        read_send_beat_index   <= 5'd0;
        read_send_byte_index   <= 6'd0;
        axi_awaddr             <= {CTRL_ADDR_WIDTH{1'b0}};
        axi_awvalid            <= 1'b0;
        axi_wdata              <= 256'd0;
        axi_wstrb              <= 32'd0;
        axi_araddr             <= {CTRL_ADDR_WIDTH{1'b0}};
        axi_arlen              <= 4'd0;
        axi_arvalid            <= 1'b0;
        aw_seen                <= 1'b0;
        w_seen                 <= 1'b0;
        ar_seen                <= 1'b0;
        status_snapshot        <= 8'd0;
        error_code             <= 8'd0;
        protocol_error         <= 1'b0;
        configured             <= 1'b0;
        write_valid            <= 1'b0;
        read_valid             <= 1'b0;
        for (clear_index = 0; clear_index < 16; clear_index = clear_index + 1)
            read_buffer[clear_index] <= 256'd0;
    end else begin
        tx_start <= 1'b0;

        case (state)
            ST_IDLE: begin
                axi_awvalid <= 1'b0;
                axi_arvalid <= 1'b0;
                aw_seen     <= 1'b0;
                w_seen      <= 1'b0;
                ar_seen     <= 1'b0;
                tx_index    <= 6'd0;

                if (uart_rx_valid && !tx_busy) begin
                    case (uart_rx_data)
                        8'h49, 8'h69: begin // I / i
                            state <= ST_SEND_INFO;
                        end

                        8'h53, 8'h73: begin // S / s
                            status_snapshot <= {
                                1'b0,
                                protocol_error,
                                context_full,
                                1'b0,
                                read_valid,
                                write_valid,
                                configured,
                                ddr_init_done
                            };
                            state <= ST_SEND_STATUS;
                        end

                        8'h43, 8'h63: begin // C / c
                            config_word       <= 24'd0;
                            config_byte_index <= 2'd0;
                            state             <= ST_RECV_CONFIG;
                        end

                        8'h57, 8'h77: begin // W / w
                            if (!ddr_init_done) begin
                                protocol_error <= 1'b1;
                                error_code     <= 8'h02;
                                state          <= ST_SEND_ERROR;
                            end else if (!configured) begin
                                protocol_error <= 1'b1;
                                error_code     <= 8'h03;
                                state          <= ST_SEND_ERROR;
                            end else if (context_full) begin
                                protocol_error <= 1'b1;
                                error_code     <= 8'h05;
                                state          <= ST_SEND_ERROR;
                            end else begin
                                written_position <= current_position;
                                upload_beat      <= 256'd0;
                                rx_byte_index    <= 6'd0;
                                write_beat_index <= 6'd0;
                                write_valid      <= 1'b0;
                                read_valid       <= 1'b0;
                                state            <= ST_RECV_WRITE;
                            end
                        end

                        8'h52, 8'h72: begin // R / r
                            if (!ddr_init_done) begin
                                protocol_error <= 1'b1;
                                error_code     <= 8'h02;
                                state          <= ST_SEND_ERROR;
                            end else if (!configured) begin
                                protocol_error <= 1'b1;
                                error_code     <= 8'h03;
                                state          <= ST_SEND_ERROR;
                            end else begin
                                read_req_word       <= 24'd0;
                                read_req_byte_index <= 2'd0;
                                read_valid          <= 1'b0;
                                state               <= ST_RECV_READ_REQ;
                            end
                        end

                        8'h5a, 8'h7a: begin // Z / z
                            if (!configured) begin
                                protocol_error <= 1'b1;
                                error_code     <= 8'h03;
                                state          <= ST_SEND_ERROR;
                            end else begin
                                current_position <= start_position;
                                written_count    <= 16'd0;
                                write_valid      <= 1'b0;
                                read_valid       <= 1'b0;
                                state            <= ST_SEND_ACK;
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
                    if (config_byte_index == 2'd2)
                        state <= ST_APPLY_CONFIG;
                    else
                        config_byte_index <= config_byte_index + 1'b1;
                end
            end

            ST_APPLY_CONFIG: begin
                if ((config_word[7:0] >= NUM_LAYERS) ||
                    (config_start_ext >= MAX_CONTEXT)) begin
                    protocol_error <= 1'b1;
                    error_code     <= 8'h04;
                    state          <= ST_SEND_ERROR;
                end else begin
                    configured_layer <= config_word[4:0];
                    start_position   <= config_word[23:8];
                    current_position <= config_word[23:8];
                    written_count    <= 16'd0;
                    configured       <= 1'b1;
                    write_valid      <= 1'b0;
                    read_valid       <= 1'b0;
                    state            <= ST_SEND_ACK;
                end
            end

            ST_RECV_WRITE: begin
                if (uart_rx_valid) begin
                    upload_beat[rx_byte_index*8 +: 8] <= uart_rx_data;
                    if (rx_byte_index == 6'd31)
                        state <= ST_SETUP_WRITE;
                    else
                        rx_byte_index <= rx_byte_index + 1'b1;
                end
            end

            ST_SETUP_WRITE: begin
                axi_awaddr  <= current_slot_base_ctrl + ({22'd0, write_beat_index} << 3);
                axi_awvalid <= 1'b1;
                axi_wdata   <= upload_beat;
                axi_wstrb   <= 32'hffff_ffff;
                aw_seen     <= 1'b0;
                w_seen      <= 1'b0;
                state       <= ST_WRITE;
            end

            ST_WRITE: begin
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
                    if (write_beat_index == TOKEN_BEATS - 1) begin
                        current_position <= current_position + 1'b1;
                        written_count    <= written_count + 1'b1;
                        write_valid      <= 1'b1;
                        state            <= ST_SEND_WRITE_ACK;
                    end else begin
                        write_beat_index <= write_beat_index + 1'b1;
                        rx_byte_index    <= 6'd0;
                        upload_beat     <= 256'd0;
                        state           <= ST_RECV_WRITE;
                    end
                end
            end

            ST_RECV_READ_REQ: begin
                if (uart_rx_valid) begin
                    read_req_word[read_req_byte_index*8 +: 8] <= uart_rx_data;
                    if (read_req_byte_index == 2'd2)
                        state <= ST_APPLY_READ_REQ;
                    else
                        read_req_byte_index <= read_req_byte_index + 1'b1;
                end
            end

            ST_APPLY_READ_REQ: begin
                if ((read_req_word[23:16] == 8'd0) ||
                    (read_req_word[23:16] > MAX_READ_TOKENS) ||
                    (read_request_end > MAX_CONTEXT)) begin
                    protocol_error <= 1'b1;
                    error_code     <= 8'h06;
                    state          <= ST_SEND_ERROR;
                end else begin
                    read_start_position <= read_req_word[15:0];
                    read_count          <= read_req_word[20:16];
                    read_base_ctrl      <= KV_BASE_CTRL + configured_layer_offset +
                                           ({12'd0, read_req_word[15:0]} << 9);
                    read_total_beats    <= ({6'd0, read_req_word[20:16]} << 6);
                    read_global_beat    <= 11'd0;
                    read_capture_index <= 5'd0;
                    read_send_beat_index <= 5'd0;
                    read_send_byte_index <= 6'd0;
                    tx_index            <= 6'd0;
                    state               <= ST_SEND_READ_HEADER;
                end
            end

            ST_SEND_READ_HEADER: begin
                if (!tx_busy && !tx_start) begin
                    case (tx_index)
                        6'd0: tx_data <= "D";
                        6'd1: tx_data <= {3'd0, configured_layer};
                        6'd2: tx_data <= read_start_position[7:0];
                        6'd3: tx_data <= read_start_position[15:8];
                        default: tx_data <= {3'd0, read_count};
                    endcase
                    tx_start <= 1'b1;
                    if (tx_index == 6'd4) begin
                        tx_index <= 6'd0;
                        state    <= ST_SETUP_READ_BURST;
                    end else begin
                        tx_index <= tx_index + 1'b1;
                    end
                end
            end

            ST_SETUP_READ_BURST: begin
                read_burst_count   <= next_burst_count;
                read_capture_index <= 5'd0;
                axi_araddr         <= read_base_ctrl + ({17'd0, read_global_beat} << 3);
                axi_arlen          <= next_burst_count - 1'b1;
                axi_arvalid        <= 1'b1;
                ar_seen            <= 1'b0;
                state              <= ST_READ_BURST;
            end

            ST_READ_BURST: begin
                if (ar_handshake) begin
                    axi_arvalid <= 1'b0;
                    ar_seen     <= 1'b1;
                end
                if (read_data_handshake) begin
                    read_buffer[read_capture_index] <= axi_rdata;
                    if (read_capture_index == read_burst_count - 1'b1) begin
                        ar_seen                  <= 1'b0;
                        read_send_beat_index     <= 5'd0;
                        read_send_byte_index     <= 6'd0;
                        state                    <= ST_SEND_READ_BYTES;
                    end else begin
                        read_capture_index <= read_capture_index + 1'b1;
                    end
                end
            end

            ST_SEND_READ_BYTES: begin
                if (!tx_busy && !tx_start) begin
                    tx_data  <= selected_read_beat[read_send_byte_index*8 +: 8];
                    tx_start <= 1'b1;
                    if (read_send_byte_index == 6'd31) begin
                        read_send_byte_index <= 6'd0;
                        if (read_send_beat_index == read_burst_count - 1'b1) begin
                            read_send_beat_index <= 5'd0;
                            if (read_global_beat + read_burst_count == read_total_beats) begin
                                read_valid <= 1'b1;
                                state      <= ST_IDLE;
                            end else begin
                                read_global_beat <= read_global_beat + read_burst_count;
                                state            <= ST_SETUP_READ_BURST;
                            end
                        end else begin
                            read_send_beat_index <= read_send_beat_index + 1'b1;
                        end
                    end else begin
                        read_send_byte_index <= read_send_byte_index + 1'b1;
                    end
                end
            end

            ST_SEND_INFO: begin
                if (!tx_busy && !tx_start) begin
                    tx_data  <= info_char(tx_index[4:0]);
                    tx_start <= 1'b1;
                    if (tx_index == 6'd21) begin
                        tx_index <= 6'd0;
                        state    <= ST_IDLE;
                    end else begin
                        tx_index <= tx_index + 1'b1;
                    end
                end
            end

            ST_SEND_STATUS: begin
                if (!tx_busy && !tx_start) begin
                    case (tx_index)
                        6'd0: tx_data <= "S";
                        6'd1: tx_data <= status_snapshot;
                        6'd2: tx_data <= {3'd0, configured_layer};
                        6'd3: tx_data <= start_position[7:0];
                        6'd4: tx_data <= start_position[15:8];
                        6'd5: tx_data <= current_position[7:0];
                        6'd6: tx_data <= current_position[15:8];
                        6'd7: tx_data <= written_count[7:0];
                        6'd8: tx_data <= written_count[15:8];
                        6'd9: tx_data <= 8'h0d;
                        default: tx_data <= 8'h0a;
                    endcase
                    tx_start <= 1'b1;
                    if (tx_index == 6'd10) begin
                        tx_index <= 6'd0;
                        state    <= ST_IDLE;
                    end else begin
                        tx_index <= tx_index + 1'b1;
                    end
                end
            end

            ST_SEND_ACK: begin
                if (!tx_busy && !tx_start) begin
                    case (tx_index)
                        6'd0: tx_data <= "K";
                        6'd1: tx_data <= 8'h0d;
                        default: tx_data <= 8'h0a;
                    endcase
                    tx_start <= 1'b1;
                    if (tx_index == 6'd2) begin
                        tx_index <= 6'd0;
                        state    <= ST_IDLE;
                    end else begin
                        tx_index <= tx_index + 1'b1;
                    end
                end
            end

            ST_SEND_WRITE_ACK: begin
                if (!tx_busy && !tx_start) begin
                    case (tx_index)
                        6'd0: tx_data <= "K";
                        6'd1: tx_data <= {3'd0, configured_layer};
                        6'd2: tx_data <= written_position[7:0];
                        6'd3: tx_data <= written_position[15:8];
                        6'd4: tx_data <= 8'h0d;
                        default: tx_data <= 8'h0a;
                    endcase
                    tx_start <= 1'b1;
                    if (tx_index == 6'd5) begin
                        tx_index <= 6'd0;
                        state    <= ST_IDLE;
                    end else begin
                        tx_index <= tx_index + 1'b1;
                    end
                end
            end

            ST_SEND_ERROR: begin
                if (!tx_busy && !tx_start) begin
                    case (tx_index)
                        6'd0: tx_data <= "E";
                        6'd1: tx_data <= error_code;
                        6'd2: tx_data <= 8'h0d;
                        default: tx_data <= 8'h0a;
                    endcase
                    tx_start <= 1'b1;
                    if (tx_index == 6'd3) begin
                        tx_index <= 6'd0;
                        state    <= ST_IDLE;
                    end else begin
                        tx_index <= tx_index + 1'b1;
                    end
                end
            end

            default: begin
                state       <= ST_IDLE;
                axi_awvalid <= 1'b0;
                axi_arvalid <= 1'b0;
                protocol_error <= 1'b1;
                error_code  <= 8'hff;
            end
        endcase
    end
end

endmodule
