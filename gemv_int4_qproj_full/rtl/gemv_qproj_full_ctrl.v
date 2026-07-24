`timescale 1ns/1ps

// layer0 q_proj 完整真实 Linear 层控制器，固定 M=896、K=896、group_size=64。
//
// UART 协议（115200 8N1）：
//   I -> "PANGU50K QPROJ FULL V1\r\n"
//   S -> 'S' + 状态字节 + "\r\n"
//   L + 固定 488320 字节载荷 -> 逐拍写入 DDR3，回复 "K\r\n"
//   G -> 执行完整层，回复 'R' + 896 个 little-endian signed int64 Q28
//
// 上传载荷：
//   activation_int8                         896 B（28 拍）
//   packed_weight_int4[896][896]         401408 B（每行 14 拍）
//   combined_scale_uq4_28[896][14]        57344 B（每行补齐为 2 拍）
//   bias_q28[896]                          28672 B（每行低 64 位有效，补齐为 1 拍）
//
// 计算期间只缓存完整激活和当前一行权重/scale。每 4 行结果组成一个 256 bit
// 数据拍立即写回 DDR3；返回阶段再从 DDR3 逐拍读取并经 UART 流式发送。
module gemv_qproj_full_ctrl #(
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

localparam integer M = 896;
localparam integer K = 896;
localparam integer ACT_BEATS = 28;
localparam integer WEIGHT_BEATS_PER_ROW = 14;
localparam integer WEIGHT_TOTAL_BEATS = M * WEIGHT_BEATS_PER_ROW;
localparam integer SCALE_BEATS_PER_ROW = 2;
localparam integer SCALE_TOTAL_BEATS = M * SCALE_BEATS_PER_ROW;
localparam integer BIAS_BEATS_PER_ROW = 1;
localparam integer BIAS_TOTAL_BEATS = M * BIAS_BEATS_PER_ROW;
localparam integer RESULT_BEATS = M / 4;
localparam integer LOAD_TOTAL_BEATS =
    ACT_BEATS + WEIGHT_TOTAL_BEATS + SCALE_TOTAL_BEATS + BIAS_TOTAL_BEATS;

// DDR3 控制器地址单位为 32 bit；一个 256 bit 数据拍占 8 个地址单位。
// 各区域均留有对齐间隔，完整层总占用小于 1 MiB。
localparam [CTRL_ADDR_WIDTH-1:0] ADDR_ACT    = 28'h0000000;
localparam [CTRL_ADDR_WIDTH-1:0] ADDR_WEIGHT = 28'h0001000;
localparam [CTRL_ADDR_WIDTH-1:0] ADDR_SCALE  = 28'h0020000;
localparam [CTRL_ADDR_WIDTH-1:0] ADDR_BIAS   = 28'h0024000;
localparam [CTRL_ADDR_WIDTH-1:0] ADDR_RESULT = 28'h0026000;

localparam [4:0] ST_IDLE               = 5'd0;
localparam [4:0] ST_RECV_LOAD          = 5'd1;
localparam [4:0] ST_SETUP_LOAD_WRITE   = 5'd2;
localparam [4:0] ST_WRITE_LOAD         = 5'd3;
localparam [4:0] ST_SETUP_ACT_READ     = 5'd4;
localparam [4:0] ST_READ_ACT           = 5'd5;
localparam [4:0] ST_SETUP_WEIGHT_READ  = 5'd6;
localparam [4:0] ST_READ_WEIGHT        = 5'd7;
localparam [4:0] ST_SETUP_SCALE_READ   = 5'd8;
localparam [4:0] ST_READ_SCALE         = 5'd9;
localparam [4:0] ST_SETUP_BIAS_READ    = 5'd10;
localparam [4:0] ST_READ_BIAS          = 5'd11;
localparam [4:0] ST_START_CORE         = 5'd12;
localparam [4:0] ST_WAIT_CORE          = 5'd13;
localparam [4:0] ST_SETUP_RESULT_WRITE = 5'd14;
localparam [4:0] ST_WRITE_RESULT       = 5'd15;
localparam [4:0] ST_SETUP_RESULT_READ  = 5'd16;
localparam [4:0] ST_READ_RESULT        = 5'd17;
localparam [4:0] ST_SEND_RESULT_PREFIX = 5'd18;
localparam [4:0] ST_SEND_RESULT_BYTES  = 5'd19;
localparam [4:0] ST_SEND_INFO          = 5'd20;
localparam [4:0] ST_SEND_STATUS        = 5'd21;
localparam [4:0] ST_SEND_ACK           = 5'd22;
localparam [4:0] ST_SEND_ERROR         = 5'd23;

reg [4:0] state;
reg [5:0] tx_index;
reg [7:0] tx_data;
reg tx_start;
wire tx_busy;
wire [7:0] rx_data;
wire rx_valid;

reg [5:0] rx_byte_index;
reg [255:0] upload_beat;
reg [13:0] load_beat_index;

reg [4:0] act_read_base_beat;
reg [4:0] active_read_burst_beats;
reg [4:0] read_beat_index;
reg [9:0] row_index;
reg [CTRL_ADDR_WIDTH-1:0] weight_row_addr;
reg [CTRL_ADDR_WIDTH-1:0] scale_row_addr;
reg [CTRL_ADDR_WIDTH-1:0] bias_row_addr;
reg [CTRL_ADDR_WIDTH-1:0] result_write_addr;
reg [255:0] bias_row_cache;
reg [255:0] result_beat_cache;

reg [7:0] result_read_beat_index;
reg [5:0] result_tx_byte_index;
reg [255:0] result_tx_cache;
reg result_prefix_sent;

reg core_start;
wire core_busy;
wire core_done;
wire signed [63:0] core_y_q28;
wire signed [63:0] selected_bias_q28 = $signed(bias_row_cache[63:0]);

reg aw_seen;
reg w_seen;
reg ar_seen;
reg [7:0] status_snapshot;
reg [7:0] error_code;

wire aw_handshake = axi_awvalid && axi_awready;
wire ar_handshake = axi_arvalid && axi_arready;
wire write_data_handshake = axi_wready && (aw_seen || aw_handshake);
wire read_data_handshake = axi_rvalid && (ar_seen || ar_handshake);

wire [4:0] act_beats_remaining = ACT_BEATS - act_read_base_beat;
wire [4:0] next_act_burst_beats =
    (act_beats_remaining > 5'd16) ? 5'd16 : act_beats_remaining;

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

gemv_qproj_full_core u_gemv_qproj_full_core (
    .clk                   (core_clk),
    .rst_n                 (core_rst_n),
    .act_load_en           ((state == ST_READ_ACT) && read_data_handshake),
    .act_load_index        (act_read_base_beat + read_beat_index),
    .act_load_data         (axi_rdata),
    .weight_load_en        ((state == ST_READ_WEIGHT) && read_data_handshake),
    .weight_load_index     (read_beat_index[3:0]),
    .weight_load_data      (axi_rdata),
    .scale_load_en         ((state == ST_READ_SCALE) && read_data_handshake),
    .scale_load_beat_index (read_beat_index[0]),
    .scale_load_data       (axi_rdata),
    .start                 (core_start),
    .bias_q28              (selected_bias_q28),
    .busy                  (core_busy),
    .done                  (core_done),
    .y_q28                 (core_y_q28)
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
            5'd9:  info_char = "Q";
            5'd10: info_char = "P";
            5'd11: info_char = "R";
            5'd12: info_char = "O";
            5'd13: info_char = "J";
            5'd14: info_char = " ";
            5'd15: info_char = "F";
            5'd16: info_char = "U";
            5'd17: info_char = "L";
            5'd18: info_char = "L";
            5'd19: info_char = " ";
            5'd20: info_char = "V";
            5'd21: info_char = "1";
            5'd22: info_char = 8'h0d;
            5'd23: info_char = 8'h0a;
            default: info_char = 8'h00;
        endcase
    end
endfunction

always @(posedge core_clk or negedge core_rst_n) begin
    if (!core_rst_n) begin
        state                   <= ST_IDLE;
        tx_index                <= 6'd0;
        tx_data                 <= 8'h00;
        tx_start                <= 1'b0;
        rx_byte_index           <= 6'd0;
        upload_beat             <= 256'd0;
        load_beat_index         <= 14'd0;
        act_read_base_beat      <= 5'd0;
        active_read_burst_beats <= 5'd0;
        read_beat_index         <= 5'd0;
        row_index               <= 10'd0;
        weight_row_addr         <= ADDR_WEIGHT;
        scale_row_addr          <= ADDR_SCALE;
        bias_row_addr           <= ADDR_BIAS;
        result_write_addr       <= ADDR_RESULT;
        bias_row_cache          <= 256'd0;
        result_beat_cache       <= 256'd0;
        result_read_beat_index  <= 8'd0;
        result_tx_byte_index    <= 6'd0;
        result_tx_cache         <= 256'd0;
        result_prefix_sent      <= 1'b0;
        core_start              <= 1'b0;
        axi_awaddr              <= {CTRL_ADDR_WIDTH{1'b0}};
        axi_awvalid             <= 1'b0;
        axi_wdata               <= 256'd0;
        axi_wstrb               <= 32'd0;
        axi_araddr              <= {CTRL_ADDR_WIDTH{1'b0}};
        axi_arlen               <= 4'd0;
        axi_arvalid             <= 1'b0;
        aw_seen                 <= 1'b0;
        w_seen                  <= 1'b0;
        ar_seen                 <= 1'b0;
        status_snapshot         <= 8'd0;
        error_code              <= 8'd0;
        protocol_error          <= 1'b0;
        loaded                  <= 1'b0;
        result_valid            <= 1'b0;
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
                            state <= ST_SEND_INFO;
                        end

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
                                load_beat_index <= 14'd0;
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
                                result_valid           <= 1'b0;
                                act_read_base_beat      <= 5'd0;
                                read_beat_index         <= 5'd0;
                                row_index               <= 10'd0;
                                weight_row_addr         <= ADDR_WEIGHT;
                                scale_row_addr          <= ADDR_SCALE;
                                bias_row_addr           <= ADDR_BIAS;
                                result_write_addr       <= ADDR_RESULT;
                                result_beat_cache       <= 256'd0;
                                result_read_beat_index  <= 8'd0;
                                result_tx_byte_index    <= 6'd0;
                                result_prefix_sent      <= 1'b0;
                                state                   <= ST_SETUP_ACT_READ;
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
                if (load_beat_index < ACT_BEATS) begin
                    axi_awaddr <= ADDR_ACT + (load_beat_index << 3);
                end else if (load_beat_index < ACT_BEATS + WEIGHT_TOTAL_BEATS) begin
                    axi_awaddr <= ADDR_WEIGHT + ((load_beat_index - ACT_BEATS) << 3);
                end else if (
                    load_beat_index < ACT_BEATS + WEIGHT_TOTAL_BEATS + SCALE_TOTAL_BEATS
                ) begin
                    axi_awaddr <= ADDR_SCALE +
                        ((load_beat_index - ACT_BEATS - WEIGHT_TOTAL_BEATS) << 3);
                end else begin
                    axi_awaddr <= ADDR_BIAS +
                        ((load_beat_index - ACT_BEATS - WEIGHT_TOTAL_BEATS -
                          SCALE_TOTAL_BEATS) << 3);
                end
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

            ST_SETUP_ACT_READ: begin
                axi_araddr              <= ADDR_ACT + (act_read_base_beat << 3);
                axi_arlen               <= next_act_burst_beats - 1'b1;
                axi_arvalid             <= 1'b1;
                ar_seen                 <= 1'b0;
                read_beat_index         <= 5'd0;
                active_read_burst_beats <= next_act_burst_beats;
                state                   <= ST_READ_ACT;
            end

            ST_READ_ACT: begin
                if (ar_handshake) begin
                    axi_arvalid <= 1'b0;
                    ar_seen     <= 1'b1;
                end
                if (read_data_handshake) begin
                    if (read_beat_index + 1'b1 == active_read_burst_beats) begin
                        ar_seen <= 1'b0;
                        if (act_read_base_beat + active_read_burst_beats == ACT_BEATS)
                            state <= ST_SETUP_WEIGHT_READ;
                        else begin
                            act_read_base_beat <=
                                act_read_base_beat + active_read_burst_beats;
                            state <= ST_SETUP_ACT_READ;
                        end
                    end else begin
                        read_beat_index <= read_beat_index + 1'b1;
                    end
                end
            end

            ST_SETUP_WEIGHT_READ: begin
                axi_araddr      <= weight_row_addr;
                axi_arlen       <= WEIGHT_BEATS_PER_ROW - 1'b1;
                axi_arvalid     <= 1'b1;
                ar_seen         <= 1'b0;
                read_beat_index <= 5'd0;
                state           <= ST_READ_WEIGHT;
            end

            ST_READ_WEIGHT: begin
                if (ar_handshake) begin
                    axi_arvalid <= 1'b0;
                    ar_seen     <= 1'b1;
                end
                if (read_data_handshake) begin
                    if (read_beat_index + 1'b1 == WEIGHT_BEATS_PER_ROW) begin
                        ar_seen <= 1'b0;
                        state   <= ST_SETUP_SCALE_READ;
                    end else begin
                        read_beat_index <= read_beat_index + 1'b1;
                    end
                end
            end

            ST_SETUP_SCALE_READ: begin
                axi_araddr      <= scale_row_addr;
                axi_arlen       <= SCALE_BEATS_PER_ROW - 1'b1;
                axi_arvalid     <= 1'b1;
                ar_seen         <= 1'b0;
                read_beat_index <= 5'd0;
                state           <= ST_READ_SCALE;
            end

            ST_READ_SCALE: begin
                if (ar_handshake) begin
                    axi_arvalid <= 1'b0;
                    ar_seen     <= 1'b1;
                end
                if (read_data_handshake) begin
                    if (read_beat_index + 1'b1 == SCALE_BEATS_PER_ROW) begin
                        ar_seen <= 1'b0;
                        state   <= ST_SETUP_BIAS_READ;
                    end else begin
                        read_beat_index <= read_beat_index + 1'b1;
                    end
                end
            end

            ST_SETUP_BIAS_READ: begin
                axi_araddr      <= bias_row_addr;
                axi_arlen       <= 4'd0;
                axi_arvalid     <= 1'b1;
                ar_seen         <= 1'b0;
                read_beat_index <= 5'd0;
                state           <= ST_READ_BIAS;
            end

            ST_READ_BIAS: begin
                if (ar_handshake) begin
                    axi_arvalid <= 1'b0;
                    ar_seen     <= 1'b1;
                end
                if (read_data_handshake) begin
                    bias_row_cache <= axi_rdata;
                    ar_seen        <= 1'b0;
                    state          <= ST_START_CORE;
                end
            end

            ST_START_CORE: begin
                core_start <= 1'b1;
                state      <= ST_WAIT_CORE;
            end

            ST_WAIT_CORE: begin
                if (core_done) begin
                    result_beat_cache[row_index[1:0]*64 +: 64] <= core_y_q28;
                    if (row_index[1:0] == 2'd3) begin
                        state <= ST_SETUP_RESULT_WRITE;
                    end else begin
                        row_index       <= row_index + 1'b1;
                        weight_row_addr <= weight_row_addr +
                            (WEIGHT_BEATS_PER_ROW << 3);
                        scale_row_addr  <= scale_row_addr +
                            (SCALE_BEATS_PER_ROW << 3);
                        bias_row_addr   <= bias_row_addr + 8;
                        state           <= ST_SETUP_WEIGHT_READ;
                    end
                end
            end

            ST_SETUP_RESULT_WRITE: begin
                axi_awaddr  <= result_write_addr;
                axi_awvalid <= 1'b1;
                axi_wdata   <= result_beat_cache;
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
                    axi_awvalid       <= 1'b0;
                    aw_seen           <= 1'b0;
                    w_seen            <= 1'b0;
                    result_beat_cache <= 256'd0;
                    if (row_index + 1'b1 == M) begin
                        result_valid           <= 1'b1;
                        result_read_beat_index  <= 8'd0;
                        result_tx_byte_index    <= 6'd0;
                        result_prefix_sent      <= 1'b0;
                        state                   <= ST_SETUP_RESULT_READ;
                    end else begin
                        row_index        <= row_index + 1'b1;
                        weight_row_addr  <= weight_row_addr +
                            (WEIGHT_BEATS_PER_ROW << 3);
                        scale_row_addr   <= scale_row_addr +
                            (SCALE_BEATS_PER_ROW << 3);
                        bias_row_addr    <= bias_row_addr + 8;
                        result_write_addr<= result_write_addr + 8;
                        state            <= ST_SETUP_WEIGHT_READ;
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
                        if (result_read_beat_index + 1'b1 == RESULT_BEATS) begin
                            state <= ST_IDLE;
                        end else begin
                            result_read_beat_index <= result_read_beat_index + 1'b1;
                            state                  <= ST_SETUP_RESULT_READ;
                        end
                    end else begin
                        result_tx_byte_index <= result_tx_byte_index + 1'b1;
                    end
                end
            end

            ST_SEND_INFO: begin
                if (!tx_busy && !tx_start) begin
                    if (tx_index < 6'd24) begin
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

endmodule
