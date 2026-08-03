`timescale 1ns/1ps

`include "transformer_block_contract.vh"

// G2 完整 Block UART/DDR3 主机控制器。
//
// UART 115200 8N1，所有整数均 little-endian：
//   I                         -> "PANGU50K G2 BLOCK V1\r\n"
//   S                         -> 'S' + flags + stage + error + "\r\n"
//   C + <4H>                  -> 配置 layer/query_position/window_start/count
//   W + <2I> + raw bytes      -> DDR3 写；header=(controller_addr, byte_length)
//   R + <2I>                  -> 'R' + <I byte_length> + raw bytes
//   P                         -> 确认本轮载荷完整
//   G                         -> 启动 22 阶段；完成后返回
//                                'D' + success + error + stage + watchdog<4B> + CRLF
//
// W/R 地址使用 DDR3 Controller 的 32-bit 单位；地址必须 256-bit beat 对齐，
// byte_length 必须是 32 的倍数。运行期间 DDR3 只归 transformer_block_ctrl，
// 主机不能注入中间结果。
module transformer_block_host_ctrl #(
    parameter integer CTRL_ADDR_WIDTH = 28,
    parameter integer CLKS_PER_BIT    = 868
)(
    input  wire                         core_clk,
    input  wire                         core_rst_n,
    input  wire                         ddr_init_done,
    input  wire                         uart_rx_i,
    output wire                         uart_tx_o,

    output wire [CTRL_ADDR_WIDTH-1:0]   axi_awaddr,
    output wire                         axi_awuser_ap,
    output wire [3:0]                   axi_awuser_id,
    output wire [3:0]                   axi_awlen,
    input  wire                         axi_awready,
    output wire                         axi_awvalid,
    output wire [255:0]                 axi_wdata,
    output wire [31:0]                  axi_wstrb,
    input  wire                         axi_wready,

    output wire [CTRL_ADDR_WIDTH-1:0]   axi_araddr,
    output wire                         axi_aruser_ap,
    output wire [3:0]                   axi_aruser_id,
    output wire [3:0]                   axi_arlen,
    input  wire                         axi_arready,
    output wire                         axi_arvalid,
    input  wire [255:0]                 axi_rdata,
    input  wire                         axi_rvalid,

    output wire [6:0]                   debug_state,
    output reg                          protocol_error,
    output reg                          configured,
    output reg                          payload_committed,
    output reg                          result_valid,
    output wire                         block_busy,
    output wire [4:0]                   block_stage,
    output wire [7:0]                   block_error_code
);

localparam [6:0] ST_IDLE              = 7'd0;
localparam [6:0] ST_RECV_CONFIG       = 7'd1;
localparam [6:0] ST_APPLY_CONFIG      = 7'd2;
localparam [6:0] ST_RECV_WRITE_HEADER = 7'd3;
localparam [6:0] ST_CHECK_WRITE       = 7'd4;
localparam [6:0] ST_RECV_WRITE_DATA   = 7'd5;
localparam [6:0] ST_SETUP_WRITE       = 7'd6;
localparam [6:0] ST_WAIT_WRITE        = 7'd7;
localparam [6:0] ST_RECV_READ_HEADER  = 7'd8;
localparam [6:0] ST_CHECK_READ        = 7'd9;
localparam [6:0] ST_SEND_READ_PREFIX  = 7'd10;
localparam [6:0] ST_SEND_READ_LENGTH  = 7'd11;
localparam [6:0] ST_SETUP_READ        = 7'd12;
localparam [6:0] ST_WAIT_READ         = 7'd13;
localparam [6:0] ST_SEND_READ_DATA    = 7'd14;
localparam [6:0] ST_START_BLOCK       = 7'd15;
localparam [6:0] ST_WAIT_BLOCK        = 7'd16;
localparam [6:0] ST_SEND_COMPLETION   = 7'd17;
localparam [6:0] ST_SEND_INFO         = 7'd18;
localparam [6:0] ST_SEND_STATUS       = 7'd19;
localparam [6:0] ST_SEND_ACK          = 7'd20;
localparam [6:0] ST_SEND_ERROR        = 7'd21;

localparam [7:0] ERR_COMMAND          = 8'h01;
localparam [7:0] ERR_DDR_NOT_READY    = 8'h02;
localparam [7:0] ERR_NOT_CONFIGURED   = 8'h03;
localparam [7:0] ERR_NOT_COMMITTED    = 8'h04;
localparam [7:0] ERR_CONFIG           = 8'h10;
localparam [7:0] ERR_ADDRESS          = 8'h11;
localparam [7:0] ERR_LENGTH           = 8'h12;
localparam [7:0] ERR_BLOCK_BASE       = 8'h40;
localparam [7:0] ERR_INTERNAL         = 8'hff;

reg [6:0] state;
reg [7:0] tx_data;
reg tx_start;
wire tx_busy;
wire [7:0] rx_data;
wire rx_valid;
reg [5:0] tx_index;
reg [7:0] error_code;
reg [7:0] status_snapshot;

reg [63:0] config_buffer;
reg [3:0] config_byte_index;
reg [4:0] cfg_layer;
reg [14:0] cfg_query_position;
reg [14:0] cfg_window_start;
reg [4:0] cfg_count;

reg [63:0] command_header;
reg [3:0] header_byte_index;
reg [CTRL_ADDR_WIDTH-1:0] transfer_base_addr;
reg [31:0] transfer_byte_length;
reg [26:0] transfer_total_beats;
reg [26:0] transfer_beat_index;
reg [5:0] transfer_byte_index;
reg [255:0] transfer_beat;
reg any_write_seen;

reg [CTRL_ADDR_WIDTH-1:0] host_awaddr;
reg host_awvalid;
reg [255:0] host_wdata;
reg [31:0] host_wstrb;
reg host_aw_seen;
reg host_w_seen;
reg [CTRL_ADDR_WIDTH-1:0] host_araddr;
reg [3:0] host_arlen;
reg host_arvalid;
reg host_ar_seen;

reg block_start;
wire block_done;
wire block_error;
wire [31:0] block_watchdog_count;
wire [3:0] block_axi_master;
reg [31:0] block_cycle_count;
reg completion_success;
reg [7:0] completion_error_code;
reg [4:0] completion_stage;
reg [31:0] completion_cycle_count;

wire [CTRL_ADDR_WIDTH-1:0] block_awaddr;
wire block_awuser_ap;
wire [3:0] block_awuser_id;
wire [3:0] block_awlen;
wire block_awvalid;
wire [255:0] block_wdata;
wire [31:0] block_wstrb;
wire [CTRL_ADDR_WIDTH-1:0] block_araddr;
wire block_aruser_ap;
wire [3:0] block_aruser_id;
wire [3:0] block_arlen;
wire block_arvalid;

wire [15:0] new_layer          = config_buffer[15:0];
wire [15:0] new_query_position = config_buffer[31:16];
wire [15:0] new_window_start   = config_buffer[47:32];
wire [15:0] new_count          = config_buffer[63:48];
wire [16:0] new_expected_query =
    {1'b0, new_window_start} + {1'b0, new_count} - 17'd1;
wire new_config_valid =
    (new_layer == 16'd0) &&
    (new_count >= 16'd1) && (new_count <= 16'd16) &&
    (new_query_position < 16'd16384) &&
    (new_window_start < 16'd16384) &&
    ({1'b0, new_query_position} == new_expected_query);

wire [31:0] new_transfer_addr = command_header[31:0];
wire [31:0] new_transfer_length = command_header[63:32];
wire [31:0] new_transfer_ctrl_words = new_transfer_length >> 2;
wire [32:0] new_transfer_end =
    {1'b0, new_transfer_addr} + {1'b0, new_transfer_ctrl_words};
wire new_transfer_addr_valid =
    (new_transfer_addr[31:CTRL_ADDR_WIDTH] == 0) &&
    (new_transfer_addr[2:0] == 3'd0) &&
    (new_transfer_end <= (33'd1 << CTRL_ADDR_WIDTH));
wire new_transfer_length_valid =
    (new_transfer_length != 32'd0) &&
    (new_transfer_length[4:0] == 5'd0);

wire block_bus_active =
    (state == ST_START_BLOCK) || (state == ST_WAIT_BLOCK) || block_busy;
wire host_aw_handshake = !block_bus_active && host_awvalid && axi_awready;
wire host_write_handshake =
    !block_bus_active && axi_wready && (host_aw_seen || host_aw_handshake);
wire host_ar_handshake = !block_bus_active && host_arvalid && axi_arready;
wire host_read_handshake =
    !block_bus_active && axi_rvalid && (host_ar_seen || host_ar_handshake);

assign axi_awaddr    = block_bus_active ? block_awaddr : host_awaddr;
assign axi_awuser_ap = block_bus_active ? block_awuser_ap : 1'b0;
assign axi_awuser_id = block_bus_active ? block_awuser_id : 4'd0;
assign axi_awlen     = block_bus_active ? block_awlen : 4'd0;
assign axi_awvalid   = block_bus_active ? block_awvalid : host_awvalid;
assign axi_wdata     = block_bus_active ? block_wdata : host_wdata;
assign axi_wstrb     = block_bus_active ? block_wstrb : host_wstrb;
assign axi_araddr    = block_bus_active ? block_araddr : host_araddr;
assign axi_aruser_ap = block_bus_active ? block_aruser_ap : 1'b0;
assign axi_aruser_id = block_bus_active ? block_aruser_id : 4'd0;
assign axi_arlen     = block_bus_active ? block_arlen : host_arlen;
assign axi_arvalid   = block_bus_active ? block_arvalid : host_arvalid;
assign debug_state   = state;

uart_rx #(
    .CLKS_PER_BIT (CLKS_PER_BIT)
) u_uart_rx (
    .clk   (core_clk),
    .rst_n (core_rst_n),
    .rx    (uart_rx_i),
    .data  (rx_data),
    .valid (rx_valid)
);

uart_tx #(
    .CLKS_PER_BIT (CLKS_PER_BIT)
) u_uart_tx (
    .clk   (core_clk),
    .rst_n (core_rst_n),
    .data  (tx_data),
    .start (tx_start),
    .tx    (uart_tx_o),
    .busy  (tx_busy)
);

transformer_block_ctrl #(
    .CTRL_ADDR_WIDTH (CTRL_ADDR_WIDTH),
    // 每个阶段独立允许 5 秒（100 MHz 下 5 亿拍），超时由 scheduler
    // 记录准确 stage/error；禁止在完整 Block 验收版本中关闭 watchdog。
    .WATCHDOG_CYCLES (32'd500000000)
) u_transformer_block_ctrl (
    .clk               (core_clk),
    .rst_n             (core_rst_n),
    .ddr_init_done     (ddr_init_done),
    .start             (block_start),
    .cfg_layer         (cfg_layer),
    .cfg_query_position(cfg_query_position),
    .cfg_window_start  (cfg_window_start),
    .cfg_count         (cfg_count),
    .axi_awaddr        (block_awaddr),
    .axi_awuser_ap     (block_awuser_ap),
    .axi_awuser_id     (block_awuser_id),
    .axi_awlen         (block_awlen),
    .axi_awready       (block_bus_active ? axi_awready : 1'b0),
    .axi_awvalid       (block_awvalid),
    .axi_wdata         (block_wdata),
    .axi_wstrb         (block_wstrb),
    .axi_wready        (block_bus_active ? axi_wready : 1'b0),
    .axi_araddr        (block_araddr),
    .axi_aruser_ap     (block_aruser_ap),
    .axi_aruser_id     (block_aruser_id),
    .axi_arlen         (block_arlen),
    .axi_arready       (block_bus_active ? axi_arready : 1'b0),
    .axi_arvalid       (block_arvalid),
    .axi_rdata         (axi_rdata),
    .axi_rvalid        (block_bus_active ? axi_rvalid : 1'b0),
    .busy              (block_busy),
    .done              (block_done),
    .error             (block_error),
    .error_code        (block_error_code),
    .current_stage     (block_stage),
    .watchdog_count    (block_watchdog_count),
    .debug_axi_master  (block_axi_master)
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
            5'd9:  info_char = "G";
            5'd10: info_char = "2";
            5'd11: info_char = " ";
            5'd12: info_char = "B";
            5'd13: info_char = "L";
            5'd14: info_char = "O";
            5'd15: info_char = "C";
            5'd16: info_char = "K";
            5'd17: info_char = " ";
            5'd18: info_char = "V";
            5'd19: info_char = "1";
            5'd20: info_char = 8'h0d;
            5'd21: info_char = 8'h0a;
            default: info_char = 8'h00;
        endcase
    end
endfunction

function [7:0] completion_byte;
    input [3:0] index;
    begin
        case (index)
            4'd0: completion_byte = "D";
            4'd1: completion_byte = {7'd0, completion_success};
            4'd2: completion_byte = completion_error_code;
            4'd3: completion_byte = {3'd0, completion_stage};
            4'd4: completion_byte = completion_cycle_count[7:0];
            4'd5: completion_byte = completion_cycle_count[15:8];
            4'd6: completion_byte = completion_cycle_count[23:16];
            4'd7: completion_byte = completion_cycle_count[31:24];
            4'd8: completion_byte = 8'h0d;
            default: completion_byte = 8'h0a;
        endcase
    end
endfunction

always @(posedge core_clk or negedge core_rst_n) begin
    if (!core_rst_n) begin
        state                <= ST_IDLE;
        tx_data              <= 8'd0;
        tx_start             <= 1'b0;
        tx_index             <= 6'd0;
        error_code           <= 8'd0;
        status_snapshot      <= 8'd0;
        config_buffer        <= 64'd0;
        config_byte_index    <= 4'd0;
        cfg_layer            <= 5'd0;
        cfg_query_position   <= 15'd0;
        cfg_window_start     <= 15'd0;
        cfg_count            <= 5'd0;
        command_header       <= 64'd0;
        header_byte_index    <= 4'd0;
        transfer_base_addr   <= {CTRL_ADDR_WIDTH{1'b0}};
        transfer_byte_length <= 32'd0;
        transfer_total_beats <= 27'd0;
        transfer_beat_index  <= 27'd0;
        transfer_byte_index  <= 6'd0;
        transfer_beat        <= 256'd0;
        any_write_seen       <= 1'b0;
        host_awaddr          <= {CTRL_ADDR_WIDTH{1'b0}};
        host_awvalid         <= 1'b0;
        host_wdata           <= 256'd0;
        host_wstrb           <= 32'd0;
        host_aw_seen         <= 1'b0;
        host_w_seen          <= 1'b0;
        host_araddr          <= {CTRL_ADDR_WIDTH{1'b0}};
        host_arlen           <= 4'd0;
        host_arvalid         <= 1'b0;
        host_ar_seen         <= 1'b0;
        block_start          <= 1'b0;
        block_cycle_count    <= 32'd0;
        completion_success   <= 1'b0;
        completion_error_code <= 8'd0;
        completion_stage     <= `G2_STAGE_IDLE;
        completion_cycle_count <= 32'd0;
        protocol_error       <= 1'b0;
        configured          <= 1'b0;
        payload_committed    <= 1'b0;
        result_valid         <= 1'b0;
    end else begin
        tx_start    <= 1'b0;
        block_start <= 1'b0;

        case (state)
            ST_IDLE: begin
                host_awvalid <= 1'b0;
                host_arvalid <= 1'b0;
                host_aw_seen <= 1'b0;
                host_w_seen  <= 1'b0;
                host_ar_seen <= 1'b0;
                tx_index     <= 6'd0;
                if (rx_valid) begin
                    case (rx_data)
                        8'h49, 8'h69: state <= ST_SEND_INFO;

                        8'h53, 8'h73: begin
                            status_snapshot <= {
                                protocol_error,
                                block_error,
                                block_busy,
                                result_valid,
                                payload_committed,
                                any_write_seen,
                                configured,
                                ddr_init_done
                            };
                            state <= ST_SEND_STATUS;
                        end

                        8'h43, 8'h63: begin
                            config_buffer     <= 64'd0;
                            config_byte_index <= 4'd0;
                            result_valid      <= 1'b0;
                            payload_committed <= 1'b0;
                            any_write_seen    <= 1'b0;
                            state             <= ST_RECV_CONFIG;
                        end

                        8'h57, 8'h77: begin
                            if (!ddr_init_done) begin
                                error_code     <= ERR_DDR_NOT_READY;
                                protocol_error <= 1'b1;
                                state          <= ST_SEND_ERROR;
                            end else begin
                                command_header    <= 64'd0;
                                header_byte_index <= 4'd0;
                                // 首个写 beat 在命令入口提前清零，避免地址合法性判定
                                // 直接扇出控制 256 位 transfer_beat 寄存器。
                                transfer_beat     <= 256'd0;
                                state             <= ST_RECV_WRITE_HEADER;
                            end
                        end

                        8'h52, 8'h72: begin
                            if (!ddr_init_done) begin
                                error_code     <= ERR_DDR_NOT_READY;
                                protocol_error <= 1'b1;
                                state          <= ST_SEND_ERROR;
                            end else begin
                                command_header    <= 64'd0;
                                header_byte_index <= 4'd0;
                                state             <= ST_RECV_READ_HEADER;
                            end
                        end

                        8'h50, 8'h70: begin
                            if (!any_write_seen) begin
                                error_code     <= ERR_NOT_COMMITTED;
                                protocol_error <= 1'b1;
                                state          <= ST_SEND_ERROR;
                            end else begin
                                payload_committed <= 1'b1;
                                protocol_error    <= 1'b0;
                                error_code        <= 8'd0;
                                state             <= ST_SEND_ACK;
                            end
                        end

                        8'h47, 8'h67: begin
                            if (!ddr_init_done) begin
                                error_code     <= ERR_DDR_NOT_READY;
                                protocol_error <= 1'b1;
                                state          <= ST_SEND_ERROR;
                            end else if (!configured) begin
                                error_code     <= ERR_NOT_CONFIGURED;
                                protocol_error <= 1'b1;
                                state          <= ST_SEND_ERROR;
                            end else if (!payload_committed) begin
                                error_code     <= ERR_NOT_COMMITTED;
                                protocol_error <= 1'b1;
                                state          <= ST_SEND_ERROR;
                            end else begin
                                result_valid <= 1'b0;
                                state        <= ST_START_BLOCK;
                            end
                        end

                        default: begin
                            error_code     <= ERR_COMMAND;
                            protocol_error <= 1'b1;
                            state          <= ST_SEND_ERROR;
                        end
                    endcase
                end
            end

            ST_RECV_CONFIG: begin
                if (rx_valid) begin
                    config_buffer[config_byte_index*8 +: 8] <= rx_data;
                    if (config_byte_index == 4'd7)
                        state <= ST_APPLY_CONFIG;
                    else
                        config_byte_index <= config_byte_index + 1'b1;
                end
            end

            ST_APPLY_CONFIG: begin
                if (!new_config_valid) begin
                    configured       <= 1'b0;
                    payload_committed <= 1'b0;
                    protocol_error    <= 1'b1;
                    error_code        <= ERR_CONFIG;
                    state             <= ST_SEND_ERROR;
                end else begin
                    cfg_layer          <= new_layer[4:0];
                    cfg_query_position <= new_query_position[14:0];
                    cfg_window_start   <= new_window_start[14:0];
                    cfg_count          <= new_count[4:0];
                    configured         <= 1'b1;
                    protocol_error     <= 1'b0;
                    error_code         <= 8'd0;
                    state              <= ST_SEND_ACK;
                end
            end

            ST_RECV_WRITE_HEADER: begin
                if (rx_valid) begin
                    command_header[header_byte_index*8 +: 8] <= rx_data;
                    if (header_byte_index == 4'd7)
                        state <= ST_CHECK_WRITE;
                    else
                        header_byte_index <= header_byte_index + 1'b1;
                end
            end

            ST_CHECK_WRITE: begin
                if (!new_transfer_addr_valid) begin
                    error_code     <= ERR_ADDRESS;
                    protocol_error <= 1'b1;
                    state          <= ST_SEND_ERROR;
                end else if (!new_transfer_length_valid) begin
                    error_code     <= ERR_LENGTH;
                    protocol_error <= 1'b1;
                    state          <= ST_SEND_ERROR;
                end else begin
                    transfer_base_addr   <= new_transfer_addr[CTRL_ADDR_WIDTH-1:0];
                    transfer_byte_length <= new_transfer_length;
                    transfer_total_beats <= new_transfer_length[31:5];
                    transfer_beat_index  <= 27'd0;
                    transfer_byte_index  <= 6'd0;
                    payload_committed    <= 1'b0;
                    state                <= ST_RECV_WRITE_DATA;
                end
            end

            ST_RECV_WRITE_DATA: begin
                if (rx_valid) begin
                    transfer_beat[transfer_byte_index*8 +: 8] <= rx_data;
                    if (transfer_byte_index == 6'd31)
                        state <= ST_SETUP_WRITE;
                    else
                        transfer_byte_index <= transfer_byte_index + 1'b1;
                end
            end

            ST_SETUP_WRITE: begin
                host_awaddr  <= transfer_base_addr + ({1'b0, transfer_beat_index} << 3);
                host_awvalid <= 1'b1;
                host_wdata   <= transfer_beat;
                host_wstrb   <= 32'hffff_ffff;
                host_aw_seen <= 1'b0;
                host_w_seen  <= 1'b0;
                state        <= ST_WAIT_WRITE;
            end

            ST_WAIT_WRITE: begin
                if (host_aw_handshake) begin
                    host_awvalid <= 1'b0;
                    host_aw_seen <= 1'b1;
                end
                if (host_write_handshake)
                    host_w_seen <= 1'b1;

                if ((host_aw_seen || host_aw_handshake) &&
                    (host_w_seen || host_write_handshake)) begin
                    host_awvalid <= 1'b0;
                    host_aw_seen <= 1'b0;
                    host_w_seen  <= 1'b0;
                    if (transfer_beat_index + 1'b1 == transfer_total_beats) begin
                        any_write_seen    <= 1'b1;
                        protocol_error    <= 1'b0;
                        error_code        <= 8'd0;
                        state             <= ST_SEND_ACK;
                    end else begin
                        transfer_beat_index <= transfer_beat_index + 1'b1;
                        transfer_byte_index <= 6'd0;
                        transfer_beat       <= 256'd0;
                        state               <= ST_RECV_WRITE_DATA;
                    end
                end
            end

            ST_RECV_READ_HEADER: begin
                if (rx_valid) begin
                    command_header[header_byte_index*8 +: 8] <= rx_data;
                    if (header_byte_index == 4'd7)
                        state <= ST_CHECK_READ;
                    else
                        header_byte_index <= header_byte_index + 1'b1;
                end
            end

            ST_CHECK_READ: begin
                if (!new_transfer_addr_valid) begin
                    error_code     <= ERR_ADDRESS;
                    protocol_error <= 1'b1;
                    state          <= ST_SEND_ERROR;
                end else if (!new_transfer_length_valid) begin
                    error_code     <= ERR_LENGTH;
                    protocol_error <= 1'b1;
                    state          <= ST_SEND_ERROR;
                end else begin
                    transfer_base_addr   <= new_transfer_addr[CTRL_ADDR_WIDTH-1:0];
                    transfer_byte_length <= new_transfer_length;
                    transfer_total_beats <= new_transfer_length[31:5];
                    transfer_beat_index  <= 27'd0;
                    transfer_byte_index  <= 6'd0;
                    tx_index             <= 6'd0;
                    state                <= ST_SEND_READ_PREFIX;
                end
            end

            ST_SEND_READ_PREFIX: begin
                if (!tx_busy && !tx_start) begin
                    tx_data  <= "R";
                    tx_start <= 1'b1;
                    tx_index <= 6'd0;
                    state    <= ST_SEND_READ_LENGTH;
                end
            end

            ST_SEND_READ_LENGTH: begin
                if (!tx_busy && !tx_start) begin
                    case (tx_index[1:0])
                        2'd0: tx_data <= transfer_byte_length[7:0];
                        2'd1: tx_data <= transfer_byte_length[15:8];
                        2'd2: tx_data <= transfer_byte_length[23:16];
                        default: tx_data <= transfer_byte_length[31:24];
                    endcase
                    tx_start <= 1'b1;
                    if (tx_index == 6'd3) begin
                        transfer_beat_index <= 27'd0;
                        state               <= ST_SETUP_READ;
                    end else begin
                        tx_index <= tx_index + 1'b1;
                    end
                end
            end

            ST_SETUP_READ: begin
                host_araddr  <= transfer_base_addr + ({1'b0, transfer_beat_index} << 3);
                host_arlen   <= 4'd0;
                host_arvalid <= 1'b1;
                host_ar_seen <= 1'b0;
                state        <= ST_WAIT_READ;
            end

            ST_WAIT_READ: begin
                if (host_ar_handshake) begin
                    host_arvalid <= 1'b0;
                    host_ar_seen <= 1'b1;
                end
                if (host_read_handshake) begin
                    transfer_beat       <= axi_rdata;
                    transfer_byte_index <= 6'd0;
                    host_ar_seen        <= 1'b0;
                    state               <= ST_SEND_READ_DATA;
                end
            end

            ST_SEND_READ_DATA: begin
                if (!tx_busy && !tx_start) begin
                    tx_data  <= transfer_beat[transfer_byte_index*8 +: 8];
                    tx_start <= 1'b1;
                    if (transfer_byte_index == 6'd31) begin
                        if (transfer_beat_index + 1'b1 == transfer_total_beats) begin
                            state <= ST_IDLE;
                        end else begin
                            transfer_beat_index <= transfer_beat_index + 1'b1;
                            state               <= ST_SETUP_READ;
                        end
                    end else begin
                        transfer_byte_index <= transfer_byte_index + 1'b1;
                    end
                end
            end

            ST_START_BLOCK: begin
                block_start       <= 1'b1;
                block_cycle_count <= 32'd0;
                state             <= ST_WAIT_BLOCK;
            end

            ST_WAIT_BLOCK: begin
                block_cycle_count <= block_cycle_count + 1'b1;
                if (block_error) begin
                    protocol_error        <= 1'b1;
                    error_code            <= ERR_BLOCK_BASE + block_error_code;
                    result_valid          <= 1'b0;
                    completion_success    <= 1'b0;
                    completion_error_code <= block_error_code;
                    completion_stage      <= block_stage;
                    completion_cycle_count<= block_cycle_count;
                    tx_index              <= 6'd0;
                    state                 <= ST_SEND_COMPLETION;
                end else if (block_done) begin
                    protocol_error        <= 1'b0;
                    error_code            <= 8'd0;
                    result_valid          <= 1'b1;
                    completion_success    <= 1'b1;
                    completion_error_code <= 8'd0;
                    completion_stage      <= block_stage;
                    completion_cycle_count<= block_cycle_count;
                    tx_index              <= 6'd0;
                    state                 <= ST_SEND_COMPLETION;
                end
            end

            ST_SEND_COMPLETION: begin
                if (!tx_busy && !tx_start) begin
                    tx_data  <= completion_byte(tx_index[3:0]);
                    tx_start <= 1'b1;
                    if (tx_index == 6'd9)
                        state <= ST_IDLE;
                    else
                        tx_index <= tx_index + 1'b1;
                end
            end

            ST_SEND_INFO: begin
                if (!tx_busy && !tx_start) begin
                    if (tx_index < 6'd22) begin
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
                    case (tx_index)
                        6'd0: tx_data <= "S";
                        6'd1: tx_data <= status_snapshot;
                        6'd2: tx_data <= {3'd0, block_stage};
                        6'd3: tx_data <= block_error_code;
                        6'd4: tx_data <= 8'h0d;
                        default: tx_data <= 8'h0a;
                    endcase
                    tx_start <= 1'b1;
                    if (tx_index == 6'd5)
                        state <= ST_IDLE;
                    else
                        tx_index <= tx_index + 1'b1;
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
                    if (tx_index == 6'd2)
                        state <= ST_IDLE;
                    else
                        tx_index <= tx_index + 1'b1;
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
                    if (tx_index == 6'd3)
                        state <= ST_IDLE;
                    else
                        tx_index <= tx_index + 1'b1;
                end
            end

            default: begin
                host_awvalid   <= 1'b0;
                host_arvalid   <= 1'b0;
                protocol_error <= 1'b1;
                error_code     <= ERR_INTERNAL;
                tx_index       <= 6'd0;
                state          <= ST_SEND_ERROR;
            end
        endcase
    end
end

wire _unused = &{1'b0, block_axi_master, block_watchdog_count,
    transfer_byte_length};

endmodule
