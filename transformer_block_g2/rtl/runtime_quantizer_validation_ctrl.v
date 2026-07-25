`timescale 1ns/1ps

// G2 Q6.10/Q28 运行时量化 DDR3 自动逐位验证控制器。
//
// UART 115200 8N1：
//   I -> "PANGU50K G2 QUANT V1\r\n"
//   S -> 'S' + flags + "\r\n"
//   C + 24 B config -> "K\r\n"
//   L + source + compact FP16 scales -> "K\r\n"
//   G -> 'R' + 96 B metadata + packed INT8 + padded UQ4.28
//
// config little-endian <4H4I>：
//   vector_length, rows, groups, matrix_id,
//   source_addr, activation_addr, raw_scale_addr, combined_scale_addr。
//
// matrix_id：0 Q、1 K、2 V、3 O、4 gate、5 up、6 down。
// 本模块同时实例化 Q10/Q28 两条 controller，仅按 matrix_id 选择一条，
// 因而一个位流即可覆盖七个真实 Linear 调用。
module runtime_quantizer_validation_ctrl #(
    parameter integer CTRL_ADDR_WIDTH = 28,
    parameter integer CLKS_PER_BIT = 868,
    parameter integer WATCHDOG_LIMIT = 50_000_000
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

    output wire [5:0]                   debug_state,
    output reg                          protocol_error,
    output reg                          configured,
    output reg                          loaded,
    output reg                          result_valid
);

localparam [5:0] ST_IDLE                = 6'd0;
localparam [5:0] ST_RECV_CONFIG         = 6'd1;
localparam [5:0] ST_APPLY_CONFIG        = 6'd2;
localparam [5:0] ST_RECV_LOAD           = 6'd3;
localparam [5:0] ST_SETUP_LOAD_WRITE    = 6'd4;
localparam [5:0] ST_WRITE_LOAD          = 6'd5;
localparam [5:0] ST_START_QUANT         = 6'd6;
localparam [5:0] ST_WAIT_QUANT          = 6'd7;
localparam [5:0] ST_FINISH_TRACE        = 6'd8;
localparam [5:0] ST_CHECK_TRACE         = 6'd9;
localparam [5:0] ST_SEND_RESULT_PREFIX  = 6'd10;
localparam [5:0] ST_SEND_RESULT_HEADER  = 6'd11;
localparam [5:0] ST_SETUP_RESULT_READ   = 6'd12;
localparam [5:0] ST_WAIT_RESULT_READ    = 6'd13;
localparam [5:0] ST_SEND_RESULT_BYTES   = 6'd14;
localparam [5:0] ST_SEND_INFO           = 6'd15;
localparam [5:0] ST_SEND_STATUS         = 6'd16;
localparam [5:0] ST_SEND_ACK            = 6'd17;
localparam [5:0] ST_SEND_ERROR          = 6'd18;

localparam [7:0] ERR_COMMAND             = 8'h01;
localparam [7:0] ERR_DDR_NOT_READY       = 8'h02;
localparam [7:0] ERR_NOT_CONFIGURED      = 8'h03;
localparam [7:0] ERR_NOT_LOADED          = 8'h04;
localparam [7:0] ERR_CONFIG              = 8'h10;
localparam [7:0] ERR_QUANTIZER_BASE      = 8'h40;
localparam [7:0] ERR_TRACE_BASE          = 8'h80;
localparam [7:0] ERR_WATCHDOG            = 8'hf0;
localparam [7:0] ERR_INTERNAL            = 8'hff;

reg [5:0] state;
reg [7:0] tx_data;
reg tx_start;
wire tx_busy;
wire [7:0] rx_data;
wire rx_valid;
reg [6:0] tx_index;
reg [7:0] error_code;
reg [7:0] status_snapshot;
reg ack_return_idle;

reg [191:0] config_buffer;
reg [4:0] config_byte_index;
reg [12:0] cfg_vector_length;
reg [12:0] cfg_rows;
reg [6:0] cfg_groups;
reg [2:0] cfg_matrix_id;
reg cfg_source_q28;
reg [CTRL_ADDR_WIDTH-1:0] cfg_source_addr;
reg [CTRL_ADDR_WIDTH-1:0] cfg_activation_addr;
reg [CTRL_ADDR_WIDTH-1:0] cfg_raw_scale_addr;
reg [CTRL_ADDR_WIDTH-1:0] cfg_combined_scale_addr;
reg [12:0] source_beats;
reg [13:0] raw_scale_beats;
reg [7:0] activation_beats;
reg [13:0] combined_scale_beats;

reg [5:0] rx_byte_index;
reg [255:0] upload_beat;
reg [13:0] load_beat_index;
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
reg result_read_combined;
reg [13:0] result_read_beat_index;
reg [255:0] result_tx_cache;
reg [5:0] result_tx_byte_index;
reg [31:0] watchdog_count;

reg q10_start;
reg q28_start;
reg trace_start;
reg trace_finish;

wire [15:0] new_vector_length = config_buffer[15:0];
wire [15:0] new_rows = config_buffer[31:16];
wire [15:0] new_groups = config_buffer[47:32];
wire [15:0] new_matrix_id = config_buffer[63:48];
wire [31:0] new_source_addr = config_buffer[95:64];
wire [31:0] new_activation_addr = config_buffer[127:96];
wire [31:0] new_raw_scale_addr = config_buffer[159:128];
wire [31:0] new_combined_scale_addr = config_buffer[191:160];
wire new_source_q28 = (new_matrix_id == 16'd3) || (new_matrix_id == 16'd6);
wire new_matrix_shape_valid =
    ((new_matrix_id == 16'd0) && (new_vector_length == 16'd896) &&
        (new_rows == 16'd896) && (new_groups == 16'd14)) ||
    (((new_matrix_id == 16'd1) || (new_matrix_id == 16'd2)) &&
        (new_vector_length == 16'd896) && (new_rows == 16'd128) &&
        (new_groups == 16'd14)) ||
    ((new_matrix_id == 16'd3) && (new_vector_length == 16'd896) &&
        (new_rows == 16'd896) && (new_groups == 16'd14)) ||
    (((new_matrix_id == 16'd4) || (new_matrix_id == 16'd5)) &&
        (new_vector_length == 16'd896) && (new_rows == 16'd4864) &&
        (new_groups == 16'd14)) ||
    ((new_matrix_id == 16'd6) && (new_vector_length == 16'd4864) &&
        (new_rows == 16'd896) && (new_groups == 16'd76));
wire new_addresses_valid =
    (new_source_addr[31:CTRL_ADDR_WIDTH] == 0) &&
    (new_activation_addr[31:CTRL_ADDR_WIDTH] == 0) &&
    (new_raw_scale_addr[31:CTRL_ADDR_WIDTH] == 0) &&
    (new_combined_scale_addr[31:CTRL_ADDR_WIDTH] == 0) &&
    (new_source_addr[2:0] == 3'd0) &&
    (new_activation_addr[2:0] == 3'd0) &&
    (new_raw_scale_addr[2:0] == 3'd0) &&
    (new_combined_scale_addr[2:0] == 3'd0);
wire new_config_valid = new_matrix_shape_valid && new_addresses_valid;
wire [19:0] new_raw_scale_values = new_rows[12:0] * new_groups[6:0];
wire [13:0] new_raw_scale_beats = new_raw_scale_values >> 4;
wire [12:0] new_source_beats = new_source_q28
    ? (new_vector_length[12:0] >> 2)
    : (new_vector_length[12:0] >> 4);
wire [7:0] new_activation_beats = new_vector_length[12:0] >> 5;
wire [13:0] new_combined_scale_beats =
    new_rows[12:0] * ((new_groups == 16'd14) ? 4'd2 : 4'd10);
wire [14:0] total_load_beats = {2'd0, source_beats} + {1'd0, raw_scale_beats};

wire host_aw_handshake = host_awvalid && axi_awready;
wire host_write_handshake = axi_wready && (host_aw_seen || host_aw_handshake);
wire host_ar_handshake = host_arvalid && axi_arready;
wire host_read_handshake = axi_rvalid && (host_ar_seen || host_ar_handshake);

wire quant_bus_active =
    (state == ST_START_QUANT) || (state == ST_WAIT_QUANT) ||
    (state == ST_FINISH_TRACE) || (state == ST_CHECK_TRACE);

wire [CTRL_ADDR_WIDTH-1:0] q10_awaddr;
wire q10_awuser_ap;
wire [3:0] q10_awuser_id;
wire [3:0] q10_awlen;
wire q10_awvalid;
wire [255:0] q10_wdata;
wire [31:0] q10_wstrb;
wire [CTRL_ADDR_WIDTH-1:0] q10_araddr;
wire q10_aruser_ap;
wire [3:0] q10_aruser_id;
wire [3:0] q10_arlen;
wire q10_arvalid;
wire q10_busy;
wire q10_done;
wire q10_error;
wire [7:0] q10_error_code;
wire [31:0] q10_saturated_count;
wire q10_all_zero;
wire [15:0] q10_max_abs_q10;
wire [23:0] q10_max_mantissa;
wire signed [9:0] q10_max_exponent;
wire [31:0] q10_max_bits;

wire [CTRL_ADDR_WIDTH-1:0] q28_awaddr;
wire q28_awuser_ap;
wire [3:0] q28_awuser_id;
wire [3:0] q28_awlen;
wire q28_awvalid;
wire [255:0] q28_wdata;
wire [31:0] q28_wstrb;
wire [CTRL_ADDR_WIDTH-1:0] q28_araddr;
wire q28_aruser_ap;
wire [3:0] q28_aruser_id;
wire [3:0] q28_arlen;
wire q28_arvalid;
wire q28_busy;
wire q28_done;
wire q28_error;
wire [7:0] q28_error_code;
wire [31:0] q28_saturated_count;
wire q28_all_zero;
wire [15:0] q28_max_abs_q10;
wire [23:0] q28_max_mantissa;
wire signed [9:0] q28_max_exponent;
wire [31:0] q28_max_bits;

wire [CTRL_ADDR_WIDTH-1:0] selected_awaddr = cfg_source_q28 ? q28_awaddr : q10_awaddr;
wire selected_awuser_ap = cfg_source_q28 ? q28_awuser_ap : q10_awuser_ap;
wire [3:0] selected_awuser_id = cfg_source_q28 ? q28_awuser_id : q10_awuser_id;
wire [3:0] selected_awlen = cfg_source_q28 ? q28_awlen : q10_awlen;
wire selected_awvalid = cfg_source_q28 ? q28_awvalid : q10_awvalid;
wire [255:0] selected_wdata = cfg_source_q28 ? q28_wdata : q10_wdata;
wire [31:0] selected_wstrb = cfg_source_q28 ? q28_wstrb : q10_wstrb;
wire [CTRL_ADDR_WIDTH-1:0] selected_araddr = cfg_source_q28 ? q28_araddr : q10_araddr;
wire selected_aruser_ap = cfg_source_q28 ? q28_aruser_ap : q10_aruser_ap;
wire [3:0] selected_aruser_id = cfg_source_q28 ? q28_aruser_id : q10_aruser_id;
wire [3:0] selected_arlen = cfg_source_q28 ? q28_arlen : q10_arlen;
wire selected_arvalid = cfg_source_q28 ? q28_arvalid : q10_arvalid;
wire selected_busy = cfg_source_q28 ? q28_busy : q10_busy;
wire selected_done = cfg_source_q28 ? q28_done : q10_done;
wire selected_error = cfg_source_q28 ? q28_error : q10_error;
wire [7:0] selected_error_code = cfg_source_q28 ? q28_error_code : q10_error_code;
wire [31:0] selected_saturated_count = cfg_source_q28
    ? q28_saturated_count : q10_saturated_count;
wire selected_all_zero = cfg_source_q28 ? q28_all_zero : q10_all_zero;
wire [15:0] selected_max_abs_q10 = cfg_source_q28
    ? q28_max_abs_q10 : q10_max_abs_q10;
wire [23:0] selected_max_mantissa = cfg_source_q28
    ? q28_max_mantissa : q10_max_mantissa;
wire signed [9:0] selected_max_exponent = cfg_source_q28
    ? q28_max_exponent : q10_max_exponent;
wire [31:0] selected_max_bits = cfg_source_q28 ? q28_max_bits : q10_max_bits;

reg result_all_zero;
reg [15:0] result_max_abs_q10;
reg [23:0] result_max_mantissa;
reg signed [9:0] result_max_exponent;
reg [31:0] result_max_bits;
reg [31:0] result_saturated_count;

wire trace_error;
wire [7:0] trace_error_code;
wire trace_complete;
wire [31:0] trace_source_read_commands;
wire [31:0] trace_source_read_beats;
wire [31:0] trace_raw_read_commands;
wire [31:0] trace_raw_read_beats;
wire [31:0] trace_activation_write_commands;
wire [31:0] trace_activation_write_beats;
wire [31:0] trace_combined_write_commands;
wire [31:0] trace_combined_write_beats;
wire trace_ar_handshake = quant_bus_active && selected_arvalid && axi_arready;
wire trace_aw_handshake = quant_bus_active && selected_awvalid && axi_awready;

assign axi_awaddr = quant_bus_active ? selected_awaddr : host_awaddr;
assign axi_awuser_ap = quant_bus_active ? selected_awuser_ap : 1'b0;
assign axi_awuser_id = quant_bus_active ? selected_awuser_id : 4'd0;
assign axi_awlen = quant_bus_active ? selected_awlen : 4'd0;
assign axi_awvalid = quant_bus_active ? selected_awvalid : host_awvalid;
assign axi_wdata = quant_bus_active ? selected_wdata : host_wdata;
assign axi_wstrb = quant_bus_active ? selected_wstrb : host_wstrb;
assign axi_araddr = quant_bus_active ? selected_araddr : host_araddr;
assign axi_aruser_ap = quant_bus_active ? selected_aruser_ap : 1'b0;
assign axi_aruser_id = quant_bus_active ? selected_aruser_id : 4'd0;
assign axi_arlen = quant_bus_active ? selected_arlen : host_arlen;
assign axi_arvalid = quant_bus_active ? selected_arvalid : host_arvalid;
assign debug_state = state;

uart_rx #(.CLKS_PER_BIT(CLKS_PER_BIT)) u_uart_rx (
    .clk(core_clk), .rst_n(core_rst_n), .rx(uart_rx_i),
    .data(rx_data), .valid(rx_valid)
);

uart_tx #(.CLKS_PER_BIT(CLKS_PER_BIT)) u_uart_tx (
    .clk(core_clk), .rst_n(core_rst_n), .data(tx_data),
    .start(tx_start), .tx(uart_tx_o), .busy(tx_busy)
);

runtime_quantizer_ctrl #(
    .SOURCE_Q28(0), .CTRL_ADDR_WIDTH(CTRL_ADDR_WIDTH)
) u_runtime_quantizer_q10 (
    .clk(core_clk), .rst_n(core_rst_n), .ddr_init_done(ddr_init_done),
    .start(q10_start), .cfg_vector_length(cfg_vector_length),
    .cfg_rows(cfg_rows), .cfg_groups(cfg_groups),
    .cfg_source_addr(cfg_source_addr),
    .cfg_activation_addr(cfg_activation_addr),
    .cfg_raw_scale_addr(cfg_raw_scale_addr),
    .cfg_combined_scale_addr(cfg_combined_scale_addr),
    .axi_awaddr(q10_awaddr), .axi_awuser_ap(q10_awuser_ap),
    .axi_awuser_id(q10_awuser_id), .axi_awlen(q10_awlen),
    .axi_awready((quant_bus_active && !cfg_source_q28) ? axi_awready : 1'b0),
    .axi_awvalid(q10_awvalid), .axi_wdata(q10_wdata), .axi_wstrb(q10_wstrb),
    .axi_wready((quant_bus_active && !cfg_source_q28) ? axi_wready : 1'b0),
    .axi_araddr(q10_araddr), .axi_aruser_ap(q10_aruser_ap),
    .axi_aruser_id(q10_aruser_id), .axi_arlen(q10_arlen),
    .axi_arready((quant_bus_active && !cfg_source_q28) ? axi_arready : 1'b0),
    .axi_arvalid(q10_arvalid), .axi_rdata(axi_rdata),
    .axi_rvalid((quant_bus_active && !cfg_source_q28) ? axi_rvalid : 1'b0),
    .busy(q10_busy), .done(q10_done), .error(q10_error),
    .error_code(q10_error_code), .saturated_count(q10_saturated_count),
    .all_zero(q10_all_zero), .max_abs_q10(q10_max_abs_q10),
    .max_mantissa_binary32(q10_max_mantissa),
    .max_exponent_binary32(q10_max_exponent),
    .max_abs_binary32_bits(q10_max_bits), .debug_state()
);

runtime_quantizer_ctrl #(
    .SOURCE_Q28(1), .CTRL_ADDR_WIDTH(CTRL_ADDR_WIDTH)
) u_runtime_quantizer_q28 (
    .clk(core_clk), .rst_n(core_rst_n), .ddr_init_done(ddr_init_done),
    .start(q28_start), .cfg_vector_length(cfg_vector_length),
    .cfg_rows(cfg_rows), .cfg_groups(cfg_groups),
    .cfg_source_addr(cfg_source_addr),
    .cfg_activation_addr(cfg_activation_addr),
    .cfg_raw_scale_addr(cfg_raw_scale_addr),
    .cfg_combined_scale_addr(cfg_combined_scale_addr),
    .axi_awaddr(q28_awaddr), .axi_awuser_ap(q28_awuser_ap),
    .axi_awuser_id(q28_awuser_id), .axi_awlen(q28_awlen),
    .axi_awready((quant_bus_active && cfg_source_q28) ? axi_awready : 1'b0),
    .axi_awvalid(q28_awvalid), .axi_wdata(q28_wdata), .axi_wstrb(q28_wstrb),
    .axi_wready((quant_bus_active && cfg_source_q28) ? axi_wready : 1'b0),
    .axi_araddr(q28_araddr), .axi_aruser_ap(q28_aruser_ap),
    .axi_aruser_id(q28_aruser_id), .axi_arlen(q28_arlen),
    .axi_arready((quant_bus_active && cfg_source_q28) ? axi_arready : 1'b0),
    .axi_arvalid(q28_arvalid), .axi_rdata(axi_rdata),
    .axi_rvalid((quant_bus_active && cfg_source_q28) ? axi_rvalid : 1'b0),
    .busy(q28_busy), .done(q28_done), .error(q28_error),
    .error_code(q28_error_code), .saturated_count(q28_saturated_count),
    .all_zero(q28_all_zero), .max_abs_q10(q28_max_abs_q10),
    .max_mantissa_binary32(q28_max_mantissa),
    .max_exponent_binary32(q28_max_exponent),
    .max_abs_binary32_bits(q28_max_bits), .debug_state()
);

runtime_quantizer_trace_checker #(
    .CTRL_ADDR_WIDTH(CTRL_ADDR_WIDTH)
) u_runtime_quantizer_trace_checker (
    .clk(core_clk), .rst_n(core_rst_n), .start(trace_start),
    .finish(trace_finish), .source_q28(cfg_source_q28),
    .vector_length(cfg_vector_length), .rows(cfg_rows), .groups(cfg_groups),
    .source_ctrl_addr(cfg_source_addr),
    .activation_ctrl_addr(cfg_activation_addr),
    .raw_scale_ctrl_addr(cfg_raw_scale_addr),
    .combined_scale_ctrl_addr(cfg_combined_scale_addr),
    .axi_araddr(selected_araddr), .axi_arlen(selected_arlen),
    .axi_ar_handshake(trace_ar_handshake),
    .axi_rvalid(quant_bus_active && axi_rvalid),
    .axi_awaddr(selected_awaddr), .axi_awlen(selected_awlen),
    .axi_aw_handshake(trace_aw_handshake),
    .axi_wready(quant_bus_active && axi_wready),
    .error(trace_error), .error_code(trace_error_code), .complete(trace_complete),
    .source_read_commands(trace_source_read_commands),
    .source_read_beats(trace_source_read_beats),
    .raw_scale_read_commands(trace_raw_read_commands),
    .raw_scale_read_beats(trace_raw_read_beats),
    .activation_write_commands(trace_activation_write_commands),
    .activation_write_beats(trace_activation_write_beats),
    .combined_write_commands(trace_combined_write_commands),
    .combined_write_beats(trace_combined_write_beats)
);

function [7:0] info_char;
    input [4:0] index;
    begin
        case (index)
            5'd0: info_char = "P";  5'd1: info_char = "A";
            5'd2: info_char = "N";  5'd3: info_char = "G";
            5'd4: info_char = "U";  5'd5: info_char = "5";
            5'd6: info_char = "0";  5'd7: info_char = "K";
            5'd8: info_char = " ";  5'd9: info_char = "G";
            5'd10: info_char = "2"; 5'd11: info_char = " ";
            5'd12: info_char = "Q"; 5'd13: info_char = "U";
            5'd14: info_char = "A"; 5'd15: info_char = "N";
            5'd16: info_char = "T"; 5'd17: info_char = " ";
            5'd18: info_char = "V"; 5'd19: info_char = "1";
            5'd20: info_char = 8'h0d; 5'd21: info_char = 8'h0a;
            default: info_char = 8'h00;
        endcase
    end
endfunction

function [31:0] result_header_word;
    input [4:0] index;
    begin
        case (index)
            5'd0: result_header_word = 32'd1;
            5'd1: result_header_word = {29'd0, cfg_matrix_id};
            5'd2: result_header_word = {31'd0, cfg_source_q28};
            5'd3: result_header_word = {19'd0, cfg_vector_length};
            5'd4: result_header_word = {19'd0, cfg_rows};
            5'd5: result_header_word = {25'd0, cfg_groups};
            5'd6: result_header_word = (cfg_groups == 7'd14) ? 32'd16 : 32'd80;
            5'd7: result_header_word = {31'd0, result_all_zero};
            5'd8: result_header_word = {16'd0, result_max_abs_q10};
            5'd9: result_header_word = {8'd0, result_max_mantissa};
            5'd10: result_header_word = {{22{result_max_exponent[9]}}, result_max_exponent};
            5'd11: result_header_word = result_max_bits;
            5'd12: result_header_word = result_saturated_count;
            5'd13: result_header_word = trace_source_read_commands;
            5'd14: result_header_word = trace_source_read_beats;
            5'd15: result_header_word = trace_raw_read_commands;
            5'd16: result_header_word = trace_raw_read_beats;
            5'd17: result_header_word = trace_activation_write_commands;
            5'd18: result_header_word = trace_activation_write_beats;
            5'd19: result_header_word = trace_combined_write_commands;
            5'd20: result_header_word = trace_combined_write_beats;
            5'd21: result_header_word = {24'd0, trace_error_code};
            default: result_header_word = 32'd0;
        endcase
    end
endfunction

function [7:0] result_header_byte;
    input [6:0] index;
    reg [31:0] word_value;
    reg [6:0] payload_index;
    begin
        if (index < 7'd8) begin
            case (index)
                7'd0: result_header_byte = "P";
                7'd1: result_header_byte = "5";
                7'd2: result_header_byte = "0";
                7'd3: result_header_byte = "Q";
                7'd4: result_header_byte = "T";
                7'd5: result_header_byte = "V";
                7'd6: result_header_byte = "1";
                default: result_header_byte = 8'h00;
            endcase
        end else begin
            payload_index = index - 7'd8;
            word_value = result_header_word(payload_index[6:2]);
            case (payload_index[1:0])
                2'd0: result_header_byte = word_value[7:0];
                2'd1: result_header_byte = word_value[15:8];
                2'd2: result_header_byte = word_value[23:16];
                default: result_header_byte = word_value[31:24];
            endcase
        end
    end
endfunction

always @(posedge core_clk or negedge core_rst_n) begin
    if (!core_rst_n) begin
        state                     <= ST_IDLE;
        tx_data                   <= 8'd0;
        tx_start                  <= 1'b0;
        tx_index                  <= 7'd0;
        error_code                <= 8'd0;
        status_snapshot           <= 8'd0;
        ack_return_idle           <= 1'b1;
        config_buffer             <= 192'd0;
        config_byte_index         <= 5'd0;
        cfg_vector_length         <= 13'd0;
        cfg_rows                  <= 13'd0;
        cfg_groups                <= 7'd0;
        cfg_matrix_id             <= 3'd0;
        cfg_source_q28            <= 1'b0;
        cfg_source_addr           <= {CTRL_ADDR_WIDTH{1'b0}};
        cfg_activation_addr       <= {CTRL_ADDR_WIDTH{1'b0}};
        cfg_raw_scale_addr        <= {CTRL_ADDR_WIDTH{1'b0}};
        cfg_combined_scale_addr   <= {CTRL_ADDR_WIDTH{1'b0}};
        source_beats              <= 13'd0;
        raw_scale_beats           <= 14'd0;
        activation_beats          <= 8'd0;
        combined_scale_beats      <= 14'd0;
        rx_byte_index             <= 6'd0;
        upload_beat               <= 256'd0;
        load_beat_index           <= 14'd0;
        host_awaddr               <= {CTRL_ADDR_WIDTH{1'b0}};
        host_awvalid              <= 1'b0;
        host_wdata                <= 256'd0;
        host_wstrb                <= 32'd0;
        host_aw_seen              <= 1'b0;
        host_w_seen               <= 1'b0;
        host_araddr               <= {CTRL_ADDR_WIDTH{1'b0}};
        host_arlen                <= 4'd0;
        host_arvalid              <= 1'b0;
        host_ar_seen              <= 1'b0;
        result_read_combined      <= 1'b0;
        result_read_beat_index    <= 14'd0;
        result_tx_cache           <= 256'd0;
        result_tx_byte_index      <= 6'd0;
        watchdog_count            <= 32'd0;
        q10_start                 <= 1'b0;
        q28_start                 <= 1'b0;
        trace_start               <= 1'b0;
        trace_finish              <= 1'b0;
        result_all_zero           <= 1'b0;
        result_max_abs_q10        <= 16'd0;
        result_max_mantissa       <= 24'd0;
        result_max_exponent       <= 10'sd0;
        result_max_bits           <= 32'd0;
        result_saturated_count    <= 32'd0;
        protocol_error            <= 1'b0;
        configured               <= 1'b0;
        loaded                   <= 1'b0;
        result_valid             <= 1'b0;
    end else begin
        tx_start     <= 1'b0;
        q10_start    <= 1'b0;
        q28_start    <= 1'b0;
        trace_start  <= 1'b0;
        trace_finish <= 1'b0;

        case (state)
            ST_IDLE: begin
                host_awvalid <= 1'b0;
                host_arvalid <= 1'b0;
                host_aw_seen <= 1'b0;
                host_w_seen  <= 1'b0;
                host_ar_seen <= 1'b0;
                tx_index     <= 7'd0;
                if (rx_valid) begin
                    case (rx_data)
                        8'h49, 8'h69: state <= ST_SEND_INFO;
                        8'h53, 8'h73: begin
                            status_snapshot <= {
                                cfg_source_q28,
                                protocol_error,
                                trace_error,
                                selected_busy,
                                result_valid,
                                loaded,
                                configured,
                                ddr_init_done
                            };
                            state <= ST_SEND_STATUS;
                        end
                        8'h43, 8'h63: begin
                            config_buffer     <= 192'd0;
                            config_byte_index <= 5'd0;
                            result_valid      <= 1'b0;
                            state             <= ST_RECV_CONFIG;
                        end
                        8'h4c, 8'h6c: begin
                            if (!ddr_init_done) begin
                                error_code <= ERR_DDR_NOT_READY;
                                protocol_error <= 1'b1;
                                state <= ST_SEND_ERROR;
                            end else if (!configured) begin
                                error_code <= ERR_NOT_CONFIGURED;
                                protocol_error <= 1'b1;
                                state <= ST_SEND_ERROR;
                            end else begin
                                upload_beat     <= 256'd0;
                                rx_byte_index   <= 6'd0;
                                load_beat_index <= 14'd0;
                                loaded          <= 1'b0;
                                result_valid    <= 1'b0;
                                state           <= ST_RECV_LOAD;
                            end
                        end
                        8'h47, 8'h67: begin
                            if (!ddr_init_done) begin
                                error_code <= ERR_DDR_NOT_READY;
                                protocol_error <= 1'b1;
                                state <= ST_SEND_ERROR;
                            end else if (!configured) begin
                                error_code <= ERR_NOT_CONFIGURED;
                                protocol_error <= 1'b1;
                                state <= ST_SEND_ERROR;
                            end else if (!loaded) begin
                                error_code <= ERR_NOT_LOADED;
                                protocol_error <= 1'b1;
                                state <= ST_SEND_ERROR;
                            end else begin
                                result_valid  <= 1'b0;
                                watchdog_count<= 32'd0;
                                state         <= ST_START_QUANT;
                            end
                        end
                        default: begin
                            error_code <= ERR_COMMAND;
                            protocol_error <= 1'b1;
                            state <= ST_SEND_ERROR;
                        end
                    endcase
                end
            end

            ST_RECV_CONFIG: begin
                if (rx_valid) begin
                    config_buffer[config_byte_index*8 +: 8] <= rx_data;
                    if (config_byte_index == 5'd23)
                        state <= ST_APPLY_CONFIG;
                    else
                        config_byte_index <= config_byte_index + 1'b1;
                end
            end

            ST_APPLY_CONFIG: begin
                if (!new_config_valid) begin
                    configured    <= 1'b0;
                    loaded        <= 1'b0;
                    protocol_error<= 1'b1;
                    error_code    <= ERR_CONFIG;
                    state         <= ST_SEND_ERROR;
                end else begin
                    cfg_vector_length       <= new_vector_length[12:0];
                    cfg_rows                <= new_rows[12:0];
                    cfg_groups              <= new_groups[6:0];
                    cfg_matrix_id           <= new_matrix_id[2:0];
                    cfg_source_q28          <= new_source_q28;
                    cfg_source_addr         <= new_source_addr[CTRL_ADDR_WIDTH-1:0];
                    cfg_activation_addr     <= new_activation_addr[CTRL_ADDR_WIDTH-1:0];
                    cfg_raw_scale_addr      <= new_raw_scale_addr[CTRL_ADDR_WIDTH-1:0];
                    cfg_combined_scale_addr <= new_combined_scale_addr[CTRL_ADDR_WIDTH-1:0];
                    source_beats            <= new_source_beats;
                    raw_scale_beats         <= new_raw_scale_beats;
                    activation_beats        <= new_activation_beats;
                    combined_scale_beats    <= new_combined_scale_beats;
                    configured              <= 1'b1;
                    loaded                  <= 1'b0;
                    result_valid            <= 1'b0;
                    protocol_error          <= 1'b0;
                    error_code              <= 8'd0;
                    tx_index                <= 7'd0;
                    ack_return_idle         <= 1'b1;
                    state                   <= ST_SEND_ACK;
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
                if (load_beat_index < source_beats)
                    host_awaddr <= cfg_source_addr + (load_beat_index << 3);
                else
                    host_awaddr <= cfg_raw_scale_addr +
                        ((load_beat_index - source_beats) << 3);
                host_awvalid <= 1'b1;
                host_wdata   <= upload_beat;
                host_wstrb   <= 32'hffff_ffff;
                host_aw_seen <= 1'b0;
                host_w_seen  <= 1'b0;
                state        <= ST_WRITE_LOAD;
            end

            ST_WRITE_LOAD: begin
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
                    if (load_beat_index + 1'b1 == total_load_beats) begin
                        loaded          <= 1'b1;
                        tx_index        <= 7'd0;
                        ack_return_idle <= 1'b1;
                        state           <= ST_SEND_ACK;
                    end else begin
                        load_beat_index <= load_beat_index + 1'b1;
                        rx_byte_index   <= 6'd0;
                        upload_beat     <= 256'd0;
                        state           <= ST_RECV_LOAD;
                    end
                end
            end

            ST_START_QUANT: begin
                trace_start <= 1'b1;
                if (cfg_source_q28)
                    q28_start <= 1'b1;
                else
                    q10_start <= 1'b1;
                watchdog_count <= 32'd0;
                state <= ST_WAIT_QUANT;
            end

            ST_WAIT_QUANT: begin
                watchdog_count <= watchdog_count + 1'b1;
                if (selected_error) begin
                    protocol_error <= 1'b1;
                    error_code <= ERR_QUANTIZER_BASE + selected_error_code;
                    state <= ST_SEND_ERROR;
                end else if (trace_error) begin
                    protocol_error <= 1'b1;
                    error_code <= ERR_TRACE_BASE + trace_error_code;
                    state <= ST_SEND_ERROR;
                end else if (watchdog_count >= WATCHDOG_LIMIT - 1) begin
                    protocol_error <= 1'b1;
                    error_code <= ERR_WATCHDOG;
                    state <= ST_SEND_ERROR;
                end else if (selected_done) begin
                    result_all_zero        <= selected_all_zero;
                    result_max_abs_q10     <= selected_max_abs_q10;
                    result_max_mantissa    <= selected_max_mantissa;
                    result_max_exponent    <= selected_max_exponent;
                    result_max_bits        <= selected_max_bits;
                    result_saturated_count <= selected_saturated_count;
                    state                  <= ST_FINISH_TRACE;
                end
            end

            ST_FINISH_TRACE: begin
                trace_finish <= 1'b1;
                state <= ST_CHECK_TRACE;
            end

            ST_CHECK_TRACE: begin
                // trace_finish 在上一状态置位，checker 到本拍才采样；complete/error
                // 会在下一拍可见，因此这里必须等待，不能把首个空拍判成失败。
                if (trace_error) begin
                    protocol_error <= 1'b1;
                    error_code <= ERR_TRACE_BASE + trace_error_code;
                    state <= ST_SEND_ERROR;
                end else if (trace_complete) begin
                    result_valid          <= 1'b1;
                    tx_index              <= 7'd0;
                    result_read_combined  <= 1'b0;
                    result_read_beat_index<= 14'd0;
                    state                 <= ST_SEND_RESULT_PREFIX;
                end
            end

            ST_SEND_RESULT_PREFIX: begin
                if (!tx_busy && !tx_start) begin
                    tx_data  <= "R";
                    tx_start <= 1'b1;
                    tx_index <= 7'd0;
                    state    <= ST_SEND_RESULT_HEADER;
                end
            end

            ST_SEND_RESULT_HEADER: begin
                if (!tx_busy && !tx_start) begin
                    tx_data  <= result_header_byte(tx_index);
                    tx_start <= 1'b1;
                    if (tx_index == 7'd95) begin
                        result_read_combined   <= 1'b0;
                        result_read_beat_index <= 14'd0;
                        state                  <= ST_SETUP_RESULT_READ;
                    end else begin
                        tx_index <= tx_index + 1'b1;
                    end
                end
            end

            ST_SETUP_RESULT_READ: begin
                host_araddr <= (result_read_combined
                    ? cfg_combined_scale_addr : cfg_activation_addr)
                    + (result_read_beat_index << 3);
                host_arlen   <= 4'd0;
                host_arvalid <= 1'b1;
                host_ar_seen <= 1'b0;
                state        <= ST_WAIT_RESULT_READ;
            end

            ST_WAIT_RESULT_READ: begin
                if (host_ar_handshake) begin
                    host_arvalid <= 1'b0;
                    host_ar_seen <= 1'b1;
                end
                if (host_read_handshake) begin
                    result_tx_cache      <= axi_rdata;
                    result_tx_byte_index <= 6'd0;
                    host_ar_seen          <= 1'b0;
                    state                 <= ST_SEND_RESULT_BYTES;
                end
            end

            ST_SEND_RESULT_BYTES: begin
                if (!tx_busy && !tx_start) begin
                    tx_data  <= result_tx_cache[result_tx_byte_index*8 +: 8];
                    tx_start <= 1'b1;
                    if (result_tx_byte_index == 6'd31) begin
                        if (!result_read_combined &&
                            (result_read_beat_index + 1'b1 == activation_beats)) begin
                            result_read_combined   <= 1'b1;
                            result_read_beat_index <= 14'd0;
                            state                  <= ST_SETUP_RESULT_READ;
                        end else if (result_read_combined &&
                            (result_read_beat_index + 1'b1 == combined_scale_beats)) begin
                            state <= ST_IDLE;
                        end else begin
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
                    if (tx_index < 7'd22) begin
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
                    if (tx_index < 7'd4) begin
                        case (tx_index)
                            7'd0: tx_data <= "S";
                            7'd1: tx_data <= status_snapshot;
                            7'd2: tx_data <= 8'h0d;
                            default: tx_data <= 8'h0a;
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
                    if (tx_index < 7'd3) begin
                        case (tx_index)
                            7'd0: tx_data <= "K";
                            7'd1: tx_data <= 8'h0d;
                            default: tx_data <= 8'h0a;
                        endcase
                        tx_start <= 1'b1;
                        tx_index <= tx_index + 1'b1;
                    end else if (ack_return_idle) begin
                        state <= ST_IDLE;
                    end else begin
                        state <= ST_IDLE;
                    end
                end
            end

            ST_SEND_ERROR: begin
                if (!tx_busy && !tx_start) begin
                    if (tx_index < 7'd4) begin
                        case (tx_index)
                            7'd0: tx_data <= "E";
                            7'd1: tx_data <= error_code;
                            7'd2: tx_data <= 8'h0d;
                            default: tx_data <= 8'h0a;
                        endcase
                        tx_start <= 1'b1;
                        tx_index <= tx_index + 1'b1;
                    end else begin
                        state <= ST_IDLE;
                    end
                end
            end

            default: begin
                host_awvalid   <= 1'b0;
                host_arvalid   <= 1'b0;
                protocol_error <= 1'b1;
                error_code     <= ERR_INTERNAL;
                tx_index       <= 7'd0;
                state          <= ST_SEND_ERROR;
            end
        endcase
    end
end

wire _unused = &{1'b0, q10_max_bits, q28_max_abs_q10, ack_return_idle};

endmodule
