`timescale 1ns/1ps

// layer0 Q/K RoPE 独立验证控制器。
//
// 输入布局：Q=[14,64]、K=[2,64]，拼接为 16 个 head，每个元素 signed int64 Q28。
// Qwen2 配对规则为 dim i 与 dim i+32，而不是相邻维度配对。
//
// UART 协议（115200 8N1）：
//   I -> "PANGU50K ROPE QK V1\r\n"
//   S -> 'S' + flags + current_position_u16 + table_index + table_count + "\r\n"
//   C + start_position_u16 + table_count_u16 -> 配置连续位置表，回复 "K\r\n"
//   L + Q/K Q28 8192B + table_count * 256B trig 行 -> 写 DDR3，回复 "K\r\n"
//   G -> 处理当前位置，回复 'R' + processed_position_u16 + 8192B Q/K Q28
//        回复完成后 current_position 和 table_index 自动加 1。
//   Z -> 将位置和表索引复位到配置起点，保留已加载数据，回复 "K\r\n"
//
// 每个 trig 行：cos_q30[32] + sin_q30[32]，均为 little-endian signed int32。
module rope_qk_ctrl #(
    parameter integer CTRL_ADDR_WIDTH = 28,
    parameter integer CLKS_PER_BIT    = 868,
    parameter integer MAX_TABLE_ROWS = 16
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
    output wire [15:0]                  debug_position,
    output reg                          protocol_error,
    output reg                          configured,
    output reg                          loaded,
    output reg                          result_valid
);

localparam integer TOTAL_HEADS = 16;
localparam integer HALF_DIM = 32;
localparam integer INPUT_BEATS = 256;
localparam integer RESULT_BEATS = 256;
localparam integer TRIG_BEATS_PER_ROW = 8;

// DDR3 控制器地址单位为 32 bit，一个 256 bit 数据拍占 8 个地址单位。
localparam [CTRL_ADDR_WIDTH-1:0] ADDR_INPUT  = 28'h0000000;
localparam [CTRL_ADDR_WIDTH-1:0] ADDR_TRIG   = 28'h0001000;
localparam [CTRL_ADDR_WIDTH-1:0] ADDR_RESULT = 28'h0002000;

localparam [5:0] ST_IDLE                = 6'd0;
localparam [5:0] ST_RECV_CONFIG         = 6'd1;
localparam [5:0] ST_APPLY_CONFIG        = 6'd2;
localparam [5:0] ST_RECV_LOAD           = 6'd3;
localparam [5:0] ST_SETUP_LOAD_WRITE    = 6'd4;
localparam [5:0] ST_WRITE_LOAD          = 6'd5;
localparam [5:0] ST_SETUP_TRIG_READ     = 6'd6;
localparam [5:0] ST_READ_TRIG           = 6'd7;
localparam [5:0] ST_SETUP_FIRST_READ    = 6'd8;
localparam [5:0] ST_READ_FIRST          = 6'd9;
localparam [5:0] ST_SETUP_SECOND_READ   = 6'd10;
localparam [5:0] ST_READ_SECOND         = 6'd11;
localparam [5:0] ST_START_CORE          = 6'd12;
localparam [5:0] ST_WAIT_CORE           = 6'd13;
localparam [5:0] ST_SETUP_FIRST_WRITE   = 6'd14;
localparam [5:0] ST_WRITE_FIRST         = 6'd15;
localparam [5:0] ST_SETUP_SECOND_WRITE  = 6'd16;
localparam [5:0] ST_WRITE_SECOND        = 6'd17;
localparam [5:0] ST_SEND_RESULT_PREFIX  = 6'd18;
localparam [5:0] ST_SETUP_RESULT_READ   = 6'd19;
localparam [5:0] ST_READ_RESULT         = 6'd20;
localparam [5:0] ST_SEND_RESULT_BYTES   = 6'd21;
localparam [5:0] ST_SEND_INFO           = 6'd22;
localparam [5:0] ST_SEND_STATUS         = 6'd23;
localparam [5:0] ST_SEND_ACK            = 6'd24;
localparam [5:0] ST_SEND_ERROR          = 6'd25;

reg [5:0] state;
reg [7:0] rx_data;
wire [7:0] uart_rx_data;
wire uart_rx_valid;
reg [7:0] tx_data;
reg tx_start;
wire tx_busy;
reg [5:0] tx_index;

reg [31:0] config_word;
reg [2:0] config_byte_index;
reg [15:0] start_position;
reg [15:0] current_position;
reg [15:0] processed_position;
reg [4:0] table_count;
reg [4:0] table_index;

reg [5:0] rx_byte_index;
reg [255:0] upload_beat;
reg [8:0] load_beat_index;

reg [3:0] trig_read_beat_index;
reg signed [31:0] cos_mem [0:HALF_DIM-1];
reg signed [31:0] sin_mem [0:HALF_DIM-1];
integer trig_lane;

reg [4:0] head_index;
reg [4:0] pair_index;
reg signed [63:0] first_input_cache;
reg signed [63:0] second_input_cache;
reg signed [63:0] first_output_cache;
reg signed [63:0] second_output_cache;

reg core_start;
wire core_busy;
wire core_done;
wire signed [63:0] core_y_first;
wire signed [63:0] core_y_second;

reg [8:0] result_read_beat_index;
reg [5:0] result_tx_byte_index;
reg [255:0] result_tx_cache;

reg aw_seen;
reg w_seen;
reg ar_seen;
reg [7:0] status_snapshot;
reg [7:0] error_code;

wire aw_handshake = axi_awvalid && axi_awready;
wire ar_handshake = axi_arvalid && axi_arready;
wire write_data_handshake = axi_wready && (aw_seen || aw_handshake);
wire read_data_handshake = axi_rvalid && (ar_seen || ar_handshake);

wire [8:0] active_load_beats = 9'd256 + {table_count, 3'b000};
wire sequence_exhausted = (table_index >= table_count);
wire [16:0] configured_end_position = {1'b0, config_word[15:0]} +
                                      {1'b0, config_word[31:16]};

wire [9:0] first_value_index = {head_index, 6'b000000} + pair_index;
wire [7:0] first_beat_index = first_value_index[9:2];
wire [7:0] second_beat_index = first_beat_index + 8'd8;
wire [1:0] value_lane = pair_index[1:0];
wire signed [31:0] selected_cos_q30 = cos_mem[pair_index];
wire signed [31:0] selected_sin_q30 = sin_mem[pair_index];

assign axi_awuser_ap = 1'b0;
assign axi_awuser_id = 4'h0;
assign axi_awlen = 4'h0;
assign axi_aruser_ap = 1'b0;
assign axi_aruser_id = 4'h0;
assign debug_state = state;
assign debug_position = current_position;

function [63:0] select_lane64;
    input [255:0] beat;
    input [1:0] lane;
    begin
        case (lane)
            2'd0: select_lane64 = beat[63:0];
            2'd1: select_lane64 = beat[127:64];
            2'd2: select_lane64 = beat[191:128];
            default: select_lane64 = beat[255:192];
        endcase
    end
endfunction

function [255:0] place_lane64;
    input [63:0] value;
    input [1:0] lane;
    begin
        place_lane64 = 256'd0;
        case (lane)
            2'd0: place_lane64[63:0] = value;
            2'd1: place_lane64[127:64] = value;
            2'd2: place_lane64[191:128] = value;
            default: place_lane64[255:192] = value;
        endcase
    end
endfunction

function [31:0] strobe_lane64;
    input [1:0] lane;
    begin
        case (lane)
            2'd0: strobe_lane64 = 32'h0000_00ff;
            2'd1: strobe_lane64 = 32'h0000_ff00;
            2'd2: strobe_lane64 = 32'h00ff_0000;
            default: strobe_lane64 = 32'hff00_0000;
        endcase
    end
endfunction

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
            5'd10: info_char = "O";
            5'd11: info_char = "P";
            5'd12: info_char = "E";
            5'd13: info_char = " ";
            5'd14: info_char = "Q";
            5'd15: info_char = "K";
            5'd16: info_char = " ";
            5'd17: info_char = "V";
            5'd18: info_char = "1";
            5'd19: info_char = 8'h0d;
            5'd20: info_char = 8'h0a;
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

rope_pair_q28_core u_rope_pair_q28_core (
    .clk          (core_clk),
    .rst_n        (core_rst_n),
    .start        (core_start),
    .x_first_q28  (first_input_cache),
    .x_second_q28 (second_input_cache),
    .cos_q30      (selected_cos_q30),
    .sin_q30      (selected_sin_q30),
    .busy         (core_busy),
    .done         (core_done),
    .y_first_q28  (core_y_first),
    .y_second_q28 (core_y_second)
);

always @(posedge core_clk or negedge core_rst_n) begin
    if (!core_rst_n) begin
        state                    <= ST_IDLE;
        rx_data                  <= 8'd0;
        tx_data                  <= 8'd0;
        tx_start                 <= 1'b0;
        tx_index                 <= 6'd0;
        config_word              <= 32'd0;
        config_byte_index        <= 3'd0;
        start_position           <= 16'd0;
        current_position         <= 16'd0;
        processed_position       <= 16'd0;
        table_count              <= 5'd0;
        table_index              <= 5'd0;
        rx_byte_index            <= 6'd0;
        upload_beat              <= 256'd0;
        load_beat_index          <= 9'd0;
        trig_read_beat_index     <= 4'd0;
        head_index               <= 5'd0;
        pair_index               <= 5'd0;
        first_input_cache        <= 64'sd0;
        second_input_cache       <= 64'sd0;
        first_output_cache       <= 64'sd0;
        second_output_cache      <= 64'sd0;
        core_start               <= 1'b0;
        result_read_beat_index   <= 9'd0;
        result_tx_byte_index     <= 6'd0;
        result_tx_cache          <= 256'd0;
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
        configured               <= 1'b0;
        loaded                   <= 1'b0;
        result_valid             <= 1'b0;
    end else begin
        tx_start   <= 1'b0;
        core_start <= 1'b0;
        if (uart_rx_valid)
            rx_data <= uart_rx_data;

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
                                2'd0,
                                sequence_exhausted,
                                core_busy,
                                result_valid,
                                loaded,
                                configured,
                                ddr_init_done
                            };
                            state <= ST_SEND_STATUS;
                        end

                        8'h43, 8'h63: begin // C / c
                            config_word       <= 32'd0;
                            config_byte_index <= 3'd0;
                            state             <= ST_RECV_CONFIG;
                        end

                        8'h4c, 8'h6c: begin // L / l
                            if (!ddr_init_done) begin
                                protocol_error <= 1'b1;
                                error_code     <= 8'h02;
                                state          <= ST_SEND_ERROR;
                            end else if (!configured) begin
                                protocol_error <= 1'b1;
                                error_code     <= 8'h03;
                                state          <= ST_SEND_ERROR;
                            end else begin
                                upload_beat       <= 256'd0;
                                rx_byte_index     <= 6'd0;
                                load_beat_index   <= 9'd0;
                                loaded            <= 1'b0;
                                result_valid      <= 1'b0;
                                table_index       <= 5'd0;
                                current_position  <= start_position;
                                state             <= ST_RECV_LOAD;
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
                            end else if (sequence_exhausted) begin
                                protocol_error <= 1'b1;
                                error_code     <= 8'h06;
                                state          <= ST_SEND_ERROR;
                            end else begin
                                processed_position   <= current_position;
                                trig_read_beat_index <= 4'd0;
                                head_index           <= 5'd0;
                                pair_index           <= 5'd0;
                                result_valid         <= 1'b0;
                                state                <= ST_SETUP_TRIG_READ;
                            end
                        end

                        8'h5a, 8'h7a: begin // Z / z
                            if (!configured) begin
                                protocol_error <= 1'b1;
                                error_code     <= 8'h03;
                                state          <= ST_SEND_ERROR;
                            end else begin
                                current_position <= start_position;
                                table_index      <= 5'd0;
                                result_valid     <= 1'b0;
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
                    if (config_byte_index == 3'd3)
                        state <= ST_APPLY_CONFIG;
                    else
                        config_byte_index <= config_byte_index + 1'b1;
                end
            end

            ST_APPLY_CONFIG: begin
                if ((config_word[31:16] == 16'd0) ||
                    (config_word[31:16] > MAX_TABLE_ROWS) ||
                    (configured_end_position > 17'd32768)) begin
                    protocol_error <= 1'b1;
                    error_code     <= 8'h05;
                    state          <= ST_SEND_ERROR;
                end else begin
                    start_position   <= config_word[15:0];
                    current_position <= config_word[15:0];
                    table_count      <= config_word[20:16];
                    table_index      <= 5'd0;
                    configured       <= 1'b1;
                    loaded           <= 1'b0;
                    result_valid     <= 1'b0;
                    state            <= ST_SEND_ACK;
                end
            end

            ST_RECV_LOAD: begin
                if (uart_rx_valid) begin
                    upload_beat[rx_byte_index*8 +: 8] <= uart_rx_data;
                    if (rx_byte_index == 6'd31)
                        state <= ST_SETUP_LOAD_WRITE;
                    else
                        rx_byte_index <= rx_byte_index + 1'b1;
                end
            end

            ST_SETUP_LOAD_WRITE: begin
                if (load_beat_index < INPUT_BEATS)
                    axi_awaddr <= ADDR_INPUT + (load_beat_index << 3);
                else
                    axi_awaddr <= ADDR_TRIG + ((load_beat_index - INPUT_BEATS) << 3);
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

                if ((aw_seen || aw_handshake) &&
                    (w_seen || write_data_handshake)) begin
                    axi_awvalid <= 1'b0;
                    aw_seen     <= 1'b0;
                    w_seen      <= 1'b0;
                    if (load_beat_index == active_load_beats - 1'b1) begin
                        loaded       <= 1'b1;
                        table_index  <= 5'd0;
                        current_position <= start_position;
                        state        <= ST_SEND_ACK;
                    end else begin
                        load_beat_index <= load_beat_index + 1'b1;
                        rx_byte_index   <= 6'd0;
                        upload_beat    <= 256'd0;
                        state          <= ST_RECV_LOAD;
                    end
                end
            end

            ST_SETUP_TRIG_READ: begin
                axi_araddr  <= ADDR_TRIG + ({23'd0, table_index} << 6);
                axi_arlen   <= 4'd7;
                axi_arvalid <= 1'b1;
                ar_seen     <= 1'b0;
                trig_read_beat_index <= 4'd0;
                state       <= ST_READ_TRIG;
            end

            ST_READ_TRIG: begin
                if (ar_handshake) begin
                    axi_arvalid <= 1'b0;
                    ar_seen     <= 1'b1;
                end
                if (read_data_handshake) begin
                    if (trig_read_beat_index < 4) begin
                        for (trig_lane = 0; trig_lane < 8; trig_lane = trig_lane + 1) begin
                            cos_mem[(trig_read_beat_index << 3) + trig_lane]
                                <= axi_rdata[trig_lane*32 +: 32];
                        end
                    end else begin
                        for (trig_lane = 0; trig_lane < 8; trig_lane = trig_lane + 1) begin
                            sin_mem[((trig_read_beat_index - 4'd4) << 3) + trig_lane]
                                <= axi_rdata[trig_lane*32 +: 32];
                        end
                    end

                    if (trig_read_beat_index == 4'd7) begin
                        ar_seen <= 1'b0;
                        state   <= ST_SETUP_FIRST_READ;
                    end else begin
                        trig_read_beat_index <= trig_read_beat_index + 1'b1;
                    end
                end
            end

            ST_SETUP_FIRST_READ: begin
                axi_araddr  <= ADDR_INPUT + ({20'd0, first_beat_index} << 3);
                axi_arlen   <= 4'd0;
                axi_arvalid <= 1'b1;
                ar_seen     <= 1'b0;
                state       <= ST_READ_FIRST;
            end

            ST_READ_FIRST: begin
                if (ar_handshake) begin
                    axi_arvalid <= 1'b0;
                    ar_seen     <= 1'b1;
                end
                if (read_data_handshake) begin
                    first_input_cache <= $signed(select_lane64(axi_rdata, value_lane));
                    ar_seen           <= 1'b0;
                    state             <= ST_SETUP_SECOND_READ;
                end
            end

            ST_SETUP_SECOND_READ: begin
                axi_araddr  <= ADDR_INPUT + ({20'd0, second_beat_index} << 3);
                axi_arlen   <= 4'd0;
                axi_arvalid <= 1'b1;
                ar_seen     <= 1'b0;
                state       <= ST_READ_SECOND;
            end

            ST_READ_SECOND: begin
                if (ar_handshake) begin
                    axi_arvalid <= 1'b0;
                    ar_seen     <= 1'b1;
                end
                if (read_data_handshake) begin
                    second_input_cache <= $signed(select_lane64(axi_rdata, value_lane));
                    ar_seen            <= 1'b0;
                    state              <= ST_START_CORE;
                end
            end

            ST_START_CORE: begin
                core_start <= 1'b1;
                state      <= ST_WAIT_CORE;
            end

            ST_WAIT_CORE: begin
                if (core_done) begin
                    first_output_cache  <= core_y_first;
                    second_output_cache <= core_y_second;
                    state               <= ST_SETUP_FIRST_WRITE;
                end
            end

            ST_SETUP_FIRST_WRITE: begin
                axi_awaddr  <= ADDR_RESULT + ({20'd0, first_beat_index} << 3);
                axi_awvalid <= 1'b1;
                axi_wdata   <= place_lane64(first_output_cache, value_lane);
                axi_wstrb   <= strobe_lane64(value_lane);
                aw_seen     <= 1'b0;
                w_seen      <= 1'b0;
                state       <= ST_WRITE_FIRST;
            end

            ST_WRITE_FIRST: begin
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
                    state       <= ST_SETUP_SECOND_WRITE;
                end
            end

            ST_SETUP_SECOND_WRITE: begin
                axi_awaddr  <= ADDR_RESULT + ({20'd0, second_beat_index} << 3);
                axi_awvalid <= 1'b1;
                axi_wdata   <= place_lane64(second_output_cache, value_lane);
                axi_wstrb   <= strobe_lane64(value_lane);
                aw_seen     <= 1'b0;
                w_seen      <= 1'b0;
                state       <= ST_WRITE_SECOND;
            end

            ST_WRITE_SECOND: begin
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
                    if (pair_index == HALF_DIM - 1) begin
                        if (head_index == TOTAL_HEADS - 1) begin
                            result_read_beat_index <= 9'd0;
                            result_tx_byte_index   <= 6'd0;
                            tx_index               <= 6'd0;
                            state                  <= ST_SEND_RESULT_PREFIX;
                        end else begin
                            head_index <= head_index + 1'b1;
                            pair_index <= 5'd0;
                            state      <= ST_SETUP_FIRST_READ;
                        end
                    end else begin
                        pair_index <= pair_index + 1'b1;
                        state      <= ST_SETUP_FIRST_READ;
                    end
                end
            end

            ST_SEND_RESULT_PREFIX: begin
                if (!tx_busy && !tx_start) begin
                    case (tx_index)
                        6'd0: tx_data <= "R";
                        6'd1: tx_data <= processed_position[7:0];
                        default: tx_data <= processed_position[15:8];
                    endcase
                    tx_start <= 1'b1;
                    if (tx_index == 6'd2) begin
                        tx_index <= 6'd0;
                        state    <= ST_SETUP_RESULT_READ;
                    end else begin
                        tx_index <= tx_index + 1'b1;
                    end
                end
            end

            ST_SETUP_RESULT_READ: begin
                axi_araddr  <= ADDR_RESULT + (result_read_beat_index << 3);
                axi_arlen   <= 4'd0;
                axi_arvalid <= 1'b1;
                ar_seen     <= 1'b0;
                state       <= ST_READ_RESULT;
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
                    state                <= ST_SEND_RESULT_BYTES;
                end
            end

            ST_SEND_RESULT_BYTES: begin
                if (!tx_busy && !tx_start) begin
                    tx_data  <= result_tx_cache[result_tx_byte_index*8 +: 8];
                    tx_start <= 1'b1;
                    if (result_tx_byte_index == 6'd31) begin
                        result_tx_byte_index <= 6'd0;
                        if (result_read_beat_index == RESULT_BEATS - 1) begin
                            result_valid <= 1'b1;
                            table_index  <= table_index + 1'b1;
                            current_position <= current_position + 1'b1;
                            state        <= ST_IDLE;
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
                    tx_data  <= info_char(tx_index[4:0]);
                    tx_start <= 1'b1;
                    if (tx_index == 6'd20) begin
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
                        6'd2: tx_data <= current_position[7:0];
                        6'd3: tx_data <= current_position[15:8];
                        6'd4: tx_data <= {3'd0, table_index};
                        6'd5: tx_data <= {3'd0, table_count};
                        6'd6: tx_data <= 8'h0d;
                        default: tx_data <= 8'h0a;
                    endcase
                    tx_start <= 1'b1;
                    if (tx_index == 6'd7) begin
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
            end
        endcase
    end
end

endmodule
