`timescale 1ns/1ps

// 运行时参数化 packed INT4 GEMV 控制器（D1.3 性能计数版本）。
//
// 支持范围：
//   1 <= M <= MAX_M
//   1 <= K <= MAX_K，尾块不足 16 个元素时由硬件屏蔽
//
// UART 协议（115200 8N1）：
//   I -> "PANGU50K GEMV PARAM V2\r\n"
//   S -> 'S' + 状态字节 + "\r\n"
//   C + uint16_le(M) + uint16_le(K) -> 配置运行时尺寸，回复 "K\r\n"
//   L + padded_x + padded_weight_rows -> 写入 DDR3，回复 "K\r\n"
//   G -> 执行 GEMV，回复 'R' + M 个 little-endian int32
//   P -> 读取性能计数，回复 'P' + 4 个 little-endian uint32：
//        激活读取周期、权重读取周期、MAC 计算周期、GEMV 总周期
//
// 上传载荷按 256 bit 数据拍补零：
//   padded_x            = ceil(K / 32) * 32 字节
//   padded_weight_rows  = M * ceil(K / 64) * 32 字节
// 每个权重行独立补齐到 32 字节边界。
module gemv_param_ctrl #(
    parameter integer CTRL_ADDR_WIDTH = 28,
    parameter integer CLKS_PER_BIT    = 868,
    parameter integer MAX_M           = 64,
    parameter integer MAX_K           = 896
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
    output reg                          config_valid,
    output reg                          loaded,
    output reg                          result_valid
);

// DDR3 控制器地址单位为 32 bit；一个 256 bit 数据拍占 8 个地址单位。
// 地址区域预留：激活最多 28 拍，权重最多 64*14 拍，结果最多 8 拍。
localparam [CTRL_ADDR_WIDTH-1:0] ADDR_ACT    = {CTRL_ADDR_WIDTH{1'b0}};
localparam [CTRL_ADDR_WIDTH-1:0] ADDR_WEIGHT = {{(CTRL_ADDR_WIDTH-9){1'b0}}, 9'h100};
localparam [CTRL_ADDR_WIDTH-1:0] ADDR_RESULT = {{(CTRL_ADDR_WIDTH-14){1'b0}}, 14'h2000};

localparam [5:0] ST_IDLE                = 6'd0;
localparam [5:0] ST_RECV_CONFIG         = 6'd1;
localparam [5:0] ST_VALIDATE_CONFIG     = 6'd2;
localparam [5:0] ST_RECV_LOAD           = 6'd3;
localparam [5:0] ST_SETUP_LOAD_WRITE    = 6'd4;
localparam [5:0] ST_WRITE_LOAD          = 6'd5;
localparam [5:0] ST_SETUP_ACT_READ      = 6'd6;
localparam [5:0] ST_READ_ACT            = 6'd7;
localparam [5:0] ST_SETUP_WEIGHT_READ   = 6'd8;
localparam [5:0] ST_READ_WEIGHT         = 6'd9;
localparam [5:0] ST_START_CORE          = 6'd10;
localparam [5:0] ST_WAIT_CORE           = 6'd11;
localparam [5:0] ST_SETUP_RESULT_WRITE  = 6'd12;
localparam [5:0] ST_WRITE_RESULT        = 6'd13;
localparam [5:0] ST_SEND_INFO           = 6'd14;
localparam [5:0] ST_SEND_STATUS         = 6'd15;
localparam [5:0] ST_SEND_ACK            = 6'd16;
localparam [5:0] ST_SEND_RESULT         = 6'd17;
localparam [5:0] ST_SEND_ERROR          = 6'd18;
localparam [5:0] ST_SEND_PERF           = 6'd19;

reg [5:0] state;
reg [8:0] tx_index;
reg [7:0] tx_data;
reg       tx_start;
wire      tx_busy;
wire [7:0] rx_data;
wire       rx_valid;

reg [31:0] config_payload;
reg [5:0]  rx_byte_index;
reg [255:0] upload_beat;
reg [9:0]  load_beat_index;

reg [6:0]  config_m;
reg [10:0] config_k;
reg [6:0]  k_blocks;
reg [4:0]  tail_elements;
reg [5:0]  act_beats;
reg [4:0]  weight_beats;
reg [3:0]  result_beats;
reg [9:0]  load_total_beats;

reg [MAX_M*32-1:0] result_cache;

reg [5:0] act_read_base_beat;
reg [4:0] active_read_burst_beats;
reg [4:0] read_beat_index;
reg [6:0] row_index;
reg [3:0] result_write_index;
reg [CTRL_ADDR_WIDTH-1:0] weight_row_addr;

reg core_start;
wire core_busy;
wire core_done;
wire signed [31:0] core_y_value;

reg aw_seen;
reg w_seen;
reg ar_seen;
reg [7:0] status_snapshot;
reg [7:0] error_code;

// D1.3 性能计数器。计数频率为 core_clk=100 MHz。
// 激活/权重读取周期包含对应 AXI 地址准备、握手等待和数据返回状态；
// MAC 周期只统计计算核心 busy 的周期；总周期从 G 命令进入执行流程开始，
// 到最后一个结果数据拍写回 DDR3 完成为止，不包含 UART 返回耗时。
reg        perf_active;
reg        perf_valid;
reg [31:0] perf_act_read_cycles;
reg [31:0] perf_weight_read_cycles;
reg [31:0] perf_mac_cycles;
reg [31:0] perf_total_cycles;
wire [127:0] perf_payload = {
    perf_total_cycles,
    perf_mac_cycles,
    perf_weight_read_cycles,
    perf_act_read_cycles
};

wire aw_handshake = axi_awvalid && axi_awready;
wire ar_handshake = axi_arvalid && axi_arready;
wire write_data_handshake = axi_wready && (aw_seen || aw_handshake);
wire read_data_handshake  = axi_rvalid && (ar_seen || ar_handshake);

wire [15:0] requested_m = config_payload[15:0];
wire [15:0] requested_k = config_payload[31:16];
wire requested_config_valid =
    (requested_m >= 16'd1) && (requested_m <= MAX_M) &&
    (requested_k >= 16'd1) && (requested_k <= MAX_K);
wire [6:0] requested_k_blocks = (requested_k + 16'd15) >> 4;
wire [4:0] requested_tail_elements =
    (requested_k[3:0] == 4'd0) ? 5'd16 : {1'b0, requested_k[3:0]};
wire [5:0] requested_act_beats = (requested_k + 16'd31) >> 5;
wire [4:0] requested_weight_beats = (requested_k + 16'd63) >> 6;
wire [3:0] requested_result_beats = (requested_m + 16'd7) >> 3;
wire [15:0] requested_total_load_beats =
    requested_act_beats + requested_m * requested_weight_beats;

wire [5:0] act_beats_remaining = act_beats - act_read_base_beat;
wire [4:0] next_act_burst_beats =
    (act_beats_remaining > 6'd16) ? 5'd16 : act_beats_remaining[4:0];
wire [8:0] result_payload_bytes = {config_m, 2'b00};
wire [3:0] final_result_outputs =
    ((result_write_index + 1'b1 == result_beats) && (config_m[2:0] != 3'd0))
        ? {1'b0, config_m[2:0]} : 4'd8;

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

gemv_param_core #(
    .MAX_K(MAX_K),
    .K_BLOCK_WIDTH(7),
    .ACT_BEAT_WIDTH(5),
    .WEIGHT_BEAT_WIDTH(4)
) u_gemv_core (
    .clk               (core_clk),
    .rst_n             (core_rst_n),
    .act_load_en       ((state == ST_READ_ACT) && read_data_handshake),
    .act_load_index    (act_read_base_beat + read_beat_index),
    .act_load_data     (axi_rdata),
    .weight_load_en    ((state == ST_READ_WEIGHT) && read_data_handshake),
    .weight_load_index (read_beat_index[3:0]),
    .weight_load_data  (axi_rdata),
    .start             (core_start),
    .k_blocks          (k_blocks),
    .tail_elements     (tail_elements),
    .busy              (core_busy),
    .done              (core_done),
    .y_value           (core_y_value)
);

function [31:0] result_strobe;
    input [3:0] valid_outputs;
    begin
        case (valid_outputs)
            4'd1: result_strobe = 32'h0000_000f;
            4'd2: result_strobe = 32'h0000_00ff;
            4'd3: result_strobe = 32'h0000_0fff;
            4'd4: result_strobe = 32'h0000_ffff;
            4'd5: result_strobe = 32'h000f_ffff;
            4'd6: result_strobe = 32'h00ff_ffff;
            4'd7: result_strobe = 32'h0fff_ffff;
            default: result_strobe = 32'hffff_ffff;
        endcase
    end
endfunction

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
            6'd14: info_char = "P";
            6'd15: info_char = "A";
            6'd16: info_char = "R";
            6'd17: info_char = "A";
            6'd18: info_char = "M";
            6'd19: info_char = " ";
            6'd20: info_char = "V";
            6'd21: info_char = "2";
            6'd22: info_char = 8'h0d;
            6'd23: info_char = 8'h0a;
            default: info_char = 8'h00;
        endcase
    end
endfunction

always @(posedge core_clk or negedge core_rst_n) begin
    if (!core_rst_n) begin
        state                   <= ST_IDLE;
        tx_index                <= 9'd0;
        tx_data                 <= 8'h00;
        tx_start                <= 1'b0;
        config_payload          <= 32'd0;
        rx_byte_index           <= 6'd0;
        upload_beat             <= 256'd0;
        load_beat_index         <= 10'd0;
        config_m                <= 7'd0;
        config_k                <= 11'd0;
        k_blocks                <= 7'd0;
        tail_elements           <= 5'd16;
        act_beats               <= 6'd0;
        weight_beats            <= 5'd0;
        result_beats            <= 4'd0;
        load_total_beats        <= 10'd0;
        result_cache            <= {MAX_M*32{1'b0}};
        act_read_base_beat      <= 6'd0;
        active_read_burst_beats <= 5'd0;
        read_beat_index         <= 5'd0;
        row_index               <= 7'd0;
        result_write_index      <= 4'd0;
        weight_row_addr         <= ADDR_WEIGHT;
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
        perf_active             <= 1'b0;
        perf_valid              <= 1'b0;
        perf_act_read_cycles    <= 32'd0;
        perf_weight_read_cycles <= 32'd0;
        perf_mac_cycles         <= 32'd0;
        perf_total_cycles       <= 32'd0;
        protocol_error          <= 1'b0;
        config_valid            <= 1'b0;
        loaded                  <= 1'b0;
        result_valid            <= 1'b0;
    end else begin
        tx_start   <= 1'b0;
        core_start <= 1'b0;

        if (perf_active) begin
            perf_total_cycles <= perf_total_cycles + 1'b1;
            if ((state == ST_SETUP_ACT_READ) || (state == ST_READ_ACT))
                perf_act_read_cycles <= perf_act_read_cycles + 1'b1;
            if ((state == ST_SETUP_WEIGHT_READ) || (state == ST_READ_WEIGHT))
                perf_weight_read_cycles <= perf_weight_read_cycles + 1'b1;
            if (core_busy)
                perf_mac_cycles <= perf_mac_cycles + 1'b1;
        end

        case (state)
            ST_IDLE: begin
                axi_awvalid <= 1'b0;
                axi_arvalid <= 1'b0;
                aw_seen     <= 1'b0;
                w_seen      <= 1'b0;
                ar_seen     <= 1'b0;
                tx_index    <= 9'd0;

                if (rx_valid) begin
                    case (rx_data)
                        8'h49, 8'h69: begin // I / i
                            state <= ST_SEND_INFO;
                        end

                        8'h53, 8'h73: begin // S / s
                            status_snapshot <= {
                                2'd0, perf_valid, core_busy, result_valid, loaded,
                                config_valid, ddr_init_done
                            };
                            state <= ST_SEND_STATUS;
                        end

                        8'h43, 8'h63: begin // C / c：配置 M、K
                            config_payload <= 32'd0;
                            rx_byte_index  <= 6'd0;
                            state          <= ST_RECV_CONFIG;
                        end

                        8'h4c, 8'h6c: begin // L / l：加载补齐后的激活和权重
                            if (!ddr_init_done) begin
                                protocol_error <= 1'b1;
                                error_code     <= 8'h02;
                                state          <= ST_SEND_ERROR;
                            end else if (!config_valid) begin
                                protocol_error <= 1'b1;
                                error_code     <= 8'h03;
                                state          <= ST_SEND_ERROR;
                            end else begin
                                upload_beat     <= 256'd0;
                                rx_byte_index   <= 6'd0;
                                load_beat_index <= 10'd0;
                                loaded          <= 1'b0;
                                result_valid    <= 1'b0;
                                perf_valid      <= 1'b0;
                                state           <= ST_RECV_LOAD;
                            end
                        end

                        8'h47, 8'h67: begin // G / g：执行参数化 GEMV
                            if (!ddr_init_done) begin
                                protocol_error <= 1'b1;
                                error_code     <= 8'h02;
                                state          <= ST_SEND_ERROR;
                            end else if (!config_valid) begin
                                protocol_error <= 1'b1;
                                error_code     <= 8'h03;
                                state          <= ST_SEND_ERROR;
                            end else if (!loaded) begin
                                protocol_error <= 1'b1;
                                error_code     <= 8'h04;
                                state          <= ST_SEND_ERROR;
                            end else begin
                                result_valid           <= 1'b0;
                                result_cache           <= {MAX_M*32{1'b0}};
                                perf_active            <= 1'b1;
                                perf_valid             <= 1'b0;
                                perf_act_read_cycles   <= 32'd0;
                                perf_weight_read_cycles<= 32'd0;
                                perf_mac_cycles        <= 32'd0;
                                perf_total_cycles      <= 32'd0;
                                act_read_base_beat     <= 6'd0;
                                read_beat_index     <= 5'd0;
                                row_index           <= 7'd0;
                                result_write_index  <= 4'd0;
                                weight_row_addr     <= ADDR_WEIGHT;
                                state               <= ST_SETUP_ACT_READ;
                            end
                        end

                        8'h50, 8'h70: begin // P / p：读取最近一次 GEMV 性能计数
                            if (!perf_valid) begin
                                protocol_error <= 1'b1;
                                error_code     <= 8'h06;
                                state          <= ST_SEND_ERROR;
                            end else begin
                                tx_index <= 9'd0;
                                state    <= ST_SEND_PERF;
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
                if (rx_valid) begin
                    config_payload[rx_byte_index*8 +: 8] <= rx_data;
                    if (rx_byte_index == 6'd3) begin
                        state <= ST_VALIDATE_CONFIG;
                    end else begin
                        rx_byte_index <= rx_byte_index + 1'b1;
                    end
                end
            end

            ST_VALIDATE_CONFIG: begin
                if (requested_config_valid) begin
                    config_m         <= requested_m[6:0];
                    config_k         <= requested_k[10:0];
                    k_blocks         <= requested_k_blocks;
                    tail_elements    <= requested_tail_elements;
                    act_beats        <= requested_act_beats;
                    weight_beats     <= requested_weight_beats;
                    result_beats     <= requested_result_beats;
                    load_total_beats <= requested_total_load_beats[9:0];
                    config_valid     <= 1'b1;
                    loaded           <= 1'b0;
                    result_valid     <= 1'b0;
                    perf_valid       <= 1'b0;
                    state            <= ST_SEND_ACK;
                end else begin
                    config_valid   <= 1'b0;
                    loaded         <= 1'b0;
                    result_valid   <= 1'b0;
                    protocol_error <= 1'b1;
                    error_code     <= 8'h05;
                    state          <= ST_SEND_ERROR;
                end
            end

            ST_RECV_LOAD: begin
                if (rx_valid) begin
                    upload_beat[rx_byte_index*8 +: 8] <= rx_data;
                    if (rx_byte_index == 6'd31) begin
                        state <= ST_SETUP_LOAD_WRITE;
                    end else begin
                        rx_byte_index <= rx_byte_index + 1'b1;
                    end
                end
            end

            ST_SETUP_LOAD_WRITE: begin
                if (load_beat_index < act_beats) begin
                    axi_awaddr <= ADDR_ACT + (load_beat_index << 3);
                end else begin
                    axi_awaddr <= ADDR_WEIGHT + ((load_beat_index - act_beats) << 3);
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

                    if (load_beat_index + 1'b1 == load_total_beats) begin
                        loaded  <= 1'b1;
                        state   <= ST_SEND_ACK;
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
                        if (act_read_base_beat + active_read_burst_beats == act_beats) begin
                            row_index       <= 7'd0;
                            weight_row_addr <= ADDR_WEIGHT;
                            state           <= ST_SETUP_WEIGHT_READ;
                        end else begin
                            act_read_base_beat <= act_read_base_beat + active_read_burst_beats;
                            state              <= ST_SETUP_ACT_READ;
                        end
                    end else begin
                        read_beat_index <= read_beat_index + 1'b1;
                    end
                end
            end

            ST_SETUP_WEIGHT_READ: begin
                axi_araddr       <= weight_row_addr;
                axi_arlen        <= weight_beats - 1'b1;
                axi_arvalid      <= 1'b1;
                ar_seen          <= 1'b0;
                read_beat_index  <= 5'd0;
                state            <= ST_READ_WEIGHT;
            end

            ST_READ_WEIGHT: begin
                if (ar_handshake) begin
                    axi_arvalid <= 1'b0;
                    ar_seen     <= 1'b1;
                end

                if (read_data_handshake) begin
                    if (read_beat_index + 1'b1 == weight_beats) begin
                        ar_seen <= 1'b0;
                        state   <= ST_START_CORE;
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
                    result_cache[row_index*32 +: 32] <= core_y_value;
                    if (row_index + 1'b1 == config_m) begin
                        result_write_index <= 4'd0;
                        state              <= ST_SETUP_RESULT_WRITE;
                    end else begin
                        row_index       <= row_index + 1'b1;
                        weight_row_addr <= weight_row_addr + (weight_beats << 3);
                        state           <= ST_SETUP_WEIGHT_READ;
                    end
                end
            end

            ST_SETUP_RESULT_WRITE: begin
                axi_awaddr  <= ADDR_RESULT + (result_write_index << 3);
                axi_awvalid <= 1'b1;
                axi_wdata   <= result_cache[result_write_index*256 +: 256];
                axi_wstrb   <= result_strobe(final_result_outputs);
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
                    if (result_write_index + 1'b1 == result_beats) begin
                        result_valid <= 1'b1;
                        perf_active  <= 1'b0;
                        perf_valid   <= 1'b1;
                        tx_index     <= 9'd0;
                        state        <= ST_SEND_RESULT;
                    end else begin
                        result_write_index <= result_write_index + 1'b1;
                        state              <= ST_SETUP_RESULT_WRITE;
                    end
                end
            end

            ST_SEND_INFO: begin
                if (!tx_busy && !tx_start) begin
                    if (tx_index < 9'd24) begin
                        tx_data  <= info_char(tx_index[5:0]);
                        tx_start <= 1'b1;
                        tx_index <= tx_index + 1'b1;
                    end else begin
                        state <= ST_IDLE;
                    end
                end
            end

            ST_SEND_STATUS: begin
                if (!tx_busy && !tx_start) begin
                    if (tx_index < 9'd4) begin
                        case (tx_index)
                            9'd0: tx_data <= "S";
                            9'd1: tx_data <= status_snapshot;
                            9'd2: tx_data <= 8'h0d;
                            9'd3: tx_data <= 8'h0a;
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
                    if (tx_index < 9'd3) begin
                        case (tx_index)
                            9'd0: tx_data <= "K";
                            9'd1: tx_data <= 8'h0d;
                            9'd2: tx_data <= 8'h0a;
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
                    if (tx_index <= result_payload_bytes) begin
                        if (tx_index == 9'd0)
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

            ST_SEND_PERF: begin
                if (!tx_busy && !tx_start) begin
                    if (tx_index < 9'd17) begin
                        if (tx_index == 9'd0)
                            tx_data <= "P";
                        else
                            tx_data <= perf_payload[(tx_index-1'b1)*8 +: 8];
                        tx_start <= 1'b1;
                        tx_index <= tx_index + 1'b1;
                    end else begin
                        state <= ST_IDLE;
                    end
                end
            end

            ST_SEND_ERROR: begin
                if (!tx_busy && !tx_start) begin
                    if (tx_index < 9'd4) begin
                        case (tx_index)
                            9'd0: tx_data <= "E";
                            9'd1: tx_data <= error_code;
                            9'd2: tx_data <= 8'h0d;
                            9'd3: tx_data <= 8'h0a;
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
                tx_index       <= 9'd0;
                state          <= ST_SEND_ERROR;
            end
        endcase
    end
end

wire _unused_config_k = &{1'b0, config_k};

endmodule
