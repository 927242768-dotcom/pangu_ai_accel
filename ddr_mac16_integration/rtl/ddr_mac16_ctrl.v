`timescale 1ns/1ps

// DDR3 + INT8 MAC16 集成验证控制器。
//
// 串口协议（115200 8N1）：
//   I                     -> "PANGU50K DDR3 MAC16 V2\r\n"
//   S                     -> 'S' + 状态字节 + "\r\n"
//   L + 16B激活 + 16B INT8权重 -> 将两组向量写入 DDR3，完成后回复 "K\r\n"
//   Q + 16B激活 + 8B packed INT4权重 -> 写入 DDR3，完成后回复 "K\r\n"
//   G                     -> 从 DDR3 以 2 拍×256 bit burst 读回数据并执行 MAC16，
//                            然后回复 'R' + little-endian int32
//
// DDR3 控制器地址单位为 32 bit。256 bit 单拍数据占 8 个地址单位：
//   0x00 : 激活向量（低 128 bit）
//   0x08 : 权重向量（INT8 使用低 128 bit；packed INT4 使用低 64 bit）
//   0x10 : 32 bit 点积结果（低 32 bit）
module ddr_mac16_ctrl #(
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
    output wire [3:0]                   axi_arlen,
    input  wire                         axi_arready,
    output reg                          axi_arvalid,

    input  wire [255:0]                 axi_rdata,
    input  wire                         axi_rvalid,

    output wire [5:0]                   debug_state,
    output reg                          protocol_error,
    output reg                          loaded,
    output reg                          result_valid
);

localparam [CTRL_ADDR_WIDTH-1:0] ADDR_ACT    = {CTRL_ADDR_WIDTH{1'b0}};
localparam [CTRL_ADDR_WIDTH-1:0] ADDR_WEIGHT = {{(CTRL_ADDR_WIDTH-4){1'b0}}, 4'h8};
localparam [CTRL_ADDR_WIDTH-1:0] ADDR_RESULT = {{(CTRL_ADDR_WIDTH-5){1'b0}}, 5'h10};

localparam [5:0] ST_IDLE            = 6'd0;
localparam [5:0] ST_RECV_LOAD       = 6'd1;
localparam [5:0] ST_WRITE_ACT       = 6'd2;
localparam [5:0] ST_WRITE_WEIGHT    = 6'd3;
localparam [5:0] ST_READ_BURST      = 6'd4;
localparam [5:0] ST_PREPARE_MAC     = 6'd5;
localparam [5:0] ST_COMPUTE         = 6'd6;
localparam [5:0] ST_SETUP_RESULT_WR = 6'd7;
localparam [5:0] ST_WRITE_RESULT    = 6'd8;
localparam [5:0] ST_SEND_INFO       = 6'd9;
localparam [5:0] ST_SEND_STATUS     = 6'd10;
localparam [5:0] ST_SEND_ACK        = 6'd11;
localparam [5:0] ST_SEND_RESULT     = 6'd12;
localparam [5:0] ST_SEND_ERROR      = 6'd13;

reg [5:0] state;
reg [5:0] rx_count;
reg [5:0] rx_last_index;
reg [5:0] tx_index;
reg [7:0] tx_data;
reg       tx_start;
wire      tx_busy;
wire [7:0] rx_data;
wire       rx_valid;

reg [127:0] upload_act_vec;
reg [127:0] upload_weight_vec;
reg [255:0] read_act_data;
reg [255:0] read_weight_data;
reg [127:0] mac_act_vec_reg;
reg [127:0] mac_weight_vec_reg;
reg weight_int4_mode;
reg signed [31:0] result_reg;
wire [127:0] unpacked_int4_weight_vec;
wire [127:0] selected_weight_vec;
wire signed [31:0] dot_result;

reg aw_seen;
reg w_seen;
reg ar_seen;
reg read_beat_count;
reg [7:0] status_snapshot;
reg [7:0] error_code;

wire aw_handshake = axi_awvalid && axi_awready;
wire ar_handshake = axi_arvalid && axi_arready;
wire write_data_handshake = axi_wready && (aw_seen || aw_handshake);
wire read_data_handshake  = axi_rvalid && (ar_seen || ar_handshake);

assign axi_awuser_ap = 1'b0;
assign axi_awuser_id = 4'h0;
assign axi_awlen     = 4'h0;
assign axi_aruser_ap = 1'b0;
assign axi_aruser_id = 4'h0;
// 一次读取两个相邻的 256 bit 数据拍：激活向量 + 权重向量。
assign axi_arlen     = 4'h1;
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

int4_unpack16 u_int4_unpack16 (
    .packed_vec   (read_weight_data[63:0]),
    .unpacked_vec (unpacked_int4_weight_vec)
);

assign selected_weight_vec = weight_int4_mode
                           ? unpacked_int4_weight_vec
                           : read_weight_data[127:0];

int8_dot16 u_int8_dot16 (
    .a_vec  (mac_act_vec_reg),
    .b_vec  (mac_weight_vec_reg),
    .result (dot_result)
);

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
            6'd9:  info_char = "D";
            6'd10: info_char = "D";
            6'd11: info_char = "R";
            6'd12: info_char = "3";
            6'd13: info_char = " ";
            6'd14: info_char = "M";
            6'd15: info_char = "A";
            6'd16: info_char = "C";
            6'd17: info_char = "1";
            6'd18: info_char = "6";
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
        state             <= ST_IDLE;
        rx_count          <= 6'd0;
        rx_last_index     <= 6'd31;
        tx_index          <= 6'd0;
        tx_data           <= 8'h00;
        tx_start          <= 1'b0;
        upload_act_vec    <= 128'd0;
        upload_weight_vec <= 128'd0;
        read_act_data     <= 256'd0;
        read_weight_data  <= 256'd0;
        mac_act_vec_reg    <= 128'd0;
        mac_weight_vec_reg <= 128'd0;
        weight_int4_mode  <= 1'b0;
        result_reg        <= 32'sd0;
        axi_awaddr        <= {CTRL_ADDR_WIDTH{1'b0}};
        axi_awvalid       <= 1'b0;
        axi_wdata         <= 256'd0;
        axi_wstrb         <= 32'd0;
        axi_araddr        <= {CTRL_ADDR_WIDTH{1'b0}};
        axi_arvalid       <= 1'b0;
        aw_seen           <= 1'b0;
        w_seen            <= 1'b0;
        ar_seen           <= 1'b0;
        read_beat_count    <= 1'b0;
        status_snapshot   <= 8'd0;
        error_code        <= 8'd0;
        protocol_error    <= 1'b0;
        loaded            <= 1'b0;
        result_valid      <= 1'b0;
    end else begin
        tx_start <= 1'b0;

        case (state)
            ST_IDLE: begin
                axi_awvalid <= 1'b0;
                axi_arvalid <= 1'b0;
                aw_seen     <= 1'b0;
                w_seen      <= 1'b0;
                ar_seen     <= 1'b0;
                rx_count    <= 6'd0;
                tx_index    <= 6'd0;

                if (rx_valid) begin
                    case (rx_data)
                        8'h49, 8'h69: begin // I / i
                            tx_index <= 6'd0;
                            state    <= ST_SEND_INFO;
                        end

                        8'h53, 8'h73: begin // S / s
                            status_snapshot <= {4'd0, weight_int4_mode, result_valid, loaded, ddr_init_done};
                            tx_index        <= 6'd0;
                            state           <= ST_SEND_STATUS;
                        end

                        8'h4c, 8'h6c: begin // L / l
                            if (ddr_init_done) begin
                                upload_act_vec    <= 128'd0;
                                upload_weight_vec <= 128'd0;
                                loaded            <= 1'b0;
                                result_valid      <= 1'b0;
                                weight_int4_mode  <= 1'b0;
                                rx_count          <= 6'd0;
                                rx_last_index     <= 6'd31;
                                state             <= ST_RECV_LOAD;
                            end else begin
                                protocol_error <= 1'b1;
                                error_code     <= 8'h02; // DDR 尚未初始化
                                tx_index       <= 6'd0;
                                state          <= ST_SEND_ERROR;
                            end
                        end

                        8'h51, 8'h71: begin // Q / q：packed INT4 权重
                            if (ddr_init_done) begin
                                upload_act_vec    <= 128'd0;
                                upload_weight_vec <= 128'd0;
                                loaded            <= 1'b0;
                                result_valid      <= 1'b0;
                                weight_int4_mode  <= 1'b1;
                                rx_count          <= 6'd0;
                                rx_last_index     <= 6'd23;
                                state             <= ST_RECV_LOAD;
                            end else begin
                                protocol_error <= 1'b1;
                                error_code     <= 8'h02;
                                tx_index       <= 6'd0;
                                state          <= ST_SEND_ERROR;
                            end
                        end

                        8'h47, 8'h67: begin // G / g
                            if (!ddr_init_done) begin
                                protocol_error <= 1'b1;
                                error_code     <= 8'h02;
                                tx_index       <= 6'd0;
                                state          <= ST_SEND_ERROR;
                            end else if (!loaded) begin
                                protocol_error <= 1'b1;
                                error_code     <= 8'h03; // 尚未加载向量
                                tx_index       <= 6'd0;
                                state          <= ST_SEND_ERROR;
                            end else begin
                                result_valid <= 1'b0;
                                axi_araddr       <= ADDR_ACT;
                                axi_arvalid      <= 1'b1;
                                ar_seen          <= 1'b0;
                                read_beat_count  <= 1'b0;
                                state            <= ST_READ_BURST;
                            end
                        end

                        default: begin
                            protocol_error <= 1'b1;
                            error_code     <= 8'h01; // 未知命令
                            tx_index       <= 6'd0;
                            state          <= ST_SEND_ERROR;
                        end
                    endcase
                end
            end

            ST_RECV_LOAD: begin
                if (rx_valid) begin
                    if (rx_count < 6'd16)
                        upload_act_vec[rx_count*8 +: 8] <= rx_data;
                    else
                        upload_weight_vec[(rx_count-6'd16)*8 +: 8] <= rx_data;

                    if (rx_count == rx_last_index) begin
                        axi_awaddr  <= ADDR_ACT;
                        axi_awvalid <= 1'b1;
                        axi_wdata   <= {128'd0, upload_act_vec};
                        axi_wstrb   <= 32'hffff_ffff;
                        aw_seen     <= 1'b0;
                        w_seen      <= 1'b0;
                        state       <= ST_WRITE_ACT;
                    end else begin
                        rx_count <= rx_count + 1'b1;
                    end
                end
            end

            ST_WRITE_ACT: begin
                if (aw_handshake) begin
                    axi_awvalid <= 1'b0;
                    aw_seen     <= 1'b1;
                end
                if (write_data_handshake)
                    w_seen <= 1'b1;

                if ((aw_seen || aw_handshake) && (w_seen || write_data_handshake)) begin
                    axi_awaddr  <= ADDR_WEIGHT;
                    axi_awvalid <= 1'b1;
                    axi_wdata   <= {128'd0, upload_weight_vec};
                    axi_wstrb   <= 32'hffff_ffff;
                    aw_seen     <= 1'b0;
                    w_seen      <= 1'b0;
                    state       <= ST_WRITE_WEIGHT;
                end
            end

            ST_WRITE_WEIGHT: begin
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
                    loaded      <= 1'b1;
                    tx_index    <= 6'd0;
                    state       <= ST_SEND_ACK;
                end
            end

            ST_READ_BURST: begin
                if (ar_handshake) begin
                    axi_arvalid <= 1'b0;
                    ar_seen     <= 1'b1;
                end

                if (read_data_handshake) begin
                    if (!read_beat_count) begin
                        // Burst beat 0，DDR 地址 0x00：INT8 激活向量。
                        read_act_data  <= axi_rdata;
                        read_beat_count <= 1'b1;
                    end else begin
                        // Burst beat 1，DDR 地址 0x08：INT8 权重向量。
                        read_weight_data <= axi_rdata;
                        read_beat_count   <= 1'b0;
                        axi_arvalid       <= 1'b0;
                        ar_seen           <= 1'b0;
                        state             <= ST_PREPARE_MAC;
                    end
                end
            end

            ST_PREPARE_MAC: begin
                // 流水级 1：完成 DDR 数据拆分、INT4 解包和输入选择。
                // 下一周期再执行 MAC16，切断 unpack/mux + 乘加的长组合路径。
                mac_act_vec_reg    <= read_act_data[127:0];
                mac_weight_vec_reg <= selected_weight_vec;
                state              <= ST_COMPUTE;
            end

            ST_COMPUTE: begin
                // 流水级 2：16 路有符号乘加。
                result_reg <= dot_result;
                state      <= ST_SETUP_RESULT_WR;
            end

            ST_SETUP_RESULT_WR: begin
                axi_awaddr  <= ADDR_RESULT;
                axi_awvalid <= 1'b1;
                axi_wdata   <= {224'd0, result_reg};
                axi_wstrb   <= 32'h0000_000f;
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
                    axi_awvalid  <= 1'b0;
                    aw_seen      <= 1'b0;
                    w_seen       <= 1'b0;
                    result_valid <= 1'b1;
                    tx_index     <= 6'd0;
                    state        <= ST_SEND_RESULT;
                end
            end

            ST_SEND_INFO: begin
                if (!tx_busy && !tx_start) begin
                    if (tx_index < 6'd24) begin
                        tx_data  <= info_char(tx_index);
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

            ST_SEND_RESULT: begin
                if (!tx_busy && !tx_start) begin
                    if (tx_index < 6'd5) begin
                        case (tx_index)
                            6'd0: tx_data <= "R";
                            6'd1: tx_data <= result_reg[7:0];
                            6'd2: tx_data <= result_reg[15:8];
                            6'd3: tx_data <= result_reg[23:16];
                            6'd4: tx_data <= result_reg[31:24];
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
                tx_index       <= 6'd0;
                axi_awvalid    <= 1'b0;
                axi_arvalid    <= 1'b0;
                state          <= ST_SEND_ERROR;
            end
        endcase
    end
end

endmodule
