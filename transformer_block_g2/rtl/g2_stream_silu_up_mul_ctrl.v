`timescale 1ns/1ps

// G2 流式 SiLU(gate) x up 控制器。
//
// 每组读取一个 16-lane Q6.10 SiLU beat 与四个 Q28 up beat，逐 lane 复现
// 已验证的完整 signed 16x64 -> 80-bit Q38、RNE >> 10 和 int64 饱和。
// 仅缓存当前 16 个元素，替代历史独立工程的 40 DRM 整向量缓存。
module g2_stream_silu_up_mul_ctrl #(
    parameter integer CTRL_ADDR_WIDTH = 28,
    parameter integer INPUT_GROUPS    = 304
)(
    input  wire                         clk,
    input  wire                         rst_n,
    input  wire                         ddr_init_done,
    input  wire                         start,
    input  wire [CTRL_ADDR_WIDTH-1:0]   cfg_silu_addr,
    input  wire [CTRL_ADDR_WIDTH-1:0]   cfg_up_addr,
    input  wire [CTRL_ADDR_WIDTH-1:0]   cfg_result_addr,

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

    output reg                          busy,
    output reg                          done,
    output reg                          error,
    output reg  [7:0]                   error_code,
    output wire [4:0]                   debug_state,
    output reg  [8:0]                   debug_group_index
);

localparam [4:0] ST_IDLE            = 5'd0;
localparam [4:0] ST_SETUP_SILU_READ = 5'd1;
localparam [4:0] ST_READ_SILU       = 5'd2;
localparam [4:0] ST_SETUP_UP_READ   = 5'd3;
localparam [4:0] ST_READ_UP         = 5'd4;
localparam [4:0] ST_CAPTURE         = 5'd5;
localparam [4:0] ST_ABS             = 5'd6;
localparam [4:0] ST_MUL_CAPTURE     = 5'd7;
localparam [4:0] ST_MUL_ACCUM       = 5'd8;
localparam [4:0] ST_ROUND           = 5'd9;
localparam [4:0] ST_SATURATE        = 5'd10;
localparam [4:0] ST_PACK            = 5'd11;
localparam [4:0] ST_SETUP_WRITE     = 5'd12;
localparam [4:0] ST_WRITE           = 5'd13;
localparam [4:0] ST_FINISH          = 5'd14;
localparam [4:0] ST_ERROR           = 5'd31;

localparam [7:0] ERR_DDR_NOT_READY = 8'h01;
localparam [7:0] ERR_INTERNAL      = 8'hff;

localparam [69:0] POSITIVE_LIMIT = {6'd0, 64'h7fff_ffff_ffff_ffff};
localparam [69:0] NEGATIVE_LIMIT = {6'd0, 64'h8000_0000_0000_0000};

reg [4:0] state;
reg [CTRL_ADDR_WIDTH-1:0] silu_base_addr;
reg [CTRL_ADDR_WIDTH-1:0] up_base_addr;
reg [CTRL_ADDR_WIDTH-1:0] result_base_addr;
reg [8:0] group_index;
reg [2:0] up_read_index;
reg [3:0] lane_index;
reg ar_seen;
reg aw_seen;
reg w_seen;

reg [255:0] silu_beat;
reg [255:0] up_beat0;
reg [255:0] up_beat1;
reg [255:0] up_beat2;
reg [255:0] up_beat3;
reg [255:0] output_pack;

reg signed [15:0] silu_lane_reg;
reg signed [63:0] up_lane_reg;
reg               product_negative_reg;
reg [15:0]        silu_magnitude_reg;
reg [63:0]        up_magnitude_reg;
reg [1:0]         mul_limb_index;
reg [31:0]        partial_product_reg;
reg [79:0]        product_magnitude_reg;
reg [69:0]        rounded_magnitude_reg;
reg [63:0]        output_value_reg;

wire aw_handshake = axi_awvalid && axi_awready;
wire ar_handshake = axi_arvalid && axi_arready;
wire write_data_handshake = axi_wready && (aw_seen || aw_handshake);
wire read_data_handshake = axi_rvalid && (ar_seen || ar_handshake);

reg [15:0] selected_silu_bits;
reg [63:0] selected_up_bits;
always @(*) begin
    selected_silu_bits = silu_beat[lane_index*16 +: 16];
    case (lane_index[3:2])
        2'd0: selected_up_bits = up_beat0[lane_index[1:0]*64 +: 64];
        2'd1: selected_up_bits = up_beat1[lane_index[1:0]*64 +: 64];
        2'd2: selected_up_bits = up_beat2[lane_index[1:0]*64 +: 64];
        default: selected_up_bits = up_beat3[lane_index[1:0]*64 +: 64];
    endcase
end

reg [15:0] selected_up_limb;
always @(*) begin
    case (mul_limb_index)
        2'd0: selected_up_limb = up_magnitude_reg[15:0];
        2'd1: selected_up_limb = up_magnitude_reg[31:16];
        2'd2: selected_up_limb = up_magnitude_reg[47:32];
        default: selected_up_limb = up_magnitude_reg[63:48];
    endcase
end

wire [31:0] partial_product_wire = silu_magnitude_reg * selected_up_limb;
wire [79:0] partial_product_ext = {{48{1'b0}}, partial_product_reg};

function [69:0] rne_shift10_unsigned80;
    input [79:0] magnitude;
    reg [69:0] quotient;
    reg [9:0] remainder;
    begin
        quotient = magnitude >> 10;
        remainder = magnitude[9:0];
        if ((remainder > 10'h200) ||
            ((remainder == 10'h200) && quotient[0]))
            quotient = quotient + 1'b1;
        rne_shift10_unsigned80 = quotient;
    end
endfunction

reg [255:0] output_pack_next;
always @(*) begin
    output_pack_next = output_pack;
    output_pack_next[lane_index[1:0]*64 +: 64] = output_value_reg;
end

assign axi_awuser_ap = 1'b0;
assign axi_awuser_id = 4'h0;
assign axi_awlen     = 4'h0;
assign axi_aruser_ap = 1'b0;
assign axi_aruser_id = 4'h0;
assign debug_state   = state;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        state                 <= ST_IDLE;
        silu_base_addr        <= {CTRL_ADDR_WIDTH{1'b0}};
        up_base_addr          <= {CTRL_ADDR_WIDTH{1'b0}};
        result_base_addr      <= {CTRL_ADDR_WIDTH{1'b0}};
        group_index           <= 9'd0;
        up_read_index         <= 3'd0;
        lane_index            <= 4'd0;
        ar_seen               <= 1'b0;
        aw_seen               <= 1'b0;
        w_seen                <= 1'b0;
        silu_beat             <= 256'd0;
        up_beat0              <= 256'd0;
        up_beat1              <= 256'd0;
        up_beat2              <= 256'd0;
        up_beat3              <= 256'd0;
        output_pack           <= 256'd0;
        silu_lane_reg         <= 16'sd0;
        up_lane_reg           <= 64'sd0;
        product_negative_reg  <= 1'b0;
        silu_magnitude_reg    <= 16'd0;
        up_magnitude_reg      <= 64'd0;
        mul_limb_index        <= 2'd0;
        partial_product_reg   <= 32'd0;
        product_magnitude_reg <= 80'd0;
        rounded_magnitude_reg <= 70'd0;
        output_value_reg      <= 64'd0;
        axi_awaddr            <= {CTRL_ADDR_WIDTH{1'b0}};
        axi_awvalid           <= 1'b0;
        axi_wdata             <= 256'd0;
        axi_wstrb             <= 32'd0;
        axi_araddr            <= {CTRL_ADDR_WIDTH{1'b0}};
        axi_arlen             <= 4'd0;
        axi_arvalid           <= 1'b0;
        busy                  <= 1'b0;
        done                  <= 1'b0;
        error                 <= 1'b0;
        error_code            <= 8'd0;
        debug_group_index     <= 9'd0;
    end else begin
        done <= 1'b0;

        case (state)
            ST_IDLE: begin
                axi_awvalid <= 1'b0;
                axi_arvalid <= 1'b0;
                aw_seen     <= 1'b0;
                w_seen      <= 1'b0;
                ar_seen     <= 1'b0;
                busy        <= 1'b0;
                if (start && !error) begin
                    if (!ddr_init_done) begin
                        error      <= 1'b1;
                        error_code <= ERR_DDR_NOT_READY;
                        state      <= ST_ERROR;
                    end else begin
                        silu_base_addr    <= cfg_silu_addr;
                        up_base_addr      <= cfg_up_addr;
                        result_base_addr  <= cfg_result_addr;
                        group_index       <= 9'd0;
                        debug_group_index <= 9'd0;
                        busy              <= 1'b1;
                        error_code        <= 8'd0;
                        state             <= ST_SETUP_SILU_READ;
                    end
                end
            end

            ST_SETUP_SILU_READ: begin
                axi_araddr  <= silu_base_addr + ({19'd0, group_index} << 3);
                axi_arlen   <= 4'd0;
                axi_arvalid <= 1'b1;
                ar_seen     <= 1'b0;
                state       <= ST_READ_SILU;
            end

            ST_READ_SILU: begin
                if (ar_handshake) begin
                    axi_arvalid <= 1'b0;
                    ar_seen     <= 1'b1;
                end
                if (read_data_handshake) begin
                    silu_beat     <= axi_rdata;
                    ar_seen       <= 1'b0;
                    up_read_index <= 3'd0;
                    state         <= ST_SETUP_UP_READ;
                end
            end

            ST_SETUP_UP_READ: begin
                axi_araddr  <= up_base_addr + ({17'd0, group_index} << 5);
                axi_arlen   <= 4'd3;
                axi_arvalid <= 1'b1;
                ar_seen     <= 1'b0;
                state       <= ST_READ_UP;
            end

            ST_READ_UP: begin
                if (ar_handshake) begin
                    axi_arvalid <= 1'b0;
                    ar_seen     <= 1'b1;
                end
                if (read_data_handshake) begin
                    case (up_read_index)
                        3'd0: up_beat0 <= axi_rdata;
                        3'd1: up_beat1 <= axi_rdata;
                        3'd2: up_beat2 <= axi_rdata;
                        default: up_beat3 <= axi_rdata;
                    endcase
                    if (up_read_index == 3'd3) begin
                        ar_seen       <= 1'b0;
                        lane_index    <= 4'd0;
                        output_pack   <= 256'd0;
                        state         <= ST_CAPTURE;
                    end else begin
                        up_read_index <= up_read_index + 1'b1;
                    end
                end
            end

            ST_CAPTURE: begin
                silu_lane_reg <= $signed(selected_silu_bits);
                up_lane_reg   <= $signed(selected_up_bits);
                state         <= ST_ABS;
            end

            ST_ABS: begin
                product_negative_reg <= silu_lane_reg[15] ^ up_lane_reg[63];
                silu_magnitude_reg <= silu_lane_reg[15] ?
                    (~silu_lane_reg + 1'b1) : silu_lane_reg;
                up_magnitude_reg <= up_lane_reg[63] ?
                    (~up_lane_reg + 1'b1) : up_lane_reg;
                mul_limb_index        <= 2'd0;
                partial_product_reg   <= 32'd0;
                product_magnitude_reg <= 80'd0;
                state                 <= ST_MUL_CAPTURE;
            end

            // 将 16x16 APM 乘法和 80-bit 移位累加拆成两个周期。
            // 原实现把 APM 输出直接接入 80-bit carry chain，综合慢角约差 0.6ns。
            ST_MUL_CAPTURE: begin
                partial_product_reg <= partial_product_wire;
                state               <= ST_MUL_ACCUM;
            end

            ST_MUL_ACCUM: begin
                case (mul_limb_index)
                    2'd0: product_magnitude_reg <= partial_product_ext;
                    2'd1: product_magnitude_reg <=
                        product_magnitude_reg + (partial_product_ext << 16);
                    2'd2: product_magnitude_reg <=
                        product_magnitude_reg + (partial_product_ext << 32);
                    default: product_magnitude_reg <=
                        product_magnitude_reg + (partial_product_ext << 48);
                endcase

                if (mul_limb_index == 2'd3) begin
                    state <= ST_ROUND;
                end else begin
                    mul_limb_index <= mul_limb_index + 1'b1;
                    state          <= ST_MUL_CAPTURE;
                end
            end

            ST_ROUND: begin
                rounded_magnitude_reg <= rne_shift10_unsigned80(product_magnitude_reg);
                state <= ST_SATURATE;
            end

            ST_SATURATE: begin
                if (product_negative_reg) begin
                    if (rounded_magnitude_reg >= NEGATIVE_LIMIT)
                        output_value_reg <= 64'h8000_0000_0000_0000;
                    else
                        output_value_reg <= (~rounded_magnitude_reg[63:0]) + 1'b1;
                end else begin
                    if (rounded_magnitude_reg > POSITIVE_LIMIT)
                        output_value_reg <= 64'h7fff_ffff_ffff_ffff;
                    else
                        output_value_reg <= rounded_magnitude_reg[63:0];
                end
                state <= ST_PACK;
            end

            ST_PACK: begin
                output_pack <= output_pack_next;
                if (lane_index[1:0] == 2'd3) begin
                    state <= ST_SETUP_WRITE;
                end else begin
                    lane_index <= lane_index + 1'b1;
                    state      <= ST_CAPTURE;
                end
            end

            ST_SETUP_WRITE: begin
                axi_awaddr <= result_base_addr +
                    ((({19'd0, group_index} << 2) + lane_index[3:2]) << 3);
                axi_awvalid <= 1'b1;
                axi_wdata   <= output_pack;
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
                    output_pack <= 256'd0;
                    if (lane_index == 4'd15) begin
                        if (group_index == INPUT_GROUPS - 1) begin
                            state <= ST_FINISH;
                        end else begin
                            group_index       <= group_index + 1'b1;
                            debug_group_index <= group_index + 1'b1;
                            state             <= ST_SETUP_SILU_READ;
                        end
                    end else begin
                        lane_index <= lane_index + 1'b1;
                        state      <= ST_CAPTURE;
                    end
                end
            end

            ST_FINISH: begin
                busy  <= 1'b0;
                done  <= 1'b1;
                state <= ST_IDLE;
            end

            ST_ERROR: begin
                busy        <= 1'b0;
                axi_awvalid <= 1'b0;
                axi_arvalid <= 1'b0;
            end

            default: begin
                error       <= 1'b1;
                error_code  <= ERR_INTERNAL;
                busy        <= 1'b0;
                axi_awvalid <= 1'b0;
                axi_arvalid <= 1'b0;
                state       <= ST_ERROR;
            end
        endcase
    end
end

endmodule
