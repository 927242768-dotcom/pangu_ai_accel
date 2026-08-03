`timescale 1ns/1ps

// G2 流式 Q28->Q6.10 残差控制器。
//
// 每次只缓存一个 hidden beat 和四个 Q28 beat，逐 lane 执行与已经验证的
// attention_residual/mlp_residual 完全相同的 signed RNE、两级 int16 饱和。
// 这样同一个实例可复用在 RESIDUAL1 与 RESIDUAL2，并避免历史独立工程中
// 20 个 DRM 的整向量缓存。
module g2_stream_residual_ctrl #(
    parameter integer CTRL_ADDR_WIDTH = 28,
    parameter integer RESULT_BEATS    = 56
)(
    input  wire                         clk,
    input  wire                         rst_n,
    input  wire                         ddr_init_done,
    input  wire                         start,
    input  wire [CTRL_ADDR_WIDTH-1:0]   cfg_hidden_addr,
    input  wire [CTRL_ADDR_WIDTH-1:0]   cfg_q28_addr,
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
    output reg  [5:0]                   debug_beat_index
);

localparam [4:0] ST_IDLE              = 5'd0;
localparam [4:0] ST_SETUP_HIDDEN_READ = 5'd1;
localparam [4:0] ST_READ_HIDDEN       = 5'd2;
localparam [4:0] ST_SETUP_Q28_READ    = 5'd3;
localparam [4:0] ST_READ_Q28          = 5'd4;
localparam [4:0] ST_CAPTURE           = 5'd5;
localparam [4:0] ST_ABS               = 5'd6;
localparam [4:0] ST_ROUND             = 5'd7;
localparam [4:0] ST_SAT_Q10           = 5'd8;
localparam [4:0] ST_ADD               = 5'd9;
localparam [4:0] ST_SAT_OUT           = 5'd10;
localparam [4:0] ST_PACK              = 5'd11;
localparam [4:0] ST_SETUP_WRITE       = 5'd12;
localparam [4:0] ST_WRITE             = 5'd13;
localparam [4:0] ST_FINISH            = 5'd14;
localparam [4:0] ST_ERROR             = 5'd31;

localparam [7:0] ERR_DDR_NOT_READY = 8'h01;
localparam [7:0] ERR_INTERNAL      = 8'hff;

reg [4:0] state;
reg [CTRL_ADDR_WIDTH-1:0] hidden_base_addr;
reg [CTRL_ADDR_WIDTH-1:0] q28_base_addr;
reg [CTRL_ADDR_WIDTH-1:0] result_base_addr;
reg [5:0] beat_index;
reg [2:0] q28_read_index;
reg [3:0] lane_index;
reg ar_seen;
reg aw_seen;
reg w_seen;

reg [255:0] hidden_beat;
reg [255:0] q28_beat0;
reg [255:0] q28_beat1;
reg [255:0] q28_beat2;
reg [255:0] q28_beat3;
reg [255:0] output_pack;

reg signed [15:0] hidden_lane_reg;
reg signed [63:0] q28_lane_reg;
reg               q28_negative_reg;
reg [63:0]        q28_magnitude_reg;
reg signed [63:0] q10_wide_reg;
reg signed [15:0] q10_reg;
reg signed [17:0] residual_sum_reg;
reg [15:0]        output_lane_reg;

wire aw_handshake = axi_awvalid && axi_awready;
wire ar_handshake = axi_arvalid && axi_arready;
wire write_data_handshake = axi_wready && (aw_seen || aw_handshake);
wire read_data_handshake = axi_rvalid && (ar_seen || ar_handshake);

reg [63:0] selected_q28_bits;
always @(*) begin
    case (lane_index[3:2])
        2'd0: selected_q28_bits = q28_beat0[lane_index[1:0]*64 +: 64];
        2'd1: selected_q28_bits = q28_beat1[lane_index[1:0]*64 +: 64];
        2'd2: selected_q28_bits = q28_beat2[lane_index[1:0]*64 +: 64];
        default: selected_q28_bits = q28_beat3[lane_index[1:0]*64 +: 64];
    endcase
end

wire signed [15:0] selected_hidden_lane =
    $signed(hidden_beat[lane_index*16 +: 16]);

function signed [63:0] rne_shift18_from_magnitude;
    input [63:0] magnitude;
    input        negative;
    reg [63:0] quotient;
    reg [17:0] remainder;
    begin
        quotient = magnitude >> 18;
        remainder = magnitude[17:0];
        if ((remainder > 18'h20000) ||
            ((remainder == 18'h20000) && quotient[0]))
            quotient = quotient + 1'b1;
        rne_shift18_from_magnitude = negative ? -$signed(quotient) : $signed(quotient);
    end
endfunction

function signed [15:0] saturate_signed16_from64;
    input signed [63:0] value;
    begin
        if (value > 64'sd32767)
            saturate_signed16_from64 = 16'sh7fff;
        else if (value < -64'sd32768)
            saturate_signed16_from64 = 16'sh8000;
        else
            saturate_signed16_from64 = value[15:0];
    end
endfunction

function [15:0] saturate_signed16_from18;
    input signed [17:0] value;
    begin
        if (value > 18'sd32767)
            saturate_signed16_from18 = 16'h7fff;
        else if (value < -18'sd32768)
            saturate_signed16_from18 = 16'h8000;
        else
            saturate_signed16_from18 = value[15:0];
    end
endfunction

reg [255:0] output_pack_next;
always @(*) begin
    output_pack_next = output_pack;
    output_pack_next[lane_index*16 +: 16] = output_lane_reg;
end

assign axi_awuser_ap = 1'b0;
assign axi_awuser_id = 4'h0;
assign axi_awlen     = 4'h0;
assign axi_aruser_ap = 1'b0;
assign axi_aruser_id = 4'h0;
assign debug_state   = state;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        state            <= ST_IDLE;
        hidden_base_addr <= {CTRL_ADDR_WIDTH{1'b0}};
        q28_base_addr    <= {CTRL_ADDR_WIDTH{1'b0}};
        result_base_addr <= {CTRL_ADDR_WIDTH{1'b0}};
        beat_index       <= 6'd0;
        q28_read_index   <= 3'd0;
        lane_index       <= 4'd0;
        ar_seen          <= 1'b0;
        aw_seen          <= 1'b0;
        w_seen           <= 1'b0;
        hidden_beat      <= 256'd0;
        q28_beat0        <= 256'd0;
        q28_beat1        <= 256'd0;
        q28_beat2        <= 256'd0;
        q28_beat3        <= 256'd0;
        output_pack      <= 256'd0;
        hidden_lane_reg  <= 16'sd0;
        q28_lane_reg     <= 64'sd0;
        q28_negative_reg <= 1'b0;
        q28_magnitude_reg<= 64'd0;
        q10_wide_reg     <= 64'sd0;
        q10_reg          <= 16'sd0;
        residual_sum_reg <= 18'sd0;
        output_lane_reg  <= 16'd0;
        axi_awaddr       <= {CTRL_ADDR_WIDTH{1'b0}};
        axi_awvalid      <= 1'b0;
        axi_wdata        <= 256'd0;
        axi_wstrb        <= 32'd0;
        axi_araddr       <= {CTRL_ADDR_WIDTH{1'b0}};
        axi_arlen        <= 4'd0;
        axi_arvalid      <= 1'b0;
        busy             <= 1'b0;
        done             <= 1'b0;
        error            <= 1'b0;
        error_code       <= 8'd0;
        debug_beat_index <= 6'd0;
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
                        hidden_base_addr <= cfg_hidden_addr;
                        q28_base_addr    <= cfg_q28_addr;
                        result_base_addr <= cfg_result_addr;
                        beat_index       <= 6'd0;
                        debug_beat_index <= 6'd0;
                        output_pack      <= 256'd0;
                        busy             <= 1'b1;
                        error_code       <= 8'd0;
                        state            <= ST_SETUP_HIDDEN_READ;
                    end
                end
            end

            ST_SETUP_HIDDEN_READ: begin
                axi_araddr  <= hidden_base_addr + ({22'd0, beat_index} << 3);
                axi_arlen   <= 4'd0;
                axi_arvalid <= 1'b1;
                ar_seen     <= 1'b0;
                state       <= ST_READ_HIDDEN;
            end

            ST_READ_HIDDEN: begin
                if (ar_handshake) begin
                    axi_arvalid <= 1'b0;
                    ar_seen     <= 1'b1;
                end
                if (read_data_handshake) begin
                    hidden_beat    <= axi_rdata;
                    ar_seen        <= 1'b0;
                    q28_read_index <= 3'd0;
                    state          <= ST_SETUP_Q28_READ;
                end
            end

            ST_SETUP_Q28_READ: begin
                axi_araddr  <= q28_base_addr + ({22'd0, beat_index} << 5);
                axi_arlen   <= 4'd3;
                axi_arvalid <= 1'b1;
                ar_seen     <= 1'b0;
                state       <= ST_READ_Q28;
            end

            ST_READ_Q28: begin
                if (ar_handshake) begin
                    axi_arvalid <= 1'b0;
                    ar_seen     <= 1'b1;
                end
                if (read_data_handshake) begin
                    case (q28_read_index)
                        3'd0: q28_beat0 <= axi_rdata;
                        3'd1: q28_beat1 <= axi_rdata;
                        3'd2: q28_beat2 <= axi_rdata;
                        default: q28_beat3 <= axi_rdata;
                    endcase
                    if (q28_read_index == 3'd3) begin
                        ar_seen     <= 1'b0;
                        lane_index  <= 4'd0;
                        output_pack <= 256'd0;
                        state       <= ST_CAPTURE;
                    end else begin
                        q28_read_index <= q28_read_index + 1'b1;
                    end
                end
            end

            ST_CAPTURE: begin
                hidden_lane_reg <= selected_hidden_lane;
                q28_lane_reg    <= $signed(selected_q28_bits);
                state           <= ST_ABS;
            end

            ST_ABS: begin
                q28_negative_reg  <= q28_lane_reg[63];
                q28_magnitude_reg <= q28_lane_reg[63] ?
                    (~q28_lane_reg + 1'b1) : q28_lane_reg;
                state <= ST_ROUND;
            end

            ST_ROUND: begin
                q10_wide_reg <= rne_shift18_from_magnitude(
                    q28_magnitude_reg,
                    q28_negative_reg
                );
                state <= ST_SAT_Q10;
            end

            ST_SAT_Q10: begin
                q10_reg <= saturate_signed16_from64(q10_wide_reg);
                state   <= ST_ADD;
            end

            ST_ADD: begin
                residual_sum_reg <=
                    {{2{hidden_lane_reg[15]}}, hidden_lane_reg} +
                    {{2{q10_reg[15]}}, q10_reg};
                state <= ST_SAT_OUT;
            end

            ST_SAT_OUT: begin
                output_lane_reg <= saturate_signed16_from18(residual_sum_reg);
                state <= ST_PACK;
            end

            ST_PACK: begin
                output_pack <= output_pack_next;
                if (lane_index == 4'd15) begin
                    state <= ST_SETUP_WRITE;
                end else begin
                    lane_index <= lane_index + 1'b1;
                    state <= ST_CAPTURE;
                end
            end

            ST_SETUP_WRITE: begin
                axi_awaddr  <= result_base_addr + ({22'd0, beat_index} << 3);
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
                    if (beat_index == RESULT_BEATS - 1) begin
                        state <= ST_FINISH;
                    end else begin
                        beat_index       <= beat_index + 1'b1;
                        debug_beat_index <= beat_index + 1'b1;
                        state            <= ST_SETUP_HIDDEN_READ;
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
