`timescale 1ns/1ps

// G2 流式 SiLU(gate) 控制器。
//
// 数值路径逐状态复现 mlp_silu_core：Q28 -> signed RNE >> 18 -> int16
// 饱和 -> 65 端点/64 段 PWL -> signed RNE >> 8 -> int16 饱和。
// 只缓存当前四个 Q28 beat 与 80 项 padded PWL 表，避免 32 DRM 的整向量缓存。
module g2_stream_silu_ctrl #(
    parameter integer CTRL_ADDR_WIDTH = 28,
    parameter integer RESULT_BEATS    = 304,
    parameter integer PWL_BEATS       = 5
)(
    input  wire                         clk,
    input  wire                         rst_n,
    input  wire                         ddr_init_done,
    input  wire                         start,
    input  wire [CTRL_ADDR_WIDTH-1:0]   cfg_gate_addr,
    input  wire [CTRL_ADDR_WIDTH-1:0]   cfg_pwl_addr,
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
    output reg  [8:0]                   debug_beat_index
);

localparam [4:0] ST_IDLE            = 5'd0;
localparam [4:0] ST_SETUP_PWL_READ  = 5'd1;
localparam [4:0] ST_READ_PWL        = 5'd2;
localparam [4:0] ST_SETUP_GATE_READ = 5'd3;
localparam [4:0] ST_READ_GATE       = 5'd4;
localparam [4:0] ST_CAPTURE         = 5'd5;
localparam [4:0] ST_ABS             = 5'd6;
localparam [4:0] ST_ROUND           = 5'd7;
localparam [4:0] ST_SAT_INPUT       = 5'd8;
localparam [4:0] ST_DISPATCH        = 5'd9;
localparam [4:0] ST_PWL_CAPTURE     = 5'd10;
localparam [4:0] ST_PWL_MULT        = 5'd11;
localparam [4:0] ST_PWL_INTERP      = 5'd12;
localparam [4:0] ST_PWL_ADD         = 5'd13;
localparam [4:0] ST_SAT_OUTPUT      = 5'd14;
localparam [4:0] ST_PACK            = 5'd15;
localparam [4:0] ST_SETUP_WRITE     = 5'd16;
localparam [4:0] ST_WRITE           = 5'd17;
localparam [4:0] ST_FINISH          = 5'd18;
localparam [4:0] ST_ERROR           = 5'd31;

localparam [7:0] ERR_DDR_NOT_READY = 8'h01;
localparam [7:0] ERR_INTERNAL      = 8'hff;

reg [4:0] state;
reg [CTRL_ADDR_WIDTH-1:0] gate_base_addr;
reg [CTRL_ADDR_WIDTH-1:0] pwl_base_addr;
reg [CTRL_ADDR_WIDTH-1:0] result_base_addr;
reg [8:0] beat_index;
reg [2:0] read_index;
reg [3:0] lane_index;
reg ar_seen;
reg aw_seen;
reg w_seen;

reg signed [15:0] pwl_mem [0:79];
reg [255:0] gate_beat0;
reg [255:0] gate_beat1;
reg [255:0] gate_beat2;
reg [255:0] gate_beat3;
reg [255:0] output_pack;

reg signed [63:0] gate_lane_reg;
reg               gate_negative_reg;
reg [63:0]        gate_magnitude_reg;
reg signed [63:0] gate_q10_wide_reg;
reg signed [15:0] gate_q10_reg;
reg signed [63:0] value_reg;
reg [5:0]         pwl_index_reg;
reg [7:0]         pwl_fraction_reg;
reg signed [16:0] pwl_endpoint0_reg;
reg signed [16:0] pwl_endpoint1_reg;
reg signed [26:0] pwl_product_reg;
reg signed [18:0] pwl_interp_reg;
reg [15:0]        output_lane_reg;

integer load_lane;
integer load_index;

wire aw_handshake = axi_awvalid && axi_awready;
wire ar_handshake = axi_arvalid && axi_arready;
wire write_data_handshake = axi_wready && (aw_seen || aw_handshake);
wire read_data_handshake = axi_rvalid && (ar_seen || ar_handshake);

reg [63:0] selected_gate_bits;
always @(*) begin
    case (lane_index[3:2])
        2'd0: selected_gate_bits = gate_beat0[lane_index[1:0]*64 +: 64];
        2'd1: selected_gate_bits = gate_beat1[lane_index[1:0]*64 +: 64];
        2'd2: selected_gate_bits = gate_beat2[lane_index[1:0]*64 +: 64];
        default: selected_gate_bits = gate_beat3[lane_index[1:0]*64 +: 64];
    endcase
end

wire signed [16:0] gate_q10_ext = {gate_q10_reg[15], gate_q10_reg};
wire [13:0] silu_offset_wire = gate_q10_ext + 17'sd8192;
wire signed [17:0] pwl_delta_wire = pwl_endpoint1_reg - pwl_endpoint0_reg;
wire signed [8:0] pwl_fraction_signed = {1'b0, pwl_fraction_reg};
wire signed [26:0] pwl_product_wire = pwl_delta_wire * pwl_fraction_signed;
wire signed [19:0] pwl_add_wire =
    {{3{pwl_endpoint0_reg[16]}}, pwl_endpoint0_reg} +
    {pwl_interp_reg[18], pwl_interp_reg};

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

function signed [18:0] rne_shift8_signed27;
    input signed [26:0] value;
    reg [26:0] magnitude;
    reg [18:0] quotient;
    reg [7:0] remainder;
    begin
        magnitude = value[26] ? (~value + 1'b1) : value;
        quotient = magnitude >> 8;
        remainder = magnitude[7:0];
        if ((remainder > 8'h80) ||
            ((remainder == 8'h80) && quotient[0]))
            quotient = quotient + 1'b1;
        rne_shift8_signed27 = value[26] ? -$signed(quotient) : $signed(quotient);
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

function [15:0] saturate_signed16_from20;
    input signed [19:0] value;
    begin
        if (value > 20'sd32767)
            saturate_signed16_from20 = 16'h7fff;
        else if (value < -20'sd32768)
            saturate_signed16_from20 = 16'h8000;
        else
            saturate_signed16_from20 = value[15:0];
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
        state              <= ST_IDLE;
        gate_base_addr     <= {CTRL_ADDR_WIDTH{1'b0}};
        pwl_base_addr      <= {CTRL_ADDR_WIDTH{1'b0}};
        result_base_addr   <= {CTRL_ADDR_WIDTH{1'b0}};
        beat_index         <= 9'd0;
        read_index         <= 3'd0;
        lane_index         <= 4'd0;
        ar_seen            <= 1'b0;
        aw_seen            <= 1'b0;
        w_seen             <= 1'b0;
        gate_beat0         <= 256'd0;
        gate_beat1         <= 256'd0;
        gate_beat2         <= 256'd0;
        gate_beat3         <= 256'd0;
        output_pack        <= 256'd0;
        gate_lane_reg      <= 64'sd0;
        gate_negative_reg  <= 1'b0;
        gate_magnitude_reg <= 64'd0;
        gate_q10_wide_reg  <= 64'sd0;
        gate_q10_reg       <= 16'sd0;
        value_reg          <= 64'sd0;
        pwl_index_reg      <= 6'd0;
        pwl_fraction_reg   <= 8'd0;
        pwl_endpoint0_reg  <= 17'sd0;
        pwl_endpoint1_reg  <= 17'sd0;
        pwl_product_reg    <= 27'sd0;
        pwl_interp_reg     <= 19'sd0;
        output_lane_reg    <= 16'd0;
        axi_awaddr         <= {CTRL_ADDR_WIDTH{1'b0}};
        axi_awvalid        <= 1'b0;
        axi_wdata          <= 256'd0;
        axi_wstrb          <= 32'd0;
        axi_araddr         <= {CTRL_ADDR_WIDTH{1'b0}};
        axi_arlen          <= 4'd0;
        axi_arvalid        <= 1'b0;
        busy               <= 1'b0;
        done               <= 1'b0;
        error              <= 1'b0;
        error_code         <= 8'd0;
        debug_beat_index   <= 9'd0;
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
                        gate_base_addr   <= cfg_gate_addr;
                        pwl_base_addr    <= cfg_pwl_addr;
                        result_base_addr <= cfg_result_addr;
                        beat_index       <= 9'd0;
                        debug_beat_index <= 9'd0;
                        read_index       <= 3'd0;
                        busy             <= 1'b1;
                        error_code       <= 8'd0;
                        state            <= ST_SETUP_PWL_READ;
                    end
                end
            end

            ST_SETUP_PWL_READ: begin
                axi_araddr  <= pwl_base_addr;
                axi_arlen   <= PWL_BEATS - 1;
                axi_arvalid <= 1'b1;
                ar_seen     <= 1'b0;
                read_index  <= 3'd0;
                state       <= ST_READ_PWL;
            end

            ST_READ_PWL: begin
                if (ar_handshake) begin
                    axi_arvalid <= 1'b0;
                    ar_seen     <= 1'b1;
                end
                if (read_data_handshake) begin
                    for (load_lane = 0; load_lane < 16; load_lane = load_lane + 1) begin
                        load_index = read_index * 16 + load_lane;
                        pwl_mem[load_index] <= axi_rdata[load_lane*16 +: 16];
                    end
                    if (read_index == PWL_BEATS - 1) begin
                        ar_seen <= 1'b0;
                        state   <= ST_SETUP_GATE_READ;
                    end else begin
                        read_index <= read_index + 1'b1;
                    end
                end
            end

            ST_SETUP_GATE_READ: begin
                axi_araddr  <= gate_base_addr + ({19'd0, beat_index} << 5);
                axi_arlen   <= 4'd3;
                axi_arvalid <= 1'b1;
                ar_seen     <= 1'b0;
                read_index  <= 3'd0;
                state       <= ST_READ_GATE;
            end

            ST_READ_GATE: begin
                if (ar_handshake) begin
                    axi_arvalid <= 1'b0;
                    ar_seen     <= 1'b1;
                end
                if (read_data_handshake) begin
                    case (read_index)
                        3'd0: gate_beat0 <= axi_rdata;
                        3'd1: gate_beat1 <= axi_rdata;
                        3'd2: gate_beat2 <= axi_rdata;
                        default: gate_beat3 <= axi_rdata;
                    endcase
                    if (read_index == 3'd3) begin
                        ar_seen     <= 1'b0;
                        lane_index  <= 4'd0;
                        output_pack <= 256'd0;
                        state       <= ST_CAPTURE;
                    end else begin
                        read_index <= read_index + 1'b1;
                    end
                end
            end

            ST_CAPTURE: begin
                gate_lane_reg <= $signed(selected_gate_bits);
                state         <= ST_ABS;
            end

            ST_ABS: begin
                gate_negative_reg  <= gate_lane_reg[63];
                gate_magnitude_reg <= gate_lane_reg[63] ?
                    (~gate_lane_reg + 1'b1) : gate_lane_reg;
                state <= ST_ROUND;
            end

            ST_ROUND: begin
                gate_q10_wide_reg <= rne_shift18_from_magnitude(
                    gate_magnitude_reg,
                    gate_negative_reg
                );
                state <= ST_SAT_INPUT;
            end

            ST_SAT_INPUT: begin
                gate_q10_reg <= saturate_signed16_from64(gate_q10_wide_reg);
                state        <= ST_DISPATCH;
            end

            ST_DISPATCH: begin
                if (gate_q10_reg < -16'sd8192) begin
                    value_reg <= 64'sd0;
                    state     <= ST_SAT_OUTPUT;
                end else if (gate_q10_reg >= 16'sd8192) begin
                    value_reg <= {{48{gate_q10_reg[15]}}, gate_q10_reg};
                    state     <= ST_SAT_OUTPUT;
                end else begin
                    pwl_index_reg    <= silu_offset_wire[13:8];
                    pwl_fraction_reg <= silu_offset_wire[7:0];
                    state            <= ST_PWL_CAPTURE;
                end
            end

            ST_PWL_CAPTURE: begin
                pwl_endpoint0_reg <= pwl_mem[pwl_index_reg];
                pwl_endpoint1_reg <= pwl_mem[pwl_index_reg + 1'b1];
                state             <= ST_PWL_MULT;
            end

            ST_PWL_MULT: begin
                pwl_product_reg <= pwl_product_wire;
                state           <= ST_PWL_INTERP;
            end

            ST_PWL_INTERP: begin
                pwl_interp_reg <= rne_shift8_signed27(pwl_product_reg);
                state          <= ST_PWL_ADD;
            end

            ST_PWL_ADD: begin
                value_reg <= {{44{pwl_add_wire[19]}}, pwl_add_wire};
                state     <= ST_SAT_OUTPUT;
            end

            ST_SAT_OUTPUT: begin
                if ((gate_q10_reg < -16'sd8192) || (gate_q10_reg >= 16'sd8192))
                    output_lane_reg <= saturate_signed16_from64(value_reg);
                else
                    output_lane_reg <= saturate_signed16_from20(value_reg[19:0]);
                state <= ST_PACK;
            end

            ST_PACK: begin
                output_pack <= output_pack_next;
                if (lane_index == 4'd15) begin
                    state <= ST_SETUP_WRITE;
                end else begin
                    lane_index <= lane_index + 1'b1;
                    state      <= ST_CAPTURE;
                end
            end

            ST_SETUP_WRITE: begin
                axi_awaddr  <= result_base_addr + ({19'd0, beat_index} << 3);
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
                        state            <= ST_SETUP_GATE_READ;
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
