`timescale 1ns/1ps

`include "transformer_block_contract.vh"

// G2 KV Cache 写入阶段。
// 把当前 K_rope=[2,64] 与 V=[2,64] 各 32 个 256-bit beat 复制到 F3 已验证
// 的 layer/token 槽；地址单位始终是 DDR3 Controller 的 32 bit。
module g2_kv_write_stage_ctrl #(
    parameter integer CTRL_ADDR_WIDTH = 28,
    parameter integer VECTOR_BEATS    = 32,
    parameter integer NUM_LAYERS      = 28,
    parameter integer MAX_CONTEXT     = 16384
)(
    input  wire                         clk,
    input  wire                         rst_n,
    input  wire                         ddr_init_done,
    input  wire                         start,
    input  wire [4:0]                   cfg_layer,
    input  wire [14:0]                  cfg_position,
    input  wire [CTRL_ADDR_WIDTH-1:0]   cfg_k_source_addr,
    input  wire [CTRL_ADDR_WIDTH-1:0]   cfg_v_source_addr,

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
    output wire [3:0]                   debug_state,
    output reg                          debug_is_v,
    output reg  [5:0]                   debug_beat_index,
    output reg  [CTRL_ADDR_WIDTH-1:0]   debug_slot_addr
);

localparam [3:0] ST_IDLE        = 4'd0;
localparam [3:0] ST_SETUP_READ  = 4'd1;
localparam [3:0] ST_READ        = 4'd2;
localparam [3:0] ST_SETUP_WRITE = 4'd3;
localparam [3:0] ST_WRITE       = 4'd4;
localparam [3:0] ST_ADVANCE     = 4'd5;
localparam [3:0] ST_FINISH      = 4'd6;
localparam [3:0] ST_ERROR       = 4'd15;

localparam [7:0] ERR_DDR_NOT_READY = 8'h01;
localparam [7:0] ERR_CONFIG        = 8'h02;
localparam [7:0] ERR_INTERNAL      = 8'hff;

reg [3:0] state;
reg [CTRL_ADDR_WIDTH-1:0] k_source_addr;
reg [CTRL_ADDR_WIDTH-1:0] v_source_addr;
reg [CTRL_ADDR_WIDTH-1:0] slot_base_addr;
reg is_v;
reg [5:0] beat_index;
reg [255:0] read_cache;
reg ar_seen;
reg aw_seen;
reg w_seen;

wire aw_handshake = axi_awvalid && axi_awready;
wire ar_handshake = axi_arvalid && axi_arready;
wire write_data_handshake = axi_wready && (aw_seen || aw_handshake);
wire write_complete = (aw_seen || aw_handshake) &&
                      (w_seen || write_data_handshake);
wire read_data_handshake = axi_rvalid && (ar_seen || ar_handshake);

wire [CTRL_ADDR_WIDTH-1:0] active_source_addr = is_v ? v_source_addr : k_source_addr;
wire [CTRL_ADDR_WIDTH-1:0] active_slot_addr = slot_base_addr +
    (is_v ? `G2_KV_V_OFFSET_CTRL : {CTRL_ADDR_WIDTH{1'b0}});
wire config_valid = (cfg_layer < NUM_LAYERS) && (cfg_position < MAX_CONTEXT);
wire [CTRL_ADDR_WIDTH-1:0] cfg_layer_offset =
    {{(CTRL_ADDR_WIDTH-5){1'b0}}, cfg_layer} << 23;
wire [CTRL_ADDR_WIDTH-1:0] cfg_position_offset =
    {{(CTRL_ADDR_WIDTH-15){1'b0}}, cfg_position} << 9;

assign axi_awuser_ap = 1'b0;
assign axi_awuser_id = 4'h0;
assign axi_awlen     = 4'h0;
assign axi_aruser_ap = 1'b0;
assign axi_aruser_id = 4'h0;
assign debug_state   = state;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        state            <= ST_IDLE;
        k_source_addr    <= {CTRL_ADDR_WIDTH{1'b0}};
        v_source_addr    <= {CTRL_ADDR_WIDTH{1'b0}};
        slot_base_addr   <= {CTRL_ADDR_WIDTH{1'b0}};
        is_v             <= 1'b0;
        beat_index       <= 6'd0;
        read_cache       <= 256'd0;
        ar_seen          <= 1'b0;
        aw_seen          <= 1'b0;
        w_seen           <= 1'b0;
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
        debug_is_v       <= 1'b0;
        debug_beat_index <= 6'd0;
        debug_slot_addr  <= {CTRL_ADDR_WIDTH{1'b0}};
    end else begin
        done <= 1'b0;

        case (state)
            ST_IDLE: begin
                axi_awvalid <= 1'b0;
                axi_arvalid <= 1'b0;
                ar_seen     <= 1'b0;
                aw_seen     <= 1'b0;
                w_seen      <= 1'b0;
                busy        <= 1'b0;
                if (start && !error) begin
                    if (!ddr_init_done) begin
                        error      <= 1'b1;
                        error_code <= ERR_DDR_NOT_READY;
                        state      <= ST_ERROR;
                    end else if (!config_valid) begin
                        error      <= 1'b1;
                        error_code <= ERR_CONFIG;
                        state      <= ST_ERROR;
                    end else begin
                        k_source_addr    <= cfg_k_source_addr;
                        v_source_addr    <= cfg_v_source_addr;
                        slot_base_addr   <= `G2_KV_BASE_CTRL_ADDR +
                                            cfg_layer_offset + cfg_position_offset;
                        is_v             <= 1'b0;
                        beat_index       <= 6'd0;
                        debug_is_v       <= 1'b0;
                        debug_beat_index <= 6'd0;
                        debug_slot_addr  <= `G2_KV_BASE_CTRL_ADDR +
                                            cfg_layer_offset + cfg_position_offset;
                        busy             <= 1'b1;
                        error_code       <= 8'd0;
                        state            <= ST_SETUP_READ;
                    end
                end
            end

            ST_SETUP_READ: begin
                axi_araddr  <= active_source_addr + ({22'd0, beat_index} << 3);
                axi_arlen   <= 4'd0;
                axi_arvalid <= 1'b1;
                ar_seen     <= 1'b0;
                state       <= ST_READ;
            end

            ST_READ: begin
                if (ar_handshake) begin
                    axi_arvalid <= 1'b0;
                    ar_seen     <= 1'b1;
                end
                if (read_data_handshake) begin
                    read_cache <= axi_rdata;
                    ar_seen    <= 1'b0;
                    state      <= ST_SETUP_WRITE;
                end
            end

            ST_SETUP_WRITE: begin
                axi_awaddr  <= active_slot_addr + ({22'd0, beat_index} << 3);
                axi_awvalid <= 1'b1;
                axi_wdata   <= read_cache;
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
                if (write_complete) begin
                    axi_awvalid <= 1'b0;
                    aw_seen     <= 1'b0;
                    w_seen      <= 1'b0;
                    state       <= ST_ADVANCE;
                end
            end

            ST_ADVANCE: begin
                if (beat_index != VECTOR_BEATS - 1) begin
                    beat_index       <= beat_index + 1'b1;
                    debug_beat_index <= beat_index + 1'b1;
                    state            <= ST_SETUP_READ;
                end else if (!is_v) begin
                    is_v             <= 1'b1;
                    beat_index       <= 6'd0;
                    debug_is_v       <= 1'b1;
                    debug_beat_index <= 6'd0;
                    debug_slot_addr  <= slot_base_addr + `G2_KV_V_OFFSET_CTRL;
                    state            <= ST_SETUP_READ;
                end else begin
                    state <= ST_FINISH;
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
