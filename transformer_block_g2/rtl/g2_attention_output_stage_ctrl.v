`timescale 1ns/1ps

`include "transformer_block_contract.vh"

// G2 Attention Output 阶段。
// 从 scratch 加载 [14,16] 概率，从 F3 KV Cache 加载窗口内 V，复用已验证
// attention_output_core，最终写出 head-major [14,64] signed Q28。
module g2_attention_output_stage_ctrl #(
    parameter integer CTRL_ADDR_WIDTH = 28,
    parameter integer PROB_BEATS      = 28,
    parameter integer V_BEATS         = 32,
    parameter integer MAX_TOKENS      = 16,
    parameter integer NUM_LAYERS      = 28,
    parameter integer MAX_CONTEXT     = 16384
)(
    input  wire                         clk,
    input  wire                         rst_n,
    input  wire                         ddr_init_done,
    input  wire                         start,
    input  wire [4:0]                   cfg_layer,
    input  wire [14:0]                  cfg_query_position,
    input  wire [14:0]                  cfg_window_start,
    input  wire [4:0]                   cfg_count,
    input  wire [CTRL_ADDR_WIDTH-1:0]   cfg_probability_addr,
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
    output wire [6:0]                   debug_state,
    output reg  [4:0]                   debug_token,
    output reg  [3:0]                   debug_head,
    output reg  [5:0]                   debug_dimension
);

localparam [6:0] ST_IDLE            = 7'd0;
localparam [6:0] ST_SETUP_PROB_READ = 7'd1;
localparam [6:0] ST_READ_PROB       = 7'd2;
localparam [6:0] ST_TOKEN_PREPARE   = 7'd3;
localparam [6:0] ST_SETUP_V_READ    = 7'd4;
localparam [6:0] ST_READ_V          = 7'd5;
localparam [6:0] ST_START_CORE      = 7'd6;
localparam [6:0] ST_WAIT_RESULT     = 7'd7;
localparam [6:0] ST_SETUP_WRITE     = 7'd8;
localparam [6:0] ST_WRITE           = 7'd9;
localparam [6:0] ST_FINISH          = 7'd10;
localparam [6:0] ST_ACK_RESULT      = 7'd11;
localparam [6:0] ST_ERROR           = 7'd127;

localparam [7:0] ERR_DDR_NOT_READY = 8'h01;
localparam [7:0] ERR_CONFIG        = 8'h02;
localparam [7:0] ERR_CORE_PROTOCOL = 8'h03;
localparam [7:0] ERR_INTERNAL      = 8'hff;

reg [6:0] state;
reg [4:0] layer_reg;
reg [14:0] window_start_reg;
reg [4:0] count_reg;
reg [CTRL_ADDR_WIDTH-1:0] probability_base_addr;
reg [CTRL_ADDR_WIDTH-1:0] result_base_addr;
reg [5:0] read_base_beat;
reg [4:0] read_beat_index;
reg [4:0] active_burst_beats;
reg [4:0] token_index;
reg [3:0] result_head_cache;
reg [5:0] result_dimension_cache;
reg [63:0] result_cache;
reg ar_seen;
reg aw_seen;
reg w_seen;
reg core_start;

wire core_busy;
wire core_result_valid;
wire [3:0] core_result_head;
wire [5:0] core_result_dimension;
wire signed [63:0] core_result_q28;
wire core_done;
reg  core_result_ready;

wire aw_handshake = axi_awvalid && axi_awready;
wire ar_handshake = axi_arvalid && axi_arready;
wire write_data_handshake = axi_wready && (aw_seen || aw_handshake);
wire write_complete = (aw_seen || aw_handshake) &&
                      (w_seen || write_data_handshake);
wire read_data_handshake = axi_rvalid && (ar_seen || ar_handshake);

wire [6:0] prob_beats_remaining = PROB_BEATS - read_base_beat;
wire [4:0] next_prob_burst_beats =
    (prob_beats_remaining > 7'd16) ? 5'd16 : prob_beats_remaining[4:0];
wire [6:0] v_beats_remaining = V_BEATS - read_base_beat;
wire [4:0] next_v_burst_beats =
    (v_beats_remaining > 7'd16) ? 5'd16 : v_beats_remaining[4:0];

wire [14:0] token_position = window_start_reg + token_index;
wire [CTRL_ADDR_WIDTH-1:0] layer_offset =
    {{(CTRL_ADDR_WIDTH-5){1'b0}}, layer_reg} << 23;
wire [CTRL_ADDR_WIDTH-1:0] token_offset =
    {{(CTRL_ADDR_WIDTH-15){1'b0}}, token_position} << 9;
wire [CTRL_ADDR_WIDTH-1:0] active_v_addr =
    `G2_KV_BASE_CTRL_ADDR + layer_offset + token_offset + `G2_KV_V_OFFSET_CTRL;

wire probability_load_en = (state == ST_READ_PROB) && read_data_handshake;
wire v_load_en = (state == ST_READ_V) && read_data_handshake;
wire [4:0] probability_load_index = read_base_beat[4:0] + read_beat_index;
wire [8:0] v_load_index = ({4'd0, token_index} << 5) +
                          {4'd0, read_base_beat[4:0]} + read_beat_index;
wire [7:0] result_beat_index = ({4'd0, result_head_cache} << 4) +
                               {6'd0, result_dimension_cache[5:2]};
wire [1:0] result_lane = result_dimension_cache[1:0];
reg [255:0] result_write_data;
reg [31:0] result_write_strobe;
always @(*) begin
    result_write_data = 256'd0;
    result_write_data[result_lane*64 +: 64] = result_cache;
    result_write_strobe = 32'd0;
    result_write_strobe[result_lane*8 +: 8] = 8'hff;
end

wire config_valid =
    (cfg_layer < NUM_LAYERS) &&
    (cfg_count >= 1) && (cfg_count <= MAX_TOKENS) &&
    (cfg_window_start < MAX_CONTEXT) &&
    (cfg_query_position < MAX_CONTEXT) &&
    (cfg_query_position == cfg_window_start + cfg_count - 1'b1);

assign axi_awuser_ap = 1'b0;
assign axi_awuser_id = 4'h0;
assign axi_awlen     = 4'h0;
assign axi_aruser_ap = 1'b0;
assign axi_aruser_id = 4'h0;
assign debug_state   = state;

attention_output_core u_attention_output_core (
    .clk                    (clk),
    .rst_n                  (rst_n),
    .probability_beat_we    (probability_load_en),
    .probability_beat_index (probability_load_index),
    .probability_beat_data  (axi_rdata),
    .v_beat_we              (v_load_en),
    .v_beat_index           (v_load_index),
    .v_beat_data            (axi_rdata),
    .start                  (core_start),
    .token_count            (count_reg),
    .result_ready           (core_result_ready),
    .busy                   (core_busy),
    .result_valid           (core_result_valid),
    .result_head            (core_result_head),
    .result_dimension       (core_result_dimension),
    .result_q28             (core_result_q28),
    .done                   (core_done)
);

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        state                  <= ST_IDLE;
        layer_reg              <= 5'd0;
        window_start_reg       <= 15'd0;
        count_reg              <= 5'd0;
        probability_base_addr  <= {CTRL_ADDR_WIDTH{1'b0}};
        result_base_addr       <= {CTRL_ADDR_WIDTH{1'b0}};
        read_base_beat         <= 6'd0;
        read_beat_index        <= 5'd0;
        active_burst_beats     <= 5'd0;
        token_index            <= 5'd0;
        result_head_cache      <= 4'd0;
        result_dimension_cache <= 6'd0;
        result_cache           <= 64'd0;
        ar_seen                <= 1'b0;
        aw_seen                <= 1'b0;
        w_seen                 <= 1'b0;
        core_start             <= 1'b0;
        core_result_ready      <= 1'b0;
        axi_awaddr             <= {CTRL_ADDR_WIDTH{1'b0}};
        axi_awvalid            <= 1'b0;
        axi_wdata              <= 256'd0;
        axi_wstrb              <= 32'd0;
        axi_araddr             <= {CTRL_ADDR_WIDTH{1'b0}};
        axi_arlen              <= 4'd0;
        axi_arvalid            <= 1'b0;
        busy                   <= 1'b0;
        done                   <= 1'b0;
        error                  <= 1'b0;
        error_code             <= 8'd0;
        debug_token            <= 5'd0;
        debug_head             <= 4'd0;
        debug_dimension        <= 6'd0;
    end else begin
        done              <= 1'b0;
        core_start        <= 1'b0;
        core_result_ready <= 1'b0;

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
                        layer_reg             <= cfg_layer;
                        window_start_reg      <= cfg_window_start;
                        count_reg             <= cfg_count;
                        probability_base_addr <= cfg_probability_addr;
                        result_base_addr      <= cfg_result_addr;
                        read_base_beat        <= 6'd0;
                        token_index           <= 5'd0;
                        debug_token           <= 5'd0;
                        debug_head            <= 4'd0;
                        debug_dimension       <= 6'd0;
                        busy                  <= 1'b1;
                        error_code            <= 8'd0;
                        state                 <= ST_SETUP_PROB_READ;
                    end
                end
            end

            ST_SETUP_PROB_READ: begin
                axi_araddr         <= probability_base_addr + ({22'd0, read_base_beat} << 3);
                axi_arlen          <= next_prob_burst_beats - 1'b1;
                axi_arvalid        <= 1'b1;
                ar_seen            <= 1'b0;
                read_beat_index    <= 5'd0;
                active_burst_beats <= next_prob_burst_beats;
                state              <= ST_READ_PROB;
            end

            ST_READ_PROB: begin
                if (ar_handshake) begin
                    axi_arvalid <= 1'b0;
                    ar_seen     <= 1'b1;
                end
                if (read_data_handshake) begin
                    if (read_beat_index + 1'b1 == active_burst_beats) begin
                        ar_seen <= 1'b0;
                        if (read_base_beat + active_burst_beats == PROB_BEATS) begin
                            read_base_beat <= 6'd0;
                            token_index    <= 5'd0;
                            state          <= ST_TOKEN_PREPARE;
                        end else begin
                            read_base_beat <= read_base_beat + active_burst_beats;
                            state <= ST_SETUP_PROB_READ;
                        end
                    end else begin
                        read_beat_index <= read_beat_index + 1'b1;
                    end
                end
            end

            ST_TOKEN_PREPARE: begin
                read_base_beat <= 6'd0;
                debug_token    <= token_index;
                state          <= ST_SETUP_V_READ;
            end

            ST_SETUP_V_READ: begin
                axi_araddr         <= active_v_addr + ({22'd0, read_base_beat} << 3);
                axi_arlen          <= next_v_burst_beats - 1'b1;
                axi_arvalid        <= 1'b1;
                ar_seen            <= 1'b0;
                read_beat_index    <= 5'd0;
                active_burst_beats <= next_v_burst_beats;
                state              <= ST_READ_V;
            end

            ST_READ_V: begin
                if (ar_handshake) begin
                    axi_arvalid <= 1'b0;
                    ar_seen     <= 1'b1;
                end
                if (read_data_handshake) begin
                    if (read_beat_index + 1'b1 == active_burst_beats) begin
                        ar_seen <= 1'b0;
                        if (read_base_beat + active_burst_beats == V_BEATS) begin
                            read_base_beat <= 6'd0;
                            if (token_index + 1'b1 == count_reg) begin
                                state <= ST_START_CORE;
                            end else begin
                                token_index <= token_index + 1'b1;
                                state       <= ST_TOKEN_PREPARE;
                            end
                        end else begin
                            read_base_beat <= read_base_beat + active_burst_beats;
                            state <= ST_SETUP_V_READ;
                        end
                    end else begin
                        read_beat_index <= read_beat_index + 1'b1;
                    end
                end
            end

            ST_START_CORE: begin
                if (core_busy) begin
                    error      <= 1'b1;
                    error_code <= ERR_CORE_PROTOCOL;
                    state      <= ST_ERROR;
                end else begin
                    core_start <= 1'b1;
                    state      <= ST_WAIT_RESULT;
                end
            end

            ST_WAIT_RESULT: begin
                if (core_result_valid) begin
                    result_head_cache      <= core_result_head;
                    result_dimension_cache <= core_result_dimension;
                    result_cache           <= core_result_q28;
                    debug_head             <= core_result_head;
                    debug_dimension        <= core_result_dimension;
                    state                  <= ST_SETUP_WRITE;
                end else if (core_done) begin
                    state <= ST_FINISH;
                end
            end

            ST_SETUP_WRITE: begin
                axi_awaddr  <= result_base_addr + ({20'd0, result_beat_index} << 3);
                axi_awvalid <= 1'b1;
                axi_wdata   <= result_write_data;
                axi_wstrb   <= result_write_strobe;
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
                    axi_awvalid      <= 1'b0;
                    aw_seen          <= 1'b0;
                    w_seen           <= 1'b0;
                    // DDR 写完成先转成一拍本地寄存确认，切断
                    // scheduler/AXI ready 到核心宽寄存器 CE 的组合路径。
                    core_result_ready <= 1'b1;
                    state             <= ST_ACK_RESULT;
                end
            end

            ST_ACK_RESULT: begin
                // 核心在本拍采样寄存后的 result_ready。最后一个结果可依据
                // 已缓存索引直接结束，其余结果返回等待下一次 result_valid。
                if ((result_head_cache == 4'd13) &&
                    (result_dimension_cache == 6'd63))
                    state <= ST_FINISH;
                else
                    state <= ST_WAIT_RESULT;
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
