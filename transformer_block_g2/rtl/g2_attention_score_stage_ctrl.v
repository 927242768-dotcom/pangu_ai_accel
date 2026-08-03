`timescale 1ns/1ps

`include "transformer_block_contract.vh"

// G2 Attention Score 阶段。
// 加载当前 Q_rope=[14,64]，按 F3 地址读取窗口内 K，复用已验证的
// attention_score_core。固定生成 [14,16] head-major score；窗口之外写 INT64_MIN。
module g2_attention_score_stage_ctrl #(
    parameter integer CTRL_ADDR_WIDTH = 28,
    parameter integer Q_BEATS         = 224,
    parameter integer K_BEATS         = 32,
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
    input  wire [CTRL_ADDR_WIDTH-1:0]   cfg_q_addr,
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
    output reg  [3:0]                   debug_head
);

localparam [6:0] ST_IDLE          = 7'd0;
localparam [6:0] ST_SETUP_Q_READ  = 7'd1;
localparam [6:0] ST_READ_Q        = 7'd2;
localparam [6:0] ST_TOKEN_PREPARE = 7'd3;
localparam [6:0] ST_SETUP_K_READ  = 7'd4;
localparam [6:0] ST_READ_K        = 7'd5;
localparam [6:0] ST_START_TOKEN   = 7'd6;
localparam [6:0] ST_WAIT_SCORE    = 7'd7;
localparam [6:0] ST_SETUP_WRITE   = 7'd8;
localparam [6:0] ST_WRITE         = 7'd9;
localparam [6:0] ST_ADVANCE_TOKEN = 7'd10;
localparam [6:0] ST_FINISH        = 7'd11;
localparam [6:0] ST_ERROR         = 7'd127;

localparam [7:0] ERR_DDR_NOT_READY = 8'h01;
localparam [7:0] ERR_CONFIG        = 8'h02;
localparam [7:0] ERR_CORE_PROTOCOL = 8'h03;
localparam [7:0] ERR_INTERNAL      = 8'hff;

reg [6:0] state;
reg [4:0] layer_reg;
reg [14:0] query_position_reg;
reg [14:0] window_start_reg;
reg [4:0] count_reg;
reg [CTRL_ADDR_WIDTH-1:0] q_base_addr;
reg [CTRL_ADDR_WIDTH-1:0] result_base_addr;
reg [7:0] q_read_base_beat;
reg [4:0] read_beat_index;
reg [4:0] active_burst_beats;
reg [4:0] token_index;
reg [63:0] score_cache;
reg [3:0] score_head_cache;
reg ar_seen;
reg aw_seen;
reg w_seen;
reg core_start_token;

wire core_busy;
wire core_score_valid;
wire [3:0] core_score_head;
wire signed [63:0] core_score_q28;
wire core_token_done;
wire core_score_ready;

wire aw_handshake = axi_awvalid && axi_awready;
wire ar_handshake = axi_arvalid && axi_arready;
wire write_data_handshake = axi_wready && (aw_seen || aw_handshake);
wire write_complete = (aw_seen || aw_handshake) &&
                      (w_seen || write_data_handshake);
wire read_data_handshake = axi_rvalid && (ar_seen || ar_handshake);

wire [8:0] q_beats_remaining = Q_BEATS - q_read_base_beat;
wire [4:0] next_q_burst_beats =
    (q_beats_remaining > 9'd16) ? 5'd16 : q_beats_remaining[4:0];
wire [5:0] k_beats_remaining = K_BEATS - q_read_base_beat[5:0];
wire [4:0] next_k_burst_beats =
    (k_beats_remaining > 6'd16) ? 5'd16 : k_beats_remaining[4:0];

wire token_masked = token_index >= count_reg;
wire [14:0] token_position = window_start_reg + token_index;
wire [CTRL_ADDR_WIDTH-1:0] layer_offset =
    {{(CTRL_ADDR_WIDTH-5){1'b0}}, layer_reg} << 23;
wire [CTRL_ADDR_WIDTH-1:0] token_offset =
    {{(CTRL_ADDR_WIDTH-15){1'b0}}, token_position} << 9;
wire [CTRL_ADDR_WIDTH-1:0] active_k_addr =
    `G2_KV_BASE_CTRL_ADDR + layer_offset + token_offset;

wire q_load_en = (state == ST_READ_Q) && read_data_handshake;
wire k_load_en = (state == ST_READ_K) && read_data_handshake;
wire [7:0] q_load_index = q_read_base_beat + read_beat_index;
wire [4:0] k_load_index = q_read_base_beat[4:0] + read_beat_index;
assign core_score_ready = (state == ST_WRITE) && write_complete;

wire [7:0] score_beat_index = {score_head_cache, 2'b00} +
                              {6'd0, token_index[3:2]};
wire [1:0] score_lane = token_index[1:0];
reg [255:0] score_write_data;
reg [31:0] score_write_strobe;
always @(*) begin
    score_write_data = 256'd0;
    score_write_data[score_lane*64 +: 64] = score_cache;
    score_write_strobe = 32'd0;
    score_write_strobe[score_lane*8 +: 8] = 8'hff;
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

attention_score_core #(
    .PIPELINE_MUL_ACCUM (1),
    .PIPELINE_MUL_SIGN  (1),
    .PIPELINE_DOT_ACCUM (1)
) u_attention_score_core (
    .clk          (clk),
    .rst_n        (rst_n),
    .q_beat_we    (q_load_en),
    .q_beat_index (q_load_index),
    .q_beat_data  (axi_rdata),
    .k_beat_we    (k_load_en),
    .k_beat_index (k_load_index),
    .k_beat_data  (axi_rdata),
    .start_token  (core_start_token),
    .token_masked (token_masked),
    .score_ready  (core_score_ready),
    .busy         (core_busy),
    .score_valid  (core_score_valid),
    .score_head   (core_score_head),
    .score_q28    (core_score_q28),
    .token_done   (core_token_done)
);

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        state               <= ST_IDLE;
        layer_reg           <= 5'd0;
        query_position_reg  <= 15'd0;
        window_start_reg    <= 15'd0;
        count_reg           <= 5'd0;
        q_base_addr         <= {CTRL_ADDR_WIDTH{1'b0}};
        result_base_addr    <= {CTRL_ADDR_WIDTH{1'b0}};
        q_read_base_beat    <= 8'd0;
        read_beat_index     <= 5'd0;
        active_burst_beats  <= 5'd0;
        token_index         <= 5'd0;
        score_cache         <= 64'd0;
        score_head_cache    <= 4'd0;
        ar_seen             <= 1'b0;
        aw_seen             <= 1'b0;
        w_seen              <= 1'b0;
        core_start_token    <= 1'b0;
        axi_awaddr          <= {CTRL_ADDR_WIDTH{1'b0}};
        axi_awvalid         <= 1'b0;
        axi_wdata           <= 256'd0;
        axi_wstrb           <= 32'd0;
        axi_araddr          <= {CTRL_ADDR_WIDTH{1'b0}};
        axi_arlen           <= 4'd0;
        axi_arvalid         <= 1'b0;
        busy                <= 1'b0;
        done                <= 1'b0;
        error               <= 1'b0;
        error_code          <= 8'd0;
        debug_token         <= 5'd0;
        debug_head          <= 4'd0;
    end else begin
        done             <= 1'b0;
        core_start_token <= 1'b0;

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
                        layer_reg          <= cfg_layer;
                        query_position_reg <= cfg_query_position;
                        window_start_reg   <= cfg_window_start;
                        count_reg          <= cfg_count;
                        q_base_addr        <= cfg_q_addr;
                        result_base_addr   <= cfg_result_addr;
                        q_read_base_beat   <= 8'd0;
                        token_index        <= 5'd0;
                        debug_token        <= 5'd0;
                        debug_head         <= 4'd0;
                        busy               <= 1'b1;
                        error_code         <= 8'd0;
                        state              <= ST_SETUP_Q_READ;
                    end
                end
            end

            ST_SETUP_Q_READ: begin
                axi_araddr         <= q_base_addr + ({20'd0, q_read_base_beat} << 3);
                axi_arlen          <= next_q_burst_beats - 1'b1;
                axi_arvalid        <= 1'b1;
                ar_seen            <= 1'b0;
                read_beat_index    <= 5'd0;
                active_burst_beats <= next_q_burst_beats;
                state              <= ST_READ_Q;
            end

            ST_READ_Q: begin
                if (ar_handshake) begin
                    axi_arvalid <= 1'b0;
                    ar_seen     <= 1'b1;
                end
                if (read_data_handshake) begin
                    if (read_beat_index + 1'b1 == active_burst_beats) begin
                        ar_seen <= 1'b0;
                        if (q_read_base_beat + active_burst_beats == Q_BEATS) begin
                            q_read_base_beat <= 8'd0;
                            state <= ST_TOKEN_PREPARE;
                        end else begin
                            q_read_base_beat <= q_read_base_beat + active_burst_beats;
                            state <= ST_SETUP_Q_READ;
                        end
                    end else begin
                        read_beat_index <= read_beat_index + 1'b1;
                    end
                end
            end

            ST_TOKEN_PREPARE: begin
                debug_token <= token_index;
                debug_head  <= 4'd0;
                if (token_masked) begin
                    state <= ST_START_TOKEN;
                end else begin
                    q_read_base_beat <= 8'd0;
                    state <= ST_SETUP_K_READ;
                end
            end

            ST_SETUP_K_READ: begin
                axi_araddr         <= active_k_addr + ({20'd0, q_read_base_beat} << 3);
                axi_arlen          <= next_k_burst_beats - 1'b1;
                axi_arvalid        <= 1'b1;
                ar_seen            <= 1'b0;
                read_beat_index    <= 5'd0;
                active_burst_beats <= next_k_burst_beats;
                state              <= ST_READ_K;
            end

            ST_READ_K: begin
                if (ar_handshake) begin
                    axi_arvalid <= 1'b0;
                    ar_seen     <= 1'b1;
                end
                if (read_data_handshake) begin
                    if (read_beat_index + 1'b1 == active_burst_beats) begin
                        ar_seen <= 1'b0;
                        if (q_read_base_beat + active_burst_beats == K_BEATS) begin
                            q_read_base_beat <= 8'd0;
                            state <= ST_START_TOKEN;
                        end else begin
                            q_read_base_beat <= q_read_base_beat + active_burst_beats;
                            state <= ST_SETUP_K_READ;
                        end
                    end else begin
                        read_beat_index <= read_beat_index + 1'b1;
                    end
                end
            end

            ST_START_TOKEN: begin
                if (core_busy) begin
                    error      <= 1'b1;
                    error_code <= ERR_CORE_PROTOCOL;
                    state      <= ST_ERROR;
                end else begin
                    core_start_token <= 1'b1;
                    state            <= ST_WAIT_SCORE;
                end
            end

            ST_WAIT_SCORE: begin
                if (core_score_valid) begin
                    score_cache      <= core_score_q28;
                    score_head_cache <= core_score_head;
                    debug_head       <= core_score_head;
                    state            <= ST_SETUP_WRITE;
                end else if (core_token_done) begin
                    state <= ST_ADVANCE_TOKEN;
                end
            end

            ST_SETUP_WRITE: begin
                axi_awaddr  <= result_base_addr + ({20'd0, score_beat_index} << 3);
                axi_awvalid <= 1'b1;
                axi_wdata   <= score_write_data;
                axi_wstrb   <= score_write_strobe;
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
                    // head13 已通过同拍 score_ready 被核心接受，且对应 DDR3
                    // 写回已经完成；包装器可直接推进 token，无需再依赖额外的
                    // token_done 观察状态。其余 head 返回等待下一项 score_valid。
                    if (score_head_cache == 4'd13)
                        state <= ST_ADVANCE_TOKEN;
                    else
                        state <= ST_WAIT_SCORE;
                end
            end

            ST_ADVANCE_TOKEN: begin
                if (token_index == MAX_TOKENS - 1) begin
                    state <= ST_FINISH;
                end else begin
                    token_index <= token_index + 1'b1;
                    state       <= ST_TOKEN_PREPARE;
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

wire _unused_query_position = &{1'b0, query_position_reg};

endmodule
