`timescale 1ns/1ps

// G2 Softmax DDR3 阶段包装器。
// 加载固定 [14,16] Q28 score 与 513 点 UQ1.31 exp LUT，复用已验证
// softmax_core，并按 head-major [14,16] 用 4-byte strobe 写回概率。
module g2_softmax_stage_ctrl #(
    parameter integer CTRL_ADDR_WIDTH = 28,
    parameter integer SCORE_BEATS     = 56,
    parameter integer LUT_BEATS       = 65
)(
    input  wire                         clk,
    input  wire                         rst_n,
    input  wire                         ddr_init_done,
    input  wire                         start,
    input  wire [CTRL_ADDR_WIDTH-1:0]   cfg_score_addr,
    input  wire [CTRL_ADDR_WIDTH-1:0]   cfg_lut_addr,
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
    output reg  [3:0]                   debug_head,
    output reg  [3:0]                   debug_token
);

localparam [6:0] ST_IDLE             = 7'd0;
localparam [6:0] ST_SETUP_SCORE_READ = 7'd1;
localparam [6:0] ST_READ_SCORE       = 7'd2;
localparam [6:0] ST_SETUP_LUT_READ   = 7'd3;
localparam [6:0] ST_READ_LUT         = 7'd4;
localparam [6:0] ST_START_CORE       = 7'd5;
localparam [6:0] ST_WAIT_PROBABILITY = 7'd6;
localparam [6:0] ST_SETUP_WRITE      = 7'd7;
localparam [6:0] ST_WRITE            = 7'd8;
localparam [6:0] ST_FINISH           = 7'd9;
localparam [6:0] ST_ERROR            = 7'd127;

localparam [7:0] ERR_DDR_NOT_READY = 8'h01;
localparam [7:0] ERR_CORE_PROTOCOL = 8'h02;
localparam [7:0] ERR_INTERNAL      = 8'hff;

reg [6:0] state;
reg [CTRL_ADDR_WIDTH-1:0] score_base_addr;
reg [CTRL_ADDR_WIDTH-1:0] lut_base_addr;
reg [CTRL_ADDR_WIDTH-1:0] result_base_addr;
reg [6:0] read_base_beat;
reg [4:0] read_beat_index;
reg [4:0] active_burst_beats;
reg [3:0] probability_head_cache;
reg [3:0] probability_token_cache;
reg [31:0] probability_cache;
reg ar_seen;
reg aw_seen;
reg w_seen;
reg core_start;

wire core_busy;
wire core_probability_valid;
wire [3:0] core_probability_head;
wire [3:0] core_probability_token;
wire [31:0] core_probability_q31;
wire core_done;
wire core_probability_ready;

wire aw_handshake = axi_awvalid && axi_awready;
wire ar_handshake = axi_arvalid && axi_arready;
wire write_data_handshake = axi_wready && (aw_seen || aw_handshake);
wire write_complete = (aw_seen || aw_handshake) &&
                      (w_seen || write_data_handshake);
wire read_data_handshake = axi_rvalid && (ar_seen || ar_handshake);

wire [7:0] score_beats_remaining = SCORE_BEATS - read_base_beat;
wire [4:0] next_score_burst_beats =
    (score_beats_remaining > 8'd16) ? 5'd16 : score_beats_remaining[4:0];
wire [7:0] lut_beats_remaining = LUT_BEATS - read_base_beat;
wire [4:0] next_lut_burst_beats =
    (lut_beats_remaining > 8'd16) ? 5'd16 : lut_beats_remaining[4:0];

wire score_load_en = (state == ST_READ_SCORE) && read_data_handshake;
wire lut_load_en   = (state == ST_READ_LUT) && read_data_handshake;
wire [6:0] load_index = read_base_beat + read_beat_index;
assign core_probability_ready = (state == ST_WRITE) && write_complete;

wire [5:0] result_beat_index =
    {probability_head_cache, 1'b0} + {5'd0, probability_token_cache[3]};
wire [2:0] result_lane = probability_token_cache[2:0];
reg [255:0] probability_write_data;
reg [31:0] probability_write_strobe;
always @(*) begin
    probability_write_data = 256'd0;
    probability_write_data[result_lane*32 +: 32] = probability_cache;
    probability_write_strobe = 32'd0;
    probability_write_strobe[result_lane*4 +: 4] = 4'hf;
end

assign axi_awuser_ap = 1'b0;
assign axi_awuser_id = 4'h0;
assign axi_awlen     = 4'h0;
assign axi_aruser_ap = 1'b0;
assign axi_aruser_id = 4'h0;
assign debug_state   = state;

softmax_core #(
    .PIPELINE_SCORE_DIFF (1)
) u_softmax_core (
    .clk               (clk),
    .rst_n             (rst_n),
    .score_beat_we     (score_load_en),
    .score_beat_index  (load_index[5:0]),
    .score_beat_data   (axi_rdata),
    .lut_beat_we       (lut_load_en),
    .lut_beat_index    (load_index),
    .lut_beat_data     (axi_rdata),
    .start             (core_start),
    .probability_ready (core_probability_ready),
    .busy              (core_busy),
    .probability_valid (core_probability_valid),
    .probability_head  (core_probability_head),
    .probability_token (core_probability_token),
    .probability_q31   (core_probability_q31),
    .done              (core_done)
);

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        state                   <= ST_IDLE;
        score_base_addr         <= {CTRL_ADDR_WIDTH{1'b0}};
        lut_base_addr           <= {CTRL_ADDR_WIDTH{1'b0}};
        result_base_addr        <= {CTRL_ADDR_WIDTH{1'b0}};
        read_base_beat          <= 7'd0;
        read_beat_index         <= 5'd0;
        active_burst_beats      <= 5'd0;
        probability_head_cache  <= 4'd0;
        probability_token_cache <= 4'd0;
        probability_cache       <= 32'd0;
        ar_seen                 <= 1'b0;
        aw_seen                 <= 1'b0;
        w_seen                  <= 1'b0;
        core_start              <= 1'b0;
        axi_awaddr              <= {CTRL_ADDR_WIDTH{1'b0}};
        axi_awvalid             <= 1'b0;
        axi_wdata               <= 256'd0;
        axi_wstrb               <= 32'd0;
        axi_araddr              <= {CTRL_ADDR_WIDTH{1'b0}};
        axi_arlen               <= 4'd0;
        axi_arvalid             <= 1'b0;
        busy                    <= 1'b0;
        done                    <= 1'b0;
        error                   <= 1'b0;
        error_code              <= 8'd0;
        debug_head              <= 4'd0;
        debug_token             <= 4'd0;
    end else begin
        done       <= 1'b0;
        core_start <= 1'b0;

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
                    end else begin
                        score_base_addr   <= cfg_score_addr;
                        lut_base_addr     <= cfg_lut_addr;
                        result_base_addr  <= cfg_result_addr;
                        read_base_beat    <= 7'd0;
                        debug_head        <= 4'd0;
                        debug_token       <= 4'd0;
                        busy              <= 1'b1;
                        error_code        <= 8'd0;
                        state             <= ST_SETUP_SCORE_READ;
                    end
                end
            end

            ST_SETUP_SCORE_READ: begin
                axi_araddr         <= score_base_addr + ({21'd0, read_base_beat} << 3);
                axi_arlen          <= next_score_burst_beats - 1'b1;
                axi_arvalid        <= 1'b1;
                ar_seen            <= 1'b0;
                read_beat_index    <= 5'd0;
                active_burst_beats <= next_score_burst_beats;
                state              <= ST_READ_SCORE;
            end

            ST_READ_SCORE: begin
                if (ar_handshake) begin
                    axi_arvalid <= 1'b0;
                    ar_seen     <= 1'b1;
                end
                if (read_data_handshake) begin
                    if (read_beat_index + 1'b1 == active_burst_beats) begin
                        ar_seen <= 1'b0;
                        if (read_base_beat + active_burst_beats == SCORE_BEATS) begin
                            read_base_beat <= 7'd0;
                            state <= ST_SETUP_LUT_READ;
                        end else begin
                            read_base_beat <= read_base_beat + active_burst_beats;
                            state <= ST_SETUP_SCORE_READ;
                        end
                    end else begin
                        read_beat_index <= read_beat_index + 1'b1;
                    end
                end
            end

            ST_SETUP_LUT_READ: begin
                axi_araddr         <= lut_base_addr + ({21'd0, read_base_beat} << 3);
                axi_arlen          <= next_lut_burst_beats - 1'b1;
                axi_arvalid        <= 1'b1;
                ar_seen            <= 1'b0;
                read_beat_index    <= 5'd0;
                active_burst_beats <= next_lut_burst_beats;
                state              <= ST_READ_LUT;
            end

            ST_READ_LUT: begin
                if (ar_handshake) begin
                    axi_arvalid <= 1'b0;
                    ar_seen     <= 1'b1;
                end
                if (read_data_handshake) begin
                    if (read_beat_index + 1'b1 == active_burst_beats) begin
                        ar_seen <= 1'b0;
                        if (read_base_beat + active_burst_beats == LUT_BEATS) begin
                            read_base_beat <= 7'd0;
                            state <= ST_START_CORE;
                        end else begin
                            read_base_beat <= read_base_beat + active_burst_beats;
                            state <= ST_SETUP_LUT_READ;
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
                    state      <= ST_WAIT_PROBABILITY;
                end
            end

            ST_WAIT_PROBABILITY: begin
                if (core_probability_valid) begin
                    probability_head_cache  <= core_probability_head;
                    probability_token_cache <= core_probability_token;
                    probability_cache       <= core_probability_q31;
                    debug_head              <= core_probability_head;
                    debug_token             <= core_probability_token;
                    state                   <= ST_SETUP_WRITE;
                end else if (core_done) begin
                    state <= ST_FINISH;
                end
            end

            ST_SETUP_WRITE: begin
                axi_awaddr  <= result_base_addr + ({22'd0, result_beat_index} << 3);
                axi_awvalid <= 1'b1;
                axi_wdata   <= probability_write_data;
                axi_wstrb   <= probability_write_strobe;
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
                    // 最后一个 (head13, token15) 已通过同拍
                    // probability_ready 被核心接受，且 DDR3 写回已经完成；
                    // 直接依据已缓存索引结束，不再增加额外 done 等待状态。
                    if ((probability_head_cache == 4'd13) &&
                        (probability_token_cache == 4'd15))
                        state <= ST_FINISH;
                    else
                        state <= ST_WAIT_PROBABILITY;
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
