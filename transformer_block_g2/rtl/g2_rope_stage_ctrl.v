`timescale 1ns/1ps

// G2 layer0 Q/K RoPE 流式阶段。
// Q=[14,64]、K=[2,64] 均为 head-major signed Q28；trig 区保存
// cos[32] 后接 sin[32] 的 signed Q1.30。每次仅缓存一组 4 个 split-half pair，
// 数值核心直接复用已经真实上板验证的 rope_pair_q28_core。
module g2_rope_stage_ctrl #(
    parameter integer CTRL_ADDR_WIDTH = 28,
    parameter integer Q_HEADS         = 14,
    parameter integer KV_HEADS        = 2,
    parameter integer GROUPS_PER_HEAD = 8,
    parameter integer TRIG_BEATS      = 8
)(
    input  wire                         clk,
    input  wire                         rst_n,
    input  wire                         ddr_init_done,
    input  wire                         start,
    input  wire [CTRL_ADDR_WIDTH-1:0]   cfg_q_source_addr,
    input  wire [CTRL_ADDR_WIDTH-1:0]   cfg_k_source_addr,
    input  wire [CTRL_ADDR_WIDTH-1:0]   cfg_trig_addr,
    input  wire [CTRL_ADDR_WIDTH-1:0]   cfg_q_result_addr,
    input  wire [CTRL_ADDR_WIDTH-1:0]   cfg_k_result_addr,

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
    output wire [5:0]                   debug_state,
    output reg                          debug_is_k,
    output reg  [3:0]                   debug_head,
    output reg  [2:0]                   debug_group
);

localparam [5:0] ST_IDLE              = 6'd0;
localparam [5:0] ST_SETUP_TRIG_READ   = 6'd1;
localparam [5:0] ST_READ_TRIG         = 6'd2;
localparam [5:0] ST_SETUP_FIRST_READ  = 6'd3;
localparam [5:0] ST_READ_FIRST        = 6'd4;
localparam [5:0] ST_SETUP_SECOND_READ = 6'd5;
localparam [5:0] ST_READ_SECOND       = 6'd6;
localparam [5:0] ST_START_PAIR        = 6'd7;
localparam [5:0] ST_WAIT_PAIR         = 6'd8;
localparam [5:0] ST_SETUP_FIRST_WRITE = 6'd9;
localparam [5:0] ST_WRITE_FIRST       = 6'd10;
localparam [5:0] ST_SETUP_SECOND_WRITE= 6'd11;
localparam [5:0] ST_WRITE_SECOND      = 6'd12;
localparam [5:0] ST_ADVANCE           = 6'd13;
localparam [5:0] ST_FINISH            = 6'd14;
localparam [5:0] ST_ERROR             = 6'd63;

localparam [7:0] ERR_DDR_NOT_READY = 8'h01;
localparam [7:0] ERR_CORE_PROTOCOL = 8'h02;
localparam [7:0] ERR_INTERNAL      = 8'hff;

reg [5:0] state;
reg [CTRL_ADDR_WIDTH-1:0] q_source_addr;
reg [CTRL_ADDR_WIDTH-1:0] k_source_addr;
reg [CTRL_ADDR_WIDTH-1:0] trig_addr;
reg [CTRL_ADDR_WIDTH-1:0] q_result_addr;
reg [CTRL_ADDR_WIDTH-1:0] k_result_addr;
reg is_k;
reg [3:0] head_index;
reg [2:0] group_index;
reg [1:0] pair_lane;
reg [3:0] trig_read_index;
reg ar_seen;
reg aw_seen;
reg w_seen;

reg signed [31:0] trig_mem [0:63];
reg [255:0] source_first_beat;
reg [255:0] source_second_beat;
reg [255:0] result_first_pack;
reg [255:0] result_second_pack;

reg core_start;
wire core_busy;
wire core_done;
wire signed [63:0] core_y_first;
wire signed [63:0] core_y_second;

integer trig_lane;
integer trig_global_index;

wire aw_handshake = axi_awvalid && axi_awready;
wire ar_handshake = axi_arvalid && axi_arready;
wire write_data_handshake = axi_wready && (aw_seen || aw_handshake);
wire write_complete = (aw_seen || aw_handshake) &&
                      (w_seen || write_data_handshake);
// AXI 读返回已经在 g2_axi_stage_mux 中寄存，绝不会与本地 AR 请求捕获同拍返回。
// 这里只依赖已寄存的 ar_seen，避免把 scheduler/仲裁选择组合路径带到 trig_mem 写使能。
wire read_data_handshake = axi_rvalid && ar_seen;

wire [CTRL_ADDR_WIDTH-1:0] active_source_base = is_k ? k_source_addr : q_source_addr;
wire [CTRL_ADDR_WIDTH-1:0] active_result_base = is_k ? k_result_addr : q_result_addr;
wire [CTRL_ADDR_WIDTH-1:0] head_ctrl_offset = {{(CTRL_ADDR_WIDTH-11){1'b0}}, head_index, 7'd0};
wire [CTRL_ADDR_WIDTH-1:0] group_ctrl_offset = {{(CTRL_ADDR_WIDTH-6){1'b0}}, group_index, 3'd0};
wire [CTRL_ADDR_WIDTH-1:0] second_half_offset = {{(CTRL_ADDR_WIDTH-7){1'b0}}, 7'd64};
wire [5:0] dimension_index = {group_index, 2'b00} + pair_lane;

wire signed [63:0] selected_first =
    $signed(source_first_beat[pair_lane*64 +: 64]);
wire signed [63:0] selected_second =
    $signed(source_second_beat[pair_lane*64 +: 64]);
wire signed [31:0] selected_cos = trig_mem[dimension_index];
wire signed [31:0] selected_sin = trig_mem[32 + dimension_index];

reg [255:0] result_first_next;
reg [255:0] result_second_next;
always @(*) begin
    result_first_next = result_first_pack;
    result_second_next = result_second_pack;
    result_first_next[pair_lane*64 +: 64] = core_y_first;
    result_second_next[pair_lane*64 +: 64] = core_y_second;
end

assign axi_awuser_ap = 1'b0;
assign axi_awuser_id = 4'h0;
assign axi_awlen     = 4'h0;
assign axi_aruser_ap = 1'b0;
assign axi_aruser_id = 4'h0;
assign debug_state   = state;

rope_pair_q28_core u_rope_pair_q28_core (
    .clk          (clk),
    .rst_n        (rst_n),
    .start        (core_start),
    .x_first_q28  (selected_first),
    .x_second_q28 (selected_second),
    .cos_q30      (selected_cos),
    .sin_q30      (selected_sin),
    .busy         (core_busy),
    .done         (core_done),
    .y_first_q28  (core_y_first),
    .y_second_q28 (core_y_second)
);

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        state               <= ST_IDLE;
        q_source_addr       <= {CTRL_ADDR_WIDTH{1'b0}};
        k_source_addr       <= {CTRL_ADDR_WIDTH{1'b0}};
        trig_addr           <= {CTRL_ADDR_WIDTH{1'b0}};
        q_result_addr       <= {CTRL_ADDR_WIDTH{1'b0}};
        k_result_addr       <= {CTRL_ADDR_WIDTH{1'b0}};
        is_k                <= 1'b0;
        head_index          <= 4'd0;
        group_index         <= 3'd0;
        pair_lane           <= 2'd0;
        trig_read_index     <= 4'd0;
        ar_seen             <= 1'b0;
        aw_seen             <= 1'b0;
        w_seen              <= 1'b0;
        source_first_beat   <= 256'd0;
        source_second_beat  <= 256'd0;
        result_first_pack   <= 256'd0;
        result_second_pack  <= 256'd0;
        core_start          <= 1'b0;
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
        debug_is_k          <= 1'b0;
        debug_head          <= 4'd0;
        debug_group         <= 3'd0;
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
                        q_source_addr      <= cfg_q_source_addr;
                        k_source_addr      <= cfg_k_source_addr;
                        trig_addr          <= cfg_trig_addr;
                        q_result_addr      <= cfg_q_result_addr;
                        k_result_addr      <= cfg_k_result_addr;
                        is_k               <= 1'b0;
                        head_index         <= 4'd0;
                        group_index        <= 3'd0;
                        pair_lane          <= 2'd0;
                        trig_read_index    <= 4'd0;
                        debug_is_k         <= 1'b0;
                        debug_head         <= 4'd0;
                        debug_group        <= 3'd0;
                        busy               <= 1'b1;
                        error_code         <= 8'd0;
                        state              <= ST_SETUP_TRIG_READ;
                    end
                end
            end

            ST_SETUP_TRIG_READ: begin
                axi_araddr      <= trig_addr;
                axi_arlen       <= TRIG_BEATS - 1;
                axi_arvalid     <= 1'b1;
                ar_seen         <= 1'b0;
                trig_read_index <= 4'd0;
                state           <= ST_READ_TRIG;
            end

            ST_READ_TRIG: begin
                if (ar_handshake) begin
                    axi_arvalid <= 1'b0;
                    ar_seen     <= 1'b1;
                end
                if (read_data_handshake) begin
                    for (trig_lane = 0; trig_lane < 8; trig_lane = trig_lane + 1) begin
                        trig_global_index = trig_read_index * 8 + trig_lane;
                        trig_mem[trig_global_index] <= axi_rdata[trig_lane*32 +: 32];
                    end
                    if (trig_read_index == TRIG_BEATS - 1) begin
                        ar_seen <= 1'b0;
                        state   <= ST_SETUP_FIRST_READ;
                    end else begin
                        trig_read_index <= trig_read_index + 1'b1;
                    end
                end
            end

            ST_SETUP_FIRST_READ: begin
                axi_araddr  <= active_source_base + head_ctrl_offset + group_ctrl_offset;
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
                    source_first_beat <= axi_rdata;
                    ar_seen           <= 1'b0;
                    state             <= ST_SETUP_SECOND_READ;
                end
            end

            ST_SETUP_SECOND_READ: begin
                axi_araddr  <= active_source_base + head_ctrl_offset +
                               second_half_offset + group_ctrl_offset;
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
                    source_second_beat <= axi_rdata;
                    ar_seen            <= 1'b0;
                    pair_lane          <= 2'd0;
                    result_first_pack  <= 256'd0;
                    result_second_pack <= 256'd0;
                    state              <= ST_START_PAIR;
                end
            end

            ST_START_PAIR: begin
                if (core_busy) begin
                    error      <= 1'b1;
                    error_code <= ERR_CORE_PROTOCOL;
                    state      <= ST_ERROR;
                end else begin
                    core_start <= 1'b1;
                    state      <= ST_WAIT_PAIR;
                end
            end

            ST_WAIT_PAIR: begin
                if (core_done) begin
                    result_first_pack  <= result_first_next;
                    result_second_pack <= result_second_next;
                    if (pair_lane == 2'd3) begin
                        state <= ST_SETUP_FIRST_WRITE;
                    end else begin
                        pair_lane <= pair_lane + 1'b1;
                        state     <= ST_START_PAIR;
                    end
                end
            end

            ST_SETUP_FIRST_WRITE: begin
                axi_awaddr  <= active_result_base + head_ctrl_offset + group_ctrl_offset;
                axi_awvalid <= 1'b1;
                axi_wdata   <= result_first_pack;
                axi_wstrb   <= 32'hffff_ffff;
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
                if (write_complete) begin
                    axi_awvalid <= 1'b0;
                    aw_seen     <= 1'b0;
                    w_seen      <= 1'b0;
                    state       <= ST_SETUP_SECOND_WRITE;
                end
            end

            ST_SETUP_SECOND_WRITE: begin
                axi_awaddr  <= active_result_base + head_ctrl_offset +
                               second_half_offset + group_ctrl_offset;
                axi_awvalid <= 1'b1;
                axi_wdata   <= result_second_pack;
                axi_wstrb   <= 32'hffff_ffff;
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
                if (write_complete) begin
                    axi_awvalid <= 1'b0;
                    aw_seen     <= 1'b0;
                    w_seen      <= 1'b0;
                    state       <= ST_ADVANCE;
                end
            end

            ST_ADVANCE: begin
                if (group_index != GROUPS_PER_HEAD - 1) begin
                    group_index <= group_index + 1'b1;
                    debug_group <= group_index + 1'b1;
                    state       <= ST_SETUP_FIRST_READ;
                end else if ((!is_k && (head_index != Q_HEADS - 1)) ||
                             (is_k && (head_index != KV_HEADS - 1))) begin
                    group_index <= 3'd0;
                    head_index  <= head_index + 1'b1;
                    debug_group <= 3'd0;
                    debug_head  <= head_index + 1'b1;
                    state       <= ST_SETUP_FIRST_READ;
                end else if (!is_k) begin
                    is_k        <= 1'b1;
                    head_index  <= 4'd0;
                    group_index <= 3'd0;
                    debug_is_k  <= 1'b1;
                    debug_head  <= 4'd0;
                    debug_group <= 3'd0;
                    state       <= ST_SETUP_FIRST_READ;
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
