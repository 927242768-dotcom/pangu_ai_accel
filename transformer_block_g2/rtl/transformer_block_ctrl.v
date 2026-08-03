`timescale 1ns/1ps

`include "transformer_block_contract.vh"

// G2 完整 layer0 Transformer Block 统一控制器。
//
// 22 个 scheduler 阶段通过 11 个可复用 DDR3 AXI 主设备顺序执行：
//   0 RMSNorm             : INPUT_RMS / POST_RMS
//   1 unified quantizer   : QKV / OPROJ / GATE_UP / DOWN QUANT
//   2 shared Linear       : q/k/v/o/gate/up/down
//   3 RoPE                : Q/K
//   4 KV writer           : current K/V
//   5 Attention Score
//   6 Softmax
//   7 Attention Output
//   8 residual            : residual1 / residual2
//   9 SiLU
//  10 SiLU * up
//
// 任意时刻 scheduler 只启动一个阶段；g2_axi_stage_mux 因而无需保存 outstanding
// transaction ID。所有阶段通过 transformer_block_contract.vh 的固定 DDR3 区域
// 交换张量，主机不得注入中间结果。
module transformer_block_ctrl #(
    parameter integer CTRL_ADDR_WIDTH    = 28,
    parameter integer ACTIVE_LAYER_COUNT = 1,
    parameter [31:0]  WATCHDOG_CYCLES    = 32'd0
)(
    input  wire                         clk,
    input  wire                         rst_n,
    input  wire                         ddr_init_done,
    input  wire                         start,
    input  wire [4:0]                   cfg_layer,
    input  wire [14:0]                  cfg_query_position,
    input  wire [14:0]                  cfg_window_start,
    input  wire [4:0]                   cfg_count,

    output wire [CTRL_ADDR_WIDTH-1:0]   axi_awaddr,
    output wire                         axi_awuser_ap,
    output wire [3:0]                   axi_awuser_id,
    output wire [3:0]                   axi_awlen,
    input  wire                         axi_awready,
    output wire                         axi_awvalid,
    output wire [255:0]                 axi_wdata,
    output wire [31:0]                  axi_wstrb,
    input  wire                         axi_wready,

    output wire [CTRL_ADDR_WIDTH-1:0]   axi_araddr,
    output wire                         axi_aruser_ap,
    output wire [3:0]                   axi_aruser_id,
    output wire [3:0]                   axi_arlen,
    input  wire                         axi_arready,
    output wire                         axi_arvalid,
    input  wire [255:0]                 axi_rdata,
    input  wire                         axi_rvalid,

    output wire                         busy,
    output wire                         done,
    output wire                         error,
    output wire [7:0]                   error_code,
    output wire [4:0]                   current_stage,
    output wire [31:0]                  watchdog_count,
    output wire [3:0]                   debug_axi_master
);

localparam integer NUM_MASTERS = 11;
localparam [3:0] MASTER_RMS       = 4'd0;
localparam [3:0] MASTER_QUANT     = 4'd1;
localparam [3:0] MASTER_LINEAR    = 4'd2;
localparam [3:0] MASTER_ROPE      = 4'd3;
localparam [3:0] MASTER_KV        = 4'd4;
localparam [3:0] MASTER_SCORE     = 4'd5;
localparam [3:0] MASTER_SOFTMAX   = 4'd6;
localparam [3:0] MASTER_ATTN_OUT  = 4'd7;
localparam [3:0] MASTER_RESIDUAL  = 4'd8;
localparam [3:0] MASTER_SILU      = 4'd9;
localparam [3:0] MASTER_SILU_MUL  = 4'd10;

localparam [1:0] QUANT_MODE_QKV     = 2'd0;
localparam [1:0] QUANT_MODE_OPROJ   = 2'd1;
localparam [1:0] QUANT_MODE_GATE_UP = 2'd2;
localparam [1:0] QUANT_MODE_DOWN    = 2'd3;

localparam [2:0] LINEAR_MODE_Q    = 3'd0;
localparam [2:0] LINEAR_MODE_K    = 3'd1;
localparam [2:0] LINEAR_MODE_V    = 3'd2;
localparam [2:0] LINEAR_MODE_O    = 3'd3;
localparam [2:0] LINEAR_MODE_GATE = 3'd4;
localparam [2:0] LINEAR_MODE_UP   = 3'd5;
localparam [2:0] LINEAR_MODE_DOWN = 3'd6;

localparam [7:0] ERR_CONFIG       = 8'h01;
localparam [7:0] ERR_DDR_NOT_READY= 8'h02;
localparam [7:0] ERR_AXI_SELECT   = 8'h03;

wire [15:0] expected_query_position =
    {1'b0, cfg_window_start} + {11'd0, cfg_count} - 16'd1;
wire configuration_valid =
    // 默认 ACTIVE_LAYER_COUNT=1，保持已验收 G2 只接受 layer0。
    // H3 wrapper 在每层参数换入完成后显式设为 24，开放真实 layer0..23；
    // 不得使用硬件容量中没有真实模型参数的 layer24..27。
    (cfg_layer < ACTIVE_LAYER_COUNT) &&
    (cfg_count >= 5'd1) &&
    (cfg_count <= `G2_MAX_WINDOW) &&
    (cfg_window_start < `G2_KV_MAX_POSITIONS) &&
    (cfg_query_position < `G2_KV_MAX_POSITIONS) &&
    ({1'b0, cfg_query_position} == expected_query_position);

reg local_error;
reg [7:0] local_error_code;
wire scheduler_start = start && configuration_valid && ddr_init_done && !local_error;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        local_error      <= 1'b0;
        local_error_code <= 8'd0;
    end else if (start && !local_error) begin
        if (!ddr_init_done) begin
            local_error      <= 1'b1;
            local_error_code <= ERR_DDR_NOT_READY;
        end else if (!configuration_valid) begin
            local_error      <= 1'b1;
            local_error_code <= ERR_CONFIG;
        end
    end
end

wire [21:0] engine_start;
wire [21:0] engine_done;
wire [21:0] engine_error;
wire scheduler_busy;
wire scheduler_done;
wire scheduler_error;
wire [7:0] scheduler_error_code;

transformer_block_scheduler #(
    .WATCHDOG_CYCLES (WATCHDOG_CYCLES)
) u_transformer_block_scheduler (
    .clk            (clk),
    .rst_n          (rst_n),
    .start          (scheduler_start),
    .engine_done    (engine_done),
    .engine_error   (engine_error),
    .engine_start   (engine_start),
    .current_stage  (current_stage),
    .busy           (scheduler_busy),
    .done           (scheduler_done),
    .error          (scheduler_error),
    .error_code     (scheduler_error_code),
    .watchdog_count (watchdog_count)
);

// -----------------------------------------------------------------------------
// 每个共享阶段的固定配置选择。
// -----------------------------------------------------------------------------
wire [CTRL_ADDR_WIDTH-1:0] rms_input_addr =
    (current_stage == `G2_STAGE_POST_RMS) ?
    `G2_ATTN_RESIDUAL_CTRL_ADDR : `G2_BLOCK_HIDDEN_CTRL_ADDR;
wire [CTRL_ADDR_WIDTH-1:0] rms_gamma_addr =
    (current_stage == `G2_STAGE_POST_RMS) ?
    `G2_POST_RMS_GAMMA_CTRL_ADDR : `G2_INPUT_RMS_GAMMA_CTRL_ADDR;
wire [CTRL_ADDR_WIDTH-1:0] rms_result_addr =
    (current_stage == `G2_STAGE_POST_RMS) ?
    `G2_POST_NORM_CTRL_ADDR : `G2_INPUT_NORM_CTRL_ADDR;
wire rms_start = engine_start[0] || engine_start[13];

reg [1:0] quant_mode;
always @(*) begin
    case (current_stage)
        `G2_STAGE_QKV_QUANT:     quant_mode = QUANT_MODE_QKV;
        `G2_STAGE_OPROJ_QUANT:   quant_mode = QUANT_MODE_OPROJ;
        `G2_STAGE_GATE_UP_QUANT: quant_mode = QUANT_MODE_GATE_UP;
        default:                 quant_mode = QUANT_MODE_DOWN;
    endcase
end
wire quant_start = engine_start[1] || engine_start[10] ||
                   engine_start[14] || engine_start[19];

reg [2:0] linear_mode;
always @(*) begin
    case (current_stage)
        `G2_STAGE_Q_LINEAR:    linear_mode = LINEAR_MODE_Q;
        `G2_STAGE_K_LINEAR:    linear_mode = LINEAR_MODE_K;
        `G2_STAGE_V_LINEAR:    linear_mode = LINEAR_MODE_V;
        `G2_STAGE_OPROJ_LINEAR:linear_mode = LINEAR_MODE_O;
        `G2_STAGE_GATE_LINEAR: linear_mode = LINEAR_MODE_GATE;
        `G2_STAGE_UP_LINEAR:   linear_mode = LINEAR_MODE_UP;
        default:               linear_mode = LINEAR_MODE_DOWN;
    endcase
end
wire linear_start = engine_start[2] || engine_start[3] || engine_start[4] ||
                    engine_start[11] || engine_start[15] || engine_start[16] ||
                    engine_start[20];

wire [CTRL_ADDR_WIDTH-1:0] residual_hidden_addr =
    (current_stage == `G2_STAGE_RESIDUAL2) ?
    `G2_ATTN_RESIDUAL_CTRL_ADDR : `G2_BLOCK_HIDDEN_CTRL_ADDR;
wire [CTRL_ADDR_WIDTH-1:0] residual_q28_addr =
    (current_stage == `G2_STAGE_RESIDUAL2) ?
    `G2_DOWN_CTRL_ADDR : `G2_OPROJ_CTRL_ADDR;
wire [CTRL_ADDR_WIDTH-1:0] residual_result_addr =
    (current_stage == `G2_STAGE_RESIDUAL2) ?
    `G2_BLOCK_OUTPUT_CTRL_ADDR : `G2_ATTN_RESIDUAL_CTRL_ADDR;
wire residual_start = engine_start[12] || engine_start[21];

// -----------------------------------------------------------------------------
// 12 个内部 AXI master 的独立信号。
// -----------------------------------------------------------------------------
wire [CTRL_ADDR_WIDTH-1:0] rms_awaddr;
wire rms_awuser_ap;
wire [3:0] rms_awuser_id;
wire [3:0] rms_awlen;
wire rms_awvalid;
wire [255:0] rms_wdata;
wire [31:0] rms_wstrb;
wire [CTRL_ADDR_WIDTH-1:0] rms_araddr;
wire rms_aruser_ap;
wire [3:0] rms_aruser_id;
wire [3:0] rms_arlen;
wire rms_arvalid;
wire rms_busy;
wire rms_done;
wire rms_error;
wire [7:0] rms_error_code;

g2_rmsnorm_stage_ctrl #(
    .CTRL_ADDR_WIDTH (CTRL_ADDR_WIDTH)
) u_g2_rmsnorm_stage_ctrl (
    .clk              (clk),
    .rst_n            (rst_n),
    .ddr_init_done    (ddr_init_done),
    .start            (rms_start),
    .cfg_input_addr   (rms_input_addr),
    .cfg_gamma_addr   (rms_gamma_addr),
    .cfg_lut_addr     (`G2_RMS_LUT_CTRL_ADDR),
    .cfg_result_addr  (rms_result_addr),
    .axi_awaddr       (rms_awaddr),
    .axi_awuser_ap    (rms_awuser_ap),
    .axi_awuser_id    (rms_awuser_id),
    .axi_awlen        (rms_awlen),
    .axi_awready      (master_awready[MASTER_RMS]),
    .axi_awvalid      (rms_awvalid),
    .axi_wdata        (rms_wdata),
    .axi_wstrb        (rms_wstrb),
    .axi_wready       (master_wready[MASTER_RMS]),
    .axi_araddr       (rms_araddr),
    .axi_aruser_ap    (rms_aruser_ap),
    .axi_aruser_id    (rms_aruser_id),
    .axi_arlen        (rms_arlen),
    .axi_arready      (master_arready[MASTER_RMS]),
    .axi_arvalid      (rms_arvalid),
    .axi_rdata        (master_rdata[MASTER_RMS*256 +: 256]),
    .axi_rvalid       (master_rvalid[MASTER_RMS]),
    .busy             (rms_busy),
    .done             (rms_done),
    .error            (rms_error),
    .error_code       (rms_error_code),
    .debug_state      (),
    .debug_read_beat  (),
    .debug_result_beat(),
    .debug_sum_squares(),
    .debug_variance_q20(),
    .debug_rsqrt_q20  ()
);

wire [CTRL_ADDR_WIDTH-1:0] quant_awaddr;
wire quant_awuser_ap;
wire [3:0] quant_awuser_id;
wire [3:0] quant_awlen;
wire quant_awvalid;
wire [255:0] quant_wdata;
wire [31:0] quant_wstrb;
wire [CTRL_ADDR_WIDTH-1:0] quant_araddr;
wire quant_aruser_ap;
wire [3:0] quant_aruser_id;
wire [3:0] quant_arlen;
wire quant_arvalid;
wire quant_busy;
wire quant_done;
wire quant_error;
wire [7:0] quant_error_code;

g2_quant_sequence_unified_ctrl #(
    .CTRL_ADDR_WIDTH (CTRL_ADDR_WIDTH)
) u_g2_quant_sequence_unified_ctrl (
    .clk              (clk),
    .rst_n            (rst_n),
    .ddr_init_done    (ddr_init_done),
    .start            (quant_start),
    .cfg_mode         (quant_mode),
    .axi_awaddr       (quant_awaddr),
    .axi_awuser_ap    (quant_awuser_ap),
    .axi_awuser_id    (quant_awuser_id),
    .axi_awlen        (quant_awlen),
    .axi_awready      (master_awready[MASTER_QUANT]),
    .axi_awvalid      (quant_awvalid),
    .axi_wdata        (quant_wdata),
    .axi_wstrb        (quant_wstrb),
    .axi_wready       (master_wready[MASTER_QUANT]),
    .axi_araddr       (quant_araddr),
    .axi_aruser_ap    (quant_aruser_ap),
    .axi_aruser_id    (quant_aruser_id),
    .axi_arlen        (quant_arlen),
    .axi_arready      (master_arready[MASTER_QUANT]),
    .axi_arvalid      (quant_arvalid),
    .axi_rdata        (master_rdata[MASTER_QUANT*256 +: 256]),
    .axi_rvalid       (master_rvalid[MASTER_QUANT]),
    .busy             (quant_busy),
    .done             (quant_done),
    .error            (quant_error),
    .error_code       (quant_error_code),
    .debug_inner_state(),
    .debug_mode       (),
    .debug_operation  ()
);

wire [CTRL_ADDR_WIDTH-1:0] linear_awaddr;
wire linear_awuser_ap;
wire [3:0] linear_awuser_id;
wire [3:0] linear_awlen;
wire linear_awvalid;
wire [255:0] linear_wdata;
wire [31:0] linear_wstrb;
wire [CTRL_ADDR_WIDTH-1:0] linear_araddr;
wire linear_aruser_ap;
wire [3:0] linear_aruser_id;
wire [3:0] linear_arlen;
wire linear_arvalid;
wire linear_busy;
wire linear_done;
wire linear_error;
wire [7:0] linear_error_code;

g2_linear_stage_ctrl #(
    .CTRL_ADDR_WIDTH (CTRL_ADDR_WIDTH)
) u_g2_linear_stage_ctrl (
    .clk              (clk),
    .rst_n            (rst_n),
    .ddr_init_done    (ddr_init_done),
    .start            (linear_start),
    .cfg_mode         (linear_mode),
    .axi_awaddr       (linear_awaddr),
    .axi_awuser_ap    (linear_awuser_ap),
    .axi_awuser_id    (linear_awuser_id),
    .axi_awlen        (linear_awlen),
    .axi_awready      (master_awready[MASTER_LINEAR]),
    .axi_awvalid      (linear_awvalid),
    .axi_wdata        (linear_wdata),
    .axi_wstrb        (linear_wstrb),
    .axi_wready       (master_wready[MASTER_LINEAR]),
    .axi_araddr       (linear_araddr),
    .axi_aruser_ap    (linear_aruser_ap),
    .axi_aruser_id    (linear_aruser_id),
    .axi_arlen        (linear_arlen),
    .axi_arready      (master_arready[MASTER_LINEAR]),
    .axi_arvalid      (linear_arvalid),
    .axi_rdata        (master_rdata[MASTER_LINEAR*256 +: 256]),
    .axi_rvalid       (master_rvalid[MASTER_LINEAR]),
    .busy             (linear_busy),
    .done             (linear_done),
    .error            (linear_error),
    .error_code       (linear_error_code),
    .current_row      (),
    .debug_state      ()
);

wire [CTRL_ADDR_WIDTH-1:0] rope_awaddr;
wire rope_awuser_ap;
wire [3:0] rope_awuser_id;
wire [3:0] rope_awlen;
wire rope_awvalid;
wire [255:0] rope_wdata;
wire [31:0] rope_wstrb;
wire [CTRL_ADDR_WIDTH-1:0] rope_araddr;
wire rope_aruser_ap;
wire [3:0] rope_aruser_id;
wire [3:0] rope_arlen;
wire rope_arvalid;
wire rope_busy;
wire rope_done;
wire rope_error;
wire [7:0] rope_error_code;

g2_rope_stage_ctrl #(
    .CTRL_ADDR_WIDTH (CTRL_ADDR_WIDTH)
) u_g2_rope_stage_ctrl (
    .clk              (clk),
    .rst_n            (rst_n),
    .ddr_init_done    (ddr_init_done),
    .start            (engine_start[5]),
    .cfg_q_source_addr(`G2_Q_Q28_CTRL_ADDR),
    .cfg_k_source_addr(`G2_K_Q28_CTRL_ADDR),
    .cfg_trig_addr    (`G2_ROPE_TRIG_CTRL_ADDR),
    .cfg_q_result_addr(`G2_Q_ROPE_CTRL_ADDR),
    .cfg_k_result_addr(`G2_K_ROPE_CTRL_ADDR),
    .axi_awaddr       (rope_awaddr),
    .axi_awuser_ap    (rope_awuser_ap),
    .axi_awuser_id    (rope_awuser_id),
    .axi_awlen        (rope_awlen),
    .axi_awready      (master_awready[MASTER_ROPE]),
    .axi_awvalid      (rope_awvalid),
    .axi_wdata        (rope_wdata),
    .axi_wstrb        (rope_wstrb),
    .axi_wready       (master_wready[MASTER_ROPE]),
    .axi_araddr       (rope_araddr),
    .axi_aruser_ap    (rope_aruser_ap),
    .axi_aruser_id    (rope_aruser_id),
    .axi_arlen        (rope_arlen),
    .axi_arready      (master_arready[MASTER_ROPE]),
    .axi_arvalid      (rope_arvalid),
    .axi_rdata        (master_rdata[MASTER_ROPE*256 +: 256]),
    .axi_rvalid       (master_rvalid[MASTER_ROPE]),
    .busy             (rope_busy),
    .done             (rope_done),
    .error            (rope_error),
    .error_code       (rope_error_code),
    .debug_state      (),
    .debug_is_k       (),
    .debug_head       (),
    .debug_group      ()
);

wire [CTRL_ADDR_WIDTH-1:0] kv_awaddr;
wire kv_awuser_ap;
wire [3:0] kv_awuser_id;
wire [3:0] kv_awlen;
wire kv_awvalid;
wire [255:0] kv_wdata;
wire [31:0] kv_wstrb;
wire [CTRL_ADDR_WIDTH-1:0] kv_araddr;
wire kv_aruser_ap;
wire [3:0] kv_aruser_id;
wire [3:0] kv_arlen;
wire kv_arvalid;
wire kv_busy;
wire kv_done;
wire kv_error;
wire [7:0] kv_error_code;

g2_kv_write_stage_ctrl #(
    .CTRL_ADDR_WIDTH (CTRL_ADDR_WIDTH)
) u_g2_kv_write_stage_ctrl (
    .clk              (clk),
    .rst_n            (rst_n),
    .ddr_init_done    (ddr_init_done),
    .start            (engine_start[6]),
    .cfg_layer        (cfg_layer),
    .cfg_position     (cfg_query_position),
    .cfg_k_source_addr(`G2_K_ROPE_CTRL_ADDR),
    .cfg_v_source_addr(`G2_V_Q28_CTRL_ADDR),
    .axi_awaddr       (kv_awaddr),
    .axi_awuser_ap    (kv_awuser_ap),
    .axi_awuser_id    (kv_awuser_id),
    .axi_awlen        (kv_awlen),
    .axi_awready      (master_awready[MASTER_KV]),
    .axi_awvalid      (kv_awvalid),
    .axi_wdata        (kv_wdata),
    .axi_wstrb        (kv_wstrb),
    .axi_wready       (master_wready[MASTER_KV]),
    .axi_araddr       (kv_araddr),
    .axi_aruser_ap    (kv_aruser_ap),
    .axi_aruser_id    (kv_aruser_id),
    .axi_arlen        (kv_arlen),
    .axi_arready      (master_arready[MASTER_KV]),
    .axi_arvalid      (kv_arvalid),
    .axi_rdata        (master_rdata[MASTER_KV*256 +: 256]),
    .axi_rvalid       (master_rvalid[MASTER_KV]),
    .busy             (kv_busy),
    .done             (kv_done),
    .error            (kv_error),
    .error_code       (kv_error_code),
    .debug_state      (),
    .debug_is_v       (),
    .debug_beat_index (),
    .debug_slot_addr  ()
);

wire [CTRL_ADDR_WIDTH-1:0] score_awaddr;
wire score_awuser_ap;
wire [3:0] score_awuser_id;
wire [3:0] score_awlen;
wire score_awvalid;
wire [255:0] score_wdata;
wire [31:0] score_wstrb;
wire [CTRL_ADDR_WIDTH-1:0] score_araddr;
wire score_aruser_ap;
wire [3:0] score_aruser_id;
wire [3:0] score_arlen;
wire score_arvalid;
wire score_busy;
wire score_done;
wire score_error;
wire [7:0] score_error_code;

g2_attention_score_stage_ctrl #(
    .CTRL_ADDR_WIDTH (CTRL_ADDR_WIDTH)
) u_g2_attention_score_stage_ctrl (
    .clk              (clk),
    .rst_n            (rst_n),
    .ddr_init_done    (ddr_init_done),
    .start            (engine_start[7]),
    .cfg_layer        (cfg_layer),
    .cfg_query_position(cfg_query_position),
    .cfg_window_start (cfg_window_start),
    .cfg_count        (cfg_count),
    .cfg_q_addr       (`G2_Q_ROPE_CTRL_ADDR),
    .cfg_result_addr  (`G2_SCORES_CTRL_ADDR),
    .axi_awaddr       (score_awaddr),
    .axi_awuser_ap    (score_awuser_ap),
    .axi_awuser_id    (score_awuser_id),
    .axi_awlen        (score_awlen),
    .axi_awready      (master_awready[MASTER_SCORE]),
    .axi_awvalid      (score_awvalid),
    .axi_wdata        (score_wdata),
    .axi_wstrb        (score_wstrb),
    .axi_wready       (master_wready[MASTER_SCORE]),
    .axi_araddr       (score_araddr),
    .axi_aruser_ap    (score_aruser_ap),
    .axi_aruser_id    (score_aruser_id),
    .axi_arlen        (score_arlen),
    .axi_arready      (master_arready[MASTER_SCORE]),
    .axi_arvalid      (score_arvalid),
    .axi_rdata        (master_rdata[MASTER_SCORE*256 +: 256]),
    .axi_rvalid       (master_rvalid[MASTER_SCORE]),
    .busy             (score_busy),
    .done             (score_done),
    .error            (score_error),
    .error_code       (score_error_code),
    .debug_state      (),
    .debug_token      (),
    .debug_head       ()
);

wire [CTRL_ADDR_WIDTH-1:0] softmax_awaddr;
wire softmax_awuser_ap;
wire [3:0] softmax_awuser_id;
wire [3:0] softmax_awlen;
wire softmax_awvalid;
wire [255:0] softmax_wdata;
wire [31:0] softmax_wstrb;
wire [CTRL_ADDR_WIDTH-1:0] softmax_araddr;
wire softmax_aruser_ap;
wire [3:0] softmax_aruser_id;
wire [3:0] softmax_arlen;
wire softmax_arvalid;
wire softmax_busy;
wire softmax_done;
wire softmax_error;
wire [7:0] softmax_error_code;

g2_softmax_stage_ctrl #(
    .CTRL_ADDR_WIDTH (CTRL_ADDR_WIDTH)
) u_g2_softmax_stage_ctrl (
    .clk              (clk),
    .rst_n            (rst_n),
    .ddr_init_done    (ddr_init_done),
    .start            (engine_start[8]),
    .cfg_score_addr   (`G2_SCORES_CTRL_ADDR),
    .cfg_lut_addr     (`G2_SOFTMAX_LUT_CTRL_ADDR),
    .cfg_result_addr  (`G2_PROBABILITIES_CTRL_ADDR),
    .axi_awaddr       (softmax_awaddr),
    .axi_awuser_ap    (softmax_awuser_ap),
    .axi_awuser_id    (softmax_awuser_id),
    .axi_awlen        (softmax_awlen),
    .axi_awready      (master_awready[MASTER_SOFTMAX]),
    .axi_awvalid      (softmax_awvalid),
    .axi_wdata        (softmax_wdata),
    .axi_wstrb        (softmax_wstrb),
    .axi_wready       (master_wready[MASTER_SOFTMAX]),
    .axi_araddr       (softmax_araddr),
    .axi_aruser_ap    (softmax_aruser_ap),
    .axi_aruser_id    (softmax_aruser_id),
    .axi_arlen        (softmax_arlen),
    .axi_arready      (master_arready[MASTER_SOFTMAX]),
    .axi_arvalid      (softmax_arvalid),
    .axi_rdata        (master_rdata[MASTER_SOFTMAX*256 +: 256]),
    .axi_rvalid       (master_rvalid[MASTER_SOFTMAX]),
    .busy             (softmax_busy),
    .done             (softmax_done),
    .error            (softmax_error),
    .error_code       (softmax_error_code),
    .debug_state      (),
    .debug_head       (),
    .debug_token      ()
);

wire [CTRL_ADDR_WIDTH-1:0] attn_out_awaddr;
wire attn_out_awuser_ap;
wire [3:0] attn_out_awuser_id;
wire [3:0] attn_out_awlen;
wire attn_out_awvalid;
wire [255:0] attn_out_wdata;
wire [31:0] attn_out_wstrb;
wire [CTRL_ADDR_WIDTH-1:0] attn_out_araddr;
wire attn_out_aruser_ap;
wire [3:0] attn_out_aruser_id;
wire [3:0] attn_out_arlen;
wire attn_out_arvalid;
wire attn_out_busy;
wire attn_out_done;
wire attn_out_error;
wire [7:0] attn_out_error_code;

g2_attention_output_stage_ctrl #(
    .CTRL_ADDR_WIDTH (CTRL_ADDR_WIDTH)
) u_g2_attention_output_stage_ctrl (
    .clk              (clk),
    .rst_n            (rst_n),
    .ddr_init_done    (ddr_init_done),
    .start            (engine_start[9]),
    .cfg_layer        (cfg_layer),
    .cfg_query_position(cfg_query_position),
    .cfg_window_start (cfg_window_start),
    .cfg_count        (cfg_count),
    .cfg_probability_addr(`G2_PROBABILITIES_CTRL_ADDR),
    .cfg_result_addr  (`G2_ATTN_CONCAT_CTRL_ADDR),
    .axi_awaddr       (attn_out_awaddr),
    .axi_awuser_ap    (attn_out_awuser_ap),
    .axi_awuser_id    (attn_out_awuser_id),
    .axi_awlen        (attn_out_awlen),
    .axi_awready      (master_awready[MASTER_ATTN_OUT]),
    .axi_awvalid      (attn_out_awvalid),
    .axi_wdata        (attn_out_wdata),
    .axi_wstrb        (attn_out_wstrb),
    .axi_wready       (master_wready[MASTER_ATTN_OUT]),
    .axi_araddr       (attn_out_araddr),
    .axi_aruser_ap    (attn_out_aruser_ap),
    .axi_aruser_id    (attn_out_aruser_id),
    .axi_arlen        (attn_out_arlen),
    .axi_arready      (master_arready[MASTER_ATTN_OUT]),
    .axi_arvalid      (attn_out_arvalid),
    .axi_rdata        (master_rdata[MASTER_ATTN_OUT*256 +: 256]),
    .axi_rvalid       (master_rvalid[MASTER_ATTN_OUT]),
    .busy             (attn_out_busy),
    .done             (attn_out_done),
    .error            (attn_out_error),
    .error_code       (attn_out_error_code),
    .debug_state      (),
    .debug_token      (),
    .debug_head       (),
    .debug_dimension  ()
);

wire [CTRL_ADDR_WIDTH-1:0] residual_awaddr;
wire residual_awuser_ap;
wire [3:0] residual_awuser_id;
wire [3:0] residual_awlen;
wire residual_awvalid;
wire [255:0] residual_wdata;
wire [31:0] residual_wstrb;
wire [CTRL_ADDR_WIDTH-1:0] residual_araddr;
wire residual_aruser_ap;
wire [3:0] residual_aruser_id;
wire [3:0] residual_arlen;
wire residual_arvalid;
wire residual_busy;
wire residual_done;
wire residual_error;
wire [7:0] residual_error_code;

g2_stream_residual_ctrl #(
    .CTRL_ADDR_WIDTH (CTRL_ADDR_WIDTH)
) u_g2_stream_residual_ctrl (
    .clk              (clk),
    .rst_n            (rst_n),
    .ddr_init_done    (ddr_init_done),
    .start            (residual_start),
    .cfg_hidden_addr  (residual_hidden_addr),
    .cfg_q28_addr     (residual_q28_addr),
    .cfg_result_addr  (residual_result_addr),
    .axi_awaddr       (residual_awaddr),
    .axi_awuser_ap    (residual_awuser_ap),
    .axi_awuser_id    (residual_awuser_id),
    .axi_awlen        (residual_awlen),
    .axi_awready      (master_awready[MASTER_RESIDUAL]),
    .axi_awvalid      (residual_awvalid),
    .axi_wdata        (residual_wdata),
    .axi_wstrb        (residual_wstrb),
    .axi_wready       (master_wready[MASTER_RESIDUAL]),
    .axi_araddr       (residual_araddr),
    .axi_aruser_ap    (residual_aruser_ap),
    .axi_aruser_id    (residual_aruser_id),
    .axi_arlen        (residual_arlen),
    .axi_arready      (master_arready[MASTER_RESIDUAL]),
    .axi_arvalid      (residual_arvalid),
    .axi_rdata        (master_rdata[MASTER_RESIDUAL*256 +: 256]),
    .axi_rvalid       (master_rvalid[MASTER_RESIDUAL]),
    .busy             (residual_busy),
    .done             (residual_done),
    .error            (residual_error),
    .error_code       (residual_error_code),
    .debug_state      (),
    .debug_beat_index ()
);

wire [CTRL_ADDR_WIDTH-1:0] silu_awaddr;
wire silu_awuser_ap;
wire [3:0] silu_awuser_id;
wire [3:0] silu_awlen;
wire silu_awvalid;
wire [255:0] silu_wdata;
wire [31:0] silu_wstrb;
wire [CTRL_ADDR_WIDTH-1:0] silu_araddr;
wire silu_aruser_ap;
wire [3:0] silu_aruser_id;
wire [3:0] silu_arlen;
wire silu_arvalid;
wire silu_busy;
wire silu_done;
wire silu_error;
wire [7:0] silu_error_code;

g2_stream_silu_ctrl #(
    .CTRL_ADDR_WIDTH (CTRL_ADDR_WIDTH)
) u_g2_stream_silu_ctrl (
    .clk              (clk),
    .rst_n            (rst_n),
    .ddr_init_done    (ddr_init_done),
    .start            (engine_start[17]),
    .cfg_gate_addr    (`G2_GATE_CTRL_ADDR),
    .cfg_pwl_addr     (`G2_SILU_PWL_CTRL_ADDR),
    .cfg_result_addr  (`G2_SILU_GATE_CTRL_ADDR),
    .axi_awaddr       (silu_awaddr),
    .axi_awuser_ap    (silu_awuser_ap),
    .axi_awuser_id    (silu_awuser_id),
    .axi_awlen        (silu_awlen),
    .axi_awready      (master_awready[MASTER_SILU]),
    .axi_awvalid      (silu_awvalid),
    .axi_wdata        (silu_wdata),
    .axi_wstrb        (silu_wstrb),
    .axi_wready       (master_wready[MASTER_SILU]),
    .axi_araddr       (silu_araddr),
    .axi_aruser_ap    (silu_aruser_ap),
    .axi_aruser_id    (silu_aruser_id),
    .axi_arlen        (silu_arlen),
    .axi_arready      (master_arready[MASTER_SILU]),
    .axi_arvalid      (silu_arvalid),
    .axi_rdata        (master_rdata[MASTER_SILU*256 +: 256]),
    .axi_rvalid       (master_rvalid[MASTER_SILU]),
    .busy             (silu_busy),
    .done             (silu_done),
    .error            (silu_error),
    .error_code       (silu_error_code),
    .debug_state      (),
    .debug_beat_index ()
);

wire [CTRL_ADDR_WIDTH-1:0] silu_mul_awaddr;
wire silu_mul_awuser_ap;
wire [3:0] silu_mul_awuser_id;
wire [3:0] silu_mul_awlen;
wire silu_mul_awvalid;
wire [255:0] silu_mul_wdata;
wire [31:0] silu_mul_wstrb;
wire [CTRL_ADDR_WIDTH-1:0] silu_mul_araddr;
wire silu_mul_aruser_ap;
wire [3:0] silu_mul_aruser_id;
wire [3:0] silu_mul_arlen;
wire silu_mul_arvalid;
wire silu_mul_busy;
wire silu_mul_done;
wire silu_mul_error;
wire [7:0] silu_mul_error_code;

g2_stream_silu_up_mul_ctrl #(
    .CTRL_ADDR_WIDTH (CTRL_ADDR_WIDTH)
) u_g2_stream_silu_up_mul_ctrl (
    .clk              (clk),
    .rst_n            (rst_n),
    .ddr_init_done    (ddr_init_done),
    .start            (engine_start[18]),
    .cfg_silu_addr    (`G2_SILU_GATE_CTRL_ADDR),
    .cfg_up_addr      (`G2_UP_CTRL_ADDR),
    .cfg_result_addr  (`G2_SILU_UP_CTRL_ADDR),
    .axi_awaddr       (silu_mul_awaddr),
    .axi_awuser_ap    (silu_mul_awuser_ap),
    .axi_awuser_id    (silu_mul_awuser_id),
    .axi_awlen        (silu_mul_awlen),
    .axi_awready      (master_awready[MASTER_SILU_MUL]),
    .axi_awvalid      (silu_mul_awvalid),
    .axi_wdata        (silu_mul_wdata),
    .axi_wstrb        (silu_mul_wstrb),
    .axi_wready       (master_wready[MASTER_SILU_MUL]),
    .axi_araddr       (silu_mul_araddr),
    .axi_aruser_ap    (silu_mul_aruser_ap),
    .axi_aruser_id    (silu_mul_aruser_id),
    .axi_arlen        (silu_mul_arlen),
    .axi_arready      (master_arready[MASTER_SILU_MUL]),
    .axi_arvalid      (silu_mul_arvalid),
    .axi_rdata        (master_rdata[MASTER_SILU_MUL*256 +: 256]),
    .axi_rvalid       (master_rvalid[MASTER_SILU_MUL]),
    .busy             (silu_mul_busy),
    .done             (silu_mul_done),
    .error            (silu_mul_error),
    .error_code       (silu_mul_error_code),
    .debug_state      (),
    .debug_group_index()
);

// -----------------------------------------------------------------------------
// scheduler 的 22 路 done/error 映射。
// -----------------------------------------------------------------------------
assign engine_done[0]  = rms_done;
assign engine_done[1]  = quant_done;
assign engine_done[2]  = linear_done;
assign engine_done[3]  = linear_done;
assign engine_done[4]  = linear_done;
assign engine_done[5]  = rope_done;
assign engine_done[6]  = kv_done;
assign engine_done[7]  = score_done;
assign engine_done[8]  = softmax_done;
assign engine_done[9]  = attn_out_done;
assign engine_done[10] = quant_done;
assign engine_done[11] = linear_done;
assign engine_done[12] = residual_done;
assign engine_done[13] = rms_done;
assign engine_done[14] = quant_done;
assign engine_done[15] = linear_done;
assign engine_done[16] = linear_done;
assign engine_done[17] = silu_done;
assign engine_done[18] = silu_mul_done;
assign engine_done[19] = quant_done;
assign engine_done[20] = linear_done;
assign engine_done[21] = residual_done;

assign engine_error[0]  = rms_error;
assign engine_error[1]  = quant_error;
assign engine_error[2]  = linear_error;
assign engine_error[3]  = linear_error;
assign engine_error[4]  = linear_error;
assign engine_error[5]  = rope_error;
assign engine_error[6]  = kv_error;
assign engine_error[7]  = score_error;
assign engine_error[8]  = softmax_error;
assign engine_error[9]  = attn_out_error;
assign engine_error[10] = quant_error;
assign engine_error[11] = linear_error;
assign engine_error[12] = residual_error;
assign engine_error[13] = rms_error;
assign engine_error[14] = quant_error;
assign engine_error[15] = linear_error;
assign engine_error[16] = linear_error;
assign engine_error[17] = silu_error;
assign engine_error[18] = silu_mul_error;
assign engine_error[19] = quant_error;
assign engine_error[20] = linear_error;
assign engine_error[21] = residual_error;

// -----------------------------------------------------------------------------
// 当前 scheduler 阶段到 11 路 AXI master 的唯一选择。
// -----------------------------------------------------------------------------
reg [3:0] selected_master;
always @(*) begin
    case (current_stage)
        `G2_STAGE_INPUT_RMS,
        `G2_STAGE_POST_RMS:
            selected_master = MASTER_RMS;

        `G2_STAGE_QKV_QUANT,
        `G2_STAGE_OPROJ_QUANT,
        `G2_STAGE_GATE_UP_QUANT,
        `G2_STAGE_DOWN_QUANT:
            selected_master = MASTER_QUANT;

        `G2_STAGE_Q_LINEAR,
        `G2_STAGE_K_LINEAR,
        `G2_STAGE_V_LINEAR,
        `G2_STAGE_OPROJ_LINEAR,
        `G2_STAGE_GATE_LINEAR,
        `G2_STAGE_UP_LINEAR,
        `G2_STAGE_DOWN_LINEAR:
            selected_master = MASTER_LINEAR;

        `G2_STAGE_ROPE:
            selected_master = MASTER_ROPE;
        `G2_STAGE_KV_WRITE:
            selected_master = MASTER_KV;
        `G2_STAGE_ATTENTION_SCORE:
            selected_master = MASTER_SCORE;
        `G2_STAGE_SOFTMAX:
            selected_master = MASTER_SOFTMAX;
        `G2_STAGE_ATTENTION_OUTPUT:
            selected_master = MASTER_ATTN_OUT;
        `G2_STAGE_RESIDUAL1,
        `G2_STAGE_RESIDUAL2:
            selected_master = MASTER_RESIDUAL;
        `G2_STAGE_SILU:
            selected_master = MASTER_SILU;
        `G2_STAGE_SILU_UP_MUL:
            selected_master = MASTER_SILU_MUL;
        default:
            selected_master = MASTER_RMS;
    endcase
end
assign debug_axi_master = selected_master;

wire [NUM_MASTERS*CTRL_ADDR_WIDTH-1:0] master_awaddr;
wire [NUM_MASTERS-1:0] master_awuser_ap;
wire [NUM_MASTERS*4-1:0] master_awuser_id;
wire [NUM_MASTERS*4-1:0] master_awlen;
wire [NUM_MASTERS-1:0] master_awvalid;
wire [NUM_MASTERS-1:0] master_awready;
wire [NUM_MASTERS*256-1:0] master_wdata;
wire [NUM_MASTERS*32-1:0] master_wstrb;
wire [NUM_MASTERS-1:0] master_wready;
wire [NUM_MASTERS*CTRL_ADDR_WIDTH-1:0] master_araddr;
wire [NUM_MASTERS-1:0] master_aruser_ap;
wire [NUM_MASTERS*4-1:0] master_aruser_id;
wire [NUM_MASTERS*4-1:0] master_arlen;
wire [NUM_MASTERS-1:0] master_arvalid;
wire [NUM_MASTERS-1:0] master_arready;
wire [NUM_MASTERS*256-1:0] master_rdata;
wire [NUM_MASTERS-1:0] master_rvalid;

assign master_awaddr = {
    silu_mul_awaddr, silu_awaddr, residual_awaddr, attn_out_awaddr,
    softmax_awaddr, score_awaddr, kv_awaddr, rope_awaddr,
    linear_awaddr, quant_awaddr, rms_awaddr
};
assign master_awuser_ap = {
    silu_mul_awuser_ap, silu_awuser_ap, residual_awuser_ap, attn_out_awuser_ap,
    softmax_awuser_ap, score_awuser_ap, kv_awuser_ap, rope_awuser_ap,
    linear_awuser_ap, quant_awuser_ap, rms_awuser_ap
};
assign master_awuser_id = {
    silu_mul_awuser_id, silu_awuser_id, residual_awuser_id, attn_out_awuser_id,
    softmax_awuser_id, score_awuser_id, kv_awuser_id, rope_awuser_id,
    linear_awuser_id, quant_awuser_id, rms_awuser_id
};
assign master_awlen = {
    silu_mul_awlen, silu_awlen, residual_awlen, attn_out_awlen,
    softmax_awlen, score_awlen, kv_awlen, rope_awlen,
    linear_awlen, quant_awlen, rms_awlen
};
assign master_awvalid = {
    silu_mul_awvalid, silu_awvalid, residual_awvalid, attn_out_awvalid,
    softmax_awvalid, score_awvalid, kv_awvalid, rope_awvalid,
    linear_awvalid, quant_awvalid, rms_awvalid
};
assign master_wdata = {
    silu_mul_wdata, silu_wdata, residual_wdata, attn_out_wdata,
    softmax_wdata, score_wdata, kv_wdata, rope_wdata,
    linear_wdata, quant_wdata, rms_wdata
};
assign master_wstrb = {
    silu_mul_wstrb, silu_wstrb, residual_wstrb, attn_out_wstrb,
    softmax_wstrb, score_wstrb, kv_wstrb, rope_wstrb,
    linear_wstrb, quant_wstrb, rms_wstrb
};
assign master_araddr = {
    silu_mul_araddr, silu_araddr, residual_araddr, attn_out_araddr,
    softmax_araddr, score_araddr, kv_araddr, rope_araddr,
    linear_araddr, quant_araddr, rms_araddr
};
assign master_aruser_ap = {
    silu_mul_aruser_ap, silu_aruser_ap, residual_aruser_ap, attn_out_aruser_ap,
    softmax_aruser_ap, score_aruser_ap, kv_aruser_ap, rope_aruser_ap,
    linear_aruser_ap, quant_aruser_ap, rms_aruser_ap
};
assign master_aruser_id = {
    silu_mul_aruser_id, silu_aruser_id, residual_aruser_id, attn_out_aruser_id,
    softmax_aruser_id, score_aruser_id, kv_aruser_id, rope_aruser_id,
    linear_aruser_id, quant_aruser_id, rms_aruser_id
};
assign master_arlen = {
    silu_mul_arlen, silu_arlen, residual_arlen, attn_out_arlen,
    softmax_arlen, score_arlen, kv_arlen, rope_arlen,
    linear_arlen, quant_arlen, rms_arlen
};
assign master_arvalid = {
    silu_mul_arvalid, silu_arvalid, residual_arvalid, attn_out_arvalid,
    softmax_arvalid, score_arvalid, kv_arvalid, rope_arvalid,
    linear_arvalid, quant_arvalid, rms_arvalid
};

wire mux_select_error;
g2_axi_stage_mux #(
    .NUM_MASTERS (NUM_MASTERS),
    .ADDR_WIDTH  (CTRL_ADDR_WIDTH)
) u_g2_axi_stage_mux (
    .clk           (clk),
    .rst_n         (rst_n),
    .select_master (selected_master),
    .m_awaddr      (master_awaddr),
    .m_awuser_ap   (master_awuser_ap),
    .m_awuser_id   (master_awuser_id),
    .m_awlen       (master_awlen),
    .m_awvalid     (master_awvalid),
    .m_awready     (master_awready),
    .m_wdata       (master_wdata),
    .m_wstrb       (master_wstrb),
    .m_wready      (master_wready),
    .m_araddr      (master_araddr),
    .m_aruser_ap   (master_aruser_ap),
    .m_aruser_id   (master_aruser_id),
    .m_arlen       (master_arlen),
    .m_arvalid     (master_arvalid),
    .m_arready     (master_arready),
    .m_rdata       (master_rdata),
    .m_rvalid      (master_rvalid),
    .axi_awaddr    (axi_awaddr),
    .axi_awuser_ap (axi_awuser_ap),
    .axi_awuser_id (axi_awuser_id),
    .axi_awlen     (axi_awlen),
    .axi_awready   (axi_awready),
    .axi_awvalid   (axi_awvalid),
    .axi_wdata     (axi_wdata),
    .axi_wstrb     (axi_wstrb),
    .axi_wready    (axi_wready),
    .axi_araddr    (axi_araddr),
    .axi_aruser_ap (axi_aruser_ap),
    .axi_aruser_id (axi_aruser_id),
    .axi_arlen     (axi_arlen),
    .axi_arready   (axi_arready),
    .axi_arvalid   (axi_arvalid),
    .axi_rdata     (axi_rdata),
    .axi_rvalid    (axi_rvalid),
    .select_error  (mux_select_error)
);

assign busy = scheduler_busy;
assign done = scheduler_done;
assign error = local_error || scheduler_error || mux_select_error;
assign error_code = local_error ? local_error_code :
                    scheduler_error ? scheduler_error_code :
                    mux_select_error ? ERR_AXI_SELECT : 8'd0;

// 独立综合时保留各子阶段状态用于结构检查；完整顶层只导出聚合状态。
wire _unused_child_status = &{
    1'b0,
    rms_busy, rms_error_code,
    quant_busy, quant_error_code,
    linear_busy, linear_error_code,
    rope_busy, rope_error_code,
    kv_busy, kv_error_code,
    score_busy, score_error_code,
    softmax_busy, softmax_error_code,
    attn_out_busy, attn_out_error_code,
    residual_busy, residual_error_code,
    silu_busy, silu_error_code,
    silu_mul_busy, silu_mul_error_code
};

endmodule
