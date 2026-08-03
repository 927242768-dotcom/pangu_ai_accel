`timescale 1ns/1ps

`include "transformer_block_contract.vh"

// G2 七个矩阵共享的 Linear 阶段配置器。
// 仅选择固定维度/地址，计算与 DDR3 流程完全由 runtime_linear_ctrl 和
// shared_linear_engine 执行。
module g2_linear_stage_ctrl #(
    parameter integer CTRL_ADDR_WIDTH = 28
)(
    input  wire                         clk,
    input  wire                         rst_n,
    input  wire                         ddr_init_done,
    input  wire                         start,
    input  wire [2:0]                   cfg_mode,

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
    output wire [12:0]                  current_row,
    output wire [4:0]                   debug_state
);

localparam [2:0] MODE_Q    = 3'd0;
localparam [2:0] MODE_K    = 3'd1;
localparam [2:0] MODE_V    = 3'd2;
localparam [2:0] MODE_O    = 3'd3;
localparam [2:0] MODE_GATE = 3'd4;
localparam [2:0] MODE_UP   = 3'd5;
localparam [2:0] MODE_DOWN = 3'd6;

reg [12:0] linear_m_rows;
reg [8:0] linear_k_blocks;
reg [6:0] linear_groups;
reg [7:0] linear_act_beats;
reg [6:0] linear_weight_beats_per_row;
reg [3:0] linear_scale_beats_per_row;
reg linear_has_bias;
reg [CTRL_ADDR_WIDTH-1:0] linear_weight_addr;
reg [CTRL_ADDR_WIDTH-1:0] linear_scale_addr;
reg [CTRL_ADDR_WIDTH-1:0] linear_bias_addr;
reg [CTRL_ADDR_WIDTH-1:0] linear_result_addr;

always @(*) begin
    linear_m_rows              = 13'd896;
    linear_k_blocks            = 9'd56;
    linear_groups              = 7'd14;
    linear_act_beats           = 8'd28;
    linear_weight_beats_per_row= 7'd14;
    linear_scale_beats_per_row = 4'd2;
    linear_has_bias            = 1'b0;
    linear_weight_addr         = `G2_Q_WEIGHT_CTRL_ADDR;
    linear_scale_addr          = `G2_Q_SCALE_CTRL_ADDR;
    linear_bias_addr           = `G2_Q_BIAS_CTRL_ADDR;
    linear_result_addr         = `G2_Q_Q28_CTRL_ADDR;

    case (cfg_mode)
        MODE_Q: begin
            linear_m_rows      = 13'd896;
            linear_has_bias    = 1'b1;
            linear_weight_addr = `G2_Q_WEIGHT_CTRL_ADDR;
            linear_scale_addr  = `G2_Q_SCALE_CTRL_ADDR;
            linear_bias_addr   = `G2_Q_BIAS_CTRL_ADDR;
            linear_result_addr = `G2_Q_Q28_CTRL_ADDR;
        end
        MODE_K: begin
            linear_m_rows      = 13'd128;
            linear_has_bias    = 1'b1;
            linear_weight_addr = `G2_K_WEIGHT_CTRL_ADDR;
            linear_scale_addr  = `G2_K_SCALE_CTRL_ADDR;
            linear_bias_addr   = `G2_K_BIAS_CTRL_ADDR;
            linear_result_addr = `G2_K_Q28_CTRL_ADDR;
        end
        MODE_V: begin
            linear_m_rows      = 13'd128;
            linear_has_bias    = 1'b1;
            linear_weight_addr = `G2_V_WEIGHT_CTRL_ADDR;
            linear_scale_addr  = `G2_V_SCALE_CTRL_ADDR;
            linear_bias_addr   = `G2_V_BIAS_CTRL_ADDR;
            linear_result_addr = `G2_V_Q28_CTRL_ADDR;
        end
        MODE_O: begin
            linear_m_rows      = 13'd896;
            linear_has_bias    = 1'b0;
            linear_weight_addr = `G2_OPROJ_WEIGHT_CTRL_ADDR;
            linear_scale_addr  = `G2_OPROJ_SCALE_CTRL_ADDR;
            linear_bias_addr   = {CTRL_ADDR_WIDTH{1'b0}};
            linear_result_addr = `G2_OPROJ_CTRL_ADDR;
        end
        MODE_GATE: begin
            linear_m_rows      = 13'd4864;
            linear_has_bias    = 1'b0;
            linear_weight_addr = `G2_GATE_WEIGHT_CTRL_ADDR;
            linear_scale_addr  = `G2_GATE_SCALE_CTRL_ADDR;
            linear_bias_addr   = {CTRL_ADDR_WIDTH{1'b0}};
            linear_result_addr = `G2_GATE_CTRL_ADDR;
        end
        MODE_UP: begin
            linear_m_rows      = 13'd4864;
            linear_has_bias    = 1'b0;
            linear_weight_addr = `G2_UP_WEIGHT_CTRL_ADDR;
            linear_scale_addr  = `G2_UP_SCALE_CTRL_ADDR;
            linear_bias_addr   = {CTRL_ADDR_WIDTH{1'b0}};
            linear_result_addr = `G2_UP_CTRL_ADDR;
        end
        default: begin
            linear_m_rows               = 13'd896;
            linear_k_blocks             = 9'd304;
            linear_groups               = 7'd76;
            linear_act_beats            = 8'd152;
            linear_weight_beats_per_row = 7'd76;
            linear_scale_beats_per_row  = 4'd10;
            linear_has_bias             = 1'b0;
            linear_weight_addr          = `G2_DOWN_WEIGHT_CTRL_ADDR;
            linear_scale_addr           = `G2_DOWN_SCALE_CTRL_ADDR;
            linear_bias_addr            = {CTRL_ADDR_WIDTH{1'b0}};
            linear_result_addr          = `G2_DOWN_CTRL_ADDR;
        end
    endcase
end

runtime_linear_ctrl #(
    .CTRL_ADDR_WIDTH (CTRL_ADDR_WIDTH)
) u_runtime_linear_ctrl (
    .clk                       (clk),
    .rst_n                     (rst_n),
    .ddr_init_done             (ddr_init_done),
    .start                     (start),
    .cfg_m_rows                (linear_m_rows),
    .cfg_k_blocks              (linear_k_blocks),
    .cfg_groups                (linear_groups),
    .cfg_act_beats             (linear_act_beats),
    .cfg_weight_beats_per_row  (linear_weight_beats_per_row),
    .cfg_scale_beats_per_row   (linear_scale_beats_per_row),
    .cfg_has_bias              (linear_has_bias),
    .cfg_act_addr              (`G2_LINEAR_ACT_INT8_CTRL_ADDR),
    .cfg_weight_addr           (linear_weight_addr),
    .cfg_scale_addr            (linear_scale_addr),
    .cfg_bias_addr             (linear_bias_addr),
    .cfg_result_addr           (linear_result_addr),
    .axi_awaddr                (axi_awaddr),
    .axi_awuser_ap             (axi_awuser_ap),
    .axi_awuser_id             (axi_awuser_id),
    .axi_awlen                 (axi_awlen),
    .axi_awready               (axi_awready),
    .axi_awvalid               (axi_awvalid),
    .axi_wdata                 (axi_wdata),
    .axi_wstrb                 (axi_wstrb),
    .axi_wready                (axi_wready),
    .axi_araddr                (axi_araddr),
    .axi_aruser_ap             (axi_aruser_ap),
    .axi_aruser_id             (axi_aruser_id),
    .axi_arlen                 (axi_arlen),
    .axi_arready               (axi_arready),
    .axi_arvalid               (axi_arvalid),
    .axi_rdata                 (axi_rdata),
    .axi_rvalid                (axi_rvalid),
    .busy                      (busy),
    .done                      (done),
    .error                     (error),
    .error_code                (error_code),
    .current_row               (current_row),
    .debug_state               (debug_state)
);

wire _unused_mode_down = (cfg_mode == MODE_DOWN);

endmodule
