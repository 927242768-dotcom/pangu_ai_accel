`timescale 1ns/1ps

`include "transformer_block_contract.vh"

// G2 四类运行时量化阶段共用的顺序控制器。
// MODE_QKV: q/k/v 三次；MODE_OPROJ: 一次；MODE_GATE_UP: gate/up 两次；
// MODE_DOWN: 一次。Q6.10 输入在统一 adapter 内精确左移 18 位后复用 Q28
// activation quantizer，因此完整 Block 只实例化一套量化硬件。
module g2_quant_sequence_unified_ctrl #(
    parameter integer CTRL_ADDR_WIDTH = 28
)(
    input  wire                         clk,
    input  wire                         rst_n,
    input  wire                         ddr_init_done,
    input  wire                         start,
    input  wire [1:0]                   cfg_mode,

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

    output reg                          busy,
    output reg                          done,
    output reg                          error,
    output reg  [7:0]                   error_code,
    output wire [3:0]                   debug_inner_state,
    output reg  [1:0]                   debug_mode,
    output reg  [1:0]                   debug_operation
);

localparam [1:0] MODE_QKV     = 2'd0;
localparam [1:0] MODE_OPROJ   = 2'd1;
localparam [1:0] MODE_GATE_UP = 2'd2;
localparam [1:0] MODE_DOWN    = 2'd3;

localparam [1:0] ST_IDLE   = 2'd0;
localparam [1:0] ST_LAUNCH = 2'd1;
localparam [1:0] ST_WAIT   = 2'd2;
localparam [1:0] ST_ERROR  = 2'd3;

localparam [7:0] ERR_INNER_BASE = 8'h20;
localparam [7:0] ERR_INTERNAL   = 8'hff;

reg [1:0] state;
reg [1:0] mode_reg;
reg [1:0] operation_index;
reg inner_start;
wire inner_busy;
wire inner_done;
wire inner_error;
wire [7:0] inner_error_code;

reg [12:0] quant_vector_length;
reg [12:0] quant_rows;
reg [6:0] quant_groups;
reg [CTRL_ADDR_WIDTH-1:0] quant_source_addr;
reg [CTRL_ADDR_WIDTH-1:0] quant_activation_addr;
reg [CTRL_ADDR_WIDTH-1:0] quant_raw_scale_addr;
reg [CTRL_ADDR_WIDTH-1:0] quant_combined_scale_addr;

wire quant_source_q28 =
    (mode_reg == MODE_OPROJ) || (mode_reg == MODE_DOWN);
wire operation_is_last =
    ((mode_reg == MODE_QKV) && (operation_index == 2'd2)) ||
    ((mode_reg == MODE_GATE_UP) && (operation_index == 2'd1)) ||
    (mode_reg == MODE_OPROJ) || (mode_reg == MODE_DOWN);

always @(*) begin
    quant_vector_length       = 13'd896;
    quant_rows                = 13'd896;
    quant_groups              = 7'd14;
    quant_source_addr         = `G2_INPUT_NORM_CTRL_ADDR;
    quant_activation_addr     = `G2_LINEAR_ACT_INT8_CTRL_ADDR;
    quant_raw_scale_addr      = `G2_Q_RAW_SCALE_CTRL_ADDR;
    quant_combined_scale_addr = `G2_Q_SCALE_CTRL_ADDR;

    case (mode_reg)
        MODE_QKV: begin
            quant_source_addr = `G2_INPUT_NORM_CTRL_ADDR;
            case (operation_index)
                2'd0: begin
                    quant_rows                = 13'd896;
                    quant_raw_scale_addr      = `G2_Q_RAW_SCALE_CTRL_ADDR;
                    quant_combined_scale_addr = `G2_Q_SCALE_CTRL_ADDR;
                end
                2'd1: begin
                    quant_rows                = 13'd128;
                    quant_raw_scale_addr      = `G2_K_RAW_SCALE_CTRL_ADDR;
                    quant_combined_scale_addr = `G2_K_SCALE_CTRL_ADDR;
                end
                default: begin
                    quant_rows                = 13'd128;
                    quant_raw_scale_addr      = `G2_V_RAW_SCALE_CTRL_ADDR;
                    quant_combined_scale_addr = `G2_V_SCALE_CTRL_ADDR;
                end
            endcase
        end

        MODE_OPROJ: begin
            quant_vector_length       = 13'd896;
            quant_rows                = 13'd896;
            quant_groups              = 7'd14;
            quant_source_addr         = `G2_ATTN_CONCAT_CTRL_ADDR;
            quant_raw_scale_addr      = `G2_OPROJ_RAW_SCALE_CTRL_ADDR;
            quant_combined_scale_addr = `G2_OPROJ_SCALE_CTRL_ADDR;
        end

        MODE_GATE_UP: begin
            quant_vector_length = 13'd896;
            quant_rows          = 13'd4864;
            quant_groups        = 7'd14;
            quant_source_addr   = `G2_POST_NORM_CTRL_ADDR;
            if (operation_index == 2'd0) begin
                quant_raw_scale_addr      = `G2_GATE_RAW_SCALE_CTRL_ADDR;
                quant_combined_scale_addr = `G2_GATE_SCALE_CTRL_ADDR;
            end else begin
                quant_raw_scale_addr      = `G2_UP_RAW_SCALE_CTRL_ADDR;
                quant_combined_scale_addr = `G2_UP_SCALE_CTRL_ADDR;
            end
        end

        default: begin
            quant_vector_length       = 13'd4864;
            quant_rows                = 13'd896;
            quant_groups              = 7'd76;
            quant_source_addr         = `G2_SILU_UP_CTRL_ADDR;
            quant_raw_scale_addr      = `G2_DOWN_RAW_SCALE_CTRL_ADDR;
            quant_combined_scale_addr = `G2_DOWN_SCALE_CTRL_ADDR;
        end
    endcase
end

runtime_quantizer_unified_ctrl #(
    .CTRL_ADDR_WIDTH (CTRL_ADDR_WIDTH)
) u_runtime_quantizer_unified_ctrl (
    .clk                    (clk),
    .rst_n                  (rst_n),
    .ddr_init_done          (ddr_init_done),
    .start                  (inner_start),
    .cfg_source_q28         (quant_source_q28),
    .cfg_vector_length      (quant_vector_length),
    .cfg_rows               (quant_rows),
    .cfg_groups             (quant_groups),
    .cfg_source_addr        (quant_source_addr),
    .cfg_activation_addr    (quant_activation_addr),
    .cfg_raw_scale_addr     (quant_raw_scale_addr),
    .cfg_combined_scale_addr(quant_combined_scale_addr),
    .axi_awaddr             (axi_awaddr),
    .axi_awuser_ap          (axi_awuser_ap),
    .axi_awuser_id          (axi_awuser_id),
    .axi_awlen              (axi_awlen),
    .axi_awready            (axi_awready),
    .axi_awvalid            (axi_awvalid),
    .axi_wdata              (axi_wdata),
    .axi_wstrb              (axi_wstrb),
    .axi_wready             (axi_wready),
    .axi_araddr             (axi_araddr),
    .axi_aruser_ap          (axi_aruser_ap),
    .axi_aruser_id          (axi_aruser_id),
    .axi_arlen              (axi_arlen),
    .axi_arready            (axi_arready),
    .axi_arvalid            (axi_arvalid),
    .axi_rdata              (axi_rdata),
    .axi_rvalid             (axi_rvalid),
    .busy                   (inner_busy),
    .done                   (inner_done),
    .error                  (inner_error),
    .error_code             (inner_error_code),
    .saturated_count        (),
    .all_zero               (),
    .max_abs_q10            (),
    .max_mantissa_binary32  (),
    .max_exponent_binary32  (),
    .max_abs_binary32_bits  (),
    .debug_state            (debug_inner_state)
);

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        state           <= ST_IDLE;
        mode_reg        <= MODE_QKV;
        operation_index <= 2'd0;
        inner_start     <= 1'b0;
        busy            <= 1'b0;
        done            <= 1'b0;
        error           <= 1'b0;
        error_code      <= 8'd0;
        debug_mode      <= MODE_QKV;
        debug_operation <= 2'd0;
    end else begin
        inner_start <= 1'b0;
        done        <= 1'b0;

        case (state)
            ST_IDLE: begin
                busy <= 1'b0;
                if (start && !error) begin
                    mode_reg        <= cfg_mode;
                    operation_index <= 2'd0;
                    debug_mode      <= cfg_mode;
                    debug_operation <= 2'd0;
                    busy            <= 1'b1;
                    error_code      <= 8'd0;
                    state           <= ST_LAUNCH;
                end
            end

            ST_LAUNCH: begin
                if (!inner_busy) begin
                    inner_start <= 1'b1;
                    state       <= ST_WAIT;
                end
            end

            ST_WAIT: begin
                if (inner_error) begin
                    error      <= 1'b1;
                    error_code <= ERR_INNER_BASE + inner_error_code;
                    busy       <= 1'b0;
                    state      <= ST_ERROR;
                end else if (inner_done) begin
                    if (operation_is_last) begin
                        busy  <= 1'b0;
                        done  <= 1'b1;
                        state <= ST_IDLE;
                    end else begin
                        operation_index <= operation_index + 1'b1;
                        debug_operation <= operation_index + 1'b1;
                        state           <= ST_LAUNCH;
                    end
                end
            end

            ST_ERROR: begin
                busy <= 1'b0;
            end

            default: begin
                error      <= 1'b1;
                error_code <= ERR_INTERNAL;
                busy       <= 1'b0;
                state      <= ST_ERROR;
            end
        endcase
    end
end

endmodule
