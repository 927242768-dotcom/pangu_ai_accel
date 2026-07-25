`timescale 1ns/1ps

// G2 单矩阵运行时量化阶段：
//   1. DDR3 source -> Q10/Q28 activation quantizer -> packed INT8 scratch；
//   2. raw FP16 weight scale -> padded UQ4.28 combined scale。
//
// SOURCE_Q28=0 用于 Q/K/V、gate/up；SOURCE_Q28=1 用于 O_proj、down_proj。
// QKV 和 gate/up 共享同一激活但拥有不同 weight scale，顶层可按矩阵连续调用本模块；
// 首版允许重复生成相同 activation，以优先保证闭环正确性。
module runtime_quantizer_ctrl #(
    parameter integer SOURCE_Q28 = 0,
    parameter integer CTRL_ADDR_WIDTH = 28
)(
    input  wire                         clk,
    input  wire                         rst_n,
    input  wire                         ddr_init_done,
    input  wire                         start,
    input  wire [12:0]                  cfg_vector_length,
    input  wire [12:0]                  cfg_rows,
    input  wire [6:0]                   cfg_groups,
    input  wire [CTRL_ADDR_WIDTH-1:0]   cfg_source_addr,
    input  wire [CTRL_ADDR_WIDTH-1:0]   cfg_activation_addr,
    input  wire [CTRL_ADDR_WIDTH-1:0]   cfg_raw_scale_addr,
    input  wire [CTRL_ADDR_WIDTH-1:0]   cfg_combined_scale_addr,

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
    output wire [31:0]                 saturated_count,
    output wire                        all_zero,
    output wire [15:0]                 max_abs_q10,
    output wire [23:0]                 max_mantissa_binary32,
    output wire signed [9:0]           max_exponent_binary32,
    output wire [31:0]                 max_abs_binary32_bits,
    output wire [3:0]                  debug_state
);

localparam [3:0] ST_IDLE        = 4'd0;
localparam [3:0] ST_START_ACT   = 4'd1;
localparam [3:0] ST_WAIT_ACT    = 4'd2;
localparam [3:0] ST_START_SCALE = 4'd3;
localparam [3:0] ST_WAIT_SCALE  = 4'd4;
localparam [3:0] ST_FINISH      = 4'd5;
localparam [3:0] ST_ERROR       = 4'd15;

localparam [7:0] ERR_ACTIVATION = 8'h40;
localparam [7:0] ERR_SCALE      = 8'h80;
localparam [7:0] ERR_INTERNAL   = 8'hff;

reg [3:0] state;
reg act_start;
reg scale_start;

wire [CTRL_ADDR_WIDTH-1:0] act_awaddr;
wire act_awuser_ap;
wire [3:0] act_awuser_id;
wire [3:0] act_awlen;
wire act_awvalid;
wire [255:0] act_wdata;
wire [31:0] act_wstrb;
wire [CTRL_ADDR_WIDTH-1:0] act_araddr;
wire act_aruser_ap;
wire [3:0] act_aruser_id;
wire [3:0] act_arlen;
wire act_arvalid;
wire act_busy;
wire act_done;
wire act_error;
wire [7:0] act_error_code;
wire [4:0] act_debug_state;

wire [CTRL_ADDR_WIDTH-1:0] scale_awaddr;
wire scale_awuser_ap;
wire [3:0] scale_awuser_id;
wire [3:0] scale_awlen;
wire scale_awvalid;
wire [255:0] scale_wdata;
wire [31:0] scale_wstrb;
wire [CTRL_ADDR_WIDTH-1:0] scale_araddr;
wire scale_aruser_ap;
wire [3:0] scale_aruser_id;
wire [3:0] scale_arlen;
wire scale_arvalid;
wire scale_busy;
wire scale_done;
wire scale_error;
wire [7:0] scale_error_code;
wire [3:0] scale_debug_state;
wire [12:0] scale_current_row;
wire [6:0] scale_current_group;

wire select_scale_bus = (state == ST_START_SCALE) || (state == ST_WAIT_SCALE);

runtime_activation_quantizer_ctrl #(
    .SOURCE_Q28(SOURCE_Q28),
    .CTRL_ADDR_WIDTH(CTRL_ADDR_WIDTH)
) u_runtime_activation_quantizer_ctrl (
    .clk                     (clk),
    .rst_n                   (rst_n),
    .ddr_init_done           (ddr_init_done),
    .start                   (act_start),
    .vector_length           (cfg_vector_length),
    .source_ctrl_addr        (cfg_source_addr),
    .activation_ctrl_addr    (cfg_activation_addr),
    .axi_awaddr              (act_awaddr),
    .axi_awuser_ap           (act_awuser_ap),
    .axi_awuser_id           (act_awuser_id),
    .axi_awlen               (act_awlen),
    .axi_awready             (select_scale_bus ? 1'b0 : axi_awready),
    .axi_awvalid             (act_awvalid),
    .axi_wdata               (act_wdata),
    .axi_wstrb               (act_wstrb),
    .axi_wready              (select_scale_bus ? 1'b0 : axi_wready),
    .axi_araddr              (act_araddr),
    .axi_aruser_ap           (act_aruser_ap),
    .axi_aruser_id           (act_aruser_id),
    .axi_arlen               (act_arlen),
    .axi_arready             (select_scale_bus ? 1'b0 : axi_arready),
    .axi_arvalid             (act_arvalid),
    .axi_rdata               (axi_rdata),
    .axi_rvalid              (select_scale_bus ? 1'b0 : axi_rvalid),
    .busy                    (act_busy),
    .done                    (act_done),
    .error                   (act_error),
    .error_code              (act_error_code),
    .all_zero                (all_zero),
    .max_abs_q10             (max_abs_q10),
    .max_mantissa_binary32   (max_mantissa_binary32),
    .max_exponent_binary32   (max_exponent_binary32),
    .max_abs_binary32_bits   (max_abs_binary32_bits),
    .debug_state             (act_debug_state)
);

runtime_scale_builder_ctrl #(
    .CTRL_ADDR_WIDTH(CTRL_ADDR_WIDTH)
) u_runtime_scale_builder_ctrl (
    .clk                     (clk),
    .rst_n                   (rst_n),
    .ddr_init_done           (ddr_init_done),
    .start                   (scale_start),
    .cfg_rows                (cfg_rows),
    .cfg_groups              (cfg_groups),
    .all_zero                (all_zero),
    .max_mantissa_binary32   (max_mantissa_binary32),
    .max_exponent_binary32   (max_exponent_binary32),
    .raw_scale_ctrl_addr     (cfg_raw_scale_addr),
    .combined_scale_ctrl_addr(cfg_combined_scale_addr),
    .axi_awaddr              (scale_awaddr),
    .axi_awuser_ap           (scale_awuser_ap),
    .axi_awuser_id           (scale_awuser_id),
    .axi_awlen               (scale_awlen),
    .axi_awready             (select_scale_bus ? axi_awready : 1'b0),
    .axi_awvalid             (scale_awvalid),
    .axi_wdata               (scale_wdata),
    .axi_wstrb               (scale_wstrb),
    .axi_wready              (select_scale_bus ? axi_wready : 1'b0),
    .axi_araddr              (scale_araddr),
    .axi_aruser_ap           (scale_aruser_ap),
    .axi_aruser_id           (scale_aruser_id),
    .axi_arlen               (scale_arlen),
    .axi_arready             (select_scale_bus ? axi_arready : 1'b0),
    .axi_arvalid             (scale_arvalid),
    .axi_rdata               (axi_rdata),
    .axi_rvalid              (select_scale_bus ? axi_rvalid : 1'b0),
    .busy                    (scale_busy),
    .done                    (scale_done),
    .error                   (scale_error),
    .error_code              (scale_error_code),
    .saturated_count         (saturated_count),
    .current_row             (scale_current_row),
    .current_group           (scale_current_group),
    .debug_state             (scale_debug_state)
);

assign axi_awaddr    = select_scale_bus ? scale_awaddr    : act_awaddr;
assign axi_awuser_ap = select_scale_bus ? scale_awuser_ap : act_awuser_ap;
assign axi_awuser_id = select_scale_bus ? scale_awuser_id : act_awuser_id;
assign axi_awlen     = select_scale_bus ? scale_awlen     : act_awlen;
assign axi_awvalid   = select_scale_bus ? scale_awvalid   : act_awvalid;
assign axi_wdata     = select_scale_bus ? scale_wdata     : act_wdata;
assign axi_wstrb     = select_scale_bus ? scale_wstrb     : act_wstrb;
assign axi_araddr    = select_scale_bus ? scale_araddr    : act_araddr;
assign axi_aruser_ap = select_scale_bus ? scale_aruser_ap : act_aruser_ap;
assign axi_aruser_id = select_scale_bus ? scale_aruser_id : act_aruser_id;
assign axi_arlen     = select_scale_bus ? scale_arlen     : act_arlen;
assign axi_arvalid   = select_scale_bus ? scale_arvalid   : act_arvalid;
assign debug_state   = state;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        state       <= ST_IDLE;
        act_start   <= 1'b0;
        scale_start <= 1'b0;
        busy        <= 1'b0;
        done        <= 1'b0;
        error       <= 1'b0;
        error_code  <= 8'd0;
    end else begin
        act_start   <= 1'b0;
        scale_start <= 1'b0;
        done        <= 1'b0;

        case (state)
            ST_IDLE: begin
                busy <= 1'b0;
                if (start && !error) begin
                    busy       <= 1'b1;
                    error_code <= 8'd0;
                    state      <= ST_START_ACT;
                end
            end

            ST_START_ACT: begin
                act_start <= 1'b1;
                state     <= ST_WAIT_ACT;
            end

            ST_WAIT_ACT: begin
                if (act_error) begin
                    error      <= 1'b1;
                    error_code <= ERR_ACTIVATION + act_error_code;
                    busy       <= 1'b0;
                    state      <= ST_ERROR;
                end else if (act_done) begin
                    state <= ST_START_SCALE;
                end
            end

            ST_START_SCALE: begin
                scale_start <= 1'b1;
                state       <= ST_WAIT_SCALE;
            end

            ST_WAIT_SCALE: begin
                if (scale_error) begin
                    error      <= 1'b1;
                    error_code <= ERR_SCALE + scale_error_code;
                    busy       <= 1'b0;
                    state      <= ST_ERROR;
                end else if (scale_done) begin
                    state <= ST_FINISH;
                end
            end

            ST_FINISH: begin
                busy <= 1'b0;
                done <= 1'b1;
                state <= ST_IDLE;
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
