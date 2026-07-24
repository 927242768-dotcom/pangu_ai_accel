`timescale 1ns/1ps

// 盘古 Logos PGL50H tied Embedding K=896 独立验证顶层。
// 顶层端口保持与已验证 DDR3 example design 一致，复用原有引脚和时序约束。
module embedding_k896_top #(
    parameter MEM_ROW_ADDR_WIDTH = 15,
    parameter MEM_COL_ADDR_WIDTH = 10,
    parameter MEM_BADDR_WIDTH    = 3,
    parameter MEM_DQ_WIDTH       = 32,
    parameter MEM_DM_WIDTH       = MEM_DQ_WIDTH/8,
    parameter MEM_DQS_WIDTH      = MEM_DQ_WIDTH/8,
    parameter CTRL_ADDR_WIDTH    = MEM_ROW_ADDR_WIDTH + MEM_BADDR_WIDTH + MEM_COL_ADDR_WIDTH
)(
    input                                  ref_clk,
    input                                  rst_board,
    output                                 pll_lock,
    output                                 ddr_init_done,

    input                                  uart_rxd,
    output                                 uart_txd,

    output                                 mem_rst_n,
    output                                 mem_ck,
    output                                 mem_ck_n,
    output                                 mem_cke,
    output                                 mem_cs_n,
    output                                 mem_ras_n,
    output                                 mem_cas_n,
    output                                 mem_we_n,
    output                                 mem_odt,
    output [MEM_ROW_ADDR_WIDTH-1:0]         mem_a,
    output [MEM_BADDR_WIDTH-1:0]            mem_ba,
    inout  [MEM_DQS_WIDTH-1:0]              mem_dqs,
    inout  [MEM_DQS_WIDTH-1:0]              mem_dqs_n,
    inout  [MEM_DQ_WIDTH-1:0]               mem_dq,
    output [MEM_DM_WIDTH-1:0]               mem_dm,

    output                                 heart_beat_led,
    output                                 err_flag_led
);

wire core_clk;
wire core_clk_rst_n;
wire resetn = rst_board;

wire [CTRL_ADDR_WIDTH-1:0] axi_awaddr;
wire                       axi_awuser_ap;
wire [3:0]                 axi_awuser_id;
wire [3:0]                 axi_awlen;
wire                       axi_awready;
wire                       axi_awvalid;
wire [MEM_DQ_WIDTH*8-1:0]  axi_wdata;
wire [MEM_DQ_WIDTH-1:0]    axi_wstrb;
wire                       axi_wready;

wire [CTRL_ADDR_WIDTH-1:0] axi_araddr;
wire                       axi_aruser_ap;
wire [3:0]                 axi_aruser_id;
wire [3:0]                 axi_arlen;
wire                       axi_arready;
wire                       axi_arvalid;
wire [MEM_DQ_WIDTH*8-1:0]  axi_rdata;
wire                       axi_rvalid;

wire [34*MEM_DQS_WIDTH-1:0] debug_data;
wire [13*MEM_DQS_WIDTH-1:0] debug_slice_state;
wire [23:0]                 debug_calib_ctrl;
wire [7:0]                  ck_dly_set_bin;
wire [MEM_DQS_WIDTH-1:0]    wl_step_ov_warning;
wire [7:0]                  dll_step;
wire                        dll_lock;
wire [MEM_DQS_WIDTH-1:0]    update_com_val_err_flag;

wire [4:0] ctrl_state;
wire protocol_error;
wire row_loaded;
wire configured;
wire result_valid;

reg [26:0] heartbeat_count;
reg heartbeat;

always @(posedge core_clk or negedge core_clk_rst_n) begin
    if (!core_clk_rst_n) begin
        heartbeat_count <= 27'd0;
        heartbeat       <= 1'b0;
    end else if (heartbeat_count == 27'd49_999_999) begin
        heartbeat_count <= 27'd0;
        heartbeat       <= ~heartbeat;
    end else begin
        heartbeat_count <= heartbeat_count + 1'b1;
    end
end

assign heart_beat_led = result_valid ? 1'b1 : (ddr_init_done ? heartbeat : 1'b0);
assign err_flag_led   = protocol_error;

ipsxb_rst_sync_v1_1 u_core_clk_rst_sync (
    .clk        (core_clk),
    .rst_n      (resetn),
    .sig_async  (1'b1),
    .sig_synced (core_clk_rst_n)
);

pangu_ddr3_x32 #(
    .MEM_ROW_WIDTH    (MEM_ROW_ADDR_WIDTH),
    .MEM_COLUMN_WIDTH (MEM_COL_ADDR_WIDTH),
    .MEM_BANK_WIDTH   (MEM_BADDR_WIDTH),
    .MEM_DQ_WIDTH     (MEM_DQ_WIDTH),
    .MEM_DM_WIDTH     (MEM_DM_WIDTH),
    .MEM_DQS_WIDTH    (MEM_DQS_WIDTH),
    .CTRL_ADDR_WIDTH  (CTRL_ADDR_WIDTH)
) I_ipsxb_ddr_top (
    .ref_clk                 (ref_clk),
    .resetn                  (resetn),
    .ddr_init_done           (ddr_init_done),
    .ddrphy_clkin            (core_clk),
    .pll_lock                (pll_lock),

    .axi_awaddr              (axi_awaddr),
    .axi_awuser_ap           (axi_awuser_ap),
    .axi_awuser_id           (axi_awuser_id),
    .axi_awlen               (axi_awlen),
    .axi_awready             (axi_awready),
    .axi_awvalid             (axi_awvalid),

    .axi_wdata               (axi_wdata),
    .axi_wstrb               (axi_wstrb),
    .axi_wready              (axi_wready),
    .axi_wusero_id           (),
    .axi_wusero_last         (),

    .axi_araddr              (axi_araddr),
    .axi_aruser_ap           (axi_aruser_ap),
    .axi_aruser_id           (axi_aruser_id),
    .axi_arlen               (axi_arlen),
    .axi_arready             (axi_arready),
    .axi_arvalid             (axi_arvalid),

    .axi_rdata               (axi_rdata),
    .axi_rid                 (),
    .axi_rlast               (),
    .axi_rvalid              (axi_rvalid),

    .apb_clk                 (1'b0),
    .apb_rst_n               (1'b0),
    .apb_sel                 (1'b0),
    .apb_enable              (1'b0),
    .apb_addr                (8'd0),
    .apb_write               (1'b0),
    .apb_ready               (),
    .apb_wdata               (16'd0),
    .apb_rdata               (),
    .apb_int                 (),

    .debug_data              (debug_data),
    .debug_slice_state       (debug_slice_state),
    .debug_calib_ctrl        (debug_calib_ctrl),
    .ck_dly_set_bin          (ck_dly_set_bin),
    .ck_dly_en               (1'b1),
    .init_ck_dly_step        (8'h30),
    .wl_step_ov_warning      (wl_step_ov_warning),
    .dll_step                (dll_step),
    .dll_lock                (dll_lock),
    .init_read_clk_ctrl      ({(2*MEM_DQS_WIDTH){1'b0}}),
    .init_slip_step          ({(4*MEM_DQS_WIDTH){1'b0}}),
    .force_read_clk_ctrl     (1'b0),
    .ddrphy_gate_update_en   (1'b1),
    .update_com_val_err_flag (update_com_val_err_flag),
    .rd_fake_stop            (1'b0),

    .mem_rst_n               (mem_rst_n),
    .mem_ck                  (mem_ck),
    .mem_ck_n                (mem_ck_n),
    .mem_cke                 (mem_cke),
    .mem_cs_n                (mem_cs_n),
    .mem_ras_n               (mem_ras_n),
    .mem_cas_n               (mem_cas_n),
    .mem_we_n                (mem_we_n),
    .mem_odt                 (mem_odt),
    .mem_a                   (mem_a),
    .mem_ba                  (mem_ba),
    .mem_dqs                 (mem_dqs),
    .mem_dqs_n               (mem_dqs_n),
    .mem_dq                  (mem_dq),
    .mem_dm                  (mem_dm)
);

embedding_k896_ctrl #(
    .CTRL_ADDR_WIDTH (CTRL_ADDR_WIDTH),
    .CLKS_PER_BIT    (868),
    .VOCAB_SIZE      (151936)
) u_embedding_k896_ctrl (
    .core_clk        (core_clk),
    .core_rst_n      (core_clk_rst_n),
    .ddr_init_done   (ddr_init_done),
    .uart_rx_i       (uart_rxd),
    .uart_tx_o       (uart_txd),

    .axi_awaddr      (axi_awaddr),
    .axi_awuser_ap   (axi_awuser_ap),
    .axi_awuser_id   (axi_awuser_id),
    .axi_awlen       (axi_awlen),
    .axi_awready     (axi_awready),
    .axi_awvalid     (axi_awvalid),
    .axi_wdata       (axi_wdata),
    .axi_wstrb       (axi_wstrb),
    .axi_wready      (axi_wready),

    .axi_araddr      (axi_araddr),
    .axi_aruser_ap   (axi_aruser_ap),
    .axi_aruser_id   (axi_aruser_id),
    .axi_arlen       (axi_arlen),
    .axi_arready     (axi_arready),
    .axi_arvalid     (axi_arvalid),
    .axi_rdata       (axi_rdata),
    .axi_rvalid      (axi_rvalid),

    .debug_state     (ctrl_state),
    .protocol_error  (protocol_error),
    .row_loaded      (row_loaded),
    .configured      (configured),
    .result_valid    (result_valid)
);

wire _unused_debug = &{1'b0, debug_data, debug_slice_state, debug_calib_ctrl,
                       ck_dly_set_bin, wl_step_ov_warning, dll_step, dll_lock,
                       update_com_val_err_flag, ctrl_state, row_loaded, configured};

endmodule
