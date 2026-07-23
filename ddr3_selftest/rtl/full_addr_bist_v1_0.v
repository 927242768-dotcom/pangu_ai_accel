`timescale 1ns/1ps

// Deterministic full-address DDR3 self-test controller.
// Sequence:
//   1. Wait for DDR3 PHY/controller calibration.
//   2. Use the vendor write engine to fill the complete memory space.
//   3. Read every address back with 16-beat AXI bursts.
//   4. Recreate the address-derived test pattern and latch any mismatch.
//
// The port list intentionally matches axi_bist_top_v1_0 so the generated
// example top can select this controller without changing the DDR3 IP wiring.
module full_addr_bist_v1_0 #(
    parameter          DATA_MASK_EN     = 0,
    parameter          CTRL_ADDR_WIDTH  = 28,
    parameter          MEM_DQ_WIDTH     = 16,
    parameter          MEM_SPACE_AW     = 18,
    parameter          DATA_PATTERN0    = 8'h55,
    parameter          DATA_PATTERN1    = 8'haa,
    parameter          DATA_PATTERN2    = 8'h7f,
    parameter          DATA_PATTERN3    = 8'h80,
    parameter          DATA_PATTERN4    = 8'h55,
    parameter          DATA_PATTERN5    = 8'haa,
    parameter          DATA_PATTERN6    = 8'h7f,
    parameter          DATA_PATTERN7    = 8'h80
)(
    input                               core_clk,
    input                               core_clk_rst_n,
    input      [1:0]                    wr_mode,
    input      [1:0]                    data_mode,
    input                               len_random_en,
    input      [3:0]                    fix_axi_len,
    input                               bist_stop,
    input                               ddrc_init_done,
    input      [3:0]                    read_repeat_num,
    input                               data_order,
    input      [7:0]                    dq_inversion,
    input                               insert_err,
    input                               manu_clear,
    output                              bist_run_led,
    output     [3:0]                    test_main_state,

    output     [CTRL_ADDR_WIDTH-1:0]    axi_awaddr,
    output                              axi_awuser_ap,
    output     [3:0]                    axi_awuser_id,
    output     [3:0]                    axi_awlen,
    input                               axi_awready,
    output                              axi_awvalid,

    output     [MEM_DQ_WIDTH*8-1:0]     axi_wdata,
    output     [MEM_DQ_WIDTH-1:0]       axi_wstrb,
    input                               axi_wready,
    output     [2:0]                    test_wr_state,

    output     [CTRL_ADDR_WIDTH-1:0]    axi_araddr,
    output                              axi_aruser_ap,
    output     [3:0]                    axi_aruser_id,
    output     [3:0]                    axi_arlen,
    input                               axi_arready,
    output                              axi_arvalid,

    input      [MEM_DQ_WIDTH*8-1:0]     axi_rdata,
    input                               axi_rvalid,
    output     [7:0]                    err_cnt,
    output                              err_flag_led,
    output     [MEM_DQ_WIDTH*8-1:0]     err_data_out,
    output     [MEM_DQ_WIDTH*8-1:0]     err_flag_out,
    output     [MEM_DQ_WIDTH*8-1:0]     exp_data_out,
    output                              next_err_flag,
    output     [15:0]                   result_bit_out,
    output     [2:0]                    test_rd_state,
    output     [MEM_DQ_WIDTH*8-1:0]     next_err_data,
    output     [MEM_DQ_WIDTH*8-1:0]     err_data_pre,
    output     [MEM_DQ_WIDTH*8-1:0]     err_data_aft
);

localparam [3:0] ST_WAIT_CAL = 4'd0;
localparam [3:0] ST_FULL_WR  = 4'd1;
localparam [3:0] ST_FULL_RD  = 4'd2;
localparam [3:0] ST_DRAIN    = 4'd3;
localparam [3:0] ST_CHECK    = 4'd4;
localparam [3:0] ST_PASS     = 4'd5;
localparam [3:0] ST_FAIL     = 4'd6;

// 29-bit value for the x32, row15/column10/bank3 configuration:
// 2^28 controller address units = 1 GiB of physical DDR3 data.
localparam [CTRL_ADDR_WIDTH:0] AXI_ADDR_MAX =
    ({{CTRL_ADDR_WIDTH{1'b0}}, 1'b1} << MEM_SPACE_AW);
localparam [CTRL_ADDR_WIDTH:0] LAST_BURST_ADDR_EXT = AXI_ADDR_MAX - 9'd128;
localparam [CTRL_ADDR_WIDTH-1:0] LAST_BURST_ADDR =
    LAST_BURST_ADDR_EXT[CTRL_ADDR_WIDTH-1:0];

reg [3:0] state;
reg       init_start;
reg       read_en;
reg [CTRL_ADDR_WIDTH-1:0] sweep_addr;
reg [4:0] final_read_beat_count;
reg [3:0] compare_drain_count;

wire init_done;
wire write_done_unused;
wire read_done_p;
wire rd_error_latched;

// The generated read engine asserts read_done_p when the read-address request
// is accepted. It does not accept another request until all return beats from
// the current burst have arrived, so incrementing sweep_addr here is safe.
always @(posedge core_clk or negedge core_clk_rst_n) begin
    if (!core_clk_rst_n) begin
        state                 <= ST_WAIT_CAL;
        init_start            <= 1'b0;
        read_en               <= 1'b0;
        sweep_addr            <= {CTRL_ADDR_WIDTH{1'b0}};
        final_read_beat_count <= 5'd0;
        compare_drain_count   <= 4'd0;
    end else begin
        case (state)
            ST_WAIT_CAL: begin
                init_start            <= 1'b0;
                read_en               <= 1'b0;
                sweep_addr            <= {CTRL_ADDR_WIDTH{1'b0}};
                final_read_beat_count <= 5'd0;
                compare_drain_count   <= 4'd0;
                if (ddrc_init_done && !bist_stop) begin
                    init_start <= 1'b1;
                    state      <= ST_FULL_WR;
                end
            end

            ST_FULL_WR: begin
                // The vendor engine performs a complete sequential fill while
                // init_start is asserted and only raises init_done after all
                // write data has drained from the AXI interface.
                init_start <= 1'b1;
                read_en    <= 1'b0;
                if (init_done) begin
                    init_start <= 1'b0;
                    sweep_addr <= {CTRL_ADDR_WIDTH{1'b0}};
                    read_en    <= 1'b1;
                    state      <= ST_FULL_RD;
                end
            end

            ST_FULL_RD: begin
                init_start <= 1'b0;
                read_en    <= !bist_stop;
                if (read_done_p) begin
                    if (sweep_addr == LAST_BURST_ADDR) begin
                        // The final address request has been accepted. Stop
                        // issuing new reads, but continue counting all 16 data
                        // beats from this last burst before declaring success.
                        read_en               <= 1'b0;
                        final_read_beat_count <= 5'd0;
                        state                 <= ST_DRAIN;
                    end else begin
                        sweep_addr <= sweep_addr + 9'd128;
                    end
                end
            end

            ST_DRAIN: begin
                init_start <= 1'b0;
                read_en    <= 1'b0;
                if (axi_rvalid) begin
                    if (final_read_beat_count == 5'd15) begin
                        final_read_beat_count <= 5'd16;
                        compare_drain_count   <= 4'd0;
                        state                 <= ST_CHECK;
                    end else begin
                        final_read_beat_count <= final_read_beat_count + 5'd1;
                    end
                end
            end

            ST_CHECK: begin
                // The generated checker has a short comparison pipeline.
                // Eight extra clocks guarantee the last beat can update the
                // latched error flag before PASS/FAIL is selected.
                init_start <= 1'b0;
                read_en    <= 1'b0;
                if (compare_drain_count == 4'd7) begin
                    state <= rd_error_latched ? ST_FAIL : ST_PASS;
                end else begin
                    compare_drain_count <= compare_drain_count + 4'd1;
                end
            end

            ST_PASS: begin
                init_start <= 1'b0;
                read_en    <= 1'b0;
                state      <= ST_PASS;
            end

            ST_FAIL: begin
                init_start <= 1'b0;
                read_en    <= 1'b0;
                state      <= ST_FAIL;
            end

            default: begin
                state      <= ST_WAIT_CAL;
                init_start <= 1'b0;
                read_en    <= 1'b0;
            end
        endcase
    end
end

assign test_main_state = state;
assign bist_run_led = (state == ST_FULL_WR) ||
                      (state == ST_FULL_RD) ||
                      (state == ST_DRAIN)   ||
                      (state == ST_CHECK);
assign err_flag_led = rd_error_latched;

// Keep all test data deterministic. The writer and reader both seed their
// PRBS data generator from the current address, so every read can recreate the
// exact value originally written without storing a golden copy on chip.
test_wr_ctrl_v1_0 #(
    .DATA_PATTERN0   (DATA_PATTERN0),
    .DATA_PATTERN1   (DATA_PATTERN1),
    .DATA_PATTERN2   (DATA_PATTERN2),
    .DATA_PATTERN3   (DATA_PATTERN3),
    .DATA_PATTERN4   (DATA_PATTERN4),
    .DATA_PATTERN5   (DATA_PATTERN5),
    .DATA_PATTERN6   (DATA_PATTERN6),
    .DATA_PATTERN7   (DATA_PATTERN7),
    .DATA_MASK_EN    (DATA_MASK_EN),
    .CTRL_ADDR_WIDTH (CTRL_ADDR_WIDTH),
    .MEM_DQ_WIDTH    (MEM_DQ_WIDTH),
    .MEM_SPACE_AW    (MEM_SPACE_AW)
) u_full_write (
    .clk             (core_clk),
    .rst_n           (core_clk_rst_n),
    .init_start      (init_start),
    .write_en        (1'b0),
    .insert_err      (1'b0),
    .write_done_p    (write_done_unused),
    .init_done       (init_done),
    .pattern_en      (1'b0),
    .random_data_en  (1'b1),
    .stress_test     (1'b0),
    .write_to_read   (1'b0),
    .read_repeat_en  (1'b0),
    .data_order      (1'b0),
    .dq_inversion    (8'h00),
    .random_rw_addr  (sweep_addr),
    .random_axi_id   (4'h0),
    .random_axi_len  (4'hf),
    .random_axi_ap   (1'b0),
    .axi_awaddr      (axi_awaddr),
    .axi_awuser_ap   (axi_awuser_ap),
    .axi_awuser_id   (axi_awuser_id),
    .axi_awlen       (axi_awlen),
    .axi_awready     (axi_awready),
    .axi_awvalid     (axi_awvalid),
    .axi_wdata       (axi_wdata),
    .axi_wstrb       (axi_wstrb),
    .axi_wready      (axi_wready),
    .test_wr_state   (test_wr_state)
);

test_rd_ctrl_v1_0 #(
    .DATA_PATTERN0   (DATA_PATTERN0),
    .DATA_PATTERN1   (DATA_PATTERN1),
    .DATA_PATTERN2   (DATA_PATTERN2),
    .DATA_PATTERN3   (DATA_PATTERN3),
    .DATA_PATTERN4   (DATA_PATTERN4),
    .DATA_PATTERN5   (DATA_PATTERN5),
    .DATA_PATTERN6   (DATA_PATTERN6),
    .DATA_PATTERN7   (DATA_PATTERN7),
    .DATA_MASK_EN    (DATA_MASK_EN),
    .CTRL_ADDR_WIDTH (CTRL_ADDR_WIDTH),
    .MEM_DQ_WIDTH    (MEM_DQ_WIDTH),
    .MEM_SPACE_AW    (MEM_SPACE_AW)
) u_full_read (
    .clk             (core_clk),
    .rst_n           (core_clk_rst_n),
    .pattern_en      (1'b0),
    .random_data_en  (1'b1),
    .read_repeat_en  (1'b0),
    .read_repeat_num (4'h0),
    .stress_test     (1'b0),
    .write_to_read   (1'b0),
    .data_order      (1'b0),
    .dq_inversion    (8'h00),
    .random_rw_addr  (sweep_addr),
    .random_axi_id   (4'h0),
    .random_axi_len  (4'hf),
    .random_axi_ap   (1'b0),
    .read_en         (read_en),
    .read_done_p     (read_done_p),
    .axi_araddr      (axi_araddr),
    .axi_aruser_ap   (axi_aruser_ap),
    .axi_aruser_id   (axi_aruser_id),
    .axi_arlen       (axi_arlen),
    .axi_arready     (axi_arready),
    .axi_arvalid     (axi_arvalid),
    .axi_rdata       (axi_rdata),
    .axi_rvalid      (axi_rvalid),
    .err_cnt         (err_cnt),
    .err_flag_led    (rd_error_latched),
    .err_data_out    (err_data_out),
    .err_flag_out    (err_flag_out),
    .exp_data_out    (exp_data_out),
    .manu_clear      (manu_clear),
    .next_err_flag   (next_err_flag),
    .result_bit_out  (result_bit_out),
    .test_rd_state   (test_rd_state),
    .next_err_data   (next_err_data),
    .err_data_pre    (err_data_pre),
    .err_data_aft    (err_data_aft)
);

// These inputs remain in the compatible interface for the vendor UART/debug
// register map. The deterministic full sweep intentionally ignores them.
wire _unused_controls = &{1'b0, wr_mode, data_mode, len_random_en,
                          fix_axi_len, read_repeat_num, data_order,
                          dq_inversion, insert_err};

endmodule
