`timescale 1ns/1ps

// G2 FP16 raw weight-scale -> padded UQ4.28 combined-scale DDR3 controller。
//
// raw FP16 按 [rows, groups] 紧凑连续存放；输出每行按 8 words/256 bit 对齐：
//   groups=14 -> padded 16 words / 2 beats
//   groups=76 -> padded 80 words / 10 beats
module runtime_scale_builder_ctrl #(
    parameter integer CTRL_ADDR_WIDTH = 28
)(
    input  wire                         clk,
    input  wire                         rst_n,
    input  wire                         ddr_init_done,
    input  wire                         start,
    input  wire [12:0]                  cfg_rows,
    input  wire [6:0]                   cfg_groups,
    input  wire                         all_zero,
    input  wire [23:0]                  max_mantissa_binary32,
    input  wire signed [9:0]            max_exponent_binary32,
    input  wire [CTRL_ADDR_WIDTH-1:0]   raw_scale_ctrl_addr,
    input  wire [CTRL_ADDR_WIDTH-1:0]   combined_scale_ctrl_addr,

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
    output wire [3:0]                   axi_arlen,
    input  wire                         axi_arready,
    output reg                          axi_arvalid,
    input  wire [255:0]                 axi_rdata,
    input  wire                         axi_rvalid,

    output reg                          busy,
    output reg                          done,
    output reg                          error,
    output reg  [7:0]                   error_code,
    output reg  [31:0]                  saturated_count,
    output reg  [12:0]                  current_row,
    output reg  [6:0]                   current_group,
    output wire [3:0]                   debug_state
);

localparam [3:0] ST_IDLE          = 4'd0;
localparam [3:0] ST_SETUP_READ    = 4'd1;
localparam [3:0] ST_READ_RAW      = 4'd2;
localparam [3:0] ST_FEED_SCALE    = 4'd3;
localparam [3:0] ST_WAIT_RESULT   = 4'd4;
localparam [3:0] ST_SETUP_WRITE   = 4'd5;
localparam [3:0] ST_WRITE_RESULT  = 4'd6;
localparam [3:0] ST_FINISH        = 4'd7;
localparam [3:0] ST_ERROR         = 4'd15;

localparam [7:0] ERR_DDR_NOT_READY = 8'h01;
localparam [7:0] ERR_CONFIG        = 8'h02;
localparam [7:0] ERR_SCALE_BUILDER = 8'h03;
localparam [7:0] ERR_INTERNAL      = 8'hff;

reg [3:0] state;
reg [12:0] rows_reg;
reg [6:0] groups_reg;
reg all_zero_reg;
reg [23:0] max_mantissa_reg;
reg signed [9:0] max_exponent_reg;
reg [CTRL_ADDR_WIDTH-1:0] raw_read_addr;
reg [CTRL_ADDR_WIDTH-1:0] combined_write_addr;
reg [255:0] raw_scale_buffer;
reg [3:0] raw_lane_index;
reg [2:0] output_lane_index;
reg [255:0] output_buffer;
reg pending_row_end;
reg pending_need_raw_read;
reg aw_seen;
reg w_seen;
reg ar_seen;

wire config_valid =
    ((cfg_rows == 13'd128) || (cfg_rows == 13'd896) || (cfg_rows == 13'd4864)) &&
    ((cfg_groups == 7'd14) || (cfg_groups == 7'd76));
wire ar_handshake = axi_arvalid && axi_arready;
wire aw_handshake = axi_awvalid && axi_awready;
wire read_handshake = axi_rvalid && (ar_seen || ar_handshake);
wire write_handshake = axi_wready && (aw_seen || aw_handshake);

wire [15:0] selected_weight_scale =
    raw_scale_buffer[raw_lane_index*16 +: 16];
wire builder_scale_ready;
wire builder_combined_valid;
wire [31:0] builder_combined_scale;
wire builder_saturated;
wire builder_error;
wire [7:0] builder_error_code;
wire feed_handshake = (state == ST_FEED_SCALE) && builder_scale_ready;
wire result_handshake = (state == ST_WAIT_RESULT) && builder_combined_valid;
wire group_is_last = current_group + 1'b1 == groups_reg;
wire output_beat_full = output_lane_index == 3'd7;
wire raw_beat_finished = raw_lane_index == 4'd15;

assign axi_awuser_ap = 1'b0;
assign axi_awuser_id = 4'h0;
assign axi_awlen = 4'h0;
assign axi_aruser_ap = 1'b0;
assign axi_aruser_id = 4'h0;
assign axi_arlen = 4'h0;
assign debug_state = state;

runtime_fp16_scale_builder u_runtime_fp16_scale_builder (
    .clk                       (clk),
    .rst_n                     (rst_n),
    .all_zero                  (all_zero_reg),
    .max_mantissa_binary32     (max_mantissa_reg),
    .max_exponent_binary32     (max_exponent_reg),
    .scale_valid               (state == ST_FEED_SCALE),
    .scale_ready               (builder_scale_ready),
    .weight_scale_fp16         (selected_weight_scale),
    .combined_valid            (builder_combined_valid),
    .combined_ready            (state == ST_WAIT_RESULT),
    .combined_scale_uq4_28     (builder_combined_scale),
    .saturated                 (builder_saturated),
    .error                     (builder_error),
    .error_code                (builder_error_code),
    .debug_state               ()
);

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        state                  <= ST_IDLE;
        rows_reg               <= 13'd0;
        groups_reg             <= 7'd0;
        all_zero_reg           <= 1'b0;
        max_mantissa_reg       <= 24'd0;
        max_exponent_reg       <= 10'sd0;
        raw_read_addr          <= {CTRL_ADDR_WIDTH{1'b0}};
        combined_write_addr    <= {CTRL_ADDR_WIDTH{1'b0}};
        raw_scale_buffer       <= 256'd0;
        raw_lane_index         <= 4'd0;
        output_lane_index      <= 3'd0;
        output_buffer          <= 256'd0;
        pending_row_end        <= 1'b0;
        pending_need_raw_read  <= 1'b0;
        aw_seen                <= 1'b0;
        w_seen                 <= 1'b0;
        ar_seen                <= 1'b0;
        axi_awaddr             <= {CTRL_ADDR_WIDTH{1'b0}};
        axi_awvalid            <= 1'b0;
        axi_wdata              <= 256'd0;
        axi_wstrb              <= 32'd0;
        axi_araddr             <= {CTRL_ADDR_WIDTH{1'b0}};
        axi_arvalid            <= 1'b0;
        busy                   <= 1'b0;
        done                   <= 1'b0;
        error                  <= 1'b0;
        error_code             <= 8'd0;
        saturated_count        <= 32'd0;
        current_row            <= 13'd0;
        current_group          <= 7'd0;
    end else begin
        done <= 1'b0;

        if (builder_error && state != ST_ERROR) begin
            error       <= 1'b1;
            error_code  <= ERR_SCALE_BUILDER + builder_error_code;
            busy        <= 1'b0;
            axi_awvalid <= 1'b0;
            axi_arvalid <= 1'b0;
            state       <= ST_ERROR;
        end else begin
            case (state)
                ST_IDLE: begin
                    busy        <= 1'b0;
                    axi_awvalid <= 1'b0;
                    axi_arvalid <= 1'b0;
                    aw_seen     <= 1'b0;
                    w_seen      <= 1'b0;
                    ar_seen     <= 1'b0;
                    if (start && !error) begin
                        if (!ddr_init_done) begin
                            error      <= 1'b1;
                            error_code <= ERR_DDR_NOT_READY;
                            state      <= ST_ERROR;
                        end else if (!config_valid || (!all_zero && max_mantissa_binary32 == 24'd0)) begin
                            error      <= 1'b1;
                            error_code <= ERR_CONFIG;
                            state      <= ST_ERROR;
                        end else begin
                            rows_reg              <= cfg_rows;
                            groups_reg            <= cfg_groups;
                            all_zero_reg          <= all_zero;
                            max_mantissa_reg      <= max_mantissa_binary32;
                            max_exponent_reg      <= max_exponent_binary32;
                            raw_read_addr         <= raw_scale_ctrl_addr;
                            combined_write_addr   <= combined_scale_ctrl_addr;
                            raw_lane_index        <= 4'd0;
                            output_lane_index     <= 3'd0;
                            output_buffer         <= 256'd0;
                            pending_row_end       <= 1'b0;
                            pending_need_raw_read <= 1'b0;
                            saturated_count       <= 32'd0;
                            current_row           <= 13'd0;
                            current_group         <= 7'd0;
                            error_code            <= 8'd0;
                            busy                  <= 1'b1;
                            state                 <= ST_SETUP_READ;
                        end
                    end
                end

                ST_SETUP_READ: begin
                    axi_araddr  <= raw_read_addr;
                    axi_arvalid <= 1'b1;
                    ar_seen     <= 1'b0;
                    state       <= ST_READ_RAW;
                end

                ST_READ_RAW: begin
                    if (ar_handshake) begin
                        axi_arvalid <= 1'b0;
                        ar_seen     <= 1'b1;
                    end
                    if (read_handshake) begin
                        raw_scale_buffer <= axi_rdata;
                        ar_seen          <= 1'b0;
                        state            <= ST_FEED_SCALE;
                    end
                end

                ST_FEED_SCALE: begin
                    if (feed_handshake)
                        state <= ST_WAIT_RESULT;
                end

                ST_WAIT_RESULT: begin
                    if (result_handshake) begin
                        output_buffer[output_lane_index*32 +: 32]
                            <= builder_combined_scale;
                        if (builder_saturated)
                            saturated_count <= saturated_count + 1'b1;

                        if (raw_beat_finished) begin
                            raw_lane_index <= 4'd0;
                            raw_read_addr  <= raw_read_addr + 8;
                        end else begin
                            raw_lane_index <= raw_lane_index + 1'b1;
                        end

                        if (group_is_last || output_beat_full) begin
                            pending_row_end       <= group_is_last;
                            pending_need_raw_read <= raw_beat_finished;
                            state                 <= ST_SETUP_WRITE;
                        end else begin
                            current_group     <= current_group + 1'b1;
                            output_lane_index <= output_lane_index + 1'b1;
                            if (raw_beat_finished)
                                state <= ST_SETUP_READ;
                            else
                                state <= ST_FEED_SCALE;
                        end
                    end
                end

                ST_SETUP_WRITE: begin
                    axi_awaddr  <= combined_write_addr;
                    axi_awvalid <= 1'b1;
                    axi_wdata   <= output_buffer;
                    axi_wstrb   <= 32'hffff_ffff;
                    aw_seen     <= 1'b0;
                    w_seen      <= 1'b0;
                    state       <= ST_WRITE_RESULT;
                end

                ST_WRITE_RESULT: begin
                    if (aw_handshake) begin
                        axi_awvalid <= 1'b0;
                        aw_seen     <= 1'b1;
                    end
                    if (write_handshake)
                        w_seen <= 1'b1;
                    if ((aw_seen || aw_handshake) && (w_seen || write_handshake)) begin
                        axi_awvalid          <= 1'b0;
                        aw_seen              <= 1'b0;
                        w_seen               <= 1'b0;
                        combined_write_addr  <= combined_write_addr + 8;
                        output_buffer        <= 256'd0;
                        output_lane_index    <= 3'd0;

                        if (pending_row_end) begin
                            current_group <= 7'd0;
                            if (current_row + 1'b1 == rows_reg) begin
                                state <= ST_FINISH;
                            end else begin
                                current_row <= current_row + 1'b1;
                                if (pending_need_raw_read)
                                    state <= ST_SETUP_READ;
                                else
                                    state <= ST_FEED_SCALE;
                            end
                        end else begin
                            current_group <= current_group + 1'b1;
                            if (pending_need_raw_read)
                                state <= ST_SETUP_READ;
                            else
                                state <= ST_FEED_SCALE;
                        end
                        pending_row_end       <= 1'b0;
                        pending_need_raw_read <= 1'b0;
                    end
                end

                ST_FINISH: begin
                    busy <= 1'b0;
                    done <= 1'b1;
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
end

endmodule
