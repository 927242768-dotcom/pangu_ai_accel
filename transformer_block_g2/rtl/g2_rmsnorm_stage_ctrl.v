`timescale 1ns/1ps

// G2 可复用 RMSNorm DDR3 阶段控制器。
// 同一个 rmsnorm_k896_core 依次服务 INPUT_RMS 与 POST_RMS；每次从 DDR3 加载
// 56 拍输入、56 拍 gamma、32 拍 rsqrt LUT，完成后把 56 拍结果写回。
module g2_rmsnorm_stage_ctrl #(
    parameter integer CTRL_ADDR_WIDTH = 28,
    parameter integer DATA_BEATS      = 56,
    parameter integer LUT_BEATS       = 32
)(
    input  wire                         clk,
    input  wire                         rst_n,
    input  wire                         ddr_init_done,
    input  wire                         start,
    input  wire [CTRL_ADDR_WIDTH-1:0]   cfg_input_addr,
    input  wire [CTRL_ADDR_WIDTH-1:0]   cfg_gamma_addr,
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
    output wire [4:0]                   debug_state,
    output reg  [5:0]                   debug_read_beat,
    output reg  [5:0]                   debug_result_beat,
    output wire [39:0]                  debug_sum_squares,
    output wire [39:0]                  debug_variance_q20,
    output wire [31:0]                  debug_rsqrt_q20
);

localparam [4:0] ST_IDLE             = 5'd0;
localparam [4:0] ST_SETUP_INPUT_READ = 5'd1;
localparam [4:0] ST_READ_INPUT       = 5'd2;
localparam [4:0] ST_SETUP_GAMMA_READ = 5'd3;
localparam [4:0] ST_READ_GAMMA       = 5'd4;
localparam [4:0] ST_SETUP_LUT_READ   = 5'd5;
localparam [4:0] ST_READ_LUT         = 5'd6;
localparam [4:0] ST_START_CORE       = 5'd7;
localparam [4:0] ST_WAIT_RESULT      = 5'd8;
localparam [4:0] ST_SETUP_WRITE      = 5'd9;
localparam [4:0] ST_WRITE            = 5'd10;
localparam [4:0] ST_WAIT_DONE        = 5'd11;
localparam [4:0] ST_FINISH           = 5'd12;
localparam [4:0] ST_ERROR            = 5'd31;

localparam [7:0] ERR_DDR_NOT_READY = 8'h01;
localparam [7:0] ERR_CORE_PROTOCOL = 8'h02;
localparam [7:0] ERR_INTERNAL      = 8'hff;

reg [4:0] state;
reg [CTRL_ADDR_WIDTH-1:0] input_base_addr;
reg [CTRL_ADDR_WIDTH-1:0] gamma_base_addr;
reg [CTRL_ADDR_WIDTH-1:0] lut_base_addr;
reg [CTRL_ADDR_WIDTH-1:0] result_base_addr;
reg [5:0] read_base_beat;
reg [4:0] read_beat_index;
reg [4:0] active_burst_beats;
reg [5:0] result_beat_index;
reg ar_seen;
reg aw_seen;
reg w_seen;
reg core_start;
reg [255:0] result_cache;

wire core_busy;
wire core_done;
wire [255:0] core_result_data;
wire core_result_valid;
wire core_result_ready;

wire aw_handshake = axi_awvalid && axi_awready;
wire ar_handshake = axi_arvalid && axi_arready;
wire write_data_handshake = axi_wready && (aw_seen || aw_handshake);
wire write_complete = (aw_seen || aw_handshake) &&
                      (w_seen || write_data_handshake);
wire read_data_handshake = axi_rvalid && (ar_seen || ar_handshake);

wire [6:0] data_beats_remaining = DATA_BEATS - read_base_beat;
wire [4:0] next_data_burst_beats =
    (data_beats_remaining > 7'd16) ? 5'd16 : data_beats_remaining[4:0];
wire [6:0] lut_beats_remaining = LUT_BEATS - read_base_beat;
wire [4:0] next_lut_burst_beats =
    (lut_beats_remaining > 7'd16) ? 5'd16 : lut_beats_remaining[4:0];

wire input_load_en = (state == ST_READ_INPUT) && read_data_handshake;
wire gamma_load_en = (state == ST_READ_GAMMA) && read_data_handshake;
wire lut_load_en   = (state == ST_READ_LUT) && read_data_handshake;
wire [5:0] load_index = read_base_beat + read_beat_index;
assign core_result_ready = (state == ST_WRITE) && write_complete;

assign axi_awuser_ap = 1'b0;
assign axi_awuser_id = 4'h0;
assign axi_awlen     = 4'h0;
assign axi_aruser_ap = 1'b0;
assign axi_aruser_id = 4'h0;
assign debug_state   = state;

rmsnorm_k896_core #(
    .PIPELINE_NORMALIZE   (1),
    .PIPELINE_X_RNE       (1),
    .PIPELINE_GAMMA_INPUT (1),
    .PIPELINE_RSQRT_SHIFT (1)
) u_rmsnorm_k896_core (
    .clk                  (clk),
    .rst_n                (rst_n),
    .input_load_en        (input_load_en),
    .input_load_index     (load_index),
    .input_load_data      (axi_rdata),
    .gamma_load_en        (gamma_load_en),
    .gamma_load_index     (load_index),
    .gamma_load_data      (axi_rdata),
    .lut_load_en          (lut_load_en),
    .lut_load_index       (load_index[4:0]),
    .lut_load_data        (axi_rdata),
    .start                (core_start),
    .busy                 (core_busy),
    .done                 (core_done),
    .result_data          (core_result_data),
    .result_valid         (core_result_valid),
    .result_ready         (core_result_ready),
    .debug_sum_squares    (debug_sum_squares),
    .debug_variance_q20   (debug_variance_q20),
    .debug_rsqrt_q20      (debug_rsqrt_q20)
);

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        state             <= ST_IDLE;
        input_base_addr   <= {CTRL_ADDR_WIDTH{1'b0}};
        gamma_base_addr   <= {CTRL_ADDR_WIDTH{1'b0}};
        lut_base_addr     <= {CTRL_ADDR_WIDTH{1'b0}};
        result_base_addr  <= {CTRL_ADDR_WIDTH{1'b0}};
        read_base_beat    <= 6'd0;
        read_beat_index   <= 5'd0;
        active_burst_beats<= 5'd0;
        result_beat_index <= 6'd0;
        ar_seen           <= 1'b0;
        aw_seen           <= 1'b0;
        w_seen            <= 1'b0;
        core_start        <= 1'b0;
        result_cache      <= 256'd0;
        axi_awaddr        <= {CTRL_ADDR_WIDTH{1'b0}};
        axi_awvalid       <= 1'b0;
        axi_wdata         <= 256'd0;
        axi_wstrb         <= 32'd0;
        axi_araddr        <= {CTRL_ADDR_WIDTH{1'b0}};
        axi_arlen         <= 4'd0;
        axi_arvalid       <= 1'b0;
        busy              <= 1'b0;
        done              <= 1'b0;
        error             <= 1'b0;
        error_code        <= 8'd0;
        debug_read_beat   <= 6'd0;
        debug_result_beat <= 6'd0;
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
                        input_base_addr   <= cfg_input_addr;
                        gamma_base_addr   <= cfg_gamma_addr;
                        lut_base_addr     <= cfg_lut_addr;
                        result_base_addr  <= cfg_result_addr;
                        read_base_beat    <= 6'd0;
                        result_beat_index <= 6'd0;
                        debug_read_beat   <= 6'd0;
                        debug_result_beat <= 6'd0;
                        busy              <= 1'b1;
                        error_code        <= 8'd0;
                        state             <= ST_SETUP_INPUT_READ;
                    end
                end
            end

            ST_SETUP_INPUT_READ: begin
                axi_araddr              <= input_base_addr + ({22'd0, read_base_beat} << 3);
                axi_arlen               <= next_data_burst_beats - 1'b1;
                axi_arvalid             <= 1'b1;
                ar_seen                 <= 1'b0;
                read_beat_index         <= 5'd0;
                active_burst_beats      <= next_data_burst_beats;
                state                   <= ST_READ_INPUT;
            end

            ST_READ_INPUT: begin
                if (ar_handshake) begin
                    axi_arvalid <= 1'b0;
                    ar_seen     <= 1'b1;
                end
                if (read_data_handshake) begin
                    debug_read_beat <= load_index;
                    if (read_beat_index + 1'b1 == active_burst_beats) begin
                        ar_seen <= 1'b0;
                        if (read_base_beat + active_burst_beats == DATA_BEATS) begin
                            read_base_beat <= 6'd0;
                            state <= ST_SETUP_GAMMA_READ;
                        end else begin
                            read_base_beat <= read_base_beat + active_burst_beats;
                            state <= ST_SETUP_INPUT_READ;
                        end
                    end else begin
                        read_beat_index <= read_beat_index + 1'b1;
                    end
                end
            end

            ST_SETUP_GAMMA_READ: begin
                axi_araddr              <= gamma_base_addr + ({22'd0, read_base_beat} << 3);
                axi_arlen               <= next_data_burst_beats - 1'b1;
                axi_arvalid             <= 1'b1;
                ar_seen                 <= 1'b0;
                read_beat_index         <= 5'd0;
                active_burst_beats      <= next_data_burst_beats;
                state                   <= ST_READ_GAMMA;
            end

            ST_READ_GAMMA: begin
                if (ar_handshake) begin
                    axi_arvalid <= 1'b0;
                    ar_seen     <= 1'b1;
                end
                if (read_data_handshake) begin
                    debug_read_beat <= load_index;
                    if (read_beat_index + 1'b1 == active_burst_beats) begin
                        ar_seen <= 1'b0;
                        if (read_base_beat + active_burst_beats == DATA_BEATS) begin
                            read_base_beat <= 6'd0;
                            state <= ST_SETUP_LUT_READ;
                        end else begin
                            read_base_beat <= read_base_beat + active_burst_beats;
                            state <= ST_SETUP_GAMMA_READ;
                        end
                    end else begin
                        read_beat_index <= read_beat_index + 1'b1;
                    end
                end
            end

            ST_SETUP_LUT_READ: begin
                axi_araddr              <= lut_base_addr + ({22'd0, read_base_beat} << 3);
                axi_arlen               <= next_lut_burst_beats - 1'b1;
                axi_arvalid             <= 1'b1;
                ar_seen                 <= 1'b0;
                read_beat_index         <= 5'd0;
                active_burst_beats      <= next_lut_burst_beats;
                state                   <= ST_READ_LUT;
            end

            ST_READ_LUT: begin
                if (ar_handshake) begin
                    axi_arvalid <= 1'b0;
                    ar_seen     <= 1'b1;
                end
                if (read_data_handshake) begin
                    debug_read_beat <= load_index;
                    if (read_beat_index + 1'b1 == active_burst_beats) begin
                        ar_seen <= 1'b0;
                        if (read_base_beat + active_burst_beats == LUT_BEATS) begin
                            read_base_beat <= 6'd0;
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
                    state      <= ST_WAIT_RESULT;
                end
            end

            ST_WAIT_RESULT: begin
                if (core_result_valid) begin
                    result_cache <= core_result_data;
                    state        <= ST_SETUP_WRITE;
                end else if (core_done) begin
                    error      <= 1'b1;
                    error_code <= ERR_CORE_PROTOCOL;
                    state      <= ST_ERROR;
                end
            end

            ST_SETUP_WRITE: begin
                axi_awaddr  <= result_base_addr + ({22'd0, result_beat_index} << 3);
                axi_awvalid <= 1'b1;
                axi_wdata   <= result_cache;
                axi_wstrb   <= 32'hffff_ffff;
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
                    if (result_beat_index == DATA_BEATS - 1) begin
                        // 最后一拍已通过同拍 result_ready 被核心接受，且 DDR3
                        // 写回已经完成；直接完成本阶段，不再增加额外 done 等待。
                        state <= ST_FINISH;
                    end else begin
                        result_beat_index <= result_beat_index + 1'b1;
                        debug_result_beat <= result_beat_index + 1'b1;
                        state             <= ST_WAIT_RESULT;
                    end
                end
            end

            ST_WAIT_DONE: begin
                if (core_done)
                    state <= ST_FINISH;
                else if (core_result_valid) begin
                    error      <= 1'b1;
                    error_code <= ERR_CORE_PROTOCOL;
                    state      <= ST_ERROR;
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
