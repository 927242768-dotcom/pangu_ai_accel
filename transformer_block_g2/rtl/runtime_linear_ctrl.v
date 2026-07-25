`timescale 1ns/1ps

// G2 运行时参数化 Linear DDR3 行控制器。
//
// 支持当前 layer0 全部 groupwise INT4 Linear：
//   Q:         M=896,  K=896,  groups=14, has_bias=1
//   K/V:       M=128,  K=896,  groups=14, has_bias=1
//   O_proj:    M=896,  K=896,  groups=14, has_bias=0
//   gate/up:   M=4864, K=896,  groups=14, has_bias=0
//   down_proj: M=896,  K=4864, groups=76, has_bias=0
//
// 地址单位与 DDR3 Controller 一致，均为 32 bit。每个 256 bit beat 地址增量为 8。
// activation 在一次矩阵执行开始时加载一次；随后逐行读取 weight/scale/bias，
// 启动 shared_linear_engine，并把四个 int64 Q28 结果合并成一拍写回。
module runtime_linear_ctrl #(
    parameter integer CTRL_ADDR_WIDTH = 28
)(
    input  wire                         clk,
    input  wire                         rst_n,
    input  wire                         ddr_init_done,

    input  wire                         start,
    input  wire [12:0]                  cfg_m_rows,
    input  wire [8:0]                   cfg_k_blocks,
    input  wire [6:0]                   cfg_groups,
    input  wire [7:0]                   cfg_act_beats,
    input  wire [6:0]                   cfg_weight_beats_per_row,
    input  wire [3:0]                   cfg_scale_beats_per_row,
    input  wire                         cfg_has_bias,
    input  wire [CTRL_ADDR_WIDTH-1:0]   cfg_act_addr,
    input  wire [CTRL_ADDR_WIDTH-1:0]   cfg_weight_addr,
    input  wire [CTRL_ADDR_WIDTH-1:0]   cfg_scale_addr,
    input  wire [CTRL_ADDR_WIDTH-1:0]   cfg_bias_addr,
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
    output reg  [12:0]                  current_row,
    output wire [4:0]                   debug_state
);

localparam [4:0] ST_IDLE               = 5'd0;
localparam [4:0] ST_SETUP_ACT_READ     = 5'd1;
localparam [4:0] ST_READ_ACT           = 5'd2;
localparam [4:0] ST_SETUP_WEIGHT_READ  = 5'd3;
localparam [4:0] ST_READ_WEIGHT        = 5'd4;
localparam [4:0] ST_SETUP_SCALE_READ   = 5'd5;
localparam [4:0] ST_READ_SCALE         = 5'd6;
localparam [4:0] ST_SETUP_BIAS_READ    = 5'd7;
localparam [4:0] ST_READ_BIAS          = 5'd8;
localparam [4:0] ST_START_CORE         = 5'd9;
localparam [4:0] ST_WAIT_CORE          = 5'd10;
localparam [4:0] ST_SETUP_RESULT_WRITE = 5'd11;
localparam [4:0] ST_WRITE_RESULT       = 5'd12;
localparam [4:0] ST_FINISH             = 5'd13;
localparam [4:0] ST_ERROR              = 5'd31;

localparam [7:0] ERR_DDR_NOT_READY = 8'h01;
localparam [7:0] ERR_CONFIG        = 8'h02;
localparam [7:0] ERR_CORE_CONFIG   = 8'h03;
localparam [7:0] ERR_INTERNAL      = 8'hff;

reg [4:0] state;
reg [12:0] m_rows_reg;
reg [8:0] k_blocks_reg;
reg [6:0] groups_reg;
reg [7:0] act_beats_reg;
reg [6:0] weight_beats_reg;
reg [3:0] scale_beats_reg;
reg has_bias_reg;
reg [CTRL_ADDR_WIDTH-1:0] act_addr_reg;
reg [CTRL_ADDR_WIDTH-1:0] weight_row_addr;
reg [CTRL_ADDR_WIDTH-1:0] scale_row_addr;
reg [CTRL_ADDR_WIDTH-1:0] bias_row_addr;
reg [CTRL_ADDR_WIDTH-1:0] result_write_addr;

reg [7:0] act_read_base_beat;
reg [6:0] weight_read_base_beat;
reg [4:0] active_read_burst_beats;
reg [4:0] read_beat_index;
reg [255:0] bias_row_cache;
reg [255:0] result_beat_cache;

reg aw_seen;
reg w_seen;
reg ar_seen;
reg core_start;
wire core_busy;
wire core_done;
wire core_config_error;
wire signed [63:0] core_y_q28;
wire signed [63:0] selected_bias_q28 =
    has_bias_reg ? $signed(bias_row_cache[63:0]) : 64'sd0;

wire aw_handshake = axi_awvalid && axi_awready;
wire ar_handshake = axi_arvalid && axi_arready;
wire write_data_handshake = axi_wready && (aw_seen || aw_handshake);
wire read_data_handshake = axi_rvalid && (ar_seen || ar_handshake);

wire [8:0] act_beats_remaining = {1'b0, act_beats_reg} - act_read_base_beat;
wire [4:0] next_act_burst_beats =
    (act_beats_remaining > 9'd16) ? 5'd16 : act_beats_remaining[4:0];
wire [7:0] weight_beats_remaining =
    {1'b0, weight_beats_reg} - weight_read_base_beat;
wire [4:0] next_weight_burst_beats =
    (weight_beats_remaining > 8'd16) ? 5'd16 : weight_beats_remaining[4:0];

wire supported_k =
    ((cfg_k_blocks == 9'd56)  && (cfg_groups == 7'd14)) ||
    ((cfg_k_blocks == 9'd304) && (cfg_groups == 7'd76));
wire supported_m =
    (cfg_m_rows == 13'd128) ||
    (cfg_m_rows == 13'd896) ||
    (cfg_m_rows == 13'd4864);
wire config_consistent =
    supported_k && supported_m &&
    (cfg_m_rows[1:0] == 2'b00) &&
    ({1'b0, cfg_act_beats} == (cfg_k_blocks >> 1)) &&
    ({2'b00, cfg_weight_beats_per_row} == (cfg_k_blocks >> 2)) &&
    ({2'b00, cfg_weight_beats_per_row} == {2'b00, cfg_groups}) &&
    (((cfg_groups + 7'd7) >> 3) == cfg_scale_beats_per_row) &&
    (cfg_act_beats != 8'd0) &&
    (cfg_weight_beats_per_row != 7'd0) &&
    (cfg_scale_beats_per_row != 4'd0);

assign axi_awuser_ap = 1'b0;
assign axi_awuser_id = 4'h0;
assign axi_awlen     = 4'h0;
assign axi_aruser_ap = 1'b0;
assign axi_aruser_id = 4'h0;
assign debug_state   = state;

shared_linear_engine u_shared_linear_engine (
    .clk                   (clk),
    .rst_n                 (rst_n),
    .act_load_en           ((state == ST_READ_ACT) && read_data_handshake),
    .act_load_index        (act_read_base_beat + read_beat_index),
    .act_load_data         (axi_rdata),
    .weight_load_en        ((state == ST_READ_WEIGHT) && read_data_handshake),
    .weight_load_index     (weight_read_base_beat + read_beat_index),
    .weight_load_data      (axi_rdata),
    .scale_load_en         ((state == ST_READ_SCALE) && read_data_handshake),
    .scale_load_beat_index (read_beat_index[3:0]),
    .scale_load_data       (axi_rdata),
    .cfg_k_blocks          (k_blocks_reg),
    .cfg_groups            (groups_reg),
    .start                 (core_start),
    .bias_q28              (selected_bias_q28),
    .busy                  (core_busy),
    .done                  (core_done),
    .config_error          (core_config_error),
    .y_q28                 (core_y_q28)
);

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        state                     <= ST_IDLE;
        m_rows_reg                <= 13'd0;
        k_blocks_reg              <= 9'd0;
        groups_reg                <= 7'd0;
        act_beats_reg             <= 8'd0;
        weight_beats_reg          <= 7'd0;
        scale_beats_reg           <= 4'd0;
        has_bias_reg              <= 1'b0;
        act_addr_reg              <= {CTRL_ADDR_WIDTH{1'b0}};
        weight_row_addr           <= {CTRL_ADDR_WIDTH{1'b0}};
        scale_row_addr            <= {CTRL_ADDR_WIDTH{1'b0}};
        bias_row_addr             <= {CTRL_ADDR_WIDTH{1'b0}};
        result_write_addr         <= {CTRL_ADDR_WIDTH{1'b0}};
        act_read_base_beat        <= 8'd0;
        weight_read_base_beat     <= 7'd0;
        active_read_burst_beats   <= 5'd0;
        read_beat_index           <= 5'd0;
        bias_row_cache            <= 256'd0;
        result_beat_cache         <= 256'd0;
        aw_seen                   <= 1'b0;
        w_seen                    <= 1'b0;
        ar_seen                   <= 1'b0;
        core_start                <= 1'b0;
        axi_awaddr                <= {CTRL_ADDR_WIDTH{1'b0}};
        axi_awvalid               <= 1'b0;
        axi_wdata                 <= 256'd0;
        axi_wstrb                 <= 32'd0;
        axi_araddr                <= {CTRL_ADDR_WIDTH{1'b0}};
        axi_arlen                 <= 4'd0;
        axi_arvalid               <= 1'b0;
        busy                      <= 1'b0;
        done                      <= 1'b0;
        error                     <= 1'b0;
        error_code                <= 8'd0;
        current_row               <= 13'd0;
    end else begin
        done       <= 1'b0;
        core_start <= 1'b0;

        case (state)
            ST_IDLE: begin
                axi_awvalid     <= 1'b0;
                axi_arvalid     <= 1'b0;
                aw_seen         <= 1'b0;
                w_seen          <= 1'b0;
                ar_seen         <= 1'b0;
                busy            <= 1'b0;
                current_row     <= 13'd0;
                if (start && !error) begin
                    if (!ddr_init_done) begin
                        error      <= 1'b1;
                        error_code <= ERR_DDR_NOT_READY;
                        state      <= ST_ERROR;
                    end else if (!config_consistent) begin
                        error      <= 1'b1;
                        error_code <= ERR_CONFIG;
                        state      <= ST_ERROR;
                    end else begin
                        m_rows_reg              <= cfg_m_rows;
                        k_blocks_reg            <= cfg_k_blocks;
                        groups_reg              <= cfg_groups;
                        act_beats_reg           <= cfg_act_beats;
                        weight_beats_reg        <= cfg_weight_beats_per_row;
                        scale_beats_reg         <= cfg_scale_beats_per_row;
                        has_bias_reg            <= cfg_has_bias;
                        act_addr_reg            <= cfg_act_addr;
                        weight_row_addr         <= cfg_weight_addr;
                        scale_row_addr          <= cfg_scale_addr;
                        bias_row_addr           <= cfg_bias_addr;
                        result_write_addr       <= cfg_result_addr;
                        act_read_base_beat      <= 8'd0;
                        weight_read_base_beat   <= 7'd0;
                        read_beat_index         <= 5'd0;
                        bias_row_cache          <= 256'd0;
                        result_beat_cache       <= 256'd0;
                        current_row             <= 13'd0;
                        busy                    <= 1'b1;
                        error_code              <= 8'd0;
                        state                   <= ST_SETUP_ACT_READ;
                    end
                end
            end

            ST_SETUP_ACT_READ: begin
                axi_araddr              <= act_addr_reg + (act_read_base_beat * 8);
                axi_arlen               <= next_act_burst_beats - 1'b1;
                axi_arvalid             <= 1'b1;
                ar_seen                 <= 1'b0;
                read_beat_index         <= 5'd0;
                active_read_burst_beats <= next_act_burst_beats;
                state                   <= ST_READ_ACT;
            end

            ST_READ_ACT: begin
                if (ar_handshake) begin
                    axi_arvalid <= 1'b0;
                    ar_seen     <= 1'b1;
                end
                if (read_data_handshake) begin
                    if (read_beat_index + 1'b1 == active_read_burst_beats) begin
                        ar_seen <= 1'b0;
                        if (act_read_base_beat + active_read_burst_beats == act_beats_reg) begin
                            weight_read_base_beat <= 7'd0;
                            state <= ST_SETUP_WEIGHT_READ;
                        end else begin
                            act_read_base_beat <=
                                act_read_base_beat + active_read_burst_beats;
                            state <= ST_SETUP_ACT_READ;
                        end
                    end else begin
                        read_beat_index <= read_beat_index + 1'b1;
                    end
                end
            end

            ST_SETUP_WEIGHT_READ: begin
                axi_araddr              <= weight_row_addr + (weight_read_base_beat * 8);
                axi_arlen               <= next_weight_burst_beats - 1'b1;
                axi_arvalid             <= 1'b1;
                ar_seen                 <= 1'b0;
                read_beat_index         <= 5'd0;
                active_read_burst_beats <= next_weight_burst_beats;
                state                   <= ST_READ_WEIGHT;
            end

            ST_READ_WEIGHT: begin
                if (ar_handshake) begin
                    axi_arvalid <= 1'b0;
                    ar_seen     <= 1'b1;
                end
                if (read_data_handshake) begin
                    if (read_beat_index + 1'b1 == active_read_burst_beats) begin
                        ar_seen <= 1'b0;
                        if (
                            weight_read_base_beat + active_read_burst_beats ==
                            weight_beats_reg
                        ) begin
                            weight_read_base_beat <= 7'd0;
                            state <= ST_SETUP_SCALE_READ;
                        end else begin
                            weight_read_base_beat <=
                                weight_read_base_beat + active_read_burst_beats;
                            state <= ST_SETUP_WEIGHT_READ;
                        end
                    end else begin
                        read_beat_index <= read_beat_index + 1'b1;
                    end
                end
            end

            ST_SETUP_SCALE_READ: begin
                axi_araddr      <= scale_row_addr;
                axi_arlen       <= scale_beats_reg - 1'b1;
                axi_arvalid     <= 1'b1;
                ar_seen         <= 1'b0;
                read_beat_index <= 5'd0;
                state           <= ST_READ_SCALE;
            end

            ST_READ_SCALE: begin
                if (ar_handshake) begin
                    axi_arvalid <= 1'b0;
                    ar_seen     <= 1'b1;
                end
                if (read_data_handshake) begin
                    if (read_beat_index + 1'b1 == scale_beats_reg) begin
                        ar_seen <= 1'b0;
                        if (has_bias_reg)
                            state <= ST_SETUP_BIAS_READ;
                        else begin
                            bias_row_cache <= 256'd0;
                            state <= ST_START_CORE;
                        end
                    end else begin
                        read_beat_index <= read_beat_index + 1'b1;
                    end
                end
            end

            ST_SETUP_BIAS_READ: begin
                axi_araddr      <= bias_row_addr;
                axi_arlen       <= 4'd0;
                axi_arvalid     <= 1'b1;
                ar_seen         <= 1'b0;
                read_beat_index <= 5'd0;
                state           <= ST_READ_BIAS;
            end

            ST_READ_BIAS: begin
                if (ar_handshake) begin
                    axi_arvalid <= 1'b0;
                    ar_seen     <= 1'b1;
                end
                if (read_data_handshake) begin
                    bias_row_cache <= axi_rdata;
                    ar_seen        <= 1'b0;
                    state          <= ST_START_CORE;
                end
            end

            ST_START_CORE: begin
                if (core_config_error) begin
                    error      <= 1'b1;
                    error_code <= ERR_CORE_CONFIG;
                    busy       <= 1'b0;
                    state      <= ST_ERROR;
                end else begin
                    core_start <= 1'b1;
                    state      <= ST_WAIT_CORE;
                end
            end

            ST_WAIT_CORE: begin
                if (core_config_error) begin
                    error      <= 1'b1;
                    error_code <= ERR_CORE_CONFIG;
                    busy       <= 1'b0;
                    state      <= ST_ERROR;
                end else if (core_done) begin
                    result_beat_cache[current_row[1:0]*64 +: 64] <= core_y_q28;
                    if (current_row[1:0] == 2'd3) begin
                        state <= ST_SETUP_RESULT_WRITE;
                    end else begin
                        current_row             <= current_row + 1'b1;
                        weight_read_base_beat   <= 7'd0;
                        weight_row_addr         <= weight_row_addr +
                            ({21'd0, weight_beats_reg} << 3);
                        scale_row_addr          <= scale_row_addr +
                            ({24'd0, scale_beats_reg} << 3);
                        if (has_bias_reg)
                            bias_row_addr <= bias_row_addr + 8;
                        state <= ST_SETUP_WEIGHT_READ;
                    end
                end
            end

            ST_SETUP_RESULT_WRITE: begin
                axi_awaddr  <= result_write_addr;
                axi_awvalid <= 1'b1;
                axi_wdata   <= result_beat_cache;
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
                if (write_data_handshake)
                    w_seen <= 1'b1;

                if ((aw_seen || aw_handshake) && (w_seen || write_data_handshake)) begin
                    axi_awvalid       <= 1'b0;
                    aw_seen           <= 1'b0;
                    w_seen            <= 1'b0;
                    result_beat_cache <= 256'd0;
                    if (current_row + 1'b1 == m_rows_reg) begin
                        state <= ST_FINISH;
                    end else begin
                        current_row             <= current_row + 1'b1;
                        weight_read_base_beat   <= 7'd0;
                        weight_row_addr         <= weight_row_addr +
                            ({21'd0, weight_beats_reg} << 3);
                        scale_row_addr          <= scale_row_addr +
                            ({24'd0, scale_beats_reg} << 3);
                        if (has_bias_reg)
                            bias_row_addr <= bias_row_addr + 8;
                        result_write_addr <= result_write_addr + 8;
                        state <= ST_SETUP_WEIGHT_READ;
                    end
                end
            end

            ST_FINISH: begin
                busy <= 1'b0;
                done <= 1'b1;
                state <= ST_IDLE;
            end

            ST_ERROR: begin
                // error/error_code 粘滞，必须 rst_n 后才能接受新 start。
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
