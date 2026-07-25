`timescale 1ns/1ps

// G2 运行时量化 controller 的 AXI 访问契约检查器。
//
// 在量化 controller 独占 DDR3 用户口期间，逐条验证：
//   source: 最多 16 beat 的连续 read burst；
//   raw scale: 单 beat 连续读取；
//   activation: 单 beat 连续写入；
//   combined scale: 单 beat 连续写入。
//
// 地址单位与 DDR3 controller 一致，均为 32 bit；一个 256 bit beat 递增 8。
module runtime_quantizer_trace_checker #(
    parameter integer CTRL_ADDR_WIDTH = 28
)(
    input  wire                         clk,
    input  wire                         rst_n,
    input  wire                         start,
    input  wire                         finish,
    input  wire                         source_q28,
    input  wire [12:0]                  vector_length,
    input  wire [12:0]                  rows,
    input  wire [6:0]                   groups,
    input  wire [CTRL_ADDR_WIDTH-1:0]   source_ctrl_addr,
    input  wire [CTRL_ADDR_WIDTH-1:0]   activation_ctrl_addr,
    input  wire [CTRL_ADDR_WIDTH-1:0]   raw_scale_ctrl_addr,
    input  wire [CTRL_ADDR_WIDTH-1:0]   combined_scale_ctrl_addr,

    input  wire [CTRL_ADDR_WIDTH-1:0]   axi_araddr,
    input  wire [3:0]                   axi_arlen,
    input  wire                         axi_ar_handshake,
    input  wire                         axi_rvalid,
    input  wire [CTRL_ADDR_WIDTH-1:0]   axi_awaddr,
    input  wire [3:0]                   axi_awlen,
    input  wire                         axi_aw_handshake,
    input  wire                         axi_wready,

    output reg                          error,
    output reg  [7:0]                   error_code,
    output reg                          complete,
    output reg  [31:0]                  source_read_commands,
    output reg  [31:0]                  source_read_beats,
    output reg  [31:0]                  raw_scale_read_commands,
    output reg  [31:0]                  raw_scale_read_beats,
    output reg  [31:0]                  activation_write_commands,
    output reg  [31:0]                  activation_write_beats,
    output reg  [31:0]                  combined_write_commands,
    output reg  [31:0]                  combined_write_beats
);

localparam [7:0] ERR_READ_OVERLAP       = 8'h01;
localparam [7:0] ERR_SOURCE_READ_ADDR   = 8'h02;
localparam [7:0] ERR_SOURCE_READ_LEN    = 8'h03;
localparam [7:0] ERR_RAW_READ_ADDR      = 8'h04;
localparam [7:0] ERR_RAW_READ_LEN       = 8'h05;
localparam [7:0] ERR_EXTRA_READ         = 8'h06;
localparam [7:0] ERR_READ_DATA_ORDER    = 8'h07;
localparam [7:0] ERR_WRITE_OVERLAP      = 8'h08;
localparam [7:0] ERR_ACT_WRITE_ADDR     = 8'h09;
localparam [7:0] ERR_ACT_WRITE_LEN      = 8'h0a;
localparam [7:0] ERR_SCALE_WRITE_ADDR   = 8'h0b;
localparam [7:0] ERR_SCALE_WRITE_LEN    = 8'h0c;
localparam [7:0] ERR_EXTRA_WRITE        = 8'h0d;
localparam [7:0] ERR_WRITE_DATA_ORDER   = 8'h0e;
localparam [7:0] ERR_FINISH_COUNTS      = 8'h0f;
localparam [7:0] ERR_INTERNAL           = 8'hff;

reg [31:0] expected_source_read_commands;
reg [31:0] expected_source_read_beats;
reg [31:0] expected_raw_scale_read_commands;
reg [31:0] expected_raw_scale_read_beats;
reg [31:0] expected_activation_write_commands;
reg [31:0] expected_activation_write_beats;
reg [31:0] expected_combined_write_commands;
reg [31:0] expected_combined_write_beats;

reg [CTRL_ADDR_WIDTH-1:0] next_source_addr;
reg [CTRL_ADDR_WIDTH-1:0] next_raw_scale_addr;
reg [CTRL_ADDR_WIDTH-1:0] next_activation_addr;
reg [CTRL_ADDR_WIDTH-1:0] next_combined_scale_addr;
reg [4:0] active_read_beats;
reg active_read_source;
reg write_pending;
reg write_pending_activation;

wire [31:0] source_beats_wire = source_q28
    ? {19'd0, vector_length} >> 2
    : {19'd0, vector_length} >> 4;
wire [19:0] raw_scale_values_wire = rows * groups;
wire [31:0] raw_scale_beats_wire = {12'd0, raw_scale_values_wire} >> 4;
wire [31:0] activation_beats_wire = {19'd0, vector_length} >> 5;
wire [4:0] padded_scale_beats_per_row_wire =
    (groups == 7'd14) ? 5'd2 :
    (groups == 7'd76) ? 5'd10 : 5'd0;
wire [31:0] combined_beats_wire = rows * padded_scale_beats_per_row_wire;
wire [31:0] source_commands_wire = (source_beats_wire + 32'd15) >> 4;

wire source_phase = source_read_commands < expected_source_read_commands;
wire raw_phase = !source_phase &&
    (raw_scale_read_commands < expected_raw_scale_read_commands);
wire [31:0] source_beats_remaining =
    expected_source_read_beats - source_read_beats;
wire [4:0] expected_source_burst_beats =
    (source_beats_remaining > 32'd16)
        ? 5'd16 : source_beats_remaining[4:0];
wire accepted_read_is_source = source_phase;
wire accepted_read_is_raw = raw_phase;

wire activation_write_phase =
    activation_write_commands < expected_activation_write_commands;
wire combined_write_phase = !activation_write_phase &&
    (combined_write_commands < expected_combined_write_commands);
wire accepted_write_is_activation = activation_write_phase;
wire accepted_write_is_combined = combined_write_phase;
wire axi_w_handshake = axi_wready && (write_pending || axi_aw_handshake);

wire counters_complete =
    (source_read_commands == expected_source_read_commands) &&
    (source_read_beats == expected_source_read_beats) &&
    (raw_scale_read_commands == expected_raw_scale_read_commands) &&
    (raw_scale_read_beats == expected_raw_scale_read_beats) &&
    (activation_write_commands == expected_activation_write_commands) &&
    (activation_write_beats == expected_activation_write_beats) &&
    (combined_write_commands == expected_combined_write_commands) &&
    (combined_write_beats == expected_combined_write_beats) &&
    (active_read_beats == 5'd0) && !write_pending;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        expected_source_read_commands   <= 32'd0;
        expected_source_read_beats      <= 32'd0;
        expected_raw_scale_read_commands<= 32'd0;
        expected_raw_scale_read_beats   <= 32'd0;
        expected_activation_write_commands <= 32'd0;
        expected_activation_write_beats <= 32'd0;
        expected_combined_write_commands<= 32'd0;
        expected_combined_write_beats   <= 32'd0;
        next_source_addr                <= {CTRL_ADDR_WIDTH{1'b0}};
        next_raw_scale_addr             <= {CTRL_ADDR_WIDTH{1'b0}};
        next_activation_addr            <= {CTRL_ADDR_WIDTH{1'b0}};
        next_combined_scale_addr        <= {CTRL_ADDR_WIDTH{1'b0}};
        active_read_beats               <= 5'd0;
        active_read_source              <= 1'b0;
        write_pending                   <= 1'b0;
        write_pending_activation        <= 1'b0;
        error                           <= 1'b0;
        error_code                      <= 8'd0;
        complete                        <= 1'b0;
        source_read_commands            <= 32'd0;
        source_read_beats               <= 32'd0;
        raw_scale_read_commands         <= 32'd0;
        raw_scale_read_beats            <= 32'd0;
        activation_write_commands       <= 32'd0;
        activation_write_beats          <= 32'd0;
        combined_write_commands         <= 32'd0;
        combined_write_beats            <= 32'd0;
    end else begin
        complete <= 1'b0;

        if (start) begin
            expected_source_read_commands    <= source_commands_wire;
            expected_source_read_beats       <= source_beats_wire;
            expected_raw_scale_read_commands <= raw_scale_beats_wire;
            expected_raw_scale_read_beats    <= raw_scale_beats_wire;
            expected_activation_write_commands <= activation_beats_wire;
            expected_activation_write_beats  <= activation_beats_wire;
            expected_combined_write_commands <= combined_beats_wire;
            expected_combined_write_beats    <= combined_beats_wire;
            next_source_addr                 <= source_ctrl_addr;
            next_raw_scale_addr              <= raw_scale_ctrl_addr;
            next_activation_addr             <= activation_ctrl_addr;
            next_combined_scale_addr         <= combined_scale_ctrl_addr;
            active_read_beats                <= 5'd0;
            active_read_source               <= 1'b0;
            write_pending                    <= 1'b0;
            write_pending_activation         <= 1'b0;
            error                            <= 1'b0;
            error_code                       <= 8'd0;
            source_read_commands             <= 32'd0;
            source_read_beats                <= 32'd0;
            raw_scale_read_commands          <= 32'd0;
            raw_scale_read_beats             <= 32'd0;
            activation_write_commands        <= 32'd0;
            activation_write_beats           <= 32'd0;
            combined_write_commands          <= 32'd0;
            combined_write_beats             <= 32'd0;
        end else if (!error) begin
            if (axi_ar_handshake) begin
                if (active_read_beats != 5'd0) begin
                    error      <= 1'b1;
                    error_code <= ERR_READ_OVERLAP;
                end else if (accepted_read_is_source) begin
                    if (axi_araddr != next_source_addr) begin
                        error      <= 1'b1;
                        error_code <= ERR_SOURCE_READ_ADDR;
                    end else if (axi_arlen + 1'b1 != expected_source_burst_beats) begin
                        error      <= 1'b1;
                        error_code <= ERR_SOURCE_READ_LEN;
                    end else begin
                        source_read_commands <= source_read_commands + 1'b1;
                        next_source_addr <= next_source_addr +
                            (({1'b0, axi_arlen} + 1'b1) << 3);
                        active_read_beats  <= {1'b0, axi_arlen} + 1'b1;
                        active_read_source <= 1'b1;
                    end
                end else if (accepted_read_is_raw) begin
                    if (axi_araddr != next_raw_scale_addr) begin
                        error      <= 1'b1;
                        error_code <= ERR_RAW_READ_ADDR;
                    end else if (axi_arlen != 4'd0) begin
                        error      <= 1'b1;
                        error_code <= ERR_RAW_READ_LEN;
                    end else begin
                        raw_scale_read_commands <= raw_scale_read_commands + 1'b1;
                        next_raw_scale_addr      <= next_raw_scale_addr + 8;
                        active_read_beats        <= 5'd1;
                        active_read_source       <= 1'b0;
                    end
                end else begin
                    error      <= 1'b1;
                    error_code <= ERR_EXTRA_READ;
                end
            end

            if (axi_rvalid) begin
                if (active_read_beats != 5'd0) begin
                    active_read_beats <= active_read_beats - 1'b1;
                    if (active_read_source)
                        source_read_beats <= source_read_beats + 1'b1;
                    else
                        raw_scale_read_beats <= raw_scale_read_beats + 1'b1;
                end else if (axi_ar_handshake && accepted_read_is_source) begin
                    active_read_beats <= {1'b0, axi_arlen};
                    source_read_beats <= source_read_beats + 1'b1;
                end else if (axi_ar_handshake && accepted_read_is_raw) begin
                    active_read_beats <= 5'd0;
                    raw_scale_read_beats <= raw_scale_read_beats + 1'b1;
                end else begin
                    error      <= 1'b1;
                    error_code <= ERR_READ_DATA_ORDER;
                end
            end

            if (axi_aw_handshake) begin
                if (write_pending) begin
                    error      <= 1'b1;
                    error_code <= ERR_WRITE_OVERLAP;
                end else if (accepted_write_is_activation) begin
                    if (axi_awaddr != next_activation_addr) begin
                        error      <= 1'b1;
                        error_code <= ERR_ACT_WRITE_ADDR;
                    end else if (axi_awlen != 4'd0) begin
                        error      <= 1'b1;
                        error_code <= ERR_ACT_WRITE_LEN;
                    end else begin
                        activation_write_commands <= activation_write_commands + 1'b1;
                        next_activation_addr       <= next_activation_addr + 8;
                        write_pending              <= 1'b1;
                        write_pending_activation   <= 1'b1;
                    end
                end else if (accepted_write_is_combined) begin
                    if (axi_awaddr != next_combined_scale_addr) begin
                        error      <= 1'b1;
                        error_code <= ERR_SCALE_WRITE_ADDR;
                    end else if (axi_awlen != 4'd0) begin
                        error      <= 1'b1;
                        error_code <= ERR_SCALE_WRITE_LEN;
                    end else begin
                        combined_write_commands <= combined_write_commands + 1'b1;
                        next_combined_scale_addr <= next_combined_scale_addr + 8;
                        write_pending            <= 1'b1;
                        write_pending_activation <= 1'b0;
                    end
                end else begin
                    error      <= 1'b1;
                    error_code <= ERR_EXTRA_WRITE;
                end
            end

            if (axi_w_handshake) begin
                if (write_pending) begin
                    write_pending <= 1'b0;
                    if (write_pending_activation)
                        activation_write_beats <= activation_write_beats + 1'b1;
                    else
                        combined_write_beats <= combined_write_beats + 1'b1;
                end else if (axi_aw_handshake && accepted_write_is_activation) begin
                    write_pending <= 1'b0;
                    activation_write_beats <= activation_write_beats + 1'b1;
                end else if (axi_aw_handshake && accepted_write_is_combined) begin
                    write_pending <= 1'b0;
                    combined_write_beats <= combined_write_beats + 1'b1;
                end else begin
                    error      <= 1'b1;
                    error_code <= ERR_WRITE_DATA_ORDER;
                end
            end

            if (finish) begin
                if (counters_complete)
                    complete <= 1'b1;
                else begin
                    error      <= 1'b1;
                    error_code <= ERR_FINISH_COUNTS;
                end
            end
        end
    end
end

wire _unused_internal = &{1'b0, ERR_INTERNAL};

endmodule
