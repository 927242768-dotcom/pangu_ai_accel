`timescale 1ns/1ps

// G2 activation quantizer 的 DDR3 adapter。
// SOURCE_Q28=0: 16 个 Q6.10/beat；SOURCE_Q28=1: 4 个 Q28/beat。
// 输出统一为 32 个 INT8/beat。
module runtime_activation_quantizer_ctrl #(
    parameter integer SOURCE_Q28 = 0,
    parameter integer CTRL_ADDR_WIDTH = 28,
    parameter integer INDEX_WIDTH = 13,
    parameter integer MAX_BURST_BEATS = 16
)(
    input  wire                         clk,
    input  wire                         rst_n,
    input  wire                         ddr_init_done,
    input  wire                         start,
    input  wire [INDEX_WIDTH-1:0]       vector_length,
    input  wire [CTRL_ADDR_WIDTH-1:0]   source_ctrl_addr,
    input  wire [CTRL_ADDR_WIDTH-1:0]   activation_ctrl_addr,

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
    output wire                        all_zero,
    output wire [15:0]                 max_abs_q10,
    output wire [23:0]                 max_mantissa_binary32,
    output wire signed [9:0]           max_exponent_binary32,
    output wire [31:0]                 max_abs_binary32_bits,
    output wire [4:0]                  debug_state
);

localparam integer SOURCE_ELEMENT_BITS = SOURCE_Q28 ? 64 : 16;
localparam integer SOURCE_ELEMENTS_PER_BEAT = 256 / SOURCE_ELEMENT_BITS;
localparam integer SOURCE_LANE_WIDTH = SOURCE_Q28 ? 2 : 4;
localparam [4:0] MAX_BURST_BEATS_5 = MAX_BURST_BEATS;

localparam [4:0] ST_IDLE             = 5'd0;
localparam [4:0] ST_START_LOAD       = 5'd1;
localparam [4:0] ST_SETUP_READ       = 5'd2;
localparam [4:0] ST_CAPTURE_BURST    = 5'd3;
localparam [4:0] ST_SEND_BURST       = 5'd4;
localparam [4:0] ST_WAIT_LOADED      = 5'd5;
localparam [4:0] ST_START_QUANT      = 5'd6;
localparam [4:0] ST_COLLECT_OUTPUT   = 5'd7;
localparam [4:0] ST_SETUP_WRITE      = 5'd8;
localparam [4:0] ST_WRITE_OUTPUT     = 5'd9;
localparam [4:0] ST_WAIT_QUANT_DONE  = 5'd10;
localparam [4:0] ST_FINISH           = 5'd11;
localparam [4:0] ST_ERROR            = 5'd31;

localparam [7:0] ERR_DDR_NOT_READY = 8'h01;
localparam [7:0] ERR_CONFIG        = 8'h02;
localparam [7:0] ERR_QUANTIZER     = 8'h03;
localparam [7:0] ERR_INTERNAL      = 8'hff;

reg [4:0] state;
reg [INDEX_WIDTH-1:0] length_reg;
reg [10:0] total_source_beats;
reg [10:0] source_beat_base;
reg [4:0] active_burst_beats;
reg [4:0] capture_beat_index;
reg [4:0] process_beat_index;
reg [SOURCE_LANE_WIDTH-1:0] process_lane_index;
reg [INDEX_WIDTH-1:0] source_element_index;
reg [255:0] burst_buffer [0:MAX_BURST_BEATS-1];
reg [255:0] activation_buffer;
reg [4:0] activation_lane_index;
reg last_pack_pending;
reg [CTRL_ADDR_WIDTH-1:0] source_addr_reg;
reg [CTRL_ADDR_WIDTH-1:0] activation_write_addr;
reg aw_seen;
reg w_seen;
reg ar_seen;
reg quant_load_start;
reg quantize_start;
reg quant_done_seen;

wire [11:0] source_beats_remaining = total_source_beats - source_beat_base;
wire [4:0] next_burst_beats =
    (source_beats_remaining > MAX_BURST_BEATS)
        ? MAX_BURST_BEATS_5
        : source_beats_remaining[4:0];
wire config_valid =
    ((vector_length == 13'd896) || (vector_length == 13'd4864)) &&
    ((vector_length % SOURCE_ELEMENTS_PER_BEAT) == 0) &&
    ((vector_length & 13'h001f) == 0);

wire ar_handshake = axi_arvalid && axi_arready;
wire aw_handshake = axi_awvalid && axi_awready;
wire write_handshake = axi_wready && (aw_seen || aw_handshake);
wire read_handshake = axi_rvalid && (ar_seen || ar_handshake);

wire selected_source_ready;
wire selected_activation_valid;
wire signed [7:0] selected_activation_int8;
wire selected_activation_last;
wire selected_load_complete;
wire selected_quant_busy;
wire selected_quant_done;
wire selected_quant_error;
wire [7:0] selected_quant_error_code;
wire selected_all_zero;
wire [15:0] selected_max_abs_q10;
wire [23:0] selected_max_mantissa;
wire signed [9:0] selected_max_exponent;
wire [31:0] selected_max_bits;

wire source_valid_to_quant = (state == ST_SEND_BURST);
wire source_last_to_quant =
    source_valid_to_quant && (source_element_index + 1'b1 == length_reg);
wire activation_ready_to_quant = (state == ST_COLLECT_OUTPUT);
wire source_send_handshake = source_valid_to_quant && selected_source_ready;
wire activation_receive_handshake =
    selected_activation_valid && activation_ready_to_quant;

wire [255:0] selected_source_beat = burst_buffer[process_beat_index];
wire signed [15:0] selected_source_q10 =
    selected_source_beat[process_lane_index*16 +: 16];
wire signed [63:0] selected_source_q28 =
    selected_source_beat[process_lane_index*64 +: 64];

assign axi_awuser_ap = 1'b0;
assign axi_awuser_id = 4'h0;
assign axi_awlen = 4'h0;
assign axi_aruser_ap = 1'b0;
assign axi_aruser_id = 4'h0;
assign all_zero = selected_all_zero;
assign max_abs_q10 = selected_max_abs_q10;
assign max_mantissa_binary32 = selected_max_mantissa;
assign max_exponent_binary32 = selected_max_exponent;
assign max_abs_binary32_bits = selected_max_bits;
assign debug_state = state;

generate
    if (SOURCE_Q28 == 0) begin : gen_q10
        wire q10_activation_valid;
        wire signed [7:0] q10_activation_int8;
        wire q10_activation_last;
        wire q10_load_complete;
        wire q10_busy;
        wire q10_done;
        wire q10_error;
        wire [7:0] q10_error_code;
        wire q10_all_zero;
        wire [15:0] q10_max_abs;
        wire [23:0] q10_max_mantissa;
        wire signed [9:0] q10_max_exponent;

        runtime_q10_activation_quantizer u_quantizer (
            .clk                     (clk),
            .rst_n                   (rst_n),
            .load_start              (quant_load_start),
            .vector_length           (length_reg),
            .source_valid            (source_valid_to_quant),
            .source_ready            (selected_source_ready),
            .source_q10              (selected_source_q10),
            .source_last             (source_last_to_quant),
            .quantize_start          (quantize_start),
            .activation_valid        (q10_activation_valid),
            .activation_ready        (activation_ready_to_quant),
            .activation_index        (),
            .activation_int8         (q10_activation_int8),
            .activation_last         (q10_activation_last),
            .max_abs_q10             (q10_max_abs),
            .max_mantissa_binary32   (q10_max_mantissa),
            .max_exponent_binary32   (q10_max_exponent),
            .all_zero                (q10_all_zero),
            .load_complete           (q10_load_complete),
            .busy                    (q10_busy),
            .done                    (q10_done),
            .error                   (q10_error),
            .error_code              (q10_error_code),
            .debug_state             ()
        );

        assign selected_activation_valid = q10_activation_valid;
        assign selected_activation_int8 = q10_activation_int8;
        assign selected_activation_last = q10_activation_last;
        assign selected_load_complete = q10_load_complete;
        assign selected_quant_busy = q10_busy;
        assign selected_quant_done = q10_done;
        assign selected_quant_error = q10_error;
        assign selected_quant_error_code = q10_error_code;
        assign selected_all_zero = q10_all_zero;
        assign selected_max_abs_q10 = q10_max_abs;
        assign selected_max_mantissa = q10_max_mantissa;
        assign selected_max_exponent = q10_max_exponent;
        assign selected_max_bits = 32'd0;
    end else begin : gen_q28
        wire q28_activation_valid;
        wire signed [7:0] q28_activation_int8;
        wire q28_activation_last;
        wire q28_load_complete;
        wire q28_busy;
        wire q28_done;
        wire q28_error;
        wire [7:0] q28_error_code;
        wire q28_all_zero;
        wire [23:0] q28_max_mantissa;
        wire signed [9:0] q28_max_exponent;
        wire [31:0] q28_max_bits;

        runtime_q28_activation_quantizer u_quantizer (
            .clk                     (clk),
            .rst_n                   (rst_n),
            .load_start              (quant_load_start),
            .vector_length           (length_reg),
            .source_valid            (source_valid_to_quant),
            .source_ready            (selected_source_ready),
            .source_q28              (selected_source_q28),
            .source_last             (source_last_to_quant),
            .quantize_start          (quantize_start),
            .activation_valid        (q28_activation_valid),
            .activation_ready        (activation_ready_to_quant),
            .activation_index        (),
            .activation_int8         (q28_activation_int8),
            .activation_last         (q28_activation_last),
            .max_mantissa_binary32   (q28_max_mantissa),
            .max_exponent_binary32   (q28_max_exponent),
            .max_abs_binary32_bits   (q28_max_bits),
            .all_zero                (q28_all_zero),
            .load_complete           (q28_load_complete),
            .busy                    (q28_busy),
            .done                    (q28_done),
            .error                   (q28_error),
            .error_code              (q28_error_code),
            .debug_state             ()
        );

        assign selected_activation_valid = q28_activation_valid;
        assign selected_activation_int8 = q28_activation_int8;
        assign selected_activation_last = q28_activation_last;
        assign selected_load_complete = q28_load_complete;
        assign selected_quant_busy = q28_busy;
        assign selected_quant_done = q28_done;
        assign selected_quant_error = q28_error;
        assign selected_quant_error_code = q28_error_code;
        assign selected_all_zero = q28_all_zero;
        assign selected_max_abs_q10 = 16'd0;
        assign selected_max_mantissa = q28_max_mantissa;
        assign selected_max_exponent = q28_max_exponent;
        assign selected_max_bits = q28_max_bits;
    end
endgenerate

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        state                   <= ST_IDLE;
        length_reg              <= {INDEX_WIDTH{1'b0}};
        total_source_beats      <= 11'd0;
        source_beat_base        <= 11'd0;
        active_burst_beats      <= 5'd0;
        capture_beat_index      <= 5'd0;
        process_beat_index      <= 5'd0;
        process_lane_index      <= {SOURCE_LANE_WIDTH{1'b0}};
        source_element_index    <= {INDEX_WIDTH{1'b0}};
        activation_buffer       <= 256'd0;
        activation_lane_index   <= 5'd0;
        last_pack_pending       <= 1'b0;
        source_addr_reg         <= {CTRL_ADDR_WIDTH{1'b0}};
        activation_write_addr   <= {CTRL_ADDR_WIDTH{1'b0}};
        aw_seen                 <= 1'b0;
        w_seen                  <= 1'b0;
        ar_seen                 <= 1'b0;
        quant_load_start        <= 1'b0;
        quantize_start          <= 1'b0;
        quant_done_seen         <= 1'b0;
        axi_awaddr              <= {CTRL_ADDR_WIDTH{1'b0}};
        axi_awvalid             <= 1'b0;
        axi_wdata               <= 256'd0;
        axi_wstrb               <= 32'd0;
        axi_araddr              <= {CTRL_ADDR_WIDTH{1'b0}};
        axi_arlen               <= 4'd0;
        axi_arvalid             <= 1'b0;
        busy                    <= 1'b0;
        done                    <= 1'b0;
        error                   <= 1'b0;
        error_code              <= 8'd0;
    end else begin
        quant_load_start <= 1'b0;
        quantize_start   <= 1'b0;
        done             <= 1'b0;
        if (selected_quant_done)
            quant_done_seen <= 1'b1;

        if (selected_quant_error && state != ST_ERROR) begin
            error       <= 1'b1;
            error_code  <= ERR_QUANTIZER + selected_quant_error_code;
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
                        end else if (!config_valid) begin
                            error      <= 1'b1;
                            error_code <= ERR_CONFIG;
                            state      <= ST_ERROR;
                        end else begin
                            length_reg            <= vector_length;
                            total_source_beats    <= vector_length / SOURCE_ELEMENTS_PER_BEAT;
                            source_beat_base      <= 11'd0;
                            source_element_index  <= {INDEX_WIDTH{1'b0}};
                            activation_buffer     <= 256'd0;
                            activation_lane_index <= 5'd0;
                            last_pack_pending     <= 1'b0;
                            source_addr_reg       <= source_ctrl_addr;
                            activation_write_addr <= activation_ctrl_addr;
                            quant_done_seen       <= 1'b0;
                            error_code            <= 8'd0;
                            busy                  <= 1'b1;
                            state                 <= ST_START_LOAD;
                        end
                    end
                end

                ST_START_LOAD: begin
                    quant_load_start <= 1'b1;
                    state            <= ST_SETUP_READ;
                end

                ST_SETUP_READ: begin
                    axi_araddr         <= source_addr_reg + (source_beat_base << 3);
                    axi_arlen          <= next_burst_beats - 1'b1;
                    axi_arvalid        <= 1'b1;
                    ar_seen            <= 1'b0;
                    active_burst_beats <= next_burst_beats;
                    capture_beat_index <= 5'd0;
                    state              <= ST_CAPTURE_BURST;
                end

                ST_CAPTURE_BURST: begin
                    if (ar_handshake) begin
                        axi_arvalid <= 1'b0;
                        ar_seen     <= 1'b1;
                    end
                    if (read_handshake) begin
                        burst_buffer[capture_beat_index] <= axi_rdata;
                        if (capture_beat_index + 1'b1 == active_burst_beats) begin
                            ar_seen            <= 1'b0;
                            process_beat_index <= 5'd0;
                            process_lane_index <= {SOURCE_LANE_WIDTH{1'b0}};
                            state              <= ST_SEND_BURST;
                        end else begin
                            capture_beat_index <= capture_beat_index + 1'b1;
                        end
                    end
                end

                ST_SEND_BURST: begin
                    if (source_send_handshake) begin
                        source_element_index <= source_element_index + 1'b1;
                        if (source_element_index + 1'b1 == length_reg) begin
                            state <= ST_WAIT_LOADED;
                        end else if (process_lane_index + 1'b1 == SOURCE_ELEMENTS_PER_BEAT) begin
                            process_lane_index <= {SOURCE_LANE_WIDTH{1'b0}};
                            if (process_beat_index + 1'b1 == active_burst_beats) begin
                                source_beat_base <= source_beat_base + active_burst_beats;
                                state            <= ST_SETUP_READ;
                            end else begin
                                process_beat_index <= process_beat_index + 1'b1;
                            end
                        end else begin
                            process_lane_index <= process_lane_index + 1'b1;
                        end
                    end
                end

                ST_WAIT_LOADED: begin
                    if (selected_load_complete)
                        state <= ST_START_QUANT;
                end

                ST_START_QUANT: begin
                    quantize_start <= 1'b1;
                    state          <= ST_COLLECT_OUTPUT;
                end

                ST_COLLECT_OUTPUT: begin
                    if (activation_receive_handshake) begin
                        activation_buffer[activation_lane_index*8 +: 8]
                            <= selected_activation_int8;
                        if (activation_lane_index == 5'd31) begin
                            activation_lane_index <= 5'd0;
                            last_pack_pending     <= selected_activation_last;
                            state                 <= ST_SETUP_WRITE;
                        end else begin
                            activation_lane_index <= activation_lane_index + 1'b1;
                        end
                    end
                end

                ST_SETUP_WRITE: begin
                    axi_awaddr  <= activation_write_addr;
                    axi_awvalid <= 1'b1;
                    axi_wdata   <= activation_buffer;
                    axi_wstrb   <= 32'hffff_ffff;
                    aw_seen     <= 1'b0;
                    w_seen      <= 1'b0;
                    state       <= ST_WRITE_OUTPUT;
                end

                ST_WRITE_OUTPUT: begin
                    if (aw_handshake) begin
                        axi_awvalid <= 1'b0;
                        aw_seen     <= 1'b1;
                    end
                    if (write_handshake)
                        w_seen <= 1'b1;
                    if ((aw_seen || aw_handshake) && (w_seen || write_handshake)) begin
                        axi_awvalid         <= 1'b0;
                        aw_seen             <= 1'b0;
                        w_seen              <= 1'b0;
                        activation_buffer   <= 256'd0;
                        activation_write_addr <= activation_write_addr + 8;
                        if (last_pack_pending) begin
                            last_pack_pending <= 1'b0;
                            state <= ST_WAIT_QUANT_DONE;
                        end else begin
                            state <= ST_COLLECT_OUTPUT;
                        end
                    end
                end

                ST_WAIT_QUANT_DONE: begin
                    if (quant_done_seen || selected_quant_done)
                        state <= ST_FINISH;
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
