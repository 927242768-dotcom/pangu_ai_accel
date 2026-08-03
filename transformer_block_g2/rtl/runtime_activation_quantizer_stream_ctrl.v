`timescale 1ns/1ps

// G2 完整 Block 的两遍 DDR3 流式 activation quantizer。
// 第一遍求最大绝对值，第二遍逐元素生成 INT8；不缓存完整 4864×64 源向量。
module runtime_activation_quantizer_stream_ctrl #(
    parameter integer CTRL_ADDR_WIDTH = 28,
    parameter integer INDEX_WIDTH = 13,
    parameter integer MAX_BURST_BEATS = 16
)(
    input  wire                         clk,
    input  wire                         rst_n,
    input  wire                         ddr_init_done,
    input  wire                         start,
    input  wire                         source_q28,
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
    output reg                          all_zero,
    output reg  [15:0]                  max_abs_q10,
    output reg  [23:0]                  max_mantissa_binary32,
    output reg  signed [9:0]            max_exponent_binary32,
    output reg  [31:0]                  max_abs_binary32_bits,
    output wire [4:0]                   debug_state
);

localparam [4:0] MAX_BURST_BEATS_5 = MAX_BURST_BEATS;
localparam [5:0] ST_IDLE=0, ST_P1_SETUP=1, ST_P1_CAPTURE=2, ST_P1_SCAN=3,
    ST_P1_UPDATE=4, ST_MAX_START=5, ST_MAX_WAIT=6, ST_P2_SETUP=7,
    ST_P2_CAPTURE=8, ST_P2_SCAN=9, ST_SRC_START=10, ST_SRC_WAIT=11,
    ST_RATIO_PREP=12, ST_RATIO_SHIFT=13, ST_DIV_START=14, ST_DIV_WAIT=15,
    ST_PACK=16, ST_WRITE_SETUP=17, ST_WRITE=18, ST_FINISH=19, ST_ERROR=63;
localparam [7:0] ERR_DDR_NOT_READY=8'h01, ERR_CONFIG=8'h02,
    ERR_CONVERTER=8'h03, ERR_EXPONENT=8'h04, ERR_DIVIDER=8'h05,
    ERR_INTERNAL=8'hff;

reg [5:0] state;
reg source_q28_reg;
reg [INDEX_WIDTH-1:0] length_reg;
reg [CTRL_ADDR_WIDTH-1:0] source_addr_reg;
reg [CTRL_ADDR_WIDTH-1:0] activation_write_addr;
reg [10:0] total_source_beats, source_beat_base;
reg [4:0] active_burst_beats, capture_beat_index, scan_beat_index;
reg [3:0] scan_lane_index;
reg [INDEX_WIDTH-1:0] element_index;
reg [255:0] burst_buffer [0:MAX_BURST_BEATS-1];
reg signed [63:0] source_work_reg;
reg [63:0] max_abs_raw;
reg source_sign_reg, source_zero_reg;
reg [23:0] source_mantissa_reg;
reg signed [9:0] source_exponent_reg;
reg signed [7:0] quant_result_reg;
reg [255:0] activation_buffer;
reg [4:0] activation_lane_index;
reg last_pack_pending, resume_needs_read;
reg ar_seen, aw_seen, w_seen;
reg [95:0] divider_numerator_reg, divider_denominator_reg;
reg [95:0] denominator_work_reg;
reg [6:0] denominator_shift_count;
reg divider_start;

wire [4:0] elements_per_beat = source_q28_reg ? 5'd4 : 5'd16;
wire [11:0] beats_remaining = total_source_beats - source_beat_base;
wire [4:0] next_burst_beats = beats_remaining > MAX_BURST_BEATS
    ? MAX_BURST_BEATS_5 : beats_remaining[4:0];
wire config_valid = ((vector_length == 13'd896) ||
                     (vector_length == 13'd4864)) &&
                    (source_q28 ? vector_length[1:0] == 0
                                : vector_length[3:0] == 0) &&
                    ((vector_length & 13'h001f) == 0);
wire ar_handshake = axi_arvalid && axi_arready;
wire aw_handshake = axi_awvalid && axi_awready;
wire w_handshake = axi_wready && (aw_seen || aw_handshake);
wire write_complete = (aw_seen || aw_handshake) && (w_seen || w_handshake);
wire read_handshake = axi_rvalid && (ar_seen || ar_handshake);

wire [255:0] selected_beat = burst_buffer[scan_beat_index];
wire signed [15:0] selected_q10 = selected_beat[scan_lane_index*16 +: 16];
wire signed [63:0] q10_extended = {{48{selected_q10[15]}}, selected_q10};
wire signed [63:0] selected_q28 = selected_beat[scan_lane_index[1:0]*64 +: 64];
wire signed [63:0] selected_source = source_q28_reg
    ? selected_q28 : (q10_extended <<< 18);
wire [63:0] source_magnitude = source_work_reg[63]
    ? (~source_work_reg[63:0] + 1'b1) : source_work_reg[63:0];
wire [63:0] max_candidate = source_magnitude > max_abs_raw
    ? source_magnitude : max_abs_raw;
wire last_element = element_index + 1'b1 == length_reg;
wire last_source_lane = scan_lane_index + 1'b1 == elements_per_beat;
wire last_burst_beat = scan_beat_index + 1'b1 == active_burst_beats;

wire converter_start = state == ST_MAX_START || state == ST_SRC_START;
wire signed [63:0] converter_input = state == ST_MAX_START
    ? $signed(max_abs_raw) : source_work_reg;
wire converter_busy, converter_done, converter_sign, converter_zero;
wire [23:0] converter_mantissa;
wire signed [9:0] converter_exponent;
wire [31:0] converter_bits;
q28_to_binary32_sequential u_converter (
    .clk(clk), .rst_n(rst_n), .start(converter_start),
    .q28_value(converter_input), .busy(converter_busy), .done(converter_done),
    .sign(converter_sign), .zero(converter_zero), .mantissa(converter_mantissa),
    .component_exponent(converter_exponent), .binary32_bits(converter_bits)
);

wire signed [10:0] exponent_difference_signed =
    $signed(max_exponent_binary32) - $signed(source_exponent_reg);
wire exponent_difference_negative = exponent_difference_signed[10];
wire [10:0] exponent_difference = exponent_difference_signed[10:0];
wire exponent_difference_too_large = exponent_difference >= 11'd72;
wire [30:0] source_mantissa_extended = {7'd0, source_mantissa_reg};
wire [30:0] ratio_numerator_base =
    (source_mantissa_extended << 7) - source_mantissa_extended;
wire [95:0] ratio_denominator_base = {{72{1'b0}}, max_mantissa_binary32};
wire divider_busy, divider_done, divider_divide_by_zero, divider_overflow;
wire [95:0] divider_quotient, divider_remainder;
unsigned_divider_rne #(.WIDTH(96)) u_divider (
    .clk(clk), .rst_n(rst_n), .start(divider_start),
    .numerator(divider_numerator_reg), .denominator(divider_denominator_reg),
    .busy(divider_busy), .done(divider_done),
    .divide_by_zero(divider_divide_by_zero), .overflow(divider_overflow),
    .quotient(divider_quotient), .remainder(divider_remainder)
);

assign axi_awuser_ap=1'b0; assign axi_awuser_id=4'h0; assign axi_awlen=4'h0;
assign axi_aruser_ap=1'b0; assign axi_aruser_id=4'h0;
assign debug_state=state[4:0];

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        state<=ST_IDLE; source_q28_reg<=0; length_reg<=0; source_addr_reg<=0;
        activation_write_addr<=0; total_source_beats<=0; source_beat_base<=0;
        active_burst_beats<=0; capture_beat_index<=0; scan_beat_index<=0;
        scan_lane_index<=0; element_index<=0; source_work_reg<=0; max_abs_raw<=0;
        source_sign_reg<=0; source_zero_reg<=1; source_mantissa_reg<=0;
        source_exponent_reg<=0; quant_result_reg<=0; activation_buffer<=0;
        activation_lane_index<=0; last_pack_pending<=0; resume_needs_read<=0;
        ar_seen<=0; aw_seen<=0; w_seen<=0; divider_numerator_reg<=0;
        divider_denominator_reg<=0; denominator_work_reg<=0;
        denominator_shift_count<=0; divider_start<=0;
        axi_awaddr<=0; axi_awvalid<=0; axi_wdata<=0; axi_wstrb<=0;
        axi_araddr<=0; axi_arlen<=0; axi_arvalid<=0;
        busy<=0; done<=0; error<=0; error_code<=0; all_zero<=1;
        max_abs_q10<=0; max_mantissa_binary32<=0; max_exponent_binary32<=0;
        max_abs_binary32_bits<=0;
    end else begin
        done<=0;
        divider_start<=0;
        case (state)
            ST_IDLE: begin
                busy<=0; axi_awvalid<=0; axi_arvalid<=0;
                ar_seen<=0; aw_seen<=0; w_seen<=0;
                if (start && !error) begin
                    if (!ddr_init_done) begin
                        error<=1; error_code<=ERR_DDR_NOT_READY; state<=ST_ERROR;
                    end else if (!config_valid) begin
                        error<=1; error_code<=ERR_CONFIG; state<=ST_ERROR;
                    end else begin
                        source_q28_reg<=source_q28;
                        length_reg<=vector_length;
                        source_addr_reg<=source_ctrl_addr;
                        activation_write_addr<=activation_ctrl_addr;
                        total_source_beats<=source_q28 ?
                            (vector_length >> 2) : (vector_length >> 4);
                        source_beat_base<=0; capture_beat_index<=0;
                        scan_beat_index<=0; scan_lane_index<=0; element_index<=0;
                        max_abs_raw<=0; all_zero<=1; max_abs_q10<=0;
                        max_mantissa_binary32<=0; max_exponent_binary32<=0;
                        max_abs_binary32_bits<=0; activation_buffer<=0;
                        activation_lane_index<=0; last_pack_pending<=0;
                        resume_needs_read<=0; error_code<=0; busy<=1;
                        state<=ST_P1_SETUP;
                    end
                end
            end

            ST_P1_SETUP: begin
                axi_araddr<=source_addr_reg + (source_beat_base << 3);
                axi_arlen<=next_burst_beats - 1'b1;
                axi_arvalid<=1; ar_seen<=0; capture_beat_index<=0;
                active_burst_beats<=next_burst_beats; state<=ST_P1_CAPTURE;
            end

            ST_P1_CAPTURE: begin
                if (ar_handshake) begin axi_arvalid<=0; ar_seen<=1; end
                if (read_handshake) begin
                    burst_buffer[capture_beat_index]<=axi_rdata;
                    if (capture_beat_index + 1'b1 == active_burst_beats) begin
                        ar_seen<=0; scan_beat_index<=0; scan_lane_index<=0;
                        state<=ST_P1_SCAN;
                    end else capture_beat_index<=capture_beat_index + 1'b1;
                end
            end

            ST_P1_SCAN: begin
                source_work_reg<=selected_source;
                state<=ST_P1_UPDATE;
            end

            ST_P1_UPDATE: begin
                max_abs_raw<=max_candidate;
                if (source_magnitude != 0) all_zero<=0;
                if (last_element) begin
                    state<=ST_MAX_START;
                end else begin
                    element_index<=element_index + 1'b1;
                    if (last_source_lane) begin
                        scan_lane_index<=0;
                        if (last_burst_beat) begin
                            source_beat_base<=source_beat_base + active_burst_beats;
                            state<=ST_P1_SETUP;
                        end else begin
                            scan_beat_index<=scan_beat_index + 1'b1;
                            state<=ST_P1_SCAN;
                        end
                    end else begin
                        scan_lane_index<=scan_lane_index + 1'b1;
                        state<=ST_P1_SCAN;
                    end
                end
            end

            ST_MAX_START: begin
                if (converter_busy) begin
                    error<=1; error_code<=ERR_CONVERTER; busy<=0; state<=ST_ERROR;
                end else state<=ST_MAX_WAIT;
            end

            ST_MAX_WAIT: begin
                if (converter_done) begin
                    max_mantissa_binary32<=converter_mantissa;
                    max_exponent_binary32<=converter_zero ? 10'sd0 : converter_exponent;
                    max_abs_binary32_bits<=converter_zero ? 32'd0 : {1'b0,converter_bits[30:0]};
                    max_abs_q10<=source_q28_reg ? 16'd0 : max_abs_raw[33:18];
                    source_beat_base<=0; capture_beat_index<=0; scan_beat_index<=0;
                    scan_lane_index<=0; element_index<=0; activation_buffer<=0;
                    activation_lane_index<=0; state<=ST_P2_SETUP;
                end
            end

            ST_P2_SETUP: begin
                axi_araddr<=source_addr_reg + (source_beat_base << 3);
                axi_arlen<=next_burst_beats - 1'b1;
                axi_arvalid<=1; ar_seen<=0; capture_beat_index<=0;
                active_burst_beats<=next_burst_beats; state<=ST_P2_CAPTURE;
            end

            ST_P2_CAPTURE: begin
                if (ar_handshake) begin axi_arvalid<=0; ar_seen<=1; end
                if (read_handshake) begin
                    burst_buffer[capture_beat_index]<=axi_rdata;
                    if (capture_beat_index + 1'b1 == active_burst_beats) begin
                        ar_seen<=0; scan_beat_index<=0; scan_lane_index<=0;
                        state<=ST_P2_SCAN;
                    end else capture_beat_index<=capture_beat_index + 1'b1;
                end
            end

            ST_P2_SCAN: begin
                source_work_reg<=selected_source;
                state<=ST_SRC_START;
            end

            ST_SRC_START: begin
                if (converter_busy) begin
                    error<=1; error_code<=ERR_CONVERTER; busy<=0; state<=ST_ERROR;
                end else state<=ST_SRC_WAIT;
            end

            ST_SRC_WAIT: begin
                if (converter_done) begin
                    source_sign_reg<=converter_sign;
                    source_zero_reg<=converter_zero;
                    source_mantissa_reg<=converter_mantissa;
                    source_exponent_reg<=converter_exponent;
                    state<=ST_RATIO_PREP;
                end
            end

            ST_RATIO_PREP: begin
                if (all_zero || source_zero_reg) begin
                    quant_result_reg<=0;
                    state<=ST_PACK;
                end else if (exponent_difference_negative ||
                             exponent_difference_too_large) begin
                    error<=1; error_code<=ERR_EXPONENT; busy<=0; state<=ST_ERROR;
                end else begin
                    divider_numerator_reg<={{65{1'b0}},ratio_numerator_base};
                    denominator_work_reg<=ratio_denominator_base;
                    denominator_shift_count<=exponent_difference[6:0];
                    if (exponent_difference==0) begin
                        divider_denominator_reg<=ratio_denominator_base;
                        state<=ST_DIV_START;
                    end else state<=ST_RATIO_SHIFT;
                end
            end

            ST_RATIO_SHIFT: begin
                denominator_work_reg<=denominator_work_reg << 1;
                if (denominator_shift_count==1) begin
                    divider_denominator_reg<=denominator_work_reg << 1;
                    denominator_shift_count<=0;
                    state<=ST_DIV_START;
                end else denominator_shift_count<=denominator_shift_count - 1'b1;
            end

            ST_DIV_START: begin
                divider_start<=1;
                state<=ST_DIV_WAIT;
            end

            ST_DIV_WAIT: begin
                if (divider_done) begin
                    if (divider_divide_by_zero || divider_overflow ||
                        (|divider_quotient[95:7]) ||
                        divider_quotient[6:0] > 7'd127) begin
                        error<=1; error_code<=ERR_DIVIDER; busy<=0; state<=ST_ERROR;
                    end else begin
                        quant_result_reg<=source_sign_reg
                            ? -$signed({1'b0,divider_quotient[6:0]})
                            :  $signed({1'b0,divider_quotient[6:0]});
                        state<=ST_PACK;
                    end
                end
            end

            ST_PACK: begin
                activation_buffer[activation_lane_index*8 +: 8]<=quant_result_reg;
                resume_needs_read<=0;
                if (!last_element) begin
                    element_index<=element_index + 1'b1;
                    if (last_source_lane) begin
                        scan_lane_index<=0;
                        if (last_burst_beat) begin
                            source_beat_base<=source_beat_base + active_burst_beats;
                            resume_needs_read<=1;
                        end else scan_beat_index<=scan_beat_index + 1'b1;
                    end else scan_lane_index<=scan_lane_index + 1'b1;
                end

                if (activation_lane_index==5'd31) begin
                    activation_lane_index<=0;
                    last_pack_pending<=last_element;
                    state<=ST_WRITE_SETUP;
                end else begin
                    activation_lane_index<=activation_lane_index + 1'b1;
                    if (last_source_lane && last_burst_beat)
                        state<=ST_P2_SETUP;
                    else
                        state<=ST_P2_SCAN;
                end
            end

            ST_WRITE_SETUP: begin
                axi_awaddr<=activation_write_addr;
                axi_awvalid<=1;
                axi_wdata<=activation_buffer;
                axi_wstrb<=32'hffff_ffff;
                aw_seen<=0; w_seen<=0;
                state<=ST_WRITE;
            end

            ST_WRITE: begin
                if (aw_handshake) begin axi_awvalid<=0; aw_seen<=1; end
                if (w_handshake) w_seen<=1;
                if (write_complete) begin
                    axi_awvalid<=0; aw_seen<=0; w_seen<=0;
                    activation_buffer<=0;
                    activation_write_addr<=activation_write_addr + 8;
                    if (last_pack_pending) begin
                        last_pack_pending<=0;
                        state<=ST_FINISH;
                    end else if (resume_needs_read) begin
                        resume_needs_read<=0;
                        state<=ST_P2_SETUP;
                    end else state<=ST_P2_SCAN;
                end
            end

            ST_FINISH: begin
                busy<=0; done<=1; state<=ST_IDLE;
            end

            ST_ERROR: begin
                busy<=0; axi_awvalid<=0; axi_arvalid<=0;
            end

            default: begin
                error<=1; error_code<=ERR_INTERNAL; busy<=0;
                axi_awvalid<=0; axi_arvalid<=0; state<=ST_ERROR;
            end
        endcase
    end
end

wire _unused=&{1'b0,converter_busy,divider_busy,divider_remainder};
endmodule
