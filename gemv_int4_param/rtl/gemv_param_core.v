`timescale 1ns/1ps

// 参数化 packed INT4 单行 GEMV 计算核心。
//
// 激活向量和当前权重行按 256 bit 数据拍写入片上缓存。计算时采用
// 同步读缓存，再拆成 16 元素 MAC16 分块，避免对 MAX_K 宽大向量做
// 动态多路选择，便于 PDS 推断片上 RAM/LUTRAM。
module gemv_param_core #(
    parameter integer MAX_K = 896,
    parameter integer K_BLOCK_WIDTH = 7,
    parameter integer ACT_BEAT_WIDTH = 5,
    parameter integer WEIGHT_BEAT_WIDTH = 4
)(
    input  wire                         clk,
    input  wire                         rst_n,

    input  wire                         act_load_en,
    input  wire [ACT_BEAT_WIDTH-1:0]    act_load_index,
    input  wire [255:0]                 act_load_data,
    input  wire                         weight_load_en,
    input  wire [WEIGHT_BEAT_WIDTH-1:0] weight_load_index,
    input  wire [255:0]                 weight_load_data,

    input  wire                         start,
    input  wire [K_BLOCK_WIDTH-1:0]     k_blocks,
    input  wire [4:0]                   tail_elements,
    output reg                          busy,
    output reg                          done,
    output reg signed [31:0]            y_value
);

localparam integer MAX_ACT_BEATS = (MAX_K + 31) / 32;
localparam integer MAX_WEIGHT_BEATS = (MAX_K + 63) / 64;

localparam [2:0] ST_IDLE        = 3'd0;
localparam [2:0] ST_READ_BLOCK  = 3'd1;
localparam [2:0] ST_PREPARE     = 3'd2;
localparam [2:0] ST_CAPTURE_MAC = 3'd3;
localparam [2:0] ST_ACCUMULATE  = 3'd4;

reg [255:0] activation_mem [0:MAX_ACT_BEATS-1];
reg [255:0] weight_mem [0:MAX_WEIGHT_BEATS-1];

reg [2:0] state;
reg [K_BLOCK_WIDTH-1:0] block_index;
reg [K_BLOCK_WIDTH-1:0] active_k_blocks;
reg [4:0] active_tail_elements;
reg [255:0] act_beat_reg;
reg [255:0] weight_beat_reg;
reg [127:0] mac_x_reg;
reg [127:0] mac_w_reg;
reg signed [31:0] dot_result_reg;
reg signed [31:0] accumulator;

wire [127:0] selected_x_block =
    block_index[0] ? act_beat_reg[255:128] : act_beat_reg[127:0];
reg [63:0] selected_w_block;
reg [127:0] masked_x_block;
reg [63:0] masked_w_block;
wire [127:0] unpacked_w_block;
wire signed [31:0] dot_result;
wire signed [31:0] accumulated_result = accumulator + dot_result_reg;

integer mask_index;
always @(*) begin
    case (block_index[1:0])
        2'd0: selected_w_block = weight_beat_reg[63:0];
        2'd1: selected_w_block = weight_beat_reg[127:64];
        2'd2: selected_w_block = weight_beat_reg[191:128];
        default: selected_w_block = weight_beat_reg[255:192];
    endcase

    masked_x_block = selected_x_block;
    masked_w_block = selected_w_block;
    if ((block_index + 1'b1 == active_k_blocks) &&
        (active_tail_elements < 5'd16)) begin
        for (mask_index = 0; mask_index < 16; mask_index = mask_index + 1) begin
            if (mask_index >= active_tail_elements) begin
                masked_x_block[mask_index*8 +: 8] = 8'd0;
                masked_w_block[mask_index*4 +: 4] = 4'd0;
            end
        end
    end
end

int4_unpack16 u_int4_unpack16 (
    .packed_vec   (masked_w_block),
    .unpacked_vec (unpacked_w_block)
);

int8_dot16 u_int8_dot16 (
    .a_vec  (mac_x_reg),
    .b_vec  (mac_w_reg),
    .result (dot_result)
);

// 缓存不复位，只有配置对应的有效数据拍会被读取。
// 独立写端口有利于综合器推断片上存储资源。
always @(posedge clk) begin
    if (act_load_en)
        activation_mem[act_load_index] <= act_load_data;
    if (weight_load_en)
        weight_mem[weight_load_index] <= weight_load_data;
end

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        state             <= ST_IDLE;
        block_index       <= {K_BLOCK_WIDTH{1'b0}};
        active_k_blocks   <= {K_BLOCK_WIDTH{1'b0}};
        active_tail_elements <= 5'd16;
        act_beat_reg      <= 256'd0;
        weight_beat_reg   <= 256'd0;
        mac_x_reg         <= 128'd0;
        mac_w_reg         <= 128'd0;
        dot_result_reg    <= 32'sd0;
        accumulator       <= 32'sd0;
        y_value           <= 32'sd0;
        busy              <= 1'b0;
        done              <= 1'b0;
    end else begin
        done <= 1'b0;

        case (state)
            ST_IDLE: begin
                busy <= 1'b0;
                if (start && (k_blocks != {K_BLOCK_WIDTH{1'b0}})) begin
                    block_index     <= {K_BLOCK_WIDTH{1'b0}};
                    active_k_blocks   <= k_blocks;
                    active_tail_elements <= tail_elements;
                    accumulator       <= 32'sd0;
                    y_value         <= 32'sd0;
                    busy            <= 1'b1;
                    state           <= ST_READ_BLOCK;
                end
            end

            ST_READ_BLOCK: begin
                // 同步读取当前激活拍和权重拍。
                act_beat_reg    <= activation_mem[block_index >> 1];
                weight_beat_reg <= weight_mem[block_index >> 2];
                state           <= ST_PREPARE;
            end

            ST_PREPARE: begin
                // 流水级 1：从缓存拍中选择当前 16 元素并完成 INT4 解包。
                mac_x_reg <= masked_x_block;
                mac_w_reg <= unpacked_w_block;
                state     <= ST_CAPTURE_MAC;
            end

            ST_CAPTURE_MAC: begin
                // 流水级 2：寄存 MAC16 结果，隔离乘加树与跨块累加器。
                dot_result_reg <= dot_result;
                state          <= ST_ACCUMULATE;
            end

            ST_ACCUMULATE: begin
                // 流水级 3：累加当前分块结果。
                if (block_index + 1'b1 == active_k_blocks) begin
                    y_value     <= accumulated_result;
                    accumulator <= 32'sd0;
                    busy        <= 1'b0;
                    done        <= 1'b1;
                    state       <= ST_IDLE;
                end else begin
                    accumulator <= accumulated_result;
                    block_index <= block_index + 1'b1;
                    state       <= ST_READ_BLOCK;
                end
            end

            default: begin
                state <= ST_IDLE;
                busy  <= 1'b0;
                done  <= 1'b0;
            end
        endcase
    end
end

endmodule
