`timescale 1ns/1ps

`include "transformer_block_contract.vh"

// G2 完整 layer0 Transformer Block 顺序调度器。
//
// 22 个 engine 位与计算阶段一一对应：
//   bit0 INPUT_RMS ... bit21 RESIDUAL2。
// 四个运行时量化阶段显式包含在调度中，禁止隐含由主机生成中间 INT8/scale。
// 每个 engine_start 仅保持一个周期；调度器随后等待对应 done/error。
// 本模块不猜测固定延迟，所有阶段都必须提供明确握手。
module transformer_block_scheduler #(
    parameter [31:0] WATCHDOG_CYCLES = 32'd100000000
)(
    input  wire         clk,
    input  wire         rst_n,
    input  wire         start,

    input  wire [21:0]  engine_done,
    input  wire [21:0]  engine_error,

    output reg  [21:0]  engine_start,
    output reg  [4:0]   current_stage,
    output reg          busy,
    output reg          done,
    output reg          error,
    output reg  [7:0]   error_code,
    output reg  [31:0]  watchdog_count
);

localparam [2:0] ST_IDLE   = 3'd0;
localparam [2:0] ST_LAUNCH = 3'd1;
localparam [2:0] ST_WAIT   = 3'd2;
localparam [2:0] ST_FINISH = 3'd3;
localparam [2:0] ST_ERROR  = 3'd4;

localparam [7:0] ERR_CHILD_BASE = 8'h40;
localparam [7:0] ERR_TIMEOUT    = 8'h80;
localparam [7:0] ERR_BAD_STAGE  = 8'hff;

reg [2:0] state;
reg [4:0] stage_reg;
wire [4:0] engine_index = stage_reg - 1'b1;
wire stage_in_range =
    (stage_reg >= `G2_STAGE_INPUT_RMS) &&
    (stage_reg <= `G2_STAGE_RESIDUAL2);
wire selected_done = stage_in_range ? engine_done[engine_index] : 1'b0;
wire selected_error = stage_in_range ? engine_error[engine_index] : 1'b0;
wire watchdog_expired =
    (WATCHDOG_CYCLES != 32'd0) &&
    (watchdog_count >= WATCHDOG_CYCLES - 1'b1);

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        state          <= ST_IDLE;
        stage_reg      <= `G2_STAGE_IDLE;
        engine_start   <= 22'd0;
        current_stage  <= `G2_STAGE_IDLE;
        busy           <= 1'b0;
        done           <= 1'b0;
        error          <= 1'b0;
        error_code     <= 8'd0;
        watchdog_count <= 32'd0;
    end else begin
        engine_start <= 22'd0;
        done         <= 1'b0;

        case (state)
            ST_IDLE: begin
                busy           <= 1'b0;
                current_stage  <= `G2_STAGE_IDLE;
                watchdog_count <= 32'd0;
                if (start && !error) begin
                    stage_reg     <= `G2_STAGE_INPUT_RMS;
                    current_stage <= `G2_STAGE_INPUT_RMS;
                    busy          <= 1'b1;
                    error_code    <= 8'd0;
                    state         <= ST_LAUNCH;
                end
            end

            ST_LAUNCH: begin
                if (!stage_in_range) begin
                    error         <= 1'b1;
                    error_code    <= ERR_BAD_STAGE;
                    current_stage <= `G2_STAGE_ERROR;
                    busy          <= 1'b0;
                    state         <= ST_ERROR;
                end else begin
                    engine_start[engine_index] <= 1'b1;
                    watchdog_count             <= 32'd0;
                    state                      <= ST_WAIT;
                end
            end

            ST_WAIT: begin
                if (selected_error) begin
                    error         <= 1'b1;
                    error_code    <= ERR_CHILD_BASE + {3'd0, stage_reg};
                    current_stage <= `G2_STAGE_ERROR;
                    busy          <= 1'b0;
                    state         <= ST_ERROR;
                end else if (selected_done) begin
                    watchdog_count <= 32'd0;
                    if (stage_reg == `G2_STAGE_RESIDUAL2) begin
                        current_stage <= `G2_STAGE_DONE;
                        state         <= ST_FINISH;
                    end else begin
                        stage_reg     <= stage_reg + 1'b1;
                        current_stage <= stage_reg + 1'b1;
                        state         <= ST_LAUNCH;
                    end
                end else if (watchdog_expired) begin
                    error         <= 1'b1;
                    error_code    <= ERR_TIMEOUT + {3'd0, stage_reg};
                    current_stage <= `G2_STAGE_ERROR;
                    busy          <= 1'b0;
                    state         <= ST_ERROR;
                end else begin
                    watchdog_count <= watchdog_count + 1'b1;
                end
            end

            ST_FINISH: begin
                busy           <= 1'b0;
                done           <= 1'b1;
                stage_reg      <= `G2_STAGE_IDLE;
                watchdog_count <= 32'd0;
                state          <= ST_IDLE;
            end

            ST_ERROR: begin
                // error/error_code 为粘滞状态；只有 rst_n 才能清除。
                busy           <= 1'b0;
                current_stage  <= `G2_STAGE_ERROR;
                watchdog_count <= watchdog_count;
            end

            default: begin
                error         <= 1'b1;
                error_code    <= ERR_BAD_STAGE;
                current_stage <= `G2_STAGE_ERROR;
                busy          <= 1'b0;
                state         <= ST_ERROR;
            end
        endcase
    end
end

endmodule
