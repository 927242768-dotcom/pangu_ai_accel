`timescale 1ns/1ps

// G2 统一 DDR3 AXI 主设备选择器。
//
// 完整 Block 调度器保证任意时刻只有一个计算阶段拥有 DDR3。本模块不仅完成
// 11 路阶段总线到唯一 DDR3 Controller 端口的选择，还在 AR/AW 请求和读返回
// 边界各加入一级寄存缓冲，避免形成：
//
//   current_stage -> master mux -> DDR ready -> stage handshake/cache enable
//
// 的超长组合路径。上游阶段看到的 ready 表示“请求已被本仲裁器缓存”，随后
// 仲裁器独立保持请求直到 DDR3 接收。所有写事务均为单 beat；读事务允许 burst，
// read_owner 在 AR 请求缓存时锁存，并用于整个返回 burst 的数据归属。
module g2_axi_stage_mux #(
    parameter integer NUM_MASTERS = 11,
    parameter integer ADDR_WIDTH  = 28
)(
    input  wire                               clk,
    input  wire                               rst_n,
    input  wire [3:0]                         select_master,

    input  wire [NUM_MASTERS*ADDR_WIDTH-1:0]  m_awaddr,
    input  wire [NUM_MASTERS-1:0]             m_awuser_ap,
    input  wire [NUM_MASTERS*4-1:0]           m_awuser_id,
    input  wire [NUM_MASTERS*4-1:0]           m_awlen,
    input  wire [NUM_MASTERS-1:0]             m_awvalid,
    output reg  [NUM_MASTERS-1:0]             m_awready,

    input  wire [NUM_MASTERS*256-1:0]         m_wdata,
    input  wire [NUM_MASTERS*32-1:0]          m_wstrb,
    output reg  [NUM_MASTERS-1:0]             m_wready,

    input  wire [NUM_MASTERS*ADDR_WIDTH-1:0]  m_araddr,
    input  wire [NUM_MASTERS-1:0]             m_aruser_ap,
    input  wire [NUM_MASTERS*4-1:0]           m_aruser_id,
    input  wire [NUM_MASTERS*4-1:0]           m_arlen,
    input  wire [NUM_MASTERS-1:0]             m_arvalid,
    output reg  [NUM_MASTERS-1:0]             m_arready,

    output reg  [NUM_MASTERS*256-1:0]         m_rdata,
    output reg  [NUM_MASTERS-1:0]             m_rvalid,

    output reg  [ADDR_WIDTH-1:0]              axi_awaddr,
    output reg                                axi_awuser_ap,
    output reg  [3:0]                         axi_awuser_id,
    output reg  [3:0]                         axi_awlen,
    input  wire                               axi_awready,
    output reg                                axi_awvalid,

    output reg  [255:0]                       axi_wdata,
    output reg  [31:0]                        axi_wstrb,
    input  wire                               axi_wready,

    output reg  [ADDR_WIDTH-1:0]              axi_araddr,
    output reg                                axi_aruser_ap,
    output reg  [3:0]                         axi_aruser_id,
    output reg  [3:0]                         axi_arlen,
    input  wire                               axi_arready,
    output reg                                axi_arvalid,

    input  wire [255:0]                       axi_rdata,
    input  wire                               axi_rvalid,

    output reg                                select_error
);

wire selected_in_range = (select_master < NUM_MASTERS);
wire selected_awvalid = selected_in_range ? m_awvalid[select_master] : 1'b0;
wire selected_arvalid = selected_in_range ? m_arvalid[select_master] : 1'b0;

// -----------------------------------------------------------------------------
// 单 beat 写请求缓冲。
// -----------------------------------------------------------------------------
reg                         write_active;
reg                         write_aw_pending;
reg                         write_w_pending;
reg [3:0]                   write_owner;
reg [ADDR_WIDTH-1:0]        write_awaddr_pipe;
reg                         write_awuser_ap_pipe;
reg [3:0]                   write_awuser_id_pipe;
reg [3:0]                   write_awlen_pipe;
reg [255:0]                 write_wdata_pipe;
reg [31:0]                  write_wstrb_pipe;
reg                         write_complete_pipe;
reg [3:0]                   write_complete_owner;

wire write_capture = !write_active && selected_awvalid;
wire write_aw_handshake =
    write_active && write_aw_pending && axi_awready;
// 只有地址已在此前周期接收，或本周期与地址同时接收时，才允许消费写数据。
wire write_w_handshake =
    write_active && write_w_pending && axi_wready &&
    (!write_aw_pending || axi_awready);
wire write_aw_done_next = !write_aw_pending || axi_awready;
wire write_w_done_next =
    !write_w_pending || (axi_wready && (!write_aw_pending || axi_awready));
wire write_finishes =
    write_active && write_aw_done_next && write_w_done_next;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        write_active         <= 1'b0;
        write_aw_pending     <= 1'b0;
        write_w_pending      <= 1'b0;
        write_owner          <= 4'd0;
        write_awaddr_pipe    <= {ADDR_WIDTH{1'b0}};
        write_awuser_ap_pipe <= 1'b0;
        write_awuser_id_pipe <= 4'd0;
        write_awlen_pipe     <= 4'd0;
        write_wdata_pipe     <= 256'd0;
        write_wstrb_pipe     <= 32'd0;
        write_complete_pipe  <= 1'b0;
        write_complete_owner <= 4'd0;
    end else begin
        write_complete_pipe <= 1'b0;

        if (write_active) begin
            if (write_aw_handshake)
                write_aw_pending <= 1'b0;
            if (write_w_handshake)
                write_w_pending <= 1'b0;
            if (write_finishes) begin
                write_active         <= 1'b0;
                write_aw_pending     <= 1'b0;
                write_w_pending      <= 1'b0;
                write_complete_pipe  <= 1'b1;
                write_complete_owner <= write_owner;
            end
        end

        if (write_capture) begin
            write_active         <= 1'b1;
            write_aw_pending     <= 1'b1;
            write_w_pending      <= 1'b1;
            write_owner          <= select_master;
            write_awaddr_pipe    <= m_awaddr[select_master*ADDR_WIDTH +: ADDR_WIDTH];
            write_awuser_ap_pipe <= m_awuser_ap[select_master];
            write_awuser_id_pipe <= m_awuser_id[select_master*4 +: 4];
            write_awlen_pipe     <= m_awlen[select_master*4 +: 4];
            write_wdata_pipe     <= m_wdata[select_master*256 +: 256];
            write_wstrb_pipe     <= m_wstrb[select_master*32 +: 32];
        end
    end
end

// -----------------------------------------------------------------------------
// 读地址请求缓冲。阶段在本地 ready 握手后即可撤销 arvalid；仲裁器继续保持
// 已寄存的请求，直到 DDR3 Controller 接收。
// -----------------------------------------------------------------------------
reg                         read_ar_pending;
reg [3:0]                   read_owner;
reg [ADDR_WIDTH-1:0]        read_araddr_pipe;
reg                         read_aruser_ap_pipe;
reg [3:0]                   read_aruser_id_pipe;
reg [3:0]                   read_arlen_pipe;

wire read_capture = !read_ar_pending && selected_arvalid;
wire read_ar_handshake = read_ar_pending && axi_arready;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        read_ar_pending     <= 1'b0;
        read_owner          <= 4'd0;
        read_araddr_pipe    <= {ADDR_WIDTH{1'b0}};
        read_aruser_ap_pipe <= 1'b0;
        read_aruser_id_pipe <= 4'd0;
        read_arlen_pipe     <= 4'd0;
    end else begin
        if (read_ar_handshake)
            read_ar_pending <= 1'b0;

        if (read_capture) begin
            read_ar_pending     <= 1'b1;
            read_owner          <= select_master;
            read_araddr_pipe    <= m_araddr[select_master*ADDR_WIDTH +: ADDR_WIDTH];
            read_aruser_ap_pipe <= m_aruser_ap[select_master];
            read_aruser_id_pipe <= m_aruser_id[select_master*4 +: 4];
            read_arlen_pipe     <= m_arlen[select_master*4 +: 4];
        end
    end
end

// -----------------------------------------------------------------------------
// DDR3 读返回边界寄存。数据广播到全部 master，但只对 read_owner 对应通道
// 产生 rvalid；下游阶段只在 rvalid=1 时采样，因此额外一拍不改变数值和顺序。
// -----------------------------------------------------------------------------
reg [255:0] return_rdata_pipe;
reg         return_rvalid_pipe;
reg [3:0]   return_master_pipe;

always @(posedge clk) begin
    return_rdata_pipe  <= axi_rdata;
    return_master_pipe <= read_owner;
end

always @(posedge clk or negedge rst_n) begin
    if (!rst_n)
        return_rvalid_pipe <= 1'b0;
    else
        return_rvalid_pipe <= axi_rvalid;
end

integer selected_bit;
always @(*) begin
    // 已寄存请求直接驱动 DDR3 Controller，输出不再依赖阶段 ready 回路。
    axi_awaddr    = write_awaddr_pipe;
    axi_awuser_ap = write_awuser_ap_pipe;
    axi_awuser_id = write_awuser_id_pipe;
    axi_awlen     = write_awlen_pipe;
    axi_awvalid   = write_active && write_aw_pending;
    axi_wdata     = write_wdata_pipe;
    axi_wstrb     = write_wstrb_pipe;

    axi_araddr    = read_araddr_pipe;
    axi_aruser_ap = read_aruser_ap_pipe;
    axi_aruser_id = read_aruser_id_pipe;
    axi_arlen     = read_arlen_pipe;
    axi_arvalid   = read_ar_pending;

    m_awready = {NUM_MASTERS{1'b0}};
    m_wready  = {NUM_MASTERS{1'b0}};
    m_arready = {NUM_MASTERS{1'b0}};
    m_rdata   = {NUM_MASTERS{return_rdata_pipe}};
    m_rvalid  = {NUM_MASTERS{1'b0}};
    select_error = 1'b0;

    selected_bit = select_master;
    if (selected_bit < NUM_MASTERS) begin
        // ready 表示请求已进入本地缓冲，而不是 DDR3 已在同拍接收。
        m_awready[selected_bit] = !write_active;
        m_arready[selected_bit] = !read_ar_pending;
    end else begin
        select_error = 1'b1;
    end

    if (write_complete_pipe && (write_complete_owner < NUM_MASTERS))
        m_wready[write_complete_owner] = 1'b1;

    if (return_rvalid_pipe && (return_master_pipe < NUM_MASTERS))
        m_rvalid[return_master_pipe] = 1'b1;
end

endmodule
