# PGL50H DDR3 x32 全地址自检固件

## 目标与边界

本固件用于盘古 50K / MES50HP 开发板上的 DDR3 初始化、全地址写入和全地址读回校验。

- FPGA：Logos `PGL50H-6IFBG484`
- DDR3 IP：Pango DDR3 Interface `v1.5`
- 数据位宽：32 bit（4 组 DQ/DQS/DM）
- 地址参数：Row 15 bit、Column 10 bit、Bank 3 bit
- 控制器地址宽度：28 bit
- 用户侧数据接口：256 bit AXI
- 覆盖容量：`2^28` 个控制器地址单位，即 1 GiB 物理 DDR3 数据空间

本工程没有使用旧的 Logos2、`base_ddr3` 或 16 位 DDR3 工程，也不表示当前系统已经能够运行完整大模型。

## 自检流程

1. 等待 PLL、DDR3 PHY 和 Controller 完成初始化/训练。
2. 从地址 0 开始，使用 16 拍 AXI 突发顺序写满整个 1 GiB 地址空间。
3. 从地址 0 开始，使用 16 拍 AXI 突发顺序读回整个地址空间。
4. 每个 256 bit 数据拍均根据当前地址重建确定性期望数据并比较。
5. 等待比较流水线排空后，锁存 PASS 或 FAIL 状态。

自检控制器源码：

`ddr3_selftest/rtl/full_addr_bist_v1_0.v`

## 已生成位流

位流路径：

`ipcore/pangu_ddr3_x32/pangu_ddr3_x32/pnr/generate_bitstream/test_ddr.sbit`

文件大小：`2,101,696 bytes`

SHA-256：

`a5759280b02366337be083399135196a16104842c09733553ea2be8156c39a3c`

配套 mask 文件：

`ipcore/pangu_ddr3_x32/pangu_ddr3_x32/pnr/generate_bitstream/test_ddr.smsk`

SHA-256：

`dfb4cb43408f83b1ddaabcb53b7c0f642c649dd227926e96c38583d6e6fc24ee`

## 构建结果

PDS：`2022.2-SP6.4 build 146967`

完整流程已经通过：

- Compile：通过
- Synthesize：通过
- Device Map：通过
- Place & Route：通过，最终未布通网络为 0
- Report Timing：通过
- Generate Netlist：通过
- Generate Bitstream：通过

主要资源占用：

- LUT：8,121 / 42,800（19%）
- FF：9,210 / 64,200（15%）
- DQSL：8 / 18（45%）
- PLL：2 / 5（40%）
- I/O：79 / 296（27%）

后布局多角时序结果：`All Constraints Met`

- 慢角最差 Setup：`0.920 ns`，TNS = `0`
- 慢角最差 Hold：`0.160 ns`，THS = `0`
- 快角最差 Setup：`1.834 ns` 以上，TNS = `0`
- 快角最差 Hold：`0.091 ns`，THS = `0`

时序报告：

`ipcore/pangu_ddr3_x32/pangu_ddr3_x32/pnr/report_timing/test_ddr.rtr`

## 板上 LED 判定

以下均为开发板用户 LED，高电平点亮：

| 引脚 | 信号 | 含义 |
|---|---|---|
| A3 | `pll_lock` | DDR3 PLL 已锁定 |
| B2 | `ddr_init_done` | DDR3 PHY/Controller 初始化训练完成 |
| B3 | `heart_beat_led` | 自检运行时闪烁；全地址自检 PASS 后常亮 |
| A2 | `err_flag_led` | 检测到任意读回错误后常亮 |

最终判定：

- PASS：A3、B2、B3 亮，A2 灭。
- FAIL：A2 亮，B3 灭。
- 若 A3 或 B2 始终不亮：优先检查参考时钟、复位、DDR3 供电、引脚映射和 PHY 训练。

## 重新构建

在以下目录执行：

`ipcore/pangu_ddr3_x32/pangu_ddr3_x32/pnr`

命令：

```text
D:\Pango\PDS_2022.2-SP6.4\bin\pds_shell.exe -project pangu_ddr3_x32.pds -run gen_bit_stream
```

PDS 接受的位流阶段名是 `gen_bit_stream`，不是 `gen_bitstream` 或 `generate_bitstream`。

## 当前硬件验证状态

2026-07-23 15:32 完成板上验证：

- COM20（Silicon Labs CP210x USB-UART）在线，参数为 `115200 8N1`。
- 配置工具识别到 `PANGO USB CABLE II` 和 `PGL50H`。
- 已将 `test_ddr.sbit` 下载到 FPGA 易失性 SRAM；下载进度达到 100%，配置 `done bit = 1`。
- UART 读取地址 `0x80` 返回 `0x00011110`：PLL 锁定、DLL 锁定、DDR3 初始化完成、PASS 指示有效、错误标志为 0。
- UART 读取地址 `0x8B` 返回 `0x00000500`：`test_main_state = 5 (PASS)`，`err_cnt = 0`。
- 间隔 3 秒再次读取，返回值保持不变，确认 PASS 状态已锁存。

结论：正确的 Logos PGL50H、FBG484、32 位 DDR3 Controller+PHY 位流已经完成真实开发板 SRAM 下载；DDR3 初始化训练成功，1 GiB 全地址写入、读回和比较自检通过，未检测到数据错误。

本次只进行了易失性 SRAM 下载，没有改写永久 Flash。