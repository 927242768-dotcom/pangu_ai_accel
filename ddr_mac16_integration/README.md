# PGL50H DDR3 + MAC16 集成验证工程

## 1. 工程目标

本工程用于验证盘古 Logos `PGL50H-6IFBG484` 上的第一条大模型计算数据通路：

`上位机 → UART → DDR3 → 256 bit AXI burst → 片上寄存缓冲 → INT4 解包/INT8 数据 → MAC16 → DDR3 结果回写 → UART → Python 对比`

工程独立于已经验证通过的 DDR3 全地址 BIST 工程，不覆盖原位流：

- 原 DDR3 BIST：`../ipcore/pangu_ddr3_x32/pangu_ddr3_x32/pnr/generate_bitstream/test_ddr.sbit`
- 本集成工程：`pnr/generate_bitstream/ddr_mac16_top.sbit`

## 2. 已实现能力

- 32 位 DDR3 Controller + PHY，用户侧数据宽度 256 bit。
- 一次 AXI 读命令返回两个连续的 256 bit 数据拍：激活向量与权重向量。
- 片上缓存两个 AXI 数据拍，并拆分低 128 bit 给 MAC16。
- 16 路有符号 `INT8 × INT8` 点积，32 位有符号累加结果。
- packed INT4 权重解包：每个字节低半字节对应偶数下标，高半字节对应奇数下标。
- INT4 按二补码符号扩展到 INT8，再复用同一个 MAC16。
- MAC 输入增加一级流水寄存器，将“DDR 数据拆分/INT4 解包”和“16 路乘加”分开，满足 100 MHz 时序。
- 计算结果写回 DDR3，并通过 UART 返回给上位机自动比较。

## 3. DDR3 地址布局

DDR3 控制器地址单位为 32 bit。一个 256 bit 数据拍占 8 个地址单位。

| 控制器地址 | 内容 |
|---|---|
| `0x00` | 16 个 INT8 激活，位于低 128 bit |
| `0x08` | INT8 模式：16 个权重位于低 128 bit；INT4 模式：16 个 packed 权重位于低 64 bit |
| `0x10` | 32 位点积结果位于低 32 bit |

## 4. UART 协议 V2

串口参数：`115200, 8N1`。

| 命令 | 请求 | 回复 |
|---|---|---|
| `I` | 固件信息 | `PANGU50K DDR3 MAC16 V2\r\n` |
| `S` | 状态查询 | `S + flags + \r\n` |
| `L` | `16B INT8激活 + 16B INT8权重` | `K\r\n` |
| `Q` | `16B INT8激活 + 8B packed INT4权重` | `K\r\n` |
| `G` | 启动 DDR3 burst 读取、MAC16 与结果回写 | `R + little-endian int32` |

状态字节：

- bit0：DDR3 初始化完成。
- bit1：输入和权重已加载。
- bit2：计算结果有效。
- bit3：当前为 INT4 权重模式。

错误回复为 `E + error_code + \r\n`。

## 5. 主要文件

| 文件 | 作用 |
|---|---|
| `rtl/ddr_mac16_top.v` | DDR3 IP、控制器、UART、LED 集成顶层 |
| `rtl/ddr_mac16_ctrl.v` | UART 协议、AXI 读写、数据缓存、MAC 调度状态机 |
| `rtl/int4_unpack16.v` | 16 个有符号 INT4 到 INT8 的并行解包与符号扩展 |
| `pnr/build_ddr_mac16.tcl` | PDS 全流程构建脚本 |
| `pnr/program_sram.tcl` | 仅写 FPGA SRAM 的 JTAG 下载脚本，不操作 Flash |
| `../tools/pangu_ddr_mac16_host.py` | 上位机固定向量与随机压力测试工具 |

## 6. 构建与下载

在 `ddr_mac16_integration/pnr` 目录执行：

```bat
D:\Pango\PDS_2022.2-SP6.4\bin\pds_shell.exe -file build_ddr_mac16.tcl -project_name ddr_mac16_top
```

仅下载到易失性 SRAM：

```bat
D:\Pango\PDS_2022.2-SP6.4\bin\cdt_cfg_shell.exe -file program_sram.tcl -work_dir .
```

最终位流：

```text
pnr/generate_bitstream/ddr_mac16_top.sbit
SHA256: e625e6dbe0e7f49915b41be805a970ea3977a72a6cb189f98c50497371b0af9f
```

## 7. 上位机验证命令

在 `pangu_ai_accel` 目录执行：

```bat
python tools\pangu_ddr_mac16_host.py --port COM20 info
python tools\pangu_ddr_mac16_host.py --port COM20 status
python tools\pangu_ddr_mac16_host.py --port COM20 dot
python tools\pangu_ddr_mac16_host.py --port COM20 stress --rounds 1000 --seed 20260723
python tools\pangu_ddr_mac16_host.py --port COM20 dot-int4
python tools\pangu_ddr_mac16_host.py --port COM20 stress-int4 --rounds 1000 --seed 20260724
```

## 8. 2026-07-23 最终验证结果

### PDS 实现与时序

- 编译、综合、Device Map、布局布线、时序分析、位流生成：全部完成。
- 布局布线：0 条未布线网络。
- 时序结论：`Design Summary : All Constraints Met.`
- 100 MHz `ddrphy_clkin` 慢速角建立 WNS：`+0.841 ns`，TNS：`0`。
- 100 MHz `ddrphy_clkin` 慢速角保持 WHS：`+0.171 ns`，THS：`0`。
- 快速角建立 WNS：`+3.210 ns`；快速角保持 WHS：`+0.100 ns`。
- 逻辑规模约为 5834 LUT、4835 FF，DDR DQSL 8 个、PLL 2 个、IO 79 个。

### 真实上板

- JTAG：识别 `PANGO USB CABLE II` 与 `PGL50H`。
- SRAM 下载进度：100%。
- FPGA `done bit=1`。
- 串口：Silicon Labs CP210x，`COM20`。
- DDR3 初始化状态：完成。
- INT8 固定向量：FPGA `272`，Python `272`，PASS。
- INT8 随机压力测试：1000/1000 PASS，约 4.49 秒。
- INT4 固定向量：FPGA `272`，Python `272`，PASS。
- INT4 随机压力测试：1000/1000 PASS，约 3.72 秒。

## 9. 当前边界与下一阶段

本工程已经完成单次点积的数据闭环与 INT4 权重解包验证，但尚未实现完整大模型推理，也尚未实现多行矩阵调度。

下一阶段从 `y = W × x` 开始：

1. 在 DDR3 中连续保存多行 packed INT4 权重。
2. 激活向量只读取一次，权重按多拍 burst 连续读取。
3. MAC16 对矩阵每一行循环执行，形成多个 32 位输出。
4. 结果批量回写 DDR3，并由 Python 对整个输出向量进行比较。
5. 闭环稳定后，再实现分块 GEMV、量化缩放/零点以及模型真实张量布局。
