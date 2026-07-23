# PGL50H packed INT4 GEMV（M=4、K=64）验证工程

## 1. 工程目标

本工程是独立于 `ddr_mac16_integration` 的固定小尺寸 GEMV 验证版本，不覆盖已经验证的单点积位流。

实现的数据闭环：

```text
Python 生成 W[4,64] INT4 和 x[64] INT8
→ UART 发送 192 字节载荷
→ FPGA 写入 DDR3
→ 激活以 2 拍 burst 读取并缓存一次
→ 4 行 packed INT4 权重以 4 拍 burst 连续读取
→ 每行拆成 4 个 MAC16 分块
→ 跨分块 INT32 累加
→ 生成 4 个 INT32 输出
→ 写回 DDR3
→ UART 返回整个输出向量
→ Python 逐元素比较
```

## 2. DDR3 地址布局

DDR3 控制器地址单位为 32 bit；一个 256 bit 数据拍占 8 个地址单位。

| 控制器地址 | 内容 |
|---|---|
| `0x00` | 激活 `x[0:31]`，32 个 INT8 |
| `0x08` | 激活 `x[32:63]`，32 个 INT8 |
| `0x10` | 权重第 0 行，64 个 packed INT4，共 32 字节 |
| `0x18` | 权重第 1 行 |
| `0x20` | 权重第 2 行 |
| `0x28` | 权重第 3 行 |
| `0x30` | 输出 `y[0:3]`，4 个 little-endian INT32，位于低 128 bit |

packed INT4 字节内顺序：低 4 bit 为偶数下标权重，高 4 bit 为奇数下标权重；均按 4 bit 二补码解释。

## 3. UART 协议

串口参数：`115200, 8N1`。

| 命令 | 请求 | 回复 |
|---|---|---|
| `I` | 固件信息 | `PANGU50K GEMV M4K64 V1\r\n` |
| `S` | 状态查询 | `S + flags + \r\n` |
| `M` | `64B INT8激活 + 128B packed INT4权重` | `K\r\n` |
| `G` | 启动 GEMV | `R + 4个little-endian int32` |

状态字节：

- bit0：DDR3 初始化完成。
- bit1：GEMV 数据已加载。
- bit2：结果有效。
- bit3：GEMV 计算核心忙。

错误回复为 `E + error_code + \r\n`。

## 4. 主要文件

| 文件 | 作用 |
|---|---|
| `rtl/gemv_m4k64_core.v` | 4 行 × 4 分块的 MAC16 调度和 INT32 累加核心 |
| `rtl/gemv_m4k64_ctrl.v` | UART、DDR3 AXI、缓存、计算和结果回写控制器 |
| `rtl/gemv_m4k64_top.v` | DDR3 IP、控制器、UART 和 LED 顶层 |
| `sim/tb_gemv_m4k64_core.v` | 固定向量核心测试平台；当前主机未安装 Icarus Verilog |
| `pnr/build_gemv_m4k64.tcl` | PDS 全流程构建脚本 |
| `pnr/program_sram.tcl` | 仅写 FPGA 易失性 SRAM，不操作 Flash |
| `../tools/pangu_gemv_m4k64_host.py` | Python 金标准、固定向量和随机压力测试工具 |

## 5. 时序流水

首版把 MAC16 乘加树和跨分块累加器放在同一周期，慢速角出现：

```text
WNS = -1.961 ns
TNS = -164.261 ns
```

最终版本采用三级计算流水：

```text
行/分块选择与 INT4 解包
→ MAC16 结果寄存
→ 跨分块 INT32 累加
```

最终多角时序：

- `Design Summary : All Constraints Met.`
- 100 MHz `ddrphy_clkin` 慢速角建立 WNS：`+0.983 ns`，TNS：`0`。
- 慢速角保持 WHS：`+0.171 ns`，THS：`0`。
- 快速角建立 WNS：`+3.276 ns`，TNS：`0`。
- 快速角保持 WHS：`+0.100 ns`，THS：`0`。
- 恢复、移除和最小脉宽检查全部通过。
- 布局布线成功，最终未布线网络：`0`。

## 6. 构建和 SRAM 下载

在 `gemv_int4_m4k64/pnr` 目录执行：

```bat
D:\Pango\PDS_2022.2-SP6.4\bin\pds_shell.exe -file build_gemv_m4k64.tcl -project_name gemv_m4k64_top
```

仅下载到易失性 SRAM：

```bat
D:\Pango\PDS_2022.2-SP6.4\bin\cdt_cfg_shell.exe -file program_sram.tcl -work_dir .
```

最终位流：

```text
pnr/generate_bitstream/gemv_m4k64_top.sbit
SHA256: 349a26b45362778849868e68475c5b8f6620bc8edb8375ebb237efbab4d352ed
```

## 7. 验证命令

```bat
python tools\pangu_gemv_m4k64_host.py selftest --rounds 1000 --seed 20260725
python tools\pangu_gemv_m4k64_host.py --port COM20 info
python tools\pangu_gemv_m4k64_host.py --port COM20 status
python tools\pangu_gemv_m4k64_host.py --port COM20 gemv
python tools\pangu_gemv_m4k64_host.py --port COM20 stress --rounds 1000 --seed 20260725
```

## 8. 2026-07-23 真实验证结果

- Python 金标准自检：1000/1000 PASS。
- PDS 编译、综合、Device Map、布局布线、时序分析和位流生成：全部成功。
- JTAG：识别 `PANGO USB CABLE II` 和 `PGL50H`。
- SRAM 下载：100%，`done bit=1`；未写 Flash。
- 串口：Silicon Labs CP210x，`COM20`。
- 固件信息：`PANGU50K GEMV M4K64 V1`。
- DDR3 初始化：完成。
- 固定向量：FPGA `[1376, -1344, 416, 256]`，Python 相同，PASS。
- 随机压力测试：1000/1000 PASS，seed=`20260725`，耗时约 `19.70` 秒。

## 9. 下一阶段

固定 `M=4、K=64` 的 D1.1 已完成。下一步进入参数化 GEMV：支持运行时 `M/K`、尾块屏蔽、更长 burst、自动地址递增和多尺寸随机测试。
