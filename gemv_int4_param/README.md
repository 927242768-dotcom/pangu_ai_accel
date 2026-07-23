# PGL50H 运行时参数化 packed INT4 GEMV 验证与性能计数工程

## 1. 工程目标

本工程在已验证的固定 `M=4、K=64` GEMV 基线上，实现 D1.2 运行时参数化：

```text
W: M × K 的有符号 packed INT4 矩阵
x: K 维有符号 INT8 激活向量
y: M 维有符号 INT32 累加结果
```

当前支持范围：

```text
1 <= M <= 64
1 <= K <= 896
```

`K` 不需要是 16 的整数倍。最后一个 MAC16 分块不足 16 个元素时，FPGA 会显式屏蔽无效激活字节和权重半字节。

本工程独立于已验证工程，不覆盖原位流：

- 固定 GEMV：`../gemv_int4_m4k64/pnr/generate_bitstream/gemv_m4k64_top.sbit`
- D1.2 参数化 GEMV 已验证位流：`pnr/generate_bitstream/gemv_param_top.sbit`
- D1.3 性能计数位流：`../gemv_int4_perf/pnr/generate_bitstream/gemv_param_top.sbit`

## 2. 数据闭环

```text
Python 产生运行时 M/K、W[M,K] 和 x[K]
→ UART 下发配置
→ 激活和逐行权重按 256 bit 对齐后写入 DDR3
→ 激活使用最长 16 拍 AXI burst 分段读取并缓存一次
→ 权重按行 burst 读取，行地址自动递增
→ 每行执行 ceil(K/16) 次 MAC16
→ 尾块硬件屏蔽
→ 得到 M 个 INT32 输出
→ 每 8 个输出一拍写回 DDR3，输出地址自动递增
→ UART 返回整个输出向量
→ Python 逐元素完全比较
```

## 3. DDR3 地址布局

DDR3 控制器地址单位为 32 bit，一个 256 bit 数据拍占 8 个地址单位。

| 控制器地址 | 内容 |
|---|---|
| `0x0000` 起 | 激活向量，按 `ceil(K/32)` 个 256 bit 数据拍存放 |
| `0x0100` 起 | packed INT4 权重；每行占 `ceil(K/64)` 个数据拍，行地址自动递增 |
| `0x2000` 起 | INT32 输出；每拍最多存放 8 个结果，地址自动递增 |

上传时采用以下补齐规则：

```text
激活区字节数 = ceil(K / 32) × 32
每个权重行字节数 = ceil(K / 64) × 32
总权重字节数 = M × 每行字节数
```

每个 packed INT4 字节中，低 4 bit 为偶数下标权重，高 4 bit 为奇数下标权重，均按 4 bit 二补码解释。奇数 K 的最后一个字节高半字节填 0，硬件仍会按真实 K 屏蔽尾部。

## 4. UART 协议 V2

串口参数：`115200, 8N1`。

| 命令 | 请求 | 回复 |
|---|---|---|
| `I` | 固件信息 | `PANGU50K GEMV PARAM V2\r\n` |
| `S` | 状态查询 | `S + flags + \r\n` |
| `C` | `uint16_le(M) + uint16_le(K)` | `K\r\n` |
| `L` | 补齐后的激活和权重载荷 | `K\r\n` |
| `G` | 启动 GEMV | `R + M个little-endian int32` |
| `P` | 读取最近一次性能计数 | `P + act_read_cycles + weight_read_cycles + mac_cycles + total_cycles`，4 个计数均为 `uint32_le` |

状态字节：

- bit0：DDR3 初始化完成；
- bit1：M/K 配置有效；
- bit2：输入和权重已加载；
- bit3：结果有效；
- bit4：计算核心忙；
- bit5：最近一次 GEMV 性能计数有效。

错误回复为 `E + error_code + \r\n`：

| 错误码 | 含义 |
|---|---|
| `0x01` | 未知命令 |
| `0x02` | DDR3 尚未初始化 |
| `0x03` | 尚未配置有效 M/K |
| `0x04` | 尚未加载数据 |
| `0x05` | M/K 超出支持范围 |
| `0x06` | 尚无有效的 GEMV 性能计数 |
| `0xFF` | FPGA 状态机异常 |

## 5. 主要文件

| 文件 | 作用 |
|---|---|
| `rtl/gemv_param_core.v` | 运行时 K、同步片上缓存、MAC16 分块累加和尾块屏蔽 |
| `rtl/gemv_param_ctrl.v` | UART 协议、DDR3 AXI、运行时 M/K、行/输出地址调度 |
| `rtl/gemv_param_top.v` | DDR3 IP、控制器、UART 和 LED 顶层 |
| `pnr/build_gemv_param.tcl` | PDS 编译到位流的完整构建脚本 |
| `pnr/program_sram.tcl` | 只写 FPGA 易失性 SRAM，不操作 Flash |
| `../tools/pangu_gemv_param_host.py` | Python 金标准、布局验证、边界和上板压力测试 |

## 6. 资源与时序

激活缓存和单行权重缓存被 PDS 推断为 4 个 `DRM18K`，避免最大 K 宽动态大多路选择。

最终资源：

- LUT：`10,715 / 42,800`，约 `25.04%`；
- Register：`8,136 / 64,200`，约 `12.67%`；
- DRM18K：`4 / 134`；
- APM：`9 / 84`。

最终布局布线：

- 0 条未布线网络；
- `Design Summary : All Constraints Met.`；
- 100 MHz `ddrphy_clkin` 慢速角建立 WNS：`+0.682 ns`，TNS：`0`；
- 慢速角保持 WHS：`+0.086 ns`，THS：`0`；
- 快速角建立 WNS：`+3.137 ns`，TNS：`0`；
- 快速角保持 WHS：`+0.001 ns`，THS：`0`；
- 恢复、移除和最小脉宽检查全部通过。

快速角保持裕量仅 `+0.001 ns`，后续修改必须继续关注保持时序，不能假设仍有充足余量。

## 7. 构建和 SRAM 下载

在 `gemv_int4_param/pnr` 目录执行：

```bat
D:\Pango\PDS_2022.2-SP6.4\bin\pds_shell.exe -file build_gemv_param.tcl -project_name gemv_param_top
```

仅下载到易失性 SRAM：

```bat
D:\Pango\PDS_2022.2-SP6.4\bin\cdt_cfg_shell.exe -file program_sram.tcl -work_dir .
```

最终位流：

```text
pnr/generate_bitstream/gemv_param_top.sbit
SHA256: 90c67a74841826b358f4a4de5e0783c587de01a296d7991c3b2a8d3fc1bcd2a3
```

## 8. 验证命令

```bat
python tools\pangu_gemv_param_host.py selftest --rounds 1000 --seed 20260728
python tools\pangu_gemv_param_host.py --port COM20 info
python tools\pangu_gemv_param_host.py --port COM20 status
python tools\pangu_gemv_param_host.py --port COM20 regression --rounds-per-shape 2 --seed 20260729
python tools\pangu_gemv_param_host.py --port COM20 stress --m 4 --k 64 --rounds 1000 --seed 20260730
python tools\pangu_gemv_param_host.py --port COM20 stress --m 16 --k 65 --rounds 1000 --seed 20260731
python tools\pangu_gemv_param_host.py --port COM20 stress --m 4 --k 895 --rounds 100 --seed 20260801
python tools\pangu_gemv_param_host.py --port COM20 boundary
python tools\pangu_gemv_param_host.py --port COM20 perf --m 4 --k 64
python tools\pangu_gemv_param_host.py --port COM20 perf --m 16 --k 65
python tools\pangu_gemv_param_host.py --port COM20 perf --m 64 --k 896
```

## 9. 2026-07-23 最终验证证据

- Python 参数化金标准：1025 例全部通过，含标准尺寸、尾块边界和固定 M4K64 回归；
- PDS 编译、综合、Device Map、布局布线、时序分析、位流生成全部成功；
- JTAG 识别 `PANGO USB CABLE II` 和 `PGL50H`；
- SRAM 下载 100%，`done bit=1`，未操作 Flash；
- D1.2 已验证固件信息：`PANGU50K GEMV PARAM V1`；
- D1.3 性能计数固件信息：`PANGU50K GEMV PARAM V2`；
- DDR3 初始化完成；
- 标准和尾块共 24 种形状，每种 1 个固定例和 2 个随机例，共 72 例全部上板通过；
- 覆盖标准组合 `M={1,4,16,64}`、`K={16,64,256,896}`；
- 尾块覆盖 `K={1,15,17,63,65,255,257,895}`；
- 固定基线 M=4、K=64：1000/1000 随机上板通过，seed=`20260730`，约 `19.89 s`；
- 尾块 M=16、K=65：1000/1000 随机上板通过，seed=`20260731`，约 `105.27 s`；
- 近最大尾块 M=4、K=895：100/100 随机上板通过，seed=`20260801`，约 `23.90 s`；
- INT32 累加边界：FPGA `[917504, -802816, 57344, 57344]`，与 Python 完全一致；
- 当前 `K<=896` 的单行绝对理论上界为 `896×128×8=917504`，不会发生 INT32 溢出。

## 10. D1.3 性能计数结果

D1.3 已在不覆盖 D1.2 位流的独立构建目录 `../gemv_int4_perf` 中完成：

- M4K64：激活/权重/MAC/总周期=`32/116/64/244`，合并 DDR3 读取带宽 `129.73 MB/s`，端到端 `0.1049 GMAC/s`，瓶颈为 DDR3 读取；
- M16K65：`33/480/320/919`，合并带宽 `218.32 MB/s`，端到端 `0.1132 GMAC/s`，瓶颈为 DDR3 读取；
- M64K896：`86/3152/14336/17912`，合并带宽 `913.16 MB/s`，端到端 `0.3201 GMAC/s`，瓶颈转为 MAC 数量/计算；
- PDS 多角时序全部满足，慢速角 100 MHz WNS=`+0.589 ns`、TNS=`0`；
- 位流 SHA256：`a727f7427143b874da278ae83d7e8a2cdeff8b82bd7c0bb4361e7a2efed73c35`；
- 24 种形状 72 例、M4K64 1000 轮、M16K65 1000 轮、M4K895 100 轮均真实上板通过。

当前唯一下一任务进入 D2：解析真实 `.p50` 模型文件头、张量目录、数据偏移和量化元数据。
