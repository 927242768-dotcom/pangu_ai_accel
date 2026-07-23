# PGL50H 参数化 packed INT4 GEMV 性能计数工程

## 1. 工程目标

本目录用于 D1.3 GEMV 性能基础设施验证。计算数据通路继续复用已经完成 D1.2 验证的参数化 GEMV RTL，构建输出放在独立目录，避免覆盖原 D1.2 位流。

支持范围保持不变：

```text
1 <= M <= 64
1 <= K <= 896
```

## 2. 新增性能计数

每次收到 `G` 命令后清零并启动计数，在最后一个输出数据拍写回 DDR3 后停止：

1. 激活 DDR3 读取周期：统计激活 AXI 读地址准备、握手等待和数据返回状态。
2. 权重 DDR3 读取周期：统计所有权重行 AXI 读地址准备、握手等待和数据返回状态。
3. MAC 计算周期：统计参数化 GEMV 核心 `busy` 为高的周期。
4. GEMV 总周期：从进入激活读取流程到最后一个输出数据拍写回完成，不包含 UART 返回耗时。

核心时钟为 100 MHz。

## 3. UART 协议增量

原有 `I/S/C/L/G` 命令保持兼容，固件信息升级为：

```text
PANGU50K GEMV PARAM V2
```

新增命令：

| 命令 | 回复 |
|---|---|
| `P` | `P + uint32_le(act_read_cycles) + uint32_le(weight_read_cycles) + uint32_le(mac_cycles) + uint32_le(total_cycles)` |

状态字节 bit5 表示最近一次性能计数有效。尚无有效计数时读取 `P`，返回错误码 `0x06`。

## 4. 上位机计算指标

`tools/pangu_gemv_param_host.py perf` 根据 FPGA 周期计数计算：

- 激活、权重和合并 DDR3 实测读取带宽；
- 核心计算阶段与端到端 GMAC/s；
- 相对单套 MAC16 理论峰值 1.6 GMAC/s 的利用率；
- DDR3 读取、MAC 计算、控制与结果写回三类周期占用；
- 当前主瓶颈分类。

## 5. 构建和验证命令

PDS 构建：

```bat
cd gemv_int4_perf\pnr
D:\Pango\PDS_2022.2-SP6.4\bin\pds_shell.exe -file build_gemv_perf.tcl -project_name gemv_param_perf
```

仅下载到 FPGA 易失性 SRAM：

```bat
D:\Pango\PDS_2022.2-SP6.4\bin\cdt_cfg_shell.exe -file program_sram.tcl -work_dir .
```

上位机性能测试：

```bat
python tools\pangu_gemv_param_host.py --port COM20 perf --m 4 --k 64
python tools\pangu_gemv_param_host.py --port COM20 perf --m 16 --k 65
python tools\pangu_gemv_param_host.py --port COM20 perf --m 64 --k 896
```

原有参数化回归仍必须通过：

```bat
python tools\pangu_gemv_param_host.py selftest --rounds 1000 --seed 20260728
python tools\pangu_gemv_param_host.py --port COM20 regression --rounds-per-shape 2 --seed 20260729
python tools\pangu_gemv_param_host.py --port COM20 stress --m 4 --k 64 --rounds 1000 --seed 20260730
python tools\pangu_gemv_param_host.py --port COM20 stress --m 16 --k 65 --rounds 1000 --seed 20260731
```

## 6. 最终验收结果（2026-07-23）

- [x] Python 金标准与性能计算公式自检通过：1025 例，seed=`20260728`。
- [x] PDS 编译、综合、Device Map、布局布线和多角时序通过，0 条未布线网络。
- [x] 位流真实下载到开发板 SRAM，进度 100%，`done bit=1`，未操作 Flash。
- [x] 固定尺寸、尾块尺寸和最大尺寸性能计数实测完成。
- [x] 参数化回归与随机压力测试通过。
- [x] 实测带宽、GMAC/s、利用率和瓶颈结论已同步到项目路线图。

资源与时序：

- LUT=`10906`、Register=`8269`、DRM18K=`4`、APM=`9`；
- `Design Summary : All Constraints Met.`；
- 慢速角 100 MHz WNS=`+0.589 ns`、TNS=`0`，WHS=`+0.142 ns`、THS=`0`；
- 快速角 WNS=`+3.074 ns`、TNS=`0`，WHS=`+0.065 ns`、THS=`0`。

代表性性能：

| 形状 | 激活读周期 | 权重读周期 | MAC周期 | 总周期 | 合并DDR3带宽 | 核心GMAC/s | 端到端GMAC/s | 端到端利用率 | 主瓶颈 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| M4K64 | 32 | 116 | 64 | 244 | 129.73 MB/s | 0.4000 | 0.1049 | 6.56% | DDR3读取 |
| M16K65 | 33 | 480 | 320 | 919 | 218.32 MB/s | 0.3250 | 0.1132 | 7.07% | DDR3读取 |
| M64K896 | 86 | 3152 | 14336 | 17912 | 913.16 MB/s | 0.4000 | 0.3201 | 20.01% | MAC数量/计算 |

真实上板回归：

- 24 种形状、72 例全部 PASS；
- M4K64：1000/1000 PASS，seed=`20260730`，约 19.79 秒；
- M16K65：1000/1000 PASS，seed=`20260731`，约 105.26 秒；
- M4K895：100/100 PASS，seed=`20260801`，约 23.90 秒；
- INT32 边界结果 `[917504, -802816, 57344, 57344]` 与 Python 一致。

最终位流：

```text
pnr/generate_bitstream/gemv_param_top.sbit
SHA256: a727f7427143b874da278ae83d7e8a2cdeff8b82bd7c0bb4361e7a2efed73c35
```
