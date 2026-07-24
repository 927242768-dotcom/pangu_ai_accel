# PGL50H 真实模型分组 UQ4.28 GEMV 验证工程

## 1. 工程目标

本工程完成 D2 阶段第一条真实模型分组定点计算闭环，固定验收对象为：

```text
张量：model.layers.0.self_attn.q_proj.weight
切片：输出行 0..3、输入列 0..895
形状：M=4、K=896
分组：group_size=64，共 14 个 group
```

计算定义与 `model_tools/linear_quant_reference.py` 完全一致：

```text
group_acc_int32 = sum(INT8 activation * INT4 weight), 每组 64 元素
product_q28     = group_acc_int32 * combined_scale_uq4_28
output_q28      = bias_q28 + sum(product_q28), signed int64
```

工程是独立目录，不覆盖以下已验证工程及位流：

- `../gemv_int4_param`
- `../gemv_int4_perf`
- `../ddr_mac16_integration`

## 2. 数据通路

```text
Python 从真实 .p50 提取 q_proj INT4 权重和 FP16 scale
→ 激活逐向量对称量化为 INT8
→ 主机预计算 UQ4.28 combined scale 和 signed Q28 bias
→ UART 上传激活、packed INT4、scale、bias
→ FPGA 写入 DDR3
→ 激活读取并缓存一次
→ 逐行读取 14 拍权重和 2 拍 scale
→ 每组执行 4 次流水 MAC16，形成 INT32 group 点积
→ signed INT32 × unsigned UQ4.28
→ signed INT64 Q28 跨组累加并加入 bias
→ 4 个 signed INT64 写回 DDR3
→ UART 返回并与 Python 逐位比较
```

## 3. 固定数据格式

- `M=4`
- `K=896`
- `group_size=64`
- 激活：signed INT8，范围 `[-127,127]`
- 权重：packed signed INT4，低半字节在前，真实模型范围 `[-7,7]`
- combined scale：unsigned `UQ4.28`，32 位小端
- bias/output：signed int64 Q28，64 位小端

固定上传载荷共 `2976` 字节：

| 区域 | 大小 | 说明 |
|---|---:|---|
| activation | 896 B | 28 个 256 bit 数据拍 |
| packed weight | 1792 B | 每行 448 B，共 4 行、56 拍 |
| combined scale | 256 B | 每行 14×uint32=56 B，补齐为 64 B、2 拍 |
| bias_q28 | 32 B | 4×signed int64，1 拍 |

## 4. DDR3 地址布局

DDR3 Controller 地址单位为 32 bit，一个 256 bit 数据拍占 8 个地址单位。

| 控制器地址 | 内容 |
|---|---|
| `0x000` | 896 B 激活 |
| `0x100` | 4 行 packed INT4 权重 |
| `0x400` | 4 行 UQ4.28 combined scale |
| `0x500` | 4 个 signed int64 bias_q28 |
| `0x600` | 4 个 signed int64 output_q28 |

## 5. UART 协议 V1

串口参数：`115200, 8N1`。

| 命令 | 请求 | 回复 |
|---|---|---|
| `I` | 无 | `PANGU50K GEMV Q28 V1\r\n` |
| `S` | 无 | `S + flags + \r\n` |
| `L` | 固定 2976 B 载荷 | `K\r\n` |
| `G` | 无 | `R + 4×little-endian signed int64` |

状态字节：

- bit0：DDR3 初始化完成；
- bit1：数据已加载；
- bit2：结果有效；
- bit3：计算核心忙。

## 6. 主要文件

| 文件 | 作用 |
|---|---|
| `rtl/int8_dot16_pipe.v` | 显式平衡流水的 16 路 signed INT8 点积 |
| `rtl/gemv_group_q28_core.v` | 64 元素分组点积、UQ4.28 乘法和 signed INT64 Q28 累加 |
| `rtl/gemv_group_q28_ctrl.v` | UART、DDR3 地址调度、逐行计算和结果返回 |
| `rtl/gemv_group_q28_top.v` | DDR3 IP、控制器、UART 和 LED 顶层 |
| `pnr/build_gemv_group_q28.tcl` | PDS 全流程构建脚本 |
| `pnr/program_sram.tcl` | 仅下载 FPGA 易失性 SRAM，不操作 Flash |
| `../tools/pangu_gemv_group_q28_host.py` | 固定真实向量、载荷自检和随机上板压力测试工具 |

## 7. 构建与下载

在 `gemv_int4_group_q28/pnr` 目录执行：

```bat
D:\Pango\PDS_2022.2-SP6.4\bin\pds_shell.exe -file build_gemv_group_q28.tcl -project_name gemv_group_q28
```

仅下载到易失性 SRAM：

```bat
D:\Pango\PDS_2022.2-SP6.4\bin\cdt_cfg_shell.exe -file program_sram.tcl -work_dir .
```

最终位流：

```text
pnr/generate_bitstream/gemv_group_q28_top.sbit
SHA256: d8c7d194d4d8ce1e5d189df39fae5fc904030fe4be6e981a5876a4df73ea17bd
```

## 8. 验证命令

在项目根目录执行：

```bat
python tools\pangu_gemv_group_q28_host.py selftest --rounds 1000 --seed 20260724 --include-real
python tools\pangu_gemv_group_q28_host.py --port COM20 info
python tools\pangu_gemv_group_q28_host.py --port COM20 status
python tools\pangu_gemv_group_q28_host.py --port COM20 fixed
python tools\pangu_gemv_group_q28_host.py --port COM20 stress --rounds 1000 --seed 20260724
```

## 9. 2026-07-24 最终验证结果

### Python 软件闭环

- 载荷打包/解包、group INT32 点积、uint32 scale 和 signed int64 累加：`1000/1000 PASS`；
- 随机种子：`20260724`；
- 真实 q_proj 固定向量确定性重建并与 JSON 清单一致；
- 固定 Q28 输出：

```text
[207253689, -173360554, 287606739, -223225713]
```

### PDS 实现与时序

- 编译、综合、Device Map、布局布线、时序分析、位流生成：全部成功；
- 最终未布线网络：0；
- 时序：`Design Summary : All Constraints Met.`；
- 慢速角 100 MHz 建立：WNS=`+0.909 ns`，TNS=`0`；
- 慢速角保持：WHS=`+0.111 ns`，THS=`0`；
- 快速角建立：WNS=`+3.041 ns`，TNS=`0`；
- 快速角保持：WHS=`+0.051 ns`，THS=`0`；
- 恢复/移除和最小脉宽均无违例；
- 资源：LUT=`8379`、FF=`7492`、DRM=`4`、APM=`12`。

首版直接复用组合 MAC16，慢速角出现 WNS=`-0.109 ns`、TNS=`-0.163 ns` 的 2 个违例端点。改为显式平衡流水 MAC16 后，最终多角时序全部通过。

### 真实上板

- JTAG 识别 `PANGO USB CABLE II` 和 `PGL50H`；
- SRAM 下载进度 100%，`done bit=1`；
- 未擦写或编程 Flash；
- 串口 `COM20`，DDR3 初始化成功；
- 固定真实向量：FPGA 四个 signed int64 与软件参考逐位完全一致；
- 边界向量：覆盖 scale bit31 和 `0xFFFFFFFF`，PASS；
- 随机分组缩放压力测试：`1000/1000 PASS`，seed=`20260724`，耗时约 `266.06` 秒。

## 10. 当前边界与下一步

本工程已经证明真实 `.p50` 权重的 group scale 可由主机转换为 UQ4.28，并在 FPGA 中完成精确的分组点积、64 位定点乘加和 bias 闭环。

当前仍固定为 q_proj 的前 4 行，尚未完成一个完整真实 Linear 层。下一步应在独立工程中扩展输出行调度和结果流式写回，完成 layer0 `q_proj` 全部输出行，并与软件量化参考逐元素比较。
