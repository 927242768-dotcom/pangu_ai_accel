# PGL50H layer0 q_proj 完整真实 Linear 层验证工程

## 1. 工程目标

本工程完成 D2 阶段第一个完整真实 Linear 层闭环，固定验收对象为：

```text
权重：model.layers.0.self_attn.q_proj.weight
bias：model.layers.0.self_attn.q_proj.bias
形状：M=896、K=896
分组：group_size=64，每行 14 个 group
输出：896 个 signed int64 Q28
```

计算定义与 `model_tools/linear_quant_reference.py` 完全一致：

```text
group_acc_int32 = sum(INT8 activation * INT4 weight), 每组 64 元素
product_q28     = group_acc_int32 * combined_scale_uq4_28
output_q28      = bias_q28 + sum(product_q28), signed int64
```

本工程位于独立目录，不覆盖以下已验证工程及位流：

- `../gemv_int4_group_q28`
- `../gemv_int4_param`
- `../gemv_int4_perf`
- `../ddr_mac16_integration`

## 2. 数据通路

```text
Python 从真实 .p50 提取完整 q_proj INT4 权重、FP16 scale 和 bias
→ 激活逐向量对称量化为 INT8
→ 主机预计算逐行 UQ4.28 combined scale 和 signed Q28 bias
→ UART 上传 488320 B 完整层载荷
→ FPGA 将载荷逐拍写入 DDR3
→ 激活读取并缓存一次
→ 逐行读取 14 拍权重、2 拍 scale 和 1 拍 padded bias
→ 每组执行 4 次流水 MAC16，形成 INT32 group 点积
→ signed INT32 × unsigned UQ4.28
→ signed INT64 Q28 跨 14 组累加并加入 bias
→ 每 4 行组成 1 个 256 bit 数据拍，立即流式写回 DDR3
→ 计算结束后从 DDR3 逐拍读取 896 个 signed int64
→ UART 流式返回并与 Python 逐元素、逐位比较
```

片上仅缓存完整激活、当前一行权重、当前一行 scale 和 4 行结果，不缓存完整 896 行输出。

## 3. 固定数据格式与上传载荷

- `M=896`
- `K=896`
- `group_size=64`
- 激活：signed INT8，范围 `[-127,127]`
- 权重：packed signed INT4，低半字节在前，真实模型范围 `[-7,7]`
- combined scale：unsigned `UQ4.28`，32 位小端
- bias/output：signed int64 Q28，64 位小端

固定上传载荷共 `488320` 字节：

| 区域 | 大小 | 说明 |
|---|---:|---|
| activation | 896 B | 28 个 256 bit 数据拍 |
| packed weight | 401408 B | 每行 448 B，共 896 行、12544 拍 |
| combined scale | 57344 B | 每行 14×uint32=56 B，补齐为 64 B、2 拍 |
| bias_q28 | 28672 B | 每行 1 个 int64，补齐为 32 B、1 拍 |

固定载荷 SHA256：

```text
71893f988816fcdcf1a58a2c7b453a4fa544224db7cf47c41d39d9aa906251aa
```

## 4. DDR3 地址布局

DDR3 Controller 地址单位为 32 bit；一个 256 bit 数据拍占 8 个地址单位。

| 控制器地址 | 内容 |
|---|---|
| `0x0000000` | 896 B 激活 |
| `0x0001000` | 896 行 packed INT4 权重 |
| `0x0020000` | 896 行 padded UQ4.28 combined scale |
| `0x0024000` | 896 行 padded signed int64 bias_q28 |
| `0x0026000` | 896 个 signed int64 output_q28 |

## 5. UART 协议 V1

串口参数：`115200, 8N1`。

| 命令 | 请求 | 回复 |
|---|---|---|
| `I` | 无 | `PANGU50K QPROJ FULL V1\r\n` |
| `S` | 无 | `S + flags + \r\n` |
| `L` | 固定 488320 B 载荷 | `K\r\n` |
| `G` | 无 | `R + 896×little-endian signed int64` |

状态字节：

- bit0：DDR3 初始化完成；
- bit1：完整层数据已加载；
- bit2：完整层结果有效；
- bit3：计算核心忙。

## 6. 主要文件

| 文件 | 作用 |
|---|---|
| `rtl/int8_dot16_pipe.v` | 显式平衡流水的 16 路 signed INT8 点积 |
| `rtl/gemv_qproj_full_core.v` | 单行 896 元素分组点积、UQ4.28 乘法和 signed INT64 累加 |
| `rtl/gemv_qproj_full_ctrl.v` | 完整层 UART、DDR3 行调度、结果流式写回与流式返回 |
| `rtl/gemv_qproj_full_top.v` | DDR3 IP、控制器、UART 和 LED 顶层 |
| `pnr/build_gemv_qproj_full.tcl` | PDS 全流程构建脚本 |
| `pnr/program_sram.tcl` | 仅下载 FPGA 易失性 SRAM，不操作 Flash |
| `../tools/pangu_gemv_qproj_full_host.py` | 完整 `.p50` 载荷、金标准、固定与随机上板验证工具 |
| `../model_tools/q_proj_full_reference.json` | 完整层固定向量关键值和 SHA256 清单 |

## 7. 构建与下载

在 `gemv_int4_qproj_full/pnr` 目录执行：

```bat
D:\Pango\PDS_2022.2-SP6.4\bin\pds_shell.exe ^
  -file build_gemv_qproj_full.tcl ^
  -project_name gemv_qproj_full
```

仅下载到易失性 SRAM：

```bat
D:\Pango\PDS_2022.2-SP6.4\bin\cdt_cfg_shell.exe ^
  -file program_sram.tcl ^
  -work_dir .
```

最终位流：

```text
pnr/generate_bitstream/gemv_qproj_full_top.sbit
SHA256: 432454b80678c11f493856cb725d791e271d86eada1b5cabccefc0d7486f8894
```

## 8. 验证命令

在项目根目录执行：

```bat
python tools\pangu_gemv_qproj_full_host.py selftest --rounds 1000 --seed 20260725
python tools\pangu_gemv_qproj_full_host.py --port COM20 info
python tools\pangu_gemv_qproj_full_host.py --port COM20 status
python tools\pangu_gemv_qproj_full_host.py --port COM20 fixed
python tools\pangu_gemv_qproj_full_host.py --port COM20 stress --rounds 3 --seed 20260725
```

## 9. 2026-07-24 最终验证结果

### Python 软件闭环

- 完整真实权重、scale、bias 只解析一次，并在 1000 组不同激活间复用；
- 固定载荷打包/解包、补齐区域和独立 Q28 重算：PASS；
- 完整层随机激活软件压力测试：`1000/1000 PASS`；
- seed 起点：`20260725`；
- 耗时：约 `25.88` 秒；
- 固定输出 SHA256：

```text
ea1f04bf4ff313dad07025ff35e66a088f13afd28d817422b89bb135f63525a0
```

固定输出前 8 行：

```text
[207253689, -173360554, 287606739, -223225713,
 -4687768431, 104298201, 301392249, 13611309]
```

固定输出后 8 行：

```text
[1043114578, -2138877247, -1060954045, 241305015,
 831408625, 67617978, 249103231, 582822115]
```

前 4 行与已验证的 `gemv_int4_group_q28` M4K896 小闭环完全一致。

### PDS 实现与时序

- 编译、综合、Device Map、布局布线、时序分析、位流生成：全部成功；
- 最终未布线网络：0；
- 时序：`Design Summary : All Constraints Met.`；
- 慢速角 100 MHz 建立：WNS=`+0.670 ns`，TNS=`0`；
- 慢速角保持：WHS=`+0.171 ns`，THS=`0`；
- 慢速角恢复：WNS=`+3.011 ns`，TNS=`0`；
- 慢速角移除：WHS=`+0.515 ns`，THS=`0`；
- 快速角建立：WNS=`+3.034 ns`，TNS=`0`；
- 快速角保持：WHS=`+0.100 ns`，THS=`0`；
- 快速角恢复：WNS=`+4.968 ns`，TNS=`0`；
- 快速角移除：WHS=`+0.319 ns`，THS=`0`；
- 最小脉宽无违例；
- 资源：LUT=`8510`、FF=`7619`、DRM=`4`、APM=`12`。

### 真实上板

- JTAG 识别 `PANGO USB CABLE II` 和 `PGL50H`；
- SRAM 下载进度 100%，`done bit=1`；
- 未擦写或编程 Flash；
- 固件：`PANGU50K QPROJ FULL V1`；
- DDR3 初始化成功；
- 固定完整层：896 个 signed int64 与 Python Q28 金标准逐位完全一致；
- 固定完整层上传、计算和回读总耗时约 `43.03` 秒；
- 随机激活完整层回归：`3/3 PASS`；
- activation seed：`20260725`、`20260726`、`20260727`；
- 随机上板总耗时约 `130.13` 秒。

## 10. 当前边界与下一步

本工程证明了真实 `.p50` 完整 Linear 层的逐行权重、scale、bias 调度，signed INT64 Q28 计算，以及完整输出流式写回和返回均可在真实 PGL50H 开发板上正确运行。

当前仍未实现 RMSNorm、Attention、MLP 或完整 Transformer Block。按照 `PROJECT_ROADMAP.md`，下一步进入 E1 RMSNorm，先确定定点格式和 `rsqrt` 近似方案，建立 Python 金标准与独立小闭环工程。
