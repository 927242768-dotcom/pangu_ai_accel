# G1 layer0 MLP gate_proj/up_proj 真实双投影工程

## 1. 阶段目标

本工程完成 Qwen2.5-0.5B layer0 MLP 的两条入口投影闭环：

```text
已验证 post_attention_layernorm [896] signed Q6.10
→ 同一份逐向量对称 INT8 激活
├─ gate_proj [4864,896] groupwise INT4 → gate [4864] signed int64 Q28
└─ up_proj   [4864,896] groupwise INT4 → up   [4864] signed int64 Q28
```

本阶段只完成 gate/up 两条真实 Linear。尚未执行 `SiLU(gate)`、`SiLU(gate) × up`、`down_proj` 或第二处残差。

## 2. 真实模型参数

| 投影 | 权重张量 | shape | `.p50` packed INT4 | 原始 FP16 scale |
|---|---|---:|---:|---:|
| gate | `model.layers.0.mlp.gate_proj.weight` | `[4864,896]` | 2179072 B | 136192 B |
| up | `model.layers.0.mlp.up_proj.weight` | `[4864,896]` | 2179072 B | 136192 B |

共同规则：

- group size：64；
- 每输出行 groups：14；
- signed symmetric INT4，zero point=0；
- `.p50` 均不存在 `gate_proj.bias` 或 `up_proj.bias`；
- 硬件载荷为保持通用 Linear 数据路，仍保留每行 32 B bias 槽，但必须全零。

## 3. 软件与硬件定点定义

1. 输入是上一阶段真实上板通过的 signed Q6.10 int16 `[896]`；
2. Q6.10 精确转换为 float32；
3. 两路共享同一份逐向量对称 INT8 激活，范围 `[-127,127]`，zero point=0，RNE；
4. 主机把 `activation_scale × weight_scale` 预量化为 unsigned UQ4.28；
5. 每个 64 元素 group 执行 signed INT32 点积；
6. 点积乘 UQ4.28 scale，并在 signed int64 Q28 中跨 14 groups 累加；
7. bias 固定为 0；输出为 `[4864]` little-endian signed int64 Q28。

软件参考对每一行额外执行独立整数重算，结果必须与通用 Linear 参考逐位相同。

## 4. 四组连贯真实固定输入

输入直接来自 G1 `post_attention_layernorm` 已验证的 1、2、6、16-token 连贯 Attention 链。

| query/count | 输入 Q10 SHA256 | gate Q28 SHA256 | up Q28 SHA256 |
|---|---|---|---|
| `0/1` | `93d2d3ee866a7923e3ce9d450ae5d6e43a05c50daeaa952cae052c4584891f80` | `4c1c79e14e8f788aeaaaea64924863f847ca276fd8b99b7406cfb6a50fbcea4e` | `9794e50eb90d560dfcfb55a2e54687ea3e3dcd06da368aeb557885c5e2a605a0` |
| `1/2` | `0ef1296dde8e999f6ac707725da227bd8f87b5da848a7a81113f422a03d0cbdf` | `42bbe0f30579275a6411abd9ad020639e0aedb9060efdccb544f5e5f4a3203c3` | `7eb40a12c870187737f47231342d02806f30a87076575cf4e00aecb361dfcc62` |
| `5/6` | `40965e0cb4d96cf8de644d4b7081df5acef34d6c24ec8cd6d448fac4943b83aa` | `869b64d81d6c5f2cacc314fd869a2e20eceee7014571b0974595a81b8acf34dc` | `6b09b2ba30bba3ffb742cbd6ebf8a257322b78f779e5c1b86ffacdc2cb96d31d` |
| `15/16` | `fa574c09c76580173c62d59bd5a682cd35bb97b70d25459dcf0ac6e3808e48b1` | `449c12f1f2904a1c4f56892a4c7049f7c785862b4c9ad9b7922b1990f161f7f6` | `03eca75bbbb7e9849f549124ddc1a0e4506f4bb64bb79daf769ec01d2d368041` |

四组输入中，gate/up 的 INT8 激活数组和 activation scale 均逐位/数值完全相同；combined scale 均未发生 UQ4.28 饱和。

## 5. UART 协议和 DDR3 载荷

串口：`115200, 8N1`。

| 命令 | 请求 | 回复 |
|---|---|---|
| `I` | 固件信息 | `PANGU50K MLP GATEUP V1\r\n` |
| `S` | 状态 | `S + flags + \r\n` |
| `L` | 上传当前 gate 或 up 的完整载荷 | `K\r\n` |
| `G` | 执行当前已加载投影 | `R + 38912 B Q28` |

每路载荷：

```text
activation_int8[896]                    896 B       28 beats
packed_weight_int4[4864][896]       2179072 B    68096 beats
combined_scale_uq4_28[4864][14]      311296 B     9728 beats（每行补齐 64 B）
zero_bias_q28[4864]                   155648 B     4864 beats（每行补齐 32 B）
合计                                  2646912 B    82716 beats
结果                                    38912 B     1216 beats
```

DDR3 控制器地址单位为 32 bit，一个 256-bit beat 占 8 个地址单位：

| 区域 | 起始地址 | 结束上界 |
|---|---:|---:|
| activation | `0x000000` | `<0x001000` |
| weight | `0x001000` | `0x086000` |
| scale | `0x090000` | `0x0A3000` |
| zero bias | `0x0A4000` | `0x0AD800` |
| result | `0x0B0000` | `0x0B2600` |

各区域互不重叠。控制器只缓存完整 896 元素激活和当前一行的 14 个权重 beat/2 个 scale beat；每 4 行结果立即组成一个 256-bit beat 写回 DDR3。

## 6. 工程隔离和主要文件

本工程未修改或覆盖已验证的 q_proj、QKV、Attention、RMSNorm 工程和位流。只复用已真实上板通过的 K=896、group=64 Q28 Linear 核：

- `../gemv_int4_qproj_full/rtl/gemv_qproj_full_core.v`
- `../gemv_int4_qproj_full/rtl/int8_dot16_pipe.v`
- `../ddr_mac16_integration/rtl/int4_unpack16.v`

新增文件：

| 文件 | 作用 |
|---|---|
| `../model_tools/mlp_gate_up_reference.py` | 双投影真实参数、四组连贯输入、独立 Q28 重算、载荷和压力参考 |
| `../model_tools/mlp_gate_up_g1_reference.json` | 四组 gate/up 输出和关键数组 SHA256 清单 |
| `../model_tools/test_mlp_gate_up_reference.py` | shape、共享激活、无 bias、载荷、独立重算和零输入测试 |
| `rtl/mlp_gate_up_ctrl.v` | 4864 行计数、DDR3 流式加载/计算/回写/回读控制器 |
| `rtl/mlp_gate_up_top.v` | 独立 DDR3/UART/LED 顶层 |
| `pnr/build_mlp_gate_up.tcl` | 独立 PDS 全流程脚本 |
| `pnr/program_sram.tcl` | 只下载易失性 SRAM |
| `../tools/pangu_mlp_gate_up_host.py` | 固件、状态、固定和随机/边界上板逐位比较 |

## 7. 软件验证

结果：

- 新增测试：6/6 PASS；
- 完整 `model_tools`：116/116 PASS，另有 11 项按既有可选环境条件跳过；
- 四组真实固定清单：PASS；
- gate/up 软件随机/边界：1000/1000 PASS，seed=`20260808`；
- 覆盖全零、交替 `INT16_MIN/MAX`、常量、稀疏、小幅值和一般/完整 int16 随机输入；
- 两路共享激活检查、无 bias 全零检查、payload 往返、独立整数 Q28 重算均通过。

命令：

```bat
python -B model_tools\mlp_gate_up_reference.py check
python -B model_tools\mlp_gate_up_reference.py stress --rounds 1000 --seed 20260808
cd model_tools
python -B -m unittest discover -v
cd ..
python -B tools\pangu_mlp_gate_up_host.py selftest --rounds 10 --seed 20260808
```

## 8. PDS 实现、资源和时序

构建：

```bat
cd mlp_gate_up_g1\pnr
D:\Pango\PDS_2022.2-SP6.4\bin\pds_shell.exe -file build_mlp_gate_up.tcl -project_name mlp_gate_up
```

Compile、Synthesize、Device Map、Place & Route、Timing 和 Bitstream 全部成功；最终未布线网络为 0。

资源：

- LUT：8548；
- FF：7628；
- distributed RAM：326；
- DRM：4；
- APM：12；
- IO：79。

`Design Summary : All Constraints Met.`

| 角 | 100 MHz core setup WNS/TNS | core hold WHS/THS |
|---|---|---|
| Slow | `+0.916 ns / 0` | `+0.157 ns / 0` |
| Fast | `+3.046 ns / 0` | `+0.089 ns / 0` |

快慢角 recovery、removal、minimum pulse width 均无违例，所有 TNS/THS/TPWS 为 0。

## 9. 位流和真实上板证据

位流：

```text
mlp_gate_up_g1/pnr/generate_bitstream/mlp_gate_up_top.sbit
大小：2101696 B
SHA256：e72959d2968a543bf3a2bcfd31f2b2c7a0d31a9888daba9ceac2d7c50cd5db6b
```

仅通过 JTAG 写 SRAM：

```bat
D:\Pango\PDS_2022.2-SP6.4\bin\cdt_cfg_shell.exe -file program_sram.tcl -work_dir .
```

实测：

- JTAG 识别 `PANGO USB CABLE II`、`PGL50H`；
- SRAM 下载 100%，`done bit=1`，未操作 Flash；
- 固件 `PANGU50K MLP GATEUP V1`，DDR3 初始化成功；
- 四组连贯真实输入的 gate/up 共 8 个完整投影全部 `4864/4864` 逐位一致；
- 单路完整上传、计算和回读约 `232.99~233.02 s`；
- gate 真实 FPGA 压力：全零、交替 INT16 极值、一般随机 `3/3 PASS`；
- up 真实 FPGA 压力：全零、交替 INT16 极值、一般随机 `3/3 PASS`；
- 双路压力合计 `6/6 PASS`，seed=`20260808`，mode index=`0/1/6`。

固定测试：

```bat
python -B tools\pangu_mlp_gate_up_host.py --port COM20 fixed --case 0 --projection gate
python -B tools\pangu_mlp_gate_up_host.py --port COM20 fixed --case 0 --projection up
rem case 1、2、3 同样分别运行 gate/up
```

压力测试：

```bat
python -B tools\pangu_mlp_gate_up_host.py --port COM20 stress --projection gate --rounds 1 --seed 20260808 --start-index 0
python -B tools\pangu_mlp_gate_up_host.py --port COM20 stress --projection gate --rounds 1 --seed 20260808 --start-index 1
python -B tools\pangu_mlp_gate_up_host.py --port COM20 stress --projection gate --rounds 1 --seed 20260808 --start-index 6
rem up 投影执行相同三组命令
```

## 10. 阶段结论和下一步

layer0 `gate_proj` 与 `up_proj` 已满足软件金标准、完整载荷、独立硬件调度、DDR3 流式回写、PDS、多角时序、JTAG SRAM、四组真实固定输入和随机/边界逐位验收条件。

当前唯一下一任务是实现 `SiLU(gate)`：先明确 gate signed int64 Q28 到非线性输入格式的 RNE/饱和边界，复用或扩展已验证 SiLU 定点定义，并独立完成软件、PDS、多角时序和真实上板逐位闭环。`SiLU(gate)` 单独通过前不得执行 `SiLU(gate) × up`。
