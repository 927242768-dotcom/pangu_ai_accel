# G1 layer0 MLP `SiLU(gate) × up` 独立闭环

本工程只验证 Qwen2.5-0.5B layer0 MLP 的逐元素乘法：

```text
SiLU(gate) [4864] signed int16 Q6.10
×
up_proj    [4864] signed int64 Q28
=
MLP product[4864] signed int64 Q28
```

两路输入直接来自已经分别完成真实 FPGA 逐位验证的 `mlp_silu_g1` 和
`mlp_gate_up_g1`，本工程不重算 gate/up，不执行 `down_proj`，也不修改任何已验证工程或位流。

## 1. 固定数值定义

对每个元素：

1. 执行完整 signed `16 × 64` 乘法；
2. 保留完整 signed 80 bit Q38 乘积，禁止隐含截断；
3. 对乘积绝对值执行 round-to-nearest-even 右移 10 位；
4. 恢复符号；
5. 显式饱和到 signed int64；
6. 输出格式为 signed int64 Q28，可直接作为后续 `down_proj` 的输入。

Python 任意精度参考在 `model_tools/mlp_silu_up_mul_reference.py` 中冻结了相同规则，包含正负
half-way tie、`INT64_MIN/MAX`、完整 80 位乘积和双向饱和测试。

## 2. 数据布局与 DDR3 地址

DDR3 Controller 地址单位为 32 bit；每个 256 bit beat 地址递增 8。

| 区域 | Controller 地址 | 内容 | 大小 |
|---|---:|---|---:|
| SiLU 输入 | `0x0000000` | 4864 个 little-endian int16 Q6.10 | 9728 B / 304 beats |
| up 输入 | `0x0001000` | 4864 个 little-endian int64 Q28 | 38912 B / 1216 beats |
| 结果 | `0x0004000` | 4864 个 little-endian int64 Q28 | 38912 B / 1216 beats |

上传载荷固定为 48640 B：先发送完整 SiLU 数组，再发送完整 up 数组。

## 3. RTL 数据通路

`mlp_silu_up_mul_core.v` 使用：

- 1 个 304×256 SiLU 缓存；
- 4 个 304×256 up bank，将连续 1216 个 up beat 重排为每组 16 个元素；
- 1 个 16×16 无符号乘法器，分四个 16-bit limb 精确重构 80-bit 乘积；
- 独立 magnitude RNE、符号恢复与 int64 饱和；
- 每 4 个 int64 结果组成一个 256-bit beat，连续输出 1216 beats。

这种结构避免实例化宽 16×64 组合乘法器，并让大数组稳定推断为片上 DRM。

## 4. UART 协议

串口固定为 115200、8N1。

| 命令 | 回复/动作 |
|---|---|
| `I` | `PANGU50K MLP SILUUP V1\r\n` |
| `S` | `S + flags + \r\n` |
| `L + 48640 B` | 将两路输入写入 DDR3，成功回复 `K\r\n` |
| `G` | 回复 `R + 38912 B`，即 4864 个 little-endian signed int64 Q28 |

状态位：bit0 DDR3 ready、bit1 loaded、bit2 result valid、bit3 core busy。

## 5. 软件验证

固定真实 query/count 为 `0/1、1/2、5/6、15/16`。四组完整结果 SHA256：

```text
278ceccc804b8f74266b6000745c1ae21d09cf47ba19041ff13cb5cbdaeac0ca
96e1191832febbb2bf246918e489725094567811869ab85bf8452ee8e6520fa9
9f01a9589fc9ee4f8b33acd9a64b8a767b37bee8f788697f0043ac395c7a28dc
297b982da2fb3ee7bd9202cd8d655dec200a9e19fee9a8c614e2e5412ae97802
```

验证结果：

- 新增单元测试 7/7 PASS；
- 完整 `model_tools` 回归 130/130 PASS；
- 固定清单和 48640 B 上传载荷往返 PASS；
- 软件随机/边界压力 1000/1000 PASS，seed=`20260815`；
- 四组真实数据没有触发 int64 饱和，但独立边界测试覆盖正负饱和。

常用命令：

```bat
python model_tools\mlp_silu_up_mul_reference.py check
python model_tools\mlp_silu_up_mul_reference.py stress --rounds 1000 --seed 20260815
python tools\pangu_mlp_silu_up_mul_host.py selftest --rounds 1000 --seed 20260815
```

## 6. PDS 构建与多角时序

默认 PnR 首次完整生成位流，但 Fast Corner core hold 仍有一条 `-0.005 ns`，因此该版本不作为
验收位流。独立 seed17/29 构建达到 `All Constraints Met`：

| 检查 | Slow Corner | Fast Corner |
|---|---:|---:|
| core setup WNS | `+0.511 ns` | `+3.050 ns` |
| core setup TNS | `0.000 ns` | `0.000 ns` |
| core hold WHS | `+0.141 ns` | `+0.065 ns` |
| core hold THS | `0.000 ns` | `0.000 ns` |

恢复、移除和最小脉宽均无违例，最终未布线网络为 0。

资源：

- 7895 LUT；
- 6910 FF；
- 70 distributed RAM；
- 40 DRM；
- 1 APM；
- 79 IO。

构建命令：

```bat
cd mlp_silu_up_mul_g1\pnr_seed17
D:\Pango\PDS_2022.2-SP6.4\bin\pds_shell.exe ^
  -file build_seed17.tcl -project_name mlp_silu_up_mul_seed17
```

验收位流：

```text
mlp_silu_up_mul_g1/pnr_seed17/generate_bitstream/mlp_silu_up_mul_top.sbit
大小：2101696 B
SHA256：a83797a8b2ec75d030fc01144e6bf51e7de0ec930fc135c1a0aba89ebf1c4336
```

## 7. JTAG SRAM 与真实 FPGA 结果

只允许下载到 FPGA 易失性 SRAM：

```bat
cd mlp_silu_up_mul_g1\pnr_seed17
D:\Pango\PDS_2022.2-SP6.4\bin\cdt_cfg_shell.exe ^
  -file program_sram.tcl -work_dir .
```

`program_sram.tcl` 只包含 `cfg_program`，不包含任何 Flash 擦除或写入命令。

2026-07-25 实测：

- JTAG 识别 `PGL50H`；
- SRAM program 成功，`done bit=1`；
- 固件标识 `PANGU50K MLP SILUUP V1`；
- DDR3 初始化成功；
- 四组真实输入均为 `4864/4864` 逐位一致，四个输出 SHA256 与固定清单完全一致；
- 固定四组耗时 30.55 秒；
- 同一随机序列 seed=`20260815` 连续 100/100 真实 FPGA 随机/边界 PASS；
- 每组 4864 个输出均与 Python 任意精度金标准逐位一致；
- 覆盖全零、RNE half-way tie、真实范围、稀疏、完整 int16/int64 bit pattern、`INT64_MIN/MAX`
  和正负饱和。

真实上板命令：

```bat
python tools\pangu_mlp_silu_up_mul_host.py --port COM20 fixed
python tools\pangu_mlp_silu_up_mul_host.py --port COM20 stress ^
  --rounds 100 --seed 20260815
```

上位机还支持 `--start-index`，可将同一固定 seed 的连续随机序列分批执行而不重复样例。

## 8. 结论与边界

`SiLU(gate) × up` 已独立完成软件、完整 PDS、多角时序、JTAG SRAM、四组连贯真实输入和
100 组随机/边界真实 FPGA 逐位验收。当前唯一允许进入的下一任务是独立 `down_proj`；在该阶段
完成前不得合并完整 MLP、残差或 Transformer Block。
