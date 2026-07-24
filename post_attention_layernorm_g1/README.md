# G1 layer0 post_attention_layernorm 独立验证工程

## 1. 阶段目标

本工程完成 Qwen2.5-0.5B layer0 MLP 入口的真实 `post_attention_layernorm` 闭环：

```text
已验证完整 layer0 Attention 子层输出 [896] signed Q6.10
→ mean(x²) + epsilon
→ LUT256 rsqrt
→ x × rsqrt × 真实 post-attention gamma
→ RNE 与 signed int16 饱和
→ MLP 输入 [896] signed Q6.10
```

真实 gamma：

```text
model.layers.0.post_attention_layernorm.weight
shape = [896]
源类型 = bfloat16
.p50 存储 = 连续 FP16，1792 B
rms_norm_eps = 1e-6
```

本阶段只完成 MLP 输入归一化，不包含 gate/up projection、SiLU、down projection 或第二处残差。

## 2. 复用与隔离边界

数值核心严格复用已经真实上板通过的 E1 `rmsnorm_k896` RTL：

- `rmsnorm_k896/rtl/rmsnorm_k896_core.v`
- `rmsnorm_k896/rtl/rmsnorm_k896_ctrl.v`
- `rmsnorm_k896/rtl/rmsnorm_k896_top.v`

但软件清单、真实输入来源、上位机、PDS 工程目录、实现数据库和位流均独立放在 G1 路径中，未修改或覆盖 E1、F6 的已验证工程和位流。

由于 UART/DDR3 控制器逐位复用 E1，固件信息仍为：

```text
PANGU50K RMSNORM K896 V1
```

该字符串表示复用的协议版本；G1 工作负载由独立 PDS 位流、真实 post-attention gamma 和独立固定清单区分。

## 3. 定点定义

| 数据 | 格式 |
|---|---|
| Attention 子层输入 | signed Q6.10 int16 `[896]` |
| gamma | signed Q6.10 int16 `[896]` |
| `sum(x²)` | unsigned 40 bit，保留 20 位小数 |
| `mean(x²)+epsilon` | Q12.20，`epsilon_q20=1` |
| rsqrt | LUT256 midpoint，unsigned UQ12.20 uint32 |
| 输出 | signed Q6.10 int16 `[896]` |

所有除法和右移采用 round-to-nearest-even；输出显式饱和到 signed int16。

## 4. 连贯真实固定输入

固定输入直接来自 F6 已真实上板逐位通过的完整 layer0 Attention 子层输出，覆盖 1、2、6、16-token 窗口。

| query/count | 输入 SHA256 | G1 输出 SHA256 |
|---|---|---|
| `0/1` | `36859690e421b96cb8db65a5760a364d165a73b63fd1121040a7d1b42c042eb7` | `93d2d3ee866a7923e3ce9d450ae5d6e43a05c50daeaa952cae052c4584891f80` |
| `1/2` | `2b4a2d9240e6e30c2afe2943fa30ac60decd47f8fc8d377ab7e530e516009378` | `0ef1296dde8e999f6ac707725da227bd8f87b5da848a7a81113f422a03d0cbdf` |
| `5/6` | `c0c0776d71e3dc97aa1a4d4e0709f38441cc82717c0c3081a79c47c30a21af10` | `40965e0cb4d96cf8de644d4b7081df5acef34d6c24ec8cd6d448fac4943b83aa` |
| `15/16` | `7e61dc1fd0eb43b231e25fe1d08b1c08342723f537b6e25924858608235fa61e` | `fa574c09c76580173c62d59bd5a682cd35bb97b70d25459dcf0ac6e3808e48b1` |

真实 gamma Q6.10 SHA256：

```text
8528c6dd7c3a26264d1af12a852c3c536ffdd75d1aa2716733a6fb185b43dd1c
```

四组固定输入和 gamma 均未发生量化截断，固定输出均未发生饱和；LUT256 相对精确 rsqrt 路径最大差值为 2 个 Q10 LSB。

## 5. UART 协议与载荷

串口参数：`115200, 8N1`。

| 命令 | 请求 | 回复 |
|---|---|---|
| `I` | 固件信息 | `PANGU50K RMSNORM K896 V1\r\n` |
| `S` | 状态查询 | `S + flags + \r\n` |
| `L` | 4608 B 固定载荷 | `K\r\n` |
| `G` | 启动计算 | `R + 1792 B signed Q6.10` |

上传载荷：

```text
input_q10[896]        1792 B
post_attn_gamma[896]  1792 B
rsqrt_lut256          1024 B
合计                  4608 B
```

## 6. 软件验证

主要文件：

| 文件 | 作用 |
|---|---|
| `../model_tools/post_attention_layernorm_reference.py` | 连贯 Attention 输入、真实 gamma、固定清单和随机/边界参考 |
| `../model_tools/post_attention_layernorm_g1_reference.json` | 四组真实固定输入与输出 SHA256 |
| `../model_tools/test_post_attention_layernorm_reference.py` | 元数据、Q10 往返、载荷、固定链和 1000 轮压力测试 |
| `../tools/pangu_post_attention_layernorm_host.py` | 软件自检、固定与随机/边界上板逐位比较 |

结果：

- G1 新增测试：5/5 PASS；
- 完整 `model_tools`：110/110 PASS，另有 11 项因本地可选条件跳过；
- 软件随机/边界：1000/1000 PASS，seed=`20260807`；
- 覆盖全零、交替 `INT16_MIN/MAX`、常量、稀疏、小幅值、一般和完整 int16 随机输入；
- 完整 int16 极端范围内，LUT256 相对精确 rsqrt 路径最大差值为 8 个 Q10 LSB。

软件命令：

```bat
python -B model_tools\post_attention_layernorm_reference.py check
python -B model_tools\post_attention_layernorm_reference.py stress --rounds 1000 --seed 20260807
python -B tools\pangu_post_attention_layernorm_host.py selftest --rounds 1000 --seed 20260807
python -B -m unittest discover -s model_tools -p "test_*.py"
```

## 7. PDS 实现与时序

构建命令：

```bat
cd post_attention_layernorm_g1\pnr
D:\Pango\PDS_2022.2-SP6.4\bin\pds_shell.exe -file build_post_attention_layernorm.tcl -project_name post_attention_layernorm_g1
```

PDS Compile、Synthesize、Device Map、Place & Route、Timing 和 Bitstream 全部成功，最终未布线网络为 0。

资源：

- LUT：8801；
- FF：7051；
- distributed RAM：70；
- DRM：12；
- APM：9；
- IO：79。

多角时序结论：`Design Summary : All Constraints Met.`

| 角 | core setup WNS/TNS | core hold WHS/THS |
|---|---|---|
| Slow | `+0.411 ns / 0` | `+0.169 ns / 0` |
| Fast | `+2.857 ns / 0` | `+0.100 ns / 0` |

恢复和移除检查在快慢角均无违例，所有 TNS/THS 为 0。

## 8. 位流与真实上板

位流：

```text
post_attention_layernorm_g1/pnr/generate_bitstream/rmsnorm_k896_top.sbit
大小：2101696 B
SHA256：b8c87ee10edf435617ab110cfdf0cf2a8d3c3ad3d3b91748c80ef04363305ec2
```

仅下载 SRAM：

```bat
D:\Pango\PDS_2022.2-SP6.4\bin\cdt_cfg_shell.exe -file program_sram.tcl -work_dir .
```

上板结果：

- JTAG 识别 `PANGO USB CABLE II` 和 `PGL50H`；
- SRAM 下载 100%，`done bit=1`，未操作 Flash；
- DDR3 初始化成功；
- 四组连贯真实固定输入全部 896/896 逐位一致，合计约 2.44 秒；
- 真实 FPGA 随机/边界 300/300 PASS，seed=`20260807`，耗时 182.74 秒。

上板命令：

```bat
python -B tools\pangu_post_attention_layernorm_host.py --port COM20 info
python -B tools\pangu_post_attention_layernorm_host.py --port COM20 status
python -B tools\pangu_post_attention_layernorm_host.py --port COM20 fixed
python -B tools\pangu_post_attention_layernorm_host.py --port COM20 stress --rounds 300 --seed 20260807
```

## 9. 阶段结论与下一步

G1 `post_attention_layernorm` 已满足软件、PDS、多角时序、独立位流、JTAG SRAM、真实固定输入和随机/边界压力的全部验收条件。

当前唯一下一任务是建立 layer0 MLP `gate_proj` 与 `up_proj` 的真实双投影闭环：二者都消费本阶段 `[896]` signed Q6.10 输出，权重形状均为 `[4864,896]`、group size 64。双投影逐位通过前不得进入 SiLU(gate) 或 `SiLU(gate) × up`。
