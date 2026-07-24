# F6 layer0 Attention 残差与完整子层闭环

## 1. 工程目标

本工程完成 Qwen2.5-0.5B layer0 Attention 的最后一步，并把此前分别验证的算子串成一条连贯的软件参考链：

```text
Attention 输入 hidden state（signed Q6.10）
→ input_layernorm
→ Q/K/V Linear
→ RoPE
→ Attention Score
→ Softmax
→ 概率 × V
→ 多头拼接
→ 真实 O_proj
→ Q28 到 Q6.10 重标定
→ 与原 hidden state 残差相加
→ 完整 layer0 Attention 子层输出（signed Q6.10）
```

硬件工程只实现最终残差入口，直接消费已经明确格式的两路数据，不覆盖任何已验证工程或位流：

- residual hidden state：`[896]` signed int16 Q6.10；
- layer0 O_proj 输出：`[896]` signed int64 Q28；
- 完整 Attention 输出：`[896]` signed int16 Q6.10。

## 2. 定点与饱和规则

### 2.1 O_proj Q28 到 Q6.10

每个 signed int64 Q28 元素执行：

```text
oproj_q10_wide = signed_RNE(oproj_q28 / 2^18)
oproj_q10 = saturate_int16(oproj_q10_wide)
```

- 右移位数：`28 - 10 = 18`；
- 舍入：对称 round-to-nearest-even；
- `INT64_MIN` 使用无符号幅值路径处理；
- 舍入结果显式饱和到 `[-32768, 32767]`。

### 2.2 残差加法

```text
attention_output_q10 = saturate_int16(
    sign_extend(residual_hidden_q10) + sign_extend(oproj_q10)
)
```

重标定饱和与最终残差饱和是两个独立边界。

## 3. 软件参考

主要文件：

| 文件 | 作用 |
|---|---|
| `../model_tools/attention_residual_reference.py` | 连贯 layer0 Attention 软件链、重标定、残差和压力参考 |
| `../model_tools/attention_residual_f6_reference.json` | 1/2/6/16-token 固定清单与 SHA256 |
| `../model_tools/test_attention_residual_reference.py` | RNE tie、INT64 极值、双重饱和和真实模型测试 |
| `../tools/pangu_attention_residual_host.py` | 软件自检、固定上板和随机/边界上板工具 |

固定软件链不再拼接此前彼此独立的测试向量。位置 `0..15` 的每个 token 都从对应 deterministic hidden state 出发，先执行真实 layer0 `input_layernorm`，再使用同一归一化输出生成 Q/K/V；K/V 历史、当前 Q、Softmax、O_proj 和残差均来自同一条连贯数据链。

固定 query/count：

| query | window | count | 最终 Q6.10 SHA256 |
|---:|---:|---:|---|
| 0 | 0..0 | 1 | `36859690e421b96cb8db65a5760a364d165a73b63fd1121040a7d1b42c042eb7` |
| 1 | 0..1 | 2 | `2b4a2d9240e6e30c2afe2943fa30ac60decd47f8fc8d377ab7e530e516009378` |
| 5 | 0..5 | 6 | `c0c0776d71e3dc97aa1a4d4e0709f38441cc82717c0c3081a79c47c30a21af10` |
| 15 | 0..15 | 16 | `7e61dc1fd0eb43b231e25fe1d08b1c08342723f537b6e25924858608235fa61e` |

## 4. RTL 架构

| 文件 | 作用 |
|---|---|
| `rtl/attention_residual_core.v` | Q28 signed RNE、两级饱和、残差加法和 256 bit 结果打包 |
| `rtl/attention_residual_ctrl.v` | UART、DDR3 上传、分段 burst 读取、结果写回和回读 |
| `rtl/attention_residual_top.v` | DDR3 IP、控制器、UART 和 LED 顶层 |
| `pnr/build_attention_residual.tcl` | PDS 全流程构建脚本 |
| `pnr/program_sram.tcl` | 仅下载易失性 SRAM，不操作 Flash |

核心缓存：

- hidden：56 个 256 bit 数据拍；
- O_proj：224 个 256 bit 数据拍，按 4 个 bank 保存，使每个输出拍可读取连续 16 个 int64；
- 输出：56 个 256 bit 数据拍，每拍 16 个 signed int16 Q6.10。

## 5. DDR3 地址布局

DDR3 Controller 地址单位为 32 bit，一个 256 bit 数据拍占 8 个地址单位。

| Controller 地址 | 内容 | 大小 |
|---|---|---:|
| `0x0000000` | residual hidden Q6.10 | 1792 B / 56 beats |
| `0x0001000` | O_proj signed Q28 | 7168 B / 224 beats |
| `0x0003000` | 最终 Attention 输出 Q6.10 | 1792 B / 56 beats |

## 6. UART 协议

串口：`115200, 8N1`。

| 命令 | 请求 | 回复 |
|---|---|---|
| `I` | 固件信息 | `PANGU50K ATTN RESIDUAL V1\r\n` |
| `S` | 状态 | `S + flags + \r\n` |
| `L` | 8960 B：hidden Q10 + O_proj Q28 | `K\r\n` |
| `G` | 启动计算 | `R + 1792 B signed Q6.10` |

状态位：

- bit0：DDR3 初始化完成；
- bit1：数据已加载；
- bit2：结果有效；
- bit3：核心忙。

错误码：`0x01` 未知命令、`0x02` DDR3 未完成、`0x04` 尚未加载、`0xFF` 状态机异常。

## 7. 构建与上板

在 `attention_residual_f6/pnr` 执行：

```bat
D:\Pango\PDS_2022.2-SP6.4\bin\pds_shell.exe -file build_attention_residual.tcl -project_name attention_residual
```

仅下载 SRAM：

```bat
D:\Pango\PDS_2022.2-SP6.4\bin\cdt_cfg_shell.exe -file program_sram.tcl -work_dir .
```

上位机：

```bat
python tools\pangu_attention_residual_host.py selftest --rounds 1000 --seed 20260806
python tools\pangu_attention_residual_host.py --port COM20 info
python tools\pangu_attention_residual_host.py --port COM20 status
python tools\pangu_attention_residual_host.py --port COM20 fixed
python tools\pangu_attention_residual_host.py --port COM20 stress --rounds 100 --seed 20260806
```

## 8. 2026-07-24 最终验证结果

### 8.1 软件

- 新增 Attention residual 单元测试：5/5 PASS；
- 完整 `model_tools` 回归：105/105 PASS；
- 连贯 1/2/6/16-token 固定链清单：PASS；
- 8960 B 上传载荷往返与 SHA256：PASS；
- signed RNE 正负 half-way tie：PASS；
- `INT64_MIN/MAX` Q28 重标定：PASS；
- Q28→Q10 正负饱和及最终残差正负饱和：PASS；
- 软件随机/边界压力：1000/1000 PASS，seed=`20260806`。

### 8.2 PDS 和时序

- Compile、Synthesize、Device Map、Place & Route、Timing、Bitstream：全部成功；
- 最终未布线网络：0；
- `Design Summary : All Constraints Met.`；
- 慢角 100 MHz setup WNS=`+1.493 ns`，TNS=`0`；
- 慢角 hold WHS=`+0.112 ns`，THS=`0`；
- 快角 setup WNS=`+3.841 ns`，TNS=`0`；
- 快角 hold WHS=`+0.051 ns`，THS=`0`；
- 慢/快角 recovery、removal 全部无违例；
- 资源：7695 LUT、6868 FF、70 distributed RAM、20 DRM、0 APM、79 IO。

位流：

```text
attention_residual_f6/pnr/generate_bitstream/attention_residual_top.sbit
大小：2101696 B
SHA256：609e1f569aa1e4579cffb995b0d7d0bc89fa34529790b35e8b26d6778226bcbd
```

### 8.3 真实板卡

- JTAG 识别 `PANGO USB CABLE II` 和 `PGL50H`；
- SRAM 下载进度 100%，`done bit=1`；
- 未操作 Flash；
- 串口 `COM20`，固件信息正确；
- DDR3 初始化成功；
- 四组连贯真实 1/2/6/16-token Attention 子层输出均 896/896 逐位一致；
- 随机/边界上板累计 300/300 PASS，分三批 seed_start=`20260806`、`20260906`、`20261006`；
- 覆盖全零 O_proj、signed RNE tie、`INT64_MIN/MAX`、一般 Q28、Q28→Q10 饱和和最终残差饱和。

## 9. 阶段结论

F6 Attention 已全部完成。当前已经具备经过真实板卡逐位验证的完整 layer0 Attention 子层：输入 RMSNorm、Q/K/V、RoPE、KV 历史、Score、Softmax、Attention 加权和、多头拼接、真实 O_proj 和第一处残差均有独立验证证据，并由本阶段建立了连贯软件参考与最终硬件闭环。

下一阶段才可进入 MLP；本工程没有修改或覆盖任何更早的已验证工程和位流。
