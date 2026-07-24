# layer0 Q/K RoPE 独立验证工程

本工程实现 Qwen2.5-0.5B layer0 的真实 Q/K Rotary Position Embedding（RoPE）定点闭环。它直接使用 F1 已验证的 Q=`[14,64]`、K=`[2,64]` head-major signed int64 Q28 输出，但建立独立 RTL、PDS 工程、位流和上位机，不覆盖 `qkv_linear_layer0` 或更早阶段成果。

## 1. 模型配置与真实配对规则

从 `model_output/yanbo_qwen25_0.5b_int4.json` 校验：

- `hidden_size=896`
- `num_attention_heads=14`
- `num_key_value_heads=2`
- `head_dim=64`
- `rotary_dim=64`
- `rope_theta=1000000`
- `max_position_embeddings=32768`

Qwen2 的 `rotate_half` **不是相邻偶奇维 `(0,1)、(2,3)` 配对**，而是将每个 head 拆成前后两半：

```text
first  = dims[0:32]
second = dims[32:64]
pair i = dim i <-> dim i+32
```

对每个 `i=0..31`：

```text
y_first  = x_first*cos(position,i) - x_second*sin(position,i)
y_second = x_second*cos(position,i) + x_first*sin(position,i)
```

inverse frequency 定义：

```text
inv_freq[i] = 1 / rope_theta^(2*i/64), i=0..31
angle       = position * inv_freq[i]
```

## 2. 定点格式与舍入

- Q/K 输入：signed int64 Q28
- Q/K 输出：signed int64 Q28
- sin/cos：signed int32 Q1.30
- 单个乘积：signed 96 bit
- 两项加/减：signed 97 bit
- 舍入：加/减完成后仅执行一次 round-to-nearest-even（RNE）右移 30 位
- 输出：显式饱和到 signed int64

软件参考给出的保守绝对误差界由 trig 量化误差和最终 Q28 RNE 误差组成。真实固定 Q/K 在位置 `0、1、2026、32767` 上的最大误差分别为：

```text
position=0      0
position=1      5.453485130147e-08
position=2026   4.564708433463e-08
position=32767  7.232674192892e-08
统一保守界      9.294017896955e-08
```

## 3. RTL 数据通路

### `rtl/rope_pair_q28_core.v`

直接 64×32 组合乘法在首版 PDS 中形成长 APM 级联，慢角 setup 曾出现 `WNS=-2.017 ns`。最终实现改为：

1. 将 64 位输入幅值拆为四个 16 位 limb；
2. 将 32 位 trig 幅值拆为两个 16 位 limb；
3. 复用一个 16×16 APM，顺序计算 8 个部分积；
4. 在 96 位幅值累加器中精确重构有符号 64×32 乘积；
5. 四个乘积依次完成；
6. 97 位加/减、绝对值、RNE、饱和分别寄存流水。

该方案牺牲吞吐率以换取数学精确、低资源和稳定 100 MHz 时序，符合 F2 功能正确优先原则。

### `rtl/rope_qk_ctrl.v`

- Q/K 合并为 16 个 head，按 head-major 顺序处理；
- 每个 head 处理 32 对 `i <-> i+32`；
- Q/K 输入、trig 表和结果位于 DDR3；
- 支持最多 16 个连续位置的 sin/cos 表；
- 每次 `G` 完成一个位置并自动递增位置索引与表索引；
- 结果保持原始 Q=`[14,64]`、K=`[2,64]` 布局。

## 4. DDR3 地址布局

DDR3 控制器地址单位为 32 bit；一个 256 bit 数据拍占 8 个地址单位。

| 区域 | 起始地址 | 内容 |
|---|---:|---|
| Input | `0x0000000` | Q 896×int64 + K 128×int64，共 8192 B |
| Trig | `0x0001000` | 每位置 cos[32]+sin[32]，Q1.30，共 256 B/位置 |
| Result | `0x0002000` | Q 896×int64 + K 128×int64，共 8192 B |

## 5. UART 协议

115200 baud、8N1。

| 命令 | 请求 | 回复 | 说明 |
|---|---|---|---|
| `I` | 1 B | `PANGU50K ROPE QK V1\r\n` | 固件信息 |
| `S` | 1 B | `S + flags + position_u16 + table_index + table_count + CRLF` | 状态 |
| `C` | `C + start_u16 + count_u16` | `K\r\n` | 配置 1..16 个连续位置 |
| `L` | `L + 8192 B Q/K + count*256 B trig` | `K\r\n` | 加载数据 |
| `G` | 1 B | `R + processed_position_u16 + 8192 B result` | 处理当前位置并自增 |
| `Z` | 1 B | `K\r\n` | 位置与表索引复位到配置起点 |

错误帧为 `E + error_code + CRLF`。

## 6. 软件参考与固定清单

- `model_tools/rope_fixed_reference.py`
- `model_tools/rope_layer0_reference.json`
- `model_tools/test_rope_fixed_reference.py`
- `tools/pangu_rope_qk_host.py`

固定位置输出 SHA256：

| Position | Q SHA256 | K SHA256 |
|---:|---|---|
| 0 | `ea1f04bf4ff313dad07025ff35e66a088f13afd28d817422b89bb135f63525a0` | `20728d329c32c722b0194032897bc3cf9a3a31323317e389d8fd7b6f78745474` |
| 1 | `aa601059c0ea93f9507a2b3bfc6b38d4dbeb85531ba19ced856cfb25d9b4612a` | `fd12ac1742b6f7ad8f01c005b36bc8494159c79acb655c4154e9cc99f2d316fb` |
| 2026 | `6c266ff09ef200af907da2796b8fb1db4e5c050f0cad15ccb62e318a5953b0d6` | `0f8625c3063eb62726c7b3bfc933af4d70652014cd4b63a0ba772916a4c02622` |
| 32767 | `b4903ff68fbdcee42aab2fa46f3030d759518e9ff4cac846c1620f6c5e886a2d` | `18263e0f4e82a7c7b67f9ee4e07376b4999b2205d093071e8faff3456da338a4` |

验证结果：

- F2 新增单元测试：7/7 PASS；
- 完整 `model_tools` 回归：55/55 PASS；
- 软件随机 Q/K 与位置压力：1000/1000 PASS，seed=`20260730`；
- 上位机软件自检：1000/1000 PASS，seed=`20260731`。

## 7. PDS 结果

构建命令：

```text
D:/Pango/PDS_2022.2-SP6.4/bin/pds_shell.exe -file build_rope_qk.tcl -project_name rope_qk
```

最终 seed：global placement 5、global route 11，并启用 route hold 修复。

- Compile、Synthesize、Device Map、P&R、Report Timing、Generate Bitstream 全部成功；
- 未布线网络：0；
- 资源：8859 LUT、9886 FF、70 distributed RAM、1 APM、0 DRM；
- `Design Summary : All Constraints Met`；
- 慢角 100 MHz setup：WNS=`+0.988 ns`、TNS=0；
- 慢角 hold：WHS=`+0.171 ns`、THS=0；
- 快角 setup：WNS=`+3.483 ns`、TNS=0；
- 快角 hold：WHS=`+0.100 ns`、THS=0；
- recovery、removal、minimum pulse width 均无违例。

位流仅保存在本地生成目录：

```text
rope_qk_layer0/pnr/generate_bitstream/rope_qk_top.sbit
SHA256=25396ffc894abc15b81ab99f62619f3694e7e662f620f3c6a89e28ae116d153a
```

## 8. 真实上板验证

- JTAG 仅下载易失性 SRAM，进度 100%，`done bit=1`，未操作 Flash；
- 固件信息：`PANGU50K ROPE QK V1`；
- DDR3 初始化成功；
- 固定位置 `0、1、2026、32767`：Q/K 全输出逐位一致；
- 连续位置 `2026..2033`：8/8 PASS；
- 自动位置递增、表索引结束状态正确；
- `Z` 复位后位置 2026 重放逐位一致；
- 随机真实上板位置：300/300 PASS，seed=`20260731`，耗时 235.59 秒。

## 9. 常用命令

```text
python tools/pangu_rope_qk_host.py selftest --rounds 1000
python tools/pangu_rope_qk_host.py --port COM20 info
python tools/pangu_rope_qk_host.py --port COM20 status
python tools/pangu_rope_qk_host.py --port COM20 fixed
python tools/pangu_rope_qk_host.py --port COM20 sequence --start 2026 --count 8 --verify-reset
python tools/pangu_rope_qk_host.py --port COM20 stress --positions 300 --seed 20260731
```

## 10. 下一任务

F2 已完成。下一步进入 F3 KV Cache：先定义 28 层、2 个 KV heads、head_dim 64、token 位置对应的 DDR3 地址布局和容量边界，再建立当前 token 写入、历史 token 顺序读取、上下文边界和防覆盖的软件参考与独立硬件闭环。
