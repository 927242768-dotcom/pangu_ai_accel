# G1 layer0 MLP `SiLU(gate)` 独立闭环

本工程完成 Qwen2.5-0.5B layer0 MLP 的独立 `SiLU(gate)` 阶段。输入直接来自上一阶段已经真实上板逐位通过的 `gate_proj` 完整输出，不重新计算 gate projection，也不提前执行 `SiLU(gate) × up`。

## 1. 本阶段边界

```text
已验证 gate_proj [4864] signed int64 Q28
→ signed RNE 右移 18 位
→ 显式饱和为 signed int16 Q6.10
→ 已验证 E2 PWL64 SiLU
→ [4864] signed int16 Q6.10
```

本工程只负责上述非线性闭环。以下内容不属于本阶段：

- `up_proj` 重算；
- `SiLU(gate) × up`；
- `down_proj`；
- MLP 第二处残差。

## 2. 固定数值定义

### 2.1 Q28 到 Q6.10

输入为 signed int64 Q28。硬件与 Python 使用完全相同的对称 signed round-to-nearest-even：

```text
q10_wide = signed_RNE(gate_q28 / 2^18)
gate_q10 = saturate_int16(q10_wide)
```

实现对 `INT64_MIN` 使用无符号二补码幅值路径，正负 half-way tie 均按偶数舍入，不使用隐含截断。

四组真实 gate 输出范围约为 `[-3.5440, 2.8183]`，均落在 PWL 主区间内，也不会触发 Q6.10 饱和；随机/边界测试仍显式覆盖 int64 极值与正负饱和。

### 2.2 PWL64 SiLU

严格复用 E2 已真实上板验证的 64 段端点 PWL：

- 输入、端点和输出：signed Q6.10；
- 主区间：`[-8, 8)`；
- 65 个端点，步长 `0.25`，端点表 1040 bit；
- 区间内线性插值乘积执行 signed RNE 右移 8 位；
- `x < -8` 时输出 `0`；
- `x >= 8` 时输出 `x`；
- 相对精确 SiLU 的完整 int16 输入域最大误差为 4 Q10 LSB。

## 3. 软件参考与固定清单

相关文件：

- `../model_tools/mlp_silu_reference.py`
- `../model_tools/mlp_silu_g1_reference.json`
- `../model_tools/test_mlp_silu_reference.py`
- `../tools/pangu_mlp_silu_host.py`

四组连贯真实输入来自 query/count=`0/1`、`1/2`、`5/6`、`15/16` 的已验证 gate projection。最终 PWL 输出 SHA256：

| query/count | `SiLU(gate)` `[4864]` Q6.10 SHA256 |
|---|---|
| `0/1` | `d3a50e88eba59160b61eccaf9a25c0d3f5dd8c5f799dbd28ede20acbd383cd18` |
| `1/2` | `4dc5e4f4d3240ce628ee7db071ed31faa570212a1a5dc56e5b01c69d9702d310` |
| `5/6` | `b807ad37514a9bd1702625666f2c13670bfa460c423a2fc53fe483a44900e9c9` |
| `15/16` | `4f16572a82b583edb041444edf7bdea5841ffcc3a5a7de71a28cae138f2e980e` |

软件验证结果：

- 新增单元测试：7/7 PASS；
- 完整 `model_tools` 回归：123/123 PASS；
- 软件随机/边界压力：1000/1000 PASS，seed=`20260809`；
- 覆盖全零、`INT64_MIN/MAX`、正负 RNE half-way tie、`±8` PWL 边界、int16 饱和边界、稀疏真实范围、一般真实范围和完整随机 int64 bit pattern；
- 四组固定上传载荷往返与 SHA256 全部一致。

## 4. DDR3 与 UART 协议

UART：115200、8N1。

| 命令 | 功能 | 回复 |
|---|---|---|
| `I` | 查询固件 | `PANGU50K MLP SILU V1\r\n` |
| `S` | 查询状态 | `S + flags + \r\n` |
| `L` + 39072 B | 加载 gate 与 PWL 端点 | `K\r\n` |
| `G` | 执行并返回结果 | `R` + 9728 B |

状态位：

- bit0：DDR3 初始化完成；
- bit1：数据已加载；
- bit2：结果有效；
- bit3：核心忙。

上传载荷：

| 区域 | 数量与格式 | 字节数 |
|---|---|---:|
| gate | 4864 个 little-endian signed int64 Q28 | 38912 |
| PWL 表 | 65 个 signed int16 Q6.10 端点，补齐到 80 项 | 160 |
| 合计 |  | 39072 |

结果为 4864 个 little-endian signed int16 Q6.10，共 9728 B。

DDR3 Controller 地址单位为 32 bit：

| 区域 | 基地址 | 说明 |
|---|---:|---|
| gate | `0x0000000` | 1216 个 256 bit beat |
| PWL | `0x0003000` | 5 个 256 bit beat |
| result | `0x0004000` | 304 个 256 bit beat |

## 5. RTL 结构

- `rtl/mlp_silu_core.v`
  - gate 输入按 4 个 DRM bank 缓存；
  - 1216 个输入 beat 重排为 304 个输出 beat；
  - 每个输出 beat 包含 16 个 Q6.10 结果；
  - 每 lane 依次完成幅值、RNE、输入饱和、PWL 读取、插值、输出饱和和打包。
- `rtl/mlp_silu_ctrl.v`
  - UART 收发；
  - DDR3 上传、burst 读取、结果写回和回读；
  - gate 最长 16 beat burst 装入片上 4-bank 缓存；
  - 计算结果逐 beat 流式写回。
- `rtl/mlp_silu_top.v`
  - 复用已验证 DDR3 Controller + PHY、时钟、复位、UART 和板级约束；
  - 保持与历史工程相同的顶层板卡接口。

## 6. PDS 构建与时序

构建脚本：

```bat
cd /d E:\50K\AI_LLM_FPGA\pangu_ai_accel\mlp_silu_g1\pnr
D:\Pango\PDS_2022.2-SP6.4\bin\pds_shell.exe -file build_mlp_silu.tcl -project_name mlp_silu
```

最终结果：

- Compile：PASS；
- Synthesize：PASS；
- Device Map：PASS；
- Place & Route：PASS，最终未布线网络 0；
- Report Timing：PASS；
- Generate Bitstream：PASS。

资源：

| 资源 | 使用量 | 可用量 | 利用率 |
|---|---:|---:|---:|
| LUT | 8024 | 42800 | 19% |
| FF | 7901 | 64200 | 13% |
| Distributed RAM | 70 | 17000 | 1% |
| DRM | 32 | 134 | 24% |
| APM | 1 | 84 | 2% |
| IO | 79 | 296 | 27% |

最终多角时序 `All Constraints Met`：

| Corner | 检查 | 最差余量 | TNS/THS |
|---|---|---:|---:|
| Slow | setup，`ddrphy_clkin` | WNS `+1.468 ns` | TNS `0` |
| Slow | hold，`ddrphy_clkin` | WHS `+0.169 ns` | THS `0` |
| Fast | setup，`ddrphy_clkin` | WNS `+3.793 ns` | TNS `0` |
| Fast | hold，`ddrphy_clkin` | WHS `+0.100 ns` | THS `0` |

Slow/Fast 的 recovery、removal 与 minimum pulse width 也无违例。

开发过程中显式 seed17/29 首次只剩 1 条 Fast hold `-0.015 ns`，未被接受；最终改用与已验证 gate/up 工程一致的默认 PnR 策略后重新完成全流程，最终报告全角为 0。失败版本没有作为正式位流使用。

## 7. 位流与下载

正式位流：

```text
pnr/generate_bitstream/mlp_silu_top.sbit
```

- 大小：2101696 B；
- SHA256：`87e643c65b70949297d54042921ac62e70454c018b6ff31f1386bbf2c8770550`。

仅下载到易失性 SRAM：

```bat
cd /d E:\50K\AI_LLM_FPGA\pangu_ai_accel\mlp_silu_g1\pnr
D:\Pango\PDS_2022.2-SP6.4\bin\cdt_cfg_shell.exe -file program_sram.tcl -work_dir .
```

JTAG 实测下载 100%，`done bit=1`。脚本只执行 `cfg_program`，没有任何 Flash 擦写或编程命令。

## 8. 真实开发板验收

板卡与接口：

- PGL50H-6IFBG484 / MES50HP；
- UART：CP210x `COM20`；
- 固件：`PANGU50K MLP SILU V1`；
- DDR3 初始化：成功。

固定真实输入：

- query/count=`0/1`：4864/4864 逐位一致；
- query/count=`1/2`：4864/4864 逐位一致；
- query/count=`5/6`：4864/4864 逐位一致；
- query/count=`15/16`：4864/4864 逐位一致；
- 四组总耗时约 17.16 秒；
- 四个输出 SHA256 与固定清单完全相同。

真实 FPGA 随机/边界压力：

- 六批各 50 组，合计 300/300 PASS；
- seeds=`20260809`、`20260810`、`20260811`、`20260812`、`20260813`、`20260814`；
- 每批轮转覆盖全部 8 类输入模式；
- 所有 4864 项结果均与 Python 金标准逐位一致。

## 9. 常用命令

软件固定清单：

```bat
python model_tools\mlp_silu_reference.py check
```

软件压力：

```bat
python model_tools\mlp_silu_reference.py stress --rounds 1000 --seed 20260809
```

上位机自检：

```bat
python tools\pangu_mlp_silu_host.py selftest --rounds 1000 --seed 20260809
```

四组真实固定输入：

```bat
python tools\pangu_mlp_silu_host.py --port COM20 fixed
```

真实 FPGA 随机/边界：

```bat
python tools\pangu_mlp_silu_host.py --port COM20 stress --rounds 50 --seed 20260809
```

## 10. 当前结论与下一任务

`SiLU(gate)` 已完整通过软件参考、固定清单、PDS 全流程、Slow/Fast 多角时序、JTAG SRAM、四组连贯真实输入和 300 组真实随机/边界验证。

G1 MLP 的唯一下一任务推进为：

```text
使用已经分别真实上板通过的 SiLU(gate) [4864] signed Q6.10
和 up_proj [4864] signed int64 Q28，首先冻结两路对齐格式、乘法位宽、
RNE 与正负饱和规则，然后独立完成 SiLU(gate) × up。
```

在该乘法阶段单独通过前，不得进入 `down_proj`。
