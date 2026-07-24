# PGL50H F6 layer0 Attention O_proj 独立验证工程

## 1. 工程目标

本工程完成 F6 的第二段闭环，固定验收对象为：

```text
输入来源：已真实上板逐位通过的 Attention `[14,64] -> [896]` head-major signed int64 Q28
权重：model.layers.0.self_attn.o_proj.weight
形状：M=896、K=896
分组：group_size=64，每行 14 个 group
bias：真实 .p50 中不存在，bias_q28 固定全 0
输出：896 个 signed int64 Q28
```

本工程位于独立目录，不覆盖 `attention_output_f6`、`softmax_f5`、`gemv_int4_qproj_full` 或更早工程及位流。

## 2. 定点与量化定义

Attention 多头拼接结果先按 Q28 解释为实数，再执行逐向量对称 INT8 量化：

```text
x_float       = x_q28 / 2^28
activation_s  = max(abs(x_float)) / 127；全零向量固定为 1.0
activation_i8 = clip(RNE(x_float / activation_s), -127, 127)
```

真实 O_proj 权重为 groupwise symmetric signed INT4，zero point 为 0。主机为每行、每 group 生成：

```text
combined_scale_uq4_28 = RNE(activation_s * weight_scale_fp16 * 2^28)
group_acc_int32       = sum(activation_i8 * weight_int4)，每组 64 元素
output_q28[row]       = sum(group_acc_int32 * combined_scale_uq4_28)
```

O_proj 在真实 `.p50` 中没有 bias 张量，因此硬件上传的 896 个 `bias_q28` 全部为 0。

## 3. 硬件复用与隔离方式

O_proj 与已验证 layer0 q_proj 都是 `M=896、K=896、group_size=64` 的完整分组 INT4 Linear，因此本工程直接复用：

- `gemv_int4_qproj_full/rtl/int8_dot16_pipe.v`
- `gemv_int4_qproj_full/rtl/gemv_qproj_full_core.v`
- `gemv_int4_qproj_full/rtl/gemv_qproj_full_ctrl.v`

新增独立顶层 `rtl/attention_oproj_top.v`，直接实例化 DDR3 IP 与上述控制器，以保持原板级约束所需的顶层层级。独立 PDS 工程、位流、软件清单和上位机均位于新的 O_proj 入口。

由于复用已验证控制器，UART 固件标识仍为：

```text
PANGU50K QPROJ FULL V1
```

该标识表示底层通用完整 Linear 协议版本；本阶段实际上传参数和金标准均为真实 O_proj。

## 4. 上传载荷与协议

协议继续使用 `115200, 8N1`：

| 命令 | 请求 | 回复 |
|---|---|---|
| `I` | 无 | `PANGU50K QPROJ FULL V1\r\n` |
| `S` | 无 | `S + flags + \r\n` |
| `L` | 固定 488320 B | `K\r\n` |
| `G` | 无 | `R + 896×little-endian signed int64 Q28` |

固定上传载荷：

| 区域 | 大小 |
|---|---:|
| activation INT8 | 896 B |
| packed O_proj INT4 weight | 401408 B |
| padded combined UQ4.28 scale | 57344 B |
| padded zero bias_q28 | 28672 B |
| 合计 | 488320 B |

四组真实固定输入的上传载荷 SHA256：

```text
case0 abb20df12c2b7e0fa9a473f3394c072a1deb7094776ce5b49691501c2ccf805c
case1 82b854727755aa726ba1c00dbb2db44f87666c4cebbe5122b0104f241afe691e
case2 0d523596b05a1835793d16d34a1891089eab0faa5d6349c6c764020e8a63a025
case3 c4981db50af0e4c2848f200eab908635e2abfcb5389dc7a799a26b3df076394d
```

## 5. 主要文件

| 文件 | 作用 |
|---|---|
| `../model_tools/attention_oproj_reference.py` | F6 Q28 输入量化、真实 O_proj 参数加载和 Q28 金标准 |
| `../model_tools/attention_oproj_f6_reference.json` | 四组真实固定输入、输出及关键数组 SHA256 |
| `../model_tools/test_attention_oproj_reference.py` | 输入转换、独立重算、真实参数、零输入和随机测试 |
| `../tools/pangu_attention_oproj_host.py` | 软件自检、固定/随机上板和逐位比较 |
| `rtl/attention_oproj_top.v` | 保持 DDR3 约束层级的独立顶层 |
| `pnr/build_attention_oproj.tcl` | 独立 PDS 全流程构建脚本 |
| `pnr/program_sram.tcl` | 仅下载易失性 SRAM，不操作 Flash |

## 6. 构建与验证命令

```bat
python -m unittest model_tools.test_attention_oproj_reference -v
python tools\pangu_attention_oproj_host.py selftest --rounds 1000 --seed 20260805

cd attention_oproj_f6\pnr
D:\Pango\PDS_2022.2-SP6.4\bin\pds_shell.exe ^
  -file build_attention_oproj.tcl ^
  -project_name attention_oproj

D:\Pango\PDS_2022.2-SP6.4\bin\cdt_cfg_shell.exe ^
  -file program_sram.tcl ^
  -work_dir .

cd ..\..
python tools\pangu_attention_oproj_host.py --port COM20 info
python tools\pangu_attention_oproj_host.py --port COM20 status
python tools\pangu_attention_oproj_host.py --port COM20 fixed --case all
python tools\pangu_attention_oproj_host.py --port COM20 stress --rounds 4 --seed 20260805
```

## 7. 2026-07-24 最终验证结果

### 软件参考

- 新增 O_proj 单元测试：`7/7 PASS`；
- 完整 `model_tools` 回归：`100/100 PASS`；
- 四组真实 F6 固定输入、真实 O_proj 参数、清单和 488320 B 载荷往返：全部 PASS；
- 随机/边界 Q28 输入软件压力：`1000/1000 PASS`，seed=`20260805`，约 35.26 秒；
- combined scale 饱和数为 0，固定输入 INT8 clipping 数为 0；
- 全零 Attention 输入严格生成全零 INT8 激活和全零 O_proj 输出。

四组固定 O_proj 输出 SHA256：

```text
case0 19008a25a59cde0f8def0c938ada397b6866dc143774b74c6ff77a2a95a7fcd5
case1 0e70753bea148c81d0bce79360d250710a1cc6ee817a40e4b6cbccf7d4f30279
case2 c0ffeb8b5a1168b661d52a34f34a5f4f12f3d075805b05b4ace346683cb8b018
case3 af63d1efc3913f597fdcd5dbe520ac782a943074301a60b249f4f25a3cf34a65
```

### PDS 实现与时序

- Compile、Synthesize、Device Map、Place & Route、Timing、Bitstream：全部成功；
- 最终未布线网络：0；
- `Design Summary : All Constraints Met.`；
- 慢角 100 MHz setup：WNS=`+0.614 ns`，TNS=`0`；
- 慢角 hold：WHS=`+0.171 ns`，THS=`0`；
- 慢角 recovery：WNS=`+2.919 ns`，TNS=`0`；
- 慢角 removal：WHS=`+0.537 ns`，THS=`0`；
- 快角 setup：WNS=`+3.023 ns`，TNS=`0`；
- 快角 hold：WHS=`+0.101 ns`，THS=`0`；
- 快角 recovery：WNS=`+4.898 ns`，TNS=`0`；
- 快角 removal：WHS=`+0.337 ns`，THS=`0`；
- 慢/快角最小脉宽均无违例；
- 资源：LUT=`8510`、FF=`7619`、distributed RAM=`326`、DRM=`4`、APM=`12`。

最终位流：

```text
pnr/generate_bitstream/attention_oproj_top.sbit
大小：2101696 B
SHA256：017517f877f29e62d945ecd3ae4ba22c2d690b6e6b92778eb0502ba7ac115533
```

### 真实上板

- JTAG 识别 `PANGO USB CABLE II` 和 `PGL50H`；
- 仅下载易失性 SRAM，进度 100%，`done bit=1`，未操作 Flash；
- DDR3 初始化成功；
- 四组真实 F6 固定输入的 896 个 O_proj 输出全部与 Python signed int64 Q28 金标准逐位一致；
- 四组上传、计算和回读耗时分别约 43.04、43.03、43.04、43.04 秒；
- 全零、随机常量、稀疏极值和完整 896 维随机 Attention Q28 输入上板回归：`4/4 PASS`，seed=`20260805`，约 172.34 秒。

## 8. 当前边界与下一步

本工程证明 F6 `[896]` Attention 多头拼接结果可以按既定逐向量 INT8 规则进入真实 layer0 O_proj，真实 `.p50` 的完整 896×896 INT4 权重、FP16 group scale、DDR3 行调度和 signed int64 Q28 输出已在 PGL50H 上逐位验证。

当前仍未完成 Attention 残差与完整 Attention 子层。唯一下一任务是把 O_proj 输出与对应的 Attention 输入残差按统一定点格式接通，建立完整 layer0 Attention 子层的软件参考和独立硬件闭环；不得提前进入 MLP。
