# layer0 Q/K/V 完整真实 Linear 统一闭环

本目录完成路线图 F1：在不覆盖任何既有验证工程和位流的前提下，复用已验证完整 `q_proj` 的逐组 Q28 数据通路，统一实现 Qwen2.5-0.5B layer0 的真实 `q_proj`、`k_proj` 和 `v_proj`。

## 1. 真实模型对象

| 投影 | 权重形状 | bias | 输出布局 | 输出行数 |
|---|---:|---:|---:|---:|
| Q | `[896,896]` | 896 | `14 heads × 64` | 896 |
| K | `[128,896]` | 128 | `2 heads × 64` | 128 |
| V | `[128,896]` | 128 | `2 heads × 64` | 128 |

三组权重均来自：

```text
model_output/yanbo_qwen25_0.5b_int4.p50
```

存储格式均为 group size 64 的分组对称 signed INT4。模型采用 GQA：14 个 Q heads、2 个 KV heads、`head_dim=64`。平坦输出采用 head-major 连续布局：

```text
q_flat.reshape(14, 64)
k_flat.reshape(2, 64)
v_flat.reshape(2, 64)
```

## 2. 硬件等价数学定义

三个投影共用同一份逐向量对称 INT8 hidden state：

```text
q_x = RNE(x / activation_scale), clamp[-127,127]
activation_scale = max(abs(x)) / 127
```

每个输出行分 14 个 group：

```text
group_acc_int32[row,group]
    = sum(q_weight_int4 * q_activation_int8)

combined_scale_uq4_28[row,group]
    = RNE(activation_scale * weight_scale_fp16 * 2^28)

output_q28[row]
    = bias_q28[row]
    + sum(group_acc_int32[row,group] * combined_scale_uq4_28[row,group])
```

输出是 little-endian signed int64 Q28。软件参考还使用独立循环重算每一组，避免只验证同一份向量化实现。

## 3. 统一载荷

每种投影均使用：

```text
activation_int8
+ packed_weight_int4（低半字节在前）
+ 每行 64 B UQ4.28 scale 区
+ 每行 32 B signed Q28 bias 区
```

| 投影 | activation | weight | scale | bias | 总载荷 |
|---|---:|---:|---:|---:|---:|
| Q | 896 B | 401408 B | 57344 B | 28672 B | 488320 B |
| K | 896 B | 57344 B | 8192 B | 4096 B | 70528 B |
| V | 896 B | 57344 B | 8192 B | 4096 B | 70528 B |

DDR3 控制器地址布局复用已验证完整 q_proj 工程：

| 区域 | 控制器地址 |
|---|---:|
| activation | `0x0000000` |
| weight | `0x0001000` |
| scale | `0x0020000` |
| bias | `0x0024000` |
| result | `0x0026000` |

K/V 只使用各区域前 128 行，不会越界或与结果区重叠。

## 4. UART 协议

115200 8N1：

| 命令 | 功能 |
|---|---|
| `I` | 返回 `PANGU50K QKV LINEAR V1\r\n` |
| `S` | 返回状态，包含 DDR ready、loaded、result valid、busy 和当前投影 |
| `Q` | 选择 Q 投影，回复 `K\r\n` |
| `K` | 选择 K 投影，回复 `K\r\n` |
| `V` | 选择 V 投影，回复 `K\r\n` |
| `L` + 当前投影载荷 | 写入 DDR3，回复 `K\r\n` |
| `G` | 运行并返回 `R` + 当前投影全部 signed int64 Q28 输出 |

选择投影会清除 loaded/result-valid，避免误用上一种投影的数据。

## 5. 文件

```text
rtl/int8_dot16_pipe.v
rtl/qkv_linear_core.v
rtl/qkv_linear_ctrl.v
rtl/qkv_linear_top.v
pnr/build_qkv_linear.tcl
pnr/program_sram.tcl
pnr_seed5/run_seed5.tcl
pnr_seed5/program_sram.tcl
../model_tools/qkv_linear_reference.py
../model_tools/qkv_layer0_reference.json
../model_tools/test_qkv_linear_reference.py
../tools/pangu_qkv_linear_host.py
```

默认种子和 seed17/29 均只在 DDR3 IP 内部出现极小快角 hold 违例，因此不作为发布位流。最终使用可复现的 seed5/11 独立构建。

## 6. 构建与软件验证

完整构建：

```bat
cd qkv_linear_layer0\pnr_seed5
D:\Pango\PDS_2022.2-SP6.4\bin\pds_shell.exe ^
  -file run_seed5.tcl -project_name qkv_linear_seed5
```

软件固定清单与 1000 轮 Q/K/V 随机 hidden state：

```bat
python tools\pangu_qkv_linear_host.py selftest ^
  --projection all --rounds 1000 --seed 20260729
```

结果：

```text
完整 model_tools 回归：48/48 PASS
QKV 软件随机 hidden state：1000/1000 PASS
```

固定输出 SHA256：

```text
Q: ea1f04bf4ff313dad07025ff35e66a088f13afd28d817422b89bb135f63525a0
K: 20728d329c32c722b0194032897bc3cf9a3a31323317e389d8fd7b6f78745474
V: 162622e05e0013ca342f28032cb280c264f428f93a197eb67dbfafd76e20a168
```

## 7. PDS 结果

seed5/11 完整执行编译、综合、Device Map、布局布线、时序分析和位流生成，最终未布线网络为 0。

资源：

```text
LUT             8503
FF              7641
Distributed RAM 326
DRM             4
APM             12
```

多角时序：`All Constraints Met`。

| 项目 | 慢角 | 快角 |
|---|---:|---:|
| setup WNS | `+0.363 ns` | `+2.985 ns` |
| setup TNS | `0` | `0` |
| hold WHS | `+0.169 ns` | `+0.100 ns` |
| hold THS | `0` | `0` |

恢复、移除和最小脉宽均无违例。

发布位流：

```text
qkv_linear_layer0/pnr_seed5/generate_bitstream/qkv_linear_top.sbit
size: 2101696 B
SHA256: e3a4b6849a5716f38d6bdd3fbd039d46f2d350a32a0417ee347462d1a8f96e26
```

## 8. 真实上板结果

2026-07-24 仅通过 JTAG 下载到易失性 SRAM，进度 100%，`done bit=1`，未操作 Flash。固件信息和状态：

```text
PANGU50K QKV LINEAR V1
DDR3初始化=是
```

固定 seed=`20260723`，Q/K/V 共用同一量化 hidden state：

- Q：896/896 个输出与 Python 逐位一致，head shape=`(14,64)`，约 43.02 秒；
- K：128/128 个输出与 Python 逐位一致，head shape=`(2,64)`，约 6.26 秒；
- V：128/128 个输出与 Python 逐位一致，head shape=`(2,64)`，约 6.26 秒；
- 真实随机 hidden state：3/3 轮完整 Q+K+V 均逐位一致，seed=`20260729..20260731`，约 166.72 秒。

固定上板命令：

```bat
python tools\pangu_qkv_linear_host.py --port COM20 fixed --projection all
```

随机上板命令：

```bat
python tools\pangu_qkv_linear_host.py --port COM20 stress ^
  --projection all --rounds 3 --seed 20260729
```

## 9. 结论

F1 已完成。当前工程不只是单独验证 q_proj，而是已在同一真实硬件工程中完成 Q/K/V 三种真实 Linear、动态输出行数、统一载荷、GQA head-major 布局、全输出回写和 Python 自动逐位比较。

下一任务是 F2 RoPE：在已验证 Q/K head 布局基础上建立位置索引、sin/cos 表、偶奇维旋转和定点误差闭环。
