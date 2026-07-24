# `.p50` 模型格式解析与提取工具

本目录负责 Qwen2.5-0.5B + LoRA 的 `.p50` 镜像导出、结构校验和真实张量提取。

## 1. 当前真实镜像

```text
model_output/yanbo_qwen25_0.5b_int4.p50
大小：263,857,920 字节（251.63 MiB）
SHA256：f0c0a22886499715fe16832b88ac59bff48fea8f3069c247437726aca6f19e9d
```

镜像内共有 290 个张量：

- 169 个二维 `int4_groupwise_symmetric` 张量；
- 121 个连续 `float16` 张量；
- INT4/FP16 主数据合计约 235.68 MiB；
- FP16 分组 scale 合计约 14.72 MiB。

二维权重形状统计：

| 形状 `[rows, columns]` | 数量 | 典型用途 |
|---|---:|---|
| `[151936, 896]` | 1 | Token Embedding，且模型标记为 tied embedding |
| `[896, 896]` | 48 | Q 投影、O 投影 |
| `[128, 896]` | 48 | K 投影、V 投影 |
| `[4864, 896]` | 48 | Gate 投影、Up 投影 |
| `[896, 4864]` | 24 | Down 投影 |

## 2. 固定头格式

固定头使用小端结构：

```python
struct.Struct("<8sIIQQIIII")
```

| 字节偏移 | 类型 | 字段 | 当前值/含义 |
|---:|---|---|---|
| 0 | `char[8]` | magic | `P50Q4V1\0` |
| 8 | `uint32` | version | `1` |
| 12 | `uint32` | header_size | `4096` |
| 16 | `uint64` | metadata_size | 当前为 `63716` |
| 24 | `uint64` | data_offset | 当前为 `528384` |
| 32 | `uint32` | tensor_count | 当前为 `290` |
| 36 | `uint32` | group_size | 当前为 `64` |
| 40 | `uint32` | flags | bit0=LoRA 已合并，bit1=tied embedding |
| 44 | `uint32` | reserved | 必须为 `0` |

结构体本身为 48 字节，整个固定头区域保留 4096 字节。内嵌 JSON 从文件偏移 4096 开始，真实张量数据区按 4 KiB 对齐。

## 3. INT4 权重格式

真实模型二维权重统一使用：

```text
scheme      = symmetric_per_row_group
weight_bits = 4
group_size  = 64
range       = [-7, 7]
zero_point  = 0（对称量化，不保存独立 zero point）
scale_dtype = float16
```

存储规则：

1. 矩阵逻辑形状为 `[输出行, 输入列]`，按输出行 row-major 存放；
2. 每一行按 64 个输入列分组；
3. 输入列不足 64 的尾部补零到 64 的整数倍；
4. 每个字节保存两个有符号 INT4：低半字节对应较小列号，高半字节对应下一列；
5. INT4 按 4 位二补码解释，导出器实际限制到 `[-7, 7]`；
6. scale 按 `[row, group]` row-major 连续保存，每个 scale 为小端 FP16；
7. 反量化公式为 `weight_fp32 = int4_value * fp16_scale`。

二维 INT4 张量的长度可由目录完全推导：

```text
padded_columns = align_up(columns, group_size)
groups_per_row = padded_columns / group_size
data_nbytes     = rows * padded_columns / 2
scale_nbytes    = rows * groups_per_row * 2
```

## 4. FP16 张量格式

一维 bias、RMSNorm 权重等张量使用连续 C-order FP16：

```text
data_nbytes = product(shape) * 2
```

FP16 张量没有 scale、padded columns 或 groups 字段。

## 5. 工具说明

| 文件 | 作用 |
|---|---|
| `export_qwen25_fpga.py` | 合并 LoRA 并导出 `.p50` 与外部 JSON |
| `p50_format.py` | 轻量解析库；校验头、目录、长度、偏移并提取张量 |
| `p50_inspect.py` | 摘要、目录查看、全量校验、行/块提取 CLI |
| `verify_p50_image.py` | 在结构全量校验基础上，对照源模型抽样验证量化误差 |
| `linear_quant_reference.py` | 真实 Linear 切片的激活 INT8、分组 scale UQ4.28 与定点输出金标准 |
| `q_proj_m4k896_reference.json` | layer0 q_proj 的 M=4、K=896 固定向量输出与各数据区 SHA256 |
| `q_proj_full_reference.json` | layer0 q_proj 完整 M=896、K=896 固定输出、上传布局与 SHA256 清单 |
| `rmsnorm_fixed_reference.py` | layer0 input_layernorm 的 Q6.10、Q12.20、LUT/NR rsqrt 与硬件等价金标准 |
| `rmsnorm_layer0_reference.json` | K=896 固定输入、真实 gamma、rsqrt LUT 和输出 SHA256 清单 |
| `elementwise_fixed_reference.py` | signed Q6.10 残差、缩放、元素乘法和 SiLU LUT/PWL 硬件等价参考 |
| `elementwise_k896_reference.json` | E2 固定边界向量、SiLU 全输入域误差和关键数组 SHA256 |
| `embedding_fixed_reference.py` | E3 Token 行地址、真实 packed INT4/FP16 scale 到 UQ4.28/Q6.10 的硬件等价参考 |
| `embedding_k896_reference.json` | E3 固定 Token 的地址、载荷、输出范围和 SHA256 清单 |
| `test_p50_format.py` | 使用独立微型镜像验证解析、解包、反量化和错误检测 |
| `test_linear_quant_reference.py` | 量化格式、1000 轮随机压力和真实 q_proj 集成测试 |
| `test_rmsnorm_fixed_reference.py` | RMSNorm RNE、边界、真实 gamma、rsqrt 和 1000 轮软件压力测试 |
| `test_elementwise_fixed_reference.py` | E2 RNE、饱和、完整 int16 SiLU 误差和 1000 轮软件压力测试 |
| `test_embedding_fixed_reference.py` | E3 Token 边界、地址、RNE、饱和、全部真实 scales 和 1000 个随机 Token 测试 |

## 6. 常用命令

全量校验真实镜像与外部 JSON：

```bat
python model_tools\p50_inspect.py verify
```

查看摘要：

```bat
python model_tools\p50_inspect.py summary --check-metadata
```

列出某类张量：

```bat
python model_tools\p50_inspect.py list --contains self_attn.q_proj.weight
```

查看一个张量的形状、偏移和长度：

```bat
python model_tools\p50_inspect.py describe ^
  --tensor model.layers.0.self_attn.q_proj.weight
```

按张量名提取任意一行：

```bat
python model_tools\p50_inspect.py row ^
  --tensor model.layers.0.self_attn.q_proj.weight ^
  --row 0 ^
  --output extracted\layer0_q_row0.npz
```

提取任意二维块，允许跨越多个量化 group：

```bat
python model_tools\p50_inspect.py block ^
  --tensor model.layers.23.mlp.gate_proj.weight ^
  --row-start 1024 --row-count 2 ^
  --column-start 60 --column-count 12 ^
  --output extracted\layer23_gate_block.npz
```

INT4 提取结果的 NPZ 包含：

- `values`：FP32 反量化结果；
- `quantized`：有符号 INT8 容器中的原始 INT4 数值；
- `scales`：该块涉及的 FP16 scales；
- `scale_group_start`：`scales` 第一个 group 的全局编号；
- 张量名、存储类型和行列范围元数据。

FP16 提取结果主要包含原始 `values` 和范围元数据。

生成真实 layer0 `q_proj` 的固定 M=4、K=896 软件参考：

```bat
python model_tools\linear_quant_reference.py ^
  --output extracted\q_proj_m4k896.npz ^
  --manifest extracted\q_proj_m4k896.json
```

完整 NPZ 包含 FPGA 后续可直接使用的激活 INT8、packed INT4 权重、FP16 weight scale、分组 INT32 累加、UQ4.28 组合 scale、bias Q28 和预期输出。仓库只提交小型 JSON 清单，完整 NPZ 可由真实 `.p50` 镜像确定性重建。

生成并验证 layer0 `q_proj` 完整 M=896、K=896 载荷和金标准：

```bat
python tools\pangu_gemv_qproj_full_host.py selftest --rounds 1000 --seed 20260725
```

该工具从真实镜像一次性提取完整权重、FP16 scale 和 bias，之后复用这些模型数据生成不同激活的逐行 signed int64 Q28 金标准，并验证 488320 B 上传载荷的打包、补齐和往返一致性。

生成 layer0 `input_layernorm` K=896 定点参考并查看 LUT/NR 比较：

```bat
python model_tools\rmsnorm_fixed_reference.py ^
  --manifest model_tools\rmsnorm_layer0_reference.json
```

运行 RMSNorm 载荷与 1000 组软件压力自检：

```bat
python tools\pangu_rmsnorm_k896_host.py selftest ^
  --rounds 1000 --seed 20260726
```

RMSNorm 第一版使用 signed Q6.10 输入、gamma 和输出，40 位平方和、Q12.20 均值/epsilon、UQ12.20 rsqrt，并统一采用 RNE 和显式饱和。完整硬件协议、DDR3 地址和上板证据见 `rmsnorm_k896/README.md`。

生成 E2 K=896 元素级固定向量并比较 SiLU 方案：

```bat
python model_tools\elementwise_fixed_reference.py ^
  --manifest model_tools\elementwise_k896_reference.json
```

运行元素级载荷和 1000 组软件压力自检：

```bat
python tools\pangu_elementwise_k896_host.py selftest ^
  --rounds 1000 --seed 20260727
```

E2 统一使用 signed Q6.10 输入、标量和输出。残差加法显式饱和；缩放和元素乘法在 signed Q12.20 中计算，经 RNE 右移 10 位后饱和。SiLU 第一版选择覆盖 `[-8,8)` 的 64 段端点分段线性方案，区间外采用 `x<-8 -> 0`、`x>=8 -> x`。完整协议和地址布局见 `elementwise_k896/README.md`。

生成 E3 真实 tied Embedding 固定 Token 清单：

```bat
python model_tools\embedding_fixed_reference.py ^
  --token-id 0 ^
  --manifest model_tools\embedding_k896_reference.json
```

运行真实 P50 Embedding 行、512 B 载荷和 1000 个随机 Token 软件自检：

```bat
python tools\pangu_embedding_k896_host.py selftest ^
  --rounds 1000 --seed 20260728
```

E3 使用 `model.embed_tokens.weight[151936,896]`。每个 Token 行固定为 448 B packed INT4、14 个 UQ4.28 scale 和 8 B padding；`row_base_ctrl_addr=token_id<<7`。每个元素执行 signed INT4 × unsigned UQ4.28，经 RNE 右移 18 位后得到 signed Q6.10 int16。完整协议、地址布局和上板证据见 `embedding_k896/README.md`。

## 7. 真实 Linear 量化与定点定义

### 7.1 激活格式

统一采用逐向量对称 INT8：

```text
qmin/qmax = -127 / 127
zero_point = 0
activation_scale = max(abs(x)) / 127
q_x = saturate[-127,127](round_rne(x / activation_scale))
```

全零向量使用 `activation_scale=1.0`，仍可精确表示。所有舍入统一为 round-to-nearest-even（RNE）。

### 7.2 分组缩放

每个输出行、每个 64 元素 group 先计算：

```text
acc[row,group] = sum(q_weight_int4 * q_activation_int8)
```

主机将激活 scale 和 `.p50` FP16 weight scale 合并：

```text
combined_scale = activation_scale * weight_scale[row,group]
combined_scale_q28 = saturate_u32(round_rne(combined_scale * 2^28))
```

`combined_scale_q28` 使用 32 位无符号 `UQ4.28`，范围 `[0, 16)`，无需 FPGA 解析 FP16。

### 7.3 输出格式

bias 也转换为带 28 位小数的有符号整数。FPGA 数据通路定义为：

```text
output_q28[row] = bias_q28[row]
                + sum(acc[row,group] * combined_scale_q28[row,group])
output_float = output_q28 / 2^28
```

分组点积使用 INT32，乘法和跨组累加使用有符号 INT64。由于组合 scale 的量化误差不超过半个 LSB，逐行理论误差上界为：

```text
fixed_error_bound = (sum(abs(acc)) + 1) * 0.5 / 2^28
```

## 8. 验证结果

2026-07-23 在真实 251.63 MiB 镜像上完成：

- 固定头、版本、flags、reserved 和数据区边界检查通过；
- 290 个张量名称唯一；
- 所有 shape、padded columns、group 数、数据长度和 scale 长度均可正确推导；
- 所有数据偏移满足 4 KiB 对齐，所有 scale 偏移满足 64 字节对齐；
- 所有数据范围均在镜像内且互不重叠；
- 外部 JSON 与镜像内嵌 JSON 逐字段完全一致；
- INT4 行、跨 group 数据块和 FP16 行提取通过；
- 独立微型镜像单元测试 5/5 PASS；
- 对照原 BF16 模型和已合并 LoRA 的 4 组抽样反量化误差全部位于理论半 scale 舍入上限内。

此阶段只确认真实模型文件格式和软件提取能力，没有修改 FPGA GEMV RTL、PDS 工程或已验证位流。

2026-07-23 继续完成真实 Linear 量化软件参考：

- 真实张量：`model.layers.0.self_attn.q_proj.weight` 与 bias；
- 切片：输出行 `0..3`，完整输入列 `0..895`，即 M=4、K=896、14 个 group；
- 固定激活：32 位 LCG，seed=`20260723`，输入值为 `1/8192` 的整数倍；
- 激活 scale：`0.0314826064222441`，INT8 饱和数为 0；
- 组合 scale 范围：`0.0001496403793 .. 0.0004270635545`，UQ4.28 饱和数为 0；
- P50 浮点基线：`[0.7752590203, -0.6386315781, 1.0810645018, -0.8347725510]`；
- 量化激活浮点参考：`[0.7720806824, -0.6458171611, 1.0714217223, -0.8315785984]`；
- 定点 Q28：`[207253689, -173360554, 287606739, -223225713]`；
- 定点反量化：`[0.7720801570, -0.6458183900, 1.0714185946, -0.8315805830]`；
- 激活量化最大绝对误差：`0.0096427795`；
- UQ4.28 最大绝对误差：`3.1277186e-6`，理论上界 `3.8200990e-5`；
- 原有解析测试与新增测试共 13/13 PASS；
- 随机软件压力测试：1000/1000 PASS，seed=`20260723`；
- 固定向量清单：`q_proj_m4k896_reference.json`，记录关键数组 SHA256；
- 本轮仍未修改 FPGA RTL、PDS 工程或任何已验证位流。

2026-07-24 完成 layer0 `q_proj` 完整 Linear 软件参考与硬件载荷：

- 完整形状：M=896、K=896、每行 14 个 group；
- 固定上传载荷：488320 B；
- 固定输出 SHA256：`ea1f04bf4ff313dad07025ff35e66a088f13afd28d817422b89bb135f63525a0`；
- 前 4 行与已验证 M4K896 小闭环逐位一致；
- 完整层固定载荷打包/解包和独立 Q28 重算通过；
- 1000 组不同激活软件压力测试全部通过，seed 起点=`20260725`，耗时约 25.88 秒；
- 固定清单：`q_proj_full_reference.json`。

2026-07-24 完成 layer0 `input_layernorm` K=896 定点软件参考：

- 真实张量：`model.layers.0.input_layernorm.weight`，连续 FP16、长度 896；
- 模型 `rms_norm_eps=1e-6`，Q12.20 中量化为 `1`；
- 输入、gamma 和输出：signed Q6.10 int16；平方和：40 位；rsqrt：UQ12.20 uint32；
- 所有转换、除法和右移使用 RNE，输出显式饱和；
- 比较 256 项中点 LUT 与 32 项种子 LUT + 一次 Newton-Raphson；第一版选择 LUT256；
- 固定向量 `sum_squares=5176164753`、`variance_q20=5776971`、`lut_rsqrt_q20=446797`；
- 固定输出 SHA256：`1f52890780e0f4cc0f734d47a4e3bdb28c3c964b8734b442d7781d4ca155a4f0`；
- 相关单元测试与既有回归合计 23/23 PASS；
- RMSNorm 软件随机压力：1000/1000 PASS，seed=`20260726`；
- 固定清单：`rmsnorm_layer0_reference.json`。

2026-07-24 建立 E2 K=896 元素级定点软件参考：

- 输入 A/B、标量 scale 和输出统一为 signed Q6.10 int16；
- 残差使用扩展加法和显式 signed int16 饱和；
- 定点缩放和元素乘法使用 signed Q12.20 乘积、RNE 右移 10 位和显式饱和；
- 在完整 65536 个 int16 输入上比较 2048 项中点 LUT 和 64 段端点 PWL；
- LUT2048 最大误差 5 Q10 LSB、平均误差 0.352692 LSB、表容量 32768 bit；
- PWL64 最大误差 4 Q10 LSB、平均误差 0.232300 LSB、端点表容量 1040 bit；
- 第一版选择 PWL64，覆盖 `[-8,8)`，尾部采用 0/x 规则；
- E2 相关单元测试 11/11 PASS；
- 完整 `model_tools` 回归 34/34 PASS；
- 软件和上传载荷随机压力 1000/1000 PASS，seed=`20260727`;
- 固定清单：`elementwise_k896_reference.json`。

2026-07-24 建立 E3 真实 tied Embedding 定点软件参考：

- 真实张量：`model.embed_tokens.weight`，shape=`[151936,896]`，group size=64，每行 14 groups；
- Token ID 有效范围：`0..151935`，DDR3 控制器行地址为 `token_id<<7`；
- 每行 512 B：448 B packed signed INT4、56 B UQ4.28 scale、8 B padding；
- 全部真实 FP16 embedding scales 均可被 UQ4.28 精确表示；
- 输出执行 RNE 右移 18 位并显式饱和为 signed Q6.10 int16；
- 固定 Token `[0,1,2026,151935]` 的固定路径与直接 Q10 逐位一致；
- E3 单元测试 11/11 PASS；完整 `model_tools` 回归 45/45 PASS；
- 真实随机 Token 软件/载荷压力 1000/1000 PASS，seed=`20260728`；
- 最大 Q6.10 量化误差 `0.00048828125`，未发生输出饱和；
- 固定清单：`embedding_k896_reference.json`。
